"""The trainer: pure-python parts run everywhere, the torch parts only where torch is installed."""

import json
import random
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pytest

from haishool import student
from haishool.schema import Record
from haishool.student import (CLASSIC_KINDS, CTX, EOS, HEADS, KIND_WORDS, LAYERS, SIMS, SPECIAL, WIDTH, Vocab, corpus,
                              corpus_ids, drop_long, drop_not_dense, extra_kind, grown_state, load_extra, load_records,
                              split_extra, tokens, verbs_of)

ROOT = Path(__file__).resolve().parents[1]
HOPS = ROOT / "data" / "hops-v4b" / "hops-train-r4.txt"
SMALL = [ROOT / "data" / "manual.jsonl", ROOT / "data" / "identity.jsonl"]
BIG = [ROOT / "data" / "records-r3-all.jsonl", ROOT / "data" / "records-300.jsonl"]


# ---- the trainer as it was before round 5, copied verbatim, so the unchanged paths are compared, not assumed


def old_vocab_itos(words):
    return SPECIAL + sorted(set(words) - set(SPECIAL))


def old_corpus(records, passes, seed, extra=None):
    samples = [r.line() for r in records] + [q for r in records for q in r.queries()] + list(extra or [])
    rng = random.Random(seed)
    stream = []
    for _ in range(passes):
        rng.shuffle(samples)
        for s in samples:
            stream.extend(tokens(s))
            stream.append(EOS)
    return stream


def old_hop_kind(line):
    w = line.split(".")[0].split()
    return w[2] if len(w) == 4 and w[0] == "q" and w[2] in ("hop", "hops") else ""


def old_split_extra(lines, holdout, seed):
    if holdout <= 0:
        return lines, []
    rng = random.Random(seed + 7)
    pairs = sorted({tuple(ln.split(".")[0].split()[1::2]) for ln in lines if old_hop_kind(ln)})
    held_pairs = {p for p in pairs if rng.random() < holdout}
    train, held = [], []
    for ln in lines:
        kind = old_hop_kind(ln)
        (held if kind and tuple(ln.split(".")[0].split()[1::2]) in held_pairs else train).append(ln)
    return train, held


def old_extra_kind(line):
    if not line.startswith("q "):
        return ""
    prompt = line.split(". a ")[0]
    if old_hop_kind(line):
        return old_hop_kind(line)
    if len(prompt.split()) == 4:
        return "yesno"
    return "view" if " according_to " in prompt else "relation"


# ---- fixtures


@pytest.fixture(scope="module")
def records() -> list[Record]:
    recs = [r for p in SMALL if p.exists() for r in load_records(p)]
    big = next((p for p in BIG if p.exists()), None)
    if big:
        recs += load_records(big)[:40]
    assert len(recs) >= 20
    return recs


@pytest.fixture
def records_path(tmp_path, records) -> Path:
    path = tmp_path / "records.jsonl"
    path.write_text("".join(json.dumps({"obj": r.obj, "values": r.values}) + "\n" for r in records), encoding="utf-8")
    return path


HOPS_LIKE = [
    "q spoon hop fork. a spoon made_of metal material_of fork.", "q spoon hops fork. a 2.",
    "q fork hop knife. a fork made_of metal material_of knife.", "q fork hops knife. a 2.",
    "q spoon made_of. a metal.", "q spoon made_of metal. a yes.", "q spoon made_of wood. a no.",
    "q jerusalem capital_of. a israel palestine contested.",
    "q jerusalem capital_of according_to palestine. a palestine.",
    "spoon links. made_of metal. found_in kitchen.",
]
TRUTH_LIKE = [
    "q calc 1 2 plus 7. a 1 9.", "q calc 3 times 4. a 1 2.", "q solve 3 x plus 4 equals 1 9. a x 5.",
    "q check 1 2 plus 7 equals 2 0. a no.", "q compare 5 and 7. a smaller.", "q steps 2 x equals 8. a x 4.",
    "q balance h 2 plus o 2. a 2 h 2 o 1.", "q judge 2 plus 2 equals 4. a yes.",
    "q carbon protons. a 6.", "q water molar_mass. a 1 8 point 0 1 5.",
    "carbon element. number 6. symbol c. protons 6.",
    "q gravity seed 7 step 2 0 clumps. a 4.", "q gravity seed 7 step 2 0 next clumps. a 5.",
    "q gravity seed 7 final clumps. a 6.", "gravity seed 7 step 2 0. radius 3 point 2. clumps 4.",
]


# ---- pure python


def test_tokens_split_on_space_and_dot():
    assert tokens("q spoon color. a silver.") == ["q", "spoon", "color", ".", "a", "silver", "."]
    assert tokens("carbon element. mass 1 2 point 0 1 1.") == ["carbon", "element", ".", "mass", "1", "2", "point", "0", "1", "1", "."]


def test_vocab_order_is_unchanged(records):
    words = [w for r in records for w in tokens(r.line())] + [w for r in records for q in r.queries() for w in tokens(q)]
    v = Vocab(words)
    assert v.itos == old_vocab_itos(words)
    assert v.itos[:5] == SPECIAL
    assert v.decode(v.encode(tokens("q spoon color. a silver."))) == tokens("q spoon color. a silver.")


def test_corpus_is_unchanged_without_the_new_paths(records):
    assert corpus(records, passes=3, seed=5) == old_corpus(records, passes=3, seed=5)
    assert corpus(records, passes=2, seed=9, extra=HOPS_LIKE) == old_corpus(records, passes=2, seed=9, extra=HOPS_LIKE)
    assert corpus(records, passes=2, seed=9, extra=HOPS_LIKE) == corpus(records, passes=2, seed=9, extra=HOPS_LIKE)
    assert corpus(records, passes=2, seed=9) != corpus(records, passes=2, seed=10)


def test_corpus_ids_is_the_corpus_encoded(records):
    extra = HOPS_LIKE + TRUTH_LIKE
    words = [w for r in records for w in tokens(r.line())] + [w for r in records for q in r.queries() for w in tokens(q)]
    full = Vocab(words + [w for ln in extra for w in tokens(ln)])
    for vocab in (full, Vocab(words), Vocab(words[:200])):  # the last two lack words, which encode() leaves out
        for passes, seed, lines in ((3, 5, extra), (2, 9, None), (1, 20261001, HOPS_LIKE)):
            ids = corpus_ids(records, passes, seed, vocab, lines)
            assert ids.itemsize == 4 and ids.typecode == "i"
            assert list(ids) == vocab.encode(old_corpus(records, passes, seed, lines))
    assert list(corpus_ids(records, 2, 7, full, extra)) == list(corpus_ids(records, 2, 7, full, extra))
    assert list(corpus_ids(records, 2, 7, full, extra)) != list(corpus_ids(records, 2, 8, full, extra))
    eos = full.stoi[EOS]
    ids = list(corpus_ids(records, 1, 1, full, extra))
    assert ids[-1] == eos and ids.count(eos) == len(records) + sum(len(r.queries()) for r in records) + len(extra)


def test_vocab_grows_by_appending_new_words():
    old = Vocab(["zeta", "a", "beta", "q", "."])
    new = Vocab.grown(old.itos, ["beta", "alpha", "new_word", ".", "q", "1"])
    assert new.itos[:len(old.itos)] == old.itos
    assert new.itos[len(old.itos):] == ["1", "alpha", "new_word"]
    prompt = tokens("q beta zeta. a")
    assert new.encode(prompt) == old.encode(prompt)
    assert new.encode(["alpha"]) == [len(old.itos) + 1]
    # a checkpoint that somehow lacks a special token gets it appended, never inserted
    odd = Vocab.grown(["x", "y"], ["z"])
    assert odd.itos[:2] == ["x", "y"] and set(odd.itos) == {"x", "y", "z", *SPECIAL}
    assert Vocab.grown(old.itos, old.itos).itos == old.itos


def test_default_sizes_are_todays_model():
    assert (LAYERS, WIDTH, HEADS, CTX) == (6, 384, 8, 96)


def test_split_extra_hop_pairs_as_before():
    rng = random.Random(3)
    nodes = [f"n{i}" for i in range(40)]
    lines = []
    for x, y in sorted({tuple(rng.sample(nodes, 2)) for _ in range(150)}):
        lines += [f"q {x} hop {y}. a {x} borders {y}.", f"q {x} hops {y}. a 1."]
    lines += [f"q {x} borders. a {nodes[0]} {nodes[1]}." for x in nodes]
    lines += [f"q {x} borders {y}. a yes." for x in nodes for y in nodes[:2] if x != y]
    lines += [f"q {x} capital_of according_to {nodes[0]}. a {nodes[1]}." for x in nodes[2:9]]
    lines += [f"{x} links. borders {nodes[0]}." for x in nodes]
    for seed in (1, 2, 20261001):
        train, held = split_extra(lines, 0.3, seed)
        assert (train, held) == old_split_extra(lines, 0.3, seed), "a round-4 file splits as in round 4"
        assert sorted(train + held) == sorted(lines) and held
        assert all(old_hop_kind(ln) for ln in held)
        pairs = Counter(tuple(ln.split(".")[0].split()[1::2]) for ln in held)
        assert all(n == 2 for n in pairs.values()), "hop and hops of a pair go together"
    # the same hop pairs are held when round-5 lines share the file
    mixed = lines + TRUTH_LIKE
    assert [ln for ln in split_extra(mixed, 0.3, 1)[1] if old_hop_kind(ln)] == old_split_extra(lines, 0.3, 1)[1]
    assert split_extra(lines, 0.0, 1) == (lines, [])


@pytest.mark.skipif(not HOPS.exists(), reason="round-4 hops file not here")
def test_split_extra_on_the_round_4_file_is_unchanged():
    lines = load_extra(HOPS)
    train, held = split_extra(lines, 0.1, 20261001)
    assert (train, held) == old_split_extra(lines, 0.1, 20261001)
    assert 0.08 < len(held) / sum(1 for ln in lines if old_hop_kind(ln)) < 0.12


def test_split_extra_holds_out_prompts_but_never_records():
    lines = [f"q calc {i} plus 1. a {i + 1}." for i in range(300)] + [f"q calc {i} plus 1. a {i + 1}." for i in range(0, 300, 10)]
    lines += [f"q e{i} protons. a {i}." for i in range(100)] + [f"e{i} element. number {i}." for i in range(100)]
    train, held = split_extra(lines, 0.3, 11)
    assert train + held and sorted(train + held) == sorted(lines)
    assert all(ln.startswith("q ") for ln in held), "record lines are always trained"
    held_prompts = {ln.split(". a ")[0] for ln in held}
    assert not any(ln.split(". a ")[0] in held_prompts for ln in train if ln.startswith("q ")), "a held prompt leaves training entirely"
    assert 0.2 < len(held_prompts) / 400 < 0.4
    assert split_extra(lines, 0.3, 11) == (train, held)
    assert split_extra(lines, 0.3, 12) != (train, held)
    assert split_extra(lines, 0.0, 11) == (lines, [])


def test_split_extra_does_not_depend_on_the_other_lines():
    first = [f"q calc {i} plus 1. a {i + 1}." for i in range(300)] + [f"q e{i} protons. a {i}." for i in range(100)]
    later = [f"q gravity seed {i} final clumps. a {i % 7}." for i in range(200)] + [f"e{i} element. number {i}." for i in range(100)]
    held_first = set(split_extra(first, 0.3, 4)[1])
    train, held = split_extra(later[:150] + first[::-1] + later[150:] + HOPS_LIKE, 0.3, 4)
    assert set(held) & set(first) == held_first, "a question held out once stays held out when files are added"
    assert not held_first & set(train)
    # the same on every machine: the draw is a hash of seed and prompt, not Python's hash()
    assert student._draw(4, "q calc 1 2 plus 7") == student._draw(4, "q calc 1 2 plus 7") == 0.005828981773594298
    assert 0 <= student._draw(5, "q calc 1 2 plus 7") < 1 and student._draw(5, "q calc 1 2 plus 7") != student._draw(4, "q calc 1 2 plus 7")


def test_split_extra_keeps_judge_lines_with_their_question():
    lines = []
    for i in range(400):
        lines += [f"q calc {i} plus 1. a {i + 1}.", f"q judge calc {i} plus 1 answer {i + 1}. a right.",
                  f"q judge calc {i} plus 1 answer {i + 2}. a wrong.", f"q judge calc {i} plus 1 answer. a wrong."]
    lines += [f"q judge calc {i} times 2 answer {2 * i}. a right." for i in range(100)]  # judged, never asked
    assert student._held_key("q judge calc 1 2 plus 7 answer 2 0. a wrong.") == "q calc 1 2 plus 7"
    assert student._held_key("q judge calc 1 2 plus 7 answer. a wrong.") == "q calc 1 2 plus 7"
    assert student._held_key("q calc 1 2 plus 7. a 1 9.") == "q calc 1 2 plus 7"
    assert student._held_key("q judge answer 3. a wrong.") == "q judge answer 3"  # no question: its own prompt
    train, held = split_extra(lines, 0.25, 6)
    assert sorted(train + held) == sorted(lines) and train and held
    questions = [{student._held_key(ln) for ln in side} for side in (train, held)]
    assert not questions[0] & questions[1], "a trained judge line would give a held-out answer away"
    assert 0.15 < len(questions[1]) / 500 < 0.35


@pytest.mark.parametrize("line,kind", [
    ("q turkey borders. a greece syria.", "relation"),
    ("q turkey borders greece. a yes.", "yesno"),
    ("q islam has_prophet according_to islam. a jesus moses muhammad.", "view"),
    ("q jerusalem capital_of palestine. a according_to palestine.", "yesno"),
    ("q turkey hop led_zeppelin. a turkey has_city istanbul.", "hop"),
    ("q turkey hops led_zeppelin. a 4.", "hops"),
    ("q force related_to mass. a yes.", "yesno"),
    ("q force is_a. a concept.", "relation"),
    ("q calc 1 2 plus 7. a 1 9.", "calc"),
    ("q solve 3 x plus 4 equals 1 9. a x 5.", "solve"),
    ("q check 1 2 plus 7 equals 2 0. a no.", "check"),
    ("q balance h 2 plus o 2. a 2 h 2 o 1.", "balance"),
    ("q compare 5 and 7. a smaller.", "compare"),
    ("q steps 2 x equals 8. a x 4.", "steps"),
    ("q judge 2 plus 2 equals 4. a yes.", "judge"),
    ("q heavier dysprosium francium. a francium.", "heavier"),
    ("q count_atoms 9 x h 2 o 2 hydrogen. a 1 8.", "count_atoms"),
    ("q lighter is_a. a tool.", "relation"),
    ("q carbon protons. a 6.", "fact"),
    ("q water molar_mass. a 1 8 point 0 1 5.", "fact"),
    ("q gravity seed 7 step 2 0 clumps. a 4.", "gravity"),
    ("q nucleo seed 1 2 final helium_fraction. a 0 point 2 5.", "nucleo"),
    # the verbs of the gates as they are written
    ("q calc 4 7 plus 3 8 steps. a 7 plus 8 is 1 5 write 5 carry 1.", "calc"),
    ("q judge calc 1 2 plus 7 answer 2 0. a wrong.", "judge"),
    ("q judge calc 1 2 plus 7 answer. a wrong.", "judge"),
    ("q element symbol fe. a iron.", "element"),
    ("q elements period 1. a hydrogen helium.", "elements"),
    ("q count elements group 3. a 4.", "count"),
    ("q higher density_g_cm3 chlorine hydrogen. a chlorine.", "higher"),
    ("q lower electronegativity cobalt gallium. a gallium.", "lower"),
    ("q neutrons mass_number 2 6 protons 1 9. a 7.", "neutrons"),
    ("q atoms c 6 h 1 2 o 6. a 2 4.", "atoms"),
    ("q molar_mass w 7 co 7. a 1 6 9 9 point 4.", "molar_mass"),
    ("q mass_percent water oxygen. a 8 8 point 8 1.", "mass_percent"),
    ("q element_count water hydrogen. a 2.", "element_count"),
    ("q heavier_molecule water methane. a water.", "heavier_molecule"),
    ("q substance formula h 2 o 1. a water.", "substance"),
    ("q substances class sugar found_in animals. a galactose lactose.", "substances"),
    ("q reactions where airbags. a sodium_azide_decomposition.", "reactions"),
    ("q forces described_by newton. a gravity normal tension.", "forces"),
    ("q stronger weak strong. a strong.", "stronger"),
    ("q longer_range strong weak. a strong.", "longer_range"),
    ("q weight m 5. a 4 9 point 0 5.", "weight"),
    ("q friction mu 0 point 3 n 5 0. a 1 5.", "friction"),
    ("q coulomb_force q 1 e minus 6 q 2 e minus 6 r 0 point 1. a 1 point 7 9 8.", "coulomb_force"),
    ("q orbital_speed m 5 point 9 7 e 2 4 r 6 point 7 7 e 6. a 7 6 7 2.", "orbital_speed"),
    # a thing and one of its keys is a fact, also when the thing shares its name with a verb or a simulation
    ("q friction unit. a newton.", "fact"),
    ("q drag formula. a half rho v 2 c_d a.", "fact"),
    ("q gravity acts_at_distance. a yes.", "fact"),
    ("q strong range. a 1 e minus 1 5.", "fact"),
    ("q air part oxygen. a 2 0 point 9.", "fact"),
    ("q calcium_oxide plus water gives. a calcium_hydroxide.", "fact"),
    ("q methane_combustion products. a carbon_dioxide water.", "fact"),
    # a round-5 key that is also a relation name keeps the round-4 bucket
    ("q water found_in. a ocean rivers atmosphere.", "relation"),
    ("q substances found_in core. a iron nickel.", "yesno"),
    # round 6: the simulation, with or without a seed
    ("q chem seed 4 2 step 1 2 next temperature. a 2 0 2 6.", "chem"),
    ("q chem valence c. a 4.", "chem"),
    ("q chem bond h he stable_below_k. a never.", "chem"),
    ("q life seed 7 param mu. a 0 point 0 2.", "life"),
    ("q planets seed 3 final habitable. a 1.", "planets"),
    ("q world seed 3 era planets rocky. a 2.", "world"),
    ("q newsim seed 3 final x. a 2.", "newsim"),
    ("carbon element. number 6. symbol c.", ""),
    ("turkey links. borders greece syria.", ""),
    ("gravity seed 7 step 2 0. clumps 4.", ""),
])
def test_extra_kind(line, kind):
    assert extra_kind(line) == kind


def test_importing_the_trainer_does_not_load_the_graph_code():
    # the chat imports load/generate from here and never needed haishool.relations (numpy)
    code = "import sys, haishool.student; sys.exit('haishool.relations' in sys.modules)"
    assert subprocess.run([sys.executable, "-c", code], cwd=ROOT).returncode == 0


def test_kind_words_cover_the_round_5_verbs():
    assert {"calc", "solve", "compare", "steps", "check", "balance", "judge"} <= KIND_WORDS
    assert not set(CLASSIC_KINDS) & KIND_WORDS and "fact" not in KIND_WORDS
    assert {"nucleo", "gravity", "planets", "chem", "life", "world"} <= SIMS
    assert not SIMS & KIND_WORDS and not SIMS & set(CLASSIC_KINDS) and "fact" not in SIMS


def test_a_file_shows_its_own_verbs():
    things = [f"q thing{i} part water. a {i}." for i in range(200)] + [f"q iron plus thing{i} gives. a rust." for i in range(29)]
    new = [f"q momentum m {i} v 2. a {2 * i}." for i in range(30)]
    twice = [f"q impulse f {i % 29} t 2. a {2 * (i % 29)}." for i in range(60)]  # 29 different prompts
    lines = things + new + twice + HOPS_LIKE + TRUTH_LIKE + [f"q thing{i} borders thing{i + 1}. a yes." for i in range(50)]
    lines += [f"q momentum hop thing{i}. a momentum borders thing{i}." for i in range(40)] + [f"momentum m {i} v 2." for i in range(40)]
    verbs = verbs_of(lines)
    assert verbs == KIND_WORDS | {"momentum"}, "a verb starts many different long questions, a thing a handful"
    assert extra_kind(new[0], verbs) == "momentum" and extra_kind(new[0]) == "fact"
    assert extra_kind(things[0], verbs) == extra_kind(things[-1], verbs) == extra_kind(twice[0], verbs) == "fact"
    assert extra_kind("q momentum unit. a kilogram_metre_per_second.", verbs) == "fact"
    assert verbs_of(HOPS_LIKE * 40) == KIND_WORDS and verbs_of([]) == KIND_WORDS
    assert verbs_of(lines, least=29) == KIND_WORDS | {"momentum", "impulse", "iron"}
    for ln in HOPS_LIKE + TRUTH_LIKE:  # the buckets of the known lines do not move
        assert extra_kind(ln, verbs) == extra_kind(ln)


def test_gate_questions_all_get_a_bucket():
    """Whatever the gates of round 5 write today: a question has a bucket, a record has none."""
    loop = pytest.importorskip("haishool.truth.loop")
    for gate in loop.load_gates()[0]:
        try:
            lines = list(gate.records()) + list(gate.generate(random.Random(5), 1500))
        except Exception:  # a gate that is being rewritten fails its own tests, not the trainer's
            continue
        verbs = verbs_of([ln.text for ln in lines])
        for ln in lines:
            assert bool(extra_kind(ln.text, verbs)) == (ln.kind != "record"), ln.text
            assert extra_kind(ln.text, verbs) in ("", "fact", ln.text.split()[1], *CLASSIC_KINDS), ln.text


@pytest.mark.skipif(not HOPS.exists(), reason="round-4 hops file not here")
def test_extra_kind_keeps_the_round_4_buckets():
    lines = load_extra(HOPS)
    assert Counter(map(extra_kind, lines)) == Counter(map(old_extra_kind, lines))
    assert max(len(tokens(ln)) for ln in lines) <= CTX


def test_drop_long_lines():
    short = "q a b. a " + " ".join(["1"] * 90) + "."  # 96 tokens
    long = "q a b. a " + " ".join(["1"] * 91) + "."  # 97 tokens
    assert len(tokens(short)) == 96 and len(tokens(long)) == 97
    assert drop_long([short, long, "x. y z."], 96) == ([short, "x. y z."], [long])
    assert drop_long([short, long], 1000) == ([short, long], [])


def test_drop_not_dense_lines():
    good = ["q calc 1 2 plus 7. a 1 9.", "carbon element. number 6. symbol c.", "q tin econf. a kr 4d10 5s2 5p2."]
    bad = ["Q calc 1 2 plus 7. a 1 9.", "q calc 1+1. a 2.", "q calc 1 2 plus 7. a 1 9", "q calc  1. a 1.", "q water mass. a 18,015.",
           "q a. a <eos>.", '{"obj": "spoon"}', "."]
    assert drop_not_dense(good + bad) == (good, bad)
    assert drop_not_dense(bad[:1] + good) == (good, bad[:1])
    assert drop_not_dense(HOPS_LIKE + TRUTH_LIKE) == (HOPS_LIKE + TRUTH_LIKE, [])


def test_load_extra_strips_blank_lines_and_a_byte_order_mark(tmp_path):
    path = tmp_path / "bom.txt"
    path.write_bytes(b"\xef\xbb\xbfq calc 1 plus 1. a 2.\r\n\r\n  carbon element. number 6.  \r\n")
    assert load_extra(path) == ["q calc 1 plus 1. a 2.", "carbon element. number 6."]
    path.write_text("q calc 1 plus 1. a 2.\n", encoding="utf-8")
    assert load_extra(path) == ["q calc 1 plus 1. a 2."] and load_extra(None) == []


@pytest.mark.skipif(student.torch is not None, reason="torch is installed here")
def test_without_torch_the_error_says_so(tmp_path):
    with pytest.raises(RuntimeError, match="torch is not installed"):
        student.load(tmp_path / "student.pt")
    with pytest.raises(RuntimeError, match="torch is not installed"):
        student.train(tmp_path / "records.jsonl", tmp_path / "out", 0.0, 1, 1, "cpu")


def test_grown_state_copies_old_rows_and_keeps_fresh_new_ones():
    np = pytest.importorskip("numpy")
    old = {"wte.weight": np.arange(12.0).reshape(4, 3), "lm_head.weight": np.arange(12.0).reshape(4, 3),
           "wpe.weight": np.ones((2, 3)), "blocks.0.w": np.full((2, 2), 7.0)}
    fresh = {"wte.weight": np.full((6, 3), -1.0), "lm_head.weight": np.full((6, 3), -1.0),
             "wpe.weight": np.zeros((5, 3)), "blocks.0.w": np.zeros((2, 2))}
    out = grown_state(old, fresh)
    for key in ("wte.weight", "lm_head.weight"):
        assert (out[key][:4] == old[key]).all() and (out[key][4:] == -1.0).all()
    assert (out["wpe.weight"][:2] == 1.0).all() and (out["wpe.weight"][2:] == 0.0).all()
    assert out["blocks.0.w"] is old["blocks.0.w"]
    with pytest.raises(ValueError):  # a vocabulary never shrinks
        grown_state({"wte.weight": np.zeros((8, 3))}, {"wte.weight": np.zeros((6, 3))})
    with pytest.raises(ValueError):  # a different width is a different model
        grown_state({"wte.weight": np.zeros((4, 3))}, {"wte.weight": np.zeros((6, 4))})
    with pytest.raises(KeyError, match="has no 'wte.weight'"):
        grown_state({}, {"wte.weight": np.zeros((6, 3))})
    with pytest.raises(KeyError, match="'blocks.6.w', which this model lacks"):  # a deeper checkpoint is not cut down silently
        grown_state({"wte.weight": np.zeros((6, 3)), "blocks.6.w": np.zeros((2, 2))}, {"wte.weight": np.zeros((6, 3))})


def test_cli_flags_reach_train(monkeypatch):
    got = {}
    monkeypatch.setattr(student, "train", lambda *a, **kw: got.update({"args": a, "kw": kw}))
    monkeypatch.setattr(sys, "argv", ["student", "train", "--records", "r.jsonl", "--out", "o", "--holdout", "0.1",
                                      "--seed", "3", "--steps", "7", "--device", "cpu", "--extra-lines", "a.txt",
                                      "--extra-lines", "b.txt", "--init", "c.pt", "--lr", "0.0005", "--layers", "2",
                                      "--width", "64", "--heads", "2", "--ctx", "64", "--eval-sample", "9", "--extra-eval", "4"])
    student.main()
    assert got["args"] == (Path("r.jsonl"), Path("o"), 0.1, 3, 7, "cpu", [Path("a.txt"), Path("b.txt")])
    assert got["kw"] == {"init": Path("c.pt"), "lr": 0.0005, "layers": 2, "width": 64, "heads": 2, "ctx": 64,
                         "eval_sample": 9, "extra_eval": 4}
    monkeypatch.setattr(sys, "argv", ["student", "train", "--records", "r.jsonl", "--out", "o", "--device", "cpu"])
    student.main()
    assert got["args"] == (Path("r.jsonl"), Path("o"), 0.0, 20261001, 4000, "cpu", None)
    assert got["kw"] == {"init": None, "lr": None, "layers": None, "width": None, "heads": None, "ctx": None,
                         "eval_sample": 1000, "extra_eval": 500}


# ---- torch


def _small_train(records_path, out, extra, **kw):
    args = dict(holdout=0.0, seed=1, steps=3, device="cpu", eval_sample=5, extra_eval=5)
    if "init" not in kw:
        args.update(layers=1, width=32, heads=2, ctx=48)
    args.update(kw)
    return student.train(records_path, out, args.pop("holdout"), args.pop("seed"), args.pop("steps"), args.pop("device"),
                         extra, **args)


def first_keys(report):
    return set(report["extra_files"][0])


def test_generate_puts_eos_before_the_prompt():
    torch = pytest.importorskip("torch")
    from haishool.model import GPT, GPTConfig
    vocab = Vocab(tokens("q spoon color. a silver."))
    torch.manual_seed(0)
    model = GPT(GPTConfig(vocab_size=len(vocab.itos), n_layer=1, n_head=2, d_model=16, ctx=16)).eval()
    seen = []
    forward = model.forward
    model.forward = lambda idx, targets=None: (seen.append(idx.clone()), forward(idx, targets))[1]
    out = student.generate(model, vocab, "q spoon color. a", max_new=2)
    assert isinstance(out, str) and seen
    assert seen[0][0, 0].item() == vocab.stoi[EOS]
    assert seen[0][0, 1:].tolist() == vocab.encode(tokens("q spoon color. a"))
    assert student.generate(model, vocab, "unknown words only") == ""


def test_grown_model_keeps_old_logits():
    torch = pytest.importorskip("torch")
    from haishool.model import GPT, GPTConfig
    torch.manual_seed(0)
    small = GPT(GPTConfig(vocab_size=10, n_layer=1, n_head=2, d_model=16, ctx=8)).eval()
    big = GPT(GPTConfig(vocab_size=14, n_layer=1, n_head=2, d_model=16, ctx=12)).eval()
    big.load_state_dict(grown_state(small.state_dict(), {k: v.clone() for k, v in big.state_dict().items()}))
    assert big.lm_head.weight is big.wte.weight, "the output layer stays tied to the embedding"
    assert torch.equal(big.wte.weight[:10], small.wte.weight) and torch.equal(big.wpe.weight[:8], small.wpe.weight)
    x = torch.tensor([[1, 4, 7, 2, 9]])
    with torch.no_grad():
        a, _ = small(x)
        b, _ = big(x)
    assert torch.allclose(a, b[..., :10], atol=1e-5)


def test_train_from_scratch_then_continue(tmp_path, records_path):
    torch = pytest.importorskip("torch")
    extra_a = tmp_path / "a.txt"
    extra_a.write_text("\n".join(HOPS_LIKE + TRUTH_LIKE[:11]) + "\n", encoding="utf-8")
    report = _small_train(records_path, tmp_path / "run1", [extra_a], holdout=0.2)
    ck = torch.load(tmp_path / "run1" / "student.pt", weights_only=True)
    assert set(ck) == {"model_state", "gpt_config", "itos"}
    assert (ck["gpt_config"]["n_layer"], ck["gpt_config"]["d_model"], ck["gpt_config"]["n_head"], ck["gpt_config"]["ctx"]) == (1, 32, 2, 48)
    assert report["lr"] == 1e-3 and report["dropped_long_lines"] == 0 and "init" not in report
    assert report["dropped_not_dense_lines"] == 0 and first_keys(report) >= {"dropped_long_lines", "dropped_not_dense_lines"}
    assert report["config"] == ck["gpt_config"] and report["vocab"] == len(ck["itos"])
    assert report["recall_trained_pairs"]["n"] == 5 and report["inference_heldout_pairs"]["n"] > 0
    assert report["extra_lines"] is report["extra_files"][0] and len(report["extra_files"]) == 1
    first = report["extra_files"][0]
    assert first["file"] == str(extra_a) and first["lines"] == len(HOPS_LIKE) + 11
    assert first["trained"] + first["held_out"] == first["lines"] and first["dropped_long_lines"] == 0
    for part in ("recall_trained", "inference_heldout"):
        assert tuple(first[part])[:5] == CLASSIC_KINDS
        assert all(set(v) == {"n", "exact"} for v in first[part].values())
    kinds = set(first["recall_trained"]) | set(first["inference_heldout"])
    assert {"calc", "solve", "check", "compare", "steps", "balance", "judge", "fact"} <= kinds
    assert sum(first[p][k]["n"] for p in ("recall_trained", "inference_heldout") for k in first[p]) == first["lines"] - 2
    assert (tmp_path / "run1" / "report.json").exists()

    model, vocab = student.load(tmp_path / "run1" / "student.pt")
    old_itos = list(vocab.itos)
    extra_b = tmp_path / "b.txt"
    long_line = "q calc " + " plus ".join(["1"] * 30) + " zzz_long. a 3 0."  # 66 tokens > ctx 48
    assert len(tokens(long_line)) > 48
    extra_b.write_text("\n".join(TRUTH_LIKE[11:] + [long_line]) + "\n", encoding="utf-8")
    report2 = _small_train(records_path, tmp_path / "run2", [extra_a, extra_b], init=tmp_path / "run1" / "student.pt", seed=2)
    assert report2["lr"] == 3e-4 and report2["dropped_long_lines"] == 1
    assert report2["init"] == {"checkpoint": str(tmp_path / "run1" / "student.pt"), "vocab_old": len(old_itos),
                               "new_words": report2["vocab"] - len(old_itos)}
    assert [f["file"] for f in report2["extra_files"]] == [str(extra_a), str(extra_b)]
    assert report2["extra_files"][1]["dropped_long_lines"] == 1 and report2["extra_files"][1]["lines"] == 5
    assert report2["extra_files"][1]["trained"] == 4 and "gravity" in report2["extra_files"][1]["recall_trained"]
    model2, vocab2 = student.load(tmp_path / "run2" / "student.pt")
    assert vocab2.itos[:len(old_itos)] == old_itos
    new_words = vocab2.itos[len(old_itos):]
    assert new_words == sorted(new_words) and {"gravity", "seed", "clumps"} <= set(new_words)
    assert "zzz_long" not in vocab2.itos and "plus" in vocab2.itos  # a dropped line adds no words
    for prompt in ("q spoon color. a", "q calc 1 2 plus 7. a", "spoon."):
        assert vocab2.encode(tokens(prompt)) == vocab.encode(tokens(prompt))
    assert model2.config.n_layer == 1 and model2.config.ctx == 48 and model2.config.vocab_size >= len(vocab2.itos)
    assert isinstance(student.generate(model2, vocab2, "q spoon color. a"), str)
    assert isinstance(student.answer(model2, vocab2, "spoon", "color"), str)
    assert student.describe(model2, vocab2, "spoon").startswith("spoon. ")

    with pytest.raises(ValueError, match="keeps its shape"):
        _small_train(records_path, tmp_path / "run3", [extra_a], init=tmp_path / "run1" / "student.pt", layers=2)
    with pytest.raises(ValueError, match="may grow, not shrink"):
        _small_train(records_path, tmp_path / "run3", [extra_a], init=tmp_path / "run1" / "student.pt", ctx=32)
    assert not (tmp_path / "run3").exists()
    report4 = _small_train(records_path, tmp_path / "run4", [], init=tmp_path / "run1" / "student.pt", ctx=64, steps=1, lr=0.0)
    model4, vocab4 = student.load(tmp_path / "run4" / "student.pt")
    assert model4.config.ctx == 64 and report4["config"]["ctx"] == 64 and vocab4.itos == old_itos
    assert torch.equal(model4.wpe.weight[:48], model.wpe.weight) and torch.equal(model4.wte.weight, model.wte.weight)
    assert "extra_files" not in report4 and "extra_lines" not in report4


def test_report_gives_long_answers_room_and_counts_dropped_lines(tmp_path, records_path, monkeypatch, capsys):
    pytest.importorskip("torch")
    steps = "q calc 4 7 plus 3 8 steps. a " + " ".join(["7 plus 8 is 1 5 write 5 carry 1"] * 6) + "."  # an answer of 60 tokens
    assert len(steps.partition(". a ")[2].rstrip(".").split()) == 60 and len(tokens(steps)) <= 96
    too_long = "q calc " + " plus ".join(["1"] * 60) + ". a 6 0."
    lines = TRUTH_LIKE + [steps, "Q Calc 1+1. a 2.", too_long, "q water mass. a 18,015."]
    path = tmp_path / "c.txt"
    path.write_bytes(b"\xef\xbb\xbf" + ("\r\n".join(lines) + "\r\n").encode())
    rooms = {}
    real = student.generate
    monkeypatch.setattr(student, "generate", lambda model, vocab, prompt, **kw: (
        rooms.__setitem__(prompt, kw.get("max_new")), real(model, vocab, prompt, **kw))[1])
    report = _small_train(records_path, tmp_path / "run", [path], ctx=96, steps=2, extra_eval=50)
    assert report["dropped_long_lines"] == 1 and report["dropped_not_dense_lines"] == 2
    entry = report["extra_files"][0]
    assert entry["lines"] == len(lines) and entry["trained"] == len(lines) - 3 and entry["held_out"] == 0
    assert entry["dropped_long_lines"] == 1 and entry["dropped_not_dense_lines"] == 2
    err = capsys.readouterr().err
    assert str(path) in err and "dropped 1 lines longer than 96 tokens and 2 not dense" in err
    assert rooms["q calc 4 7 plus 3 8 steps. a"] == 61, "the whole answer and its final dot fit"
    assert rooms["q calc 1 2 plus 7. a"] == 40 and set(rooms.values()) == {None, 40, 61}  # None: answer() for the records
    _, vocab = student.load(tmp_path / "run" / "student.pt")
    assert not {"Q", "Calc", "1+1", "18,015", "\ufeffq"} & set(vocab.itos), "a dropped line adds no words"
    assert all(w == w.strip() and w.isascii() for w in vocab.itos) and {"carry", "write"} <= set(vocab.itos)
    assert sum(v["n"] for v in entry["recall_trained"].values()) == len(lines) - 3 - 2  # minus the two record lines


def test_training_without_new_flags_is_unchanged(tmp_path, records_path):
    torch = pytest.importorskip("torch")
    one = _small_train(records_path, tmp_path / "one", None, seed=4)
    two = _small_train(records_path, tmp_path / "two", [], seed=4)
    a = torch.load(tmp_path / "one" / "student.pt", weights_only=True)
    b = torch.load(tmp_path / "two" / "student.pt", weights_only=True)
    assert a["itos"] == b["itos"] and a["gpt_config"] == b["gpt_config"]
    assert all(torch.equal(a["model_state"][k], b["model_state"][k]) for k in a["model_state"])
    assert {k: v for k, v in one.items() if k != "log"} == {k: v for k, v in two.items() if k != "log"}
    assert one["train_tokens_per_pass"] == len(old_corpus(load_records(records_path), 1, 4)) and one["lr"] == 1e-3
