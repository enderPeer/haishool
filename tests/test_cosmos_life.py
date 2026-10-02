import copy
import hashlib
import json
import random
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest

from haishool.cosmos import Rollout, Simulation, life, state_line
from haishool.student import tokens
from haishool.truth import is_dense, num, parse_num

SIM = life.simulation()
MAX_TOKENS = 80
SEEDS = (1, 2, 3, 7, 11)
#: sha256 (first 16 hex digits) of the whole rollout; a machine that computes anything else
#: must not judge lines written here
DIGESTS = {1: "adb5acdd45499bb1", 7: "936210babc0720fd", 1234: "307c5389a705ed37"}
OTHER_PROMPTS = (
    "turkey capital", "q spoon color", "carbon protons", "water molar_mass", "calc 1 2 plus 7",
    "check 1 2 plus 7 equals 2 0", "solve 3 x plus 4 equals 1 9", "chem valence c",
    "gravity seed 7 step 2 0 clumps", "gravity seed 7 step 1 2 next clumps", "gravity seed 7 final clumps",
    "chem seed 4 2 step 1 2 ch4", "chem seed 4 2 final water_fraction", "chem seed 3 final molecules",
    "nucleo seed 1 step 1 5 helium", "nucleo seed 1 final helium", "planets seed 3 step 1 4 gas",
    "planets seed 3 final habitable", "world seed 3 era planets", "world seed 3 final replicators",
)


def n_tokens(text: str) -> int:
    return len(text.replace(".", " . ").split())


def digest(r: Rollout) -> str:
    return hashlib.sha256(json.dumps([r.params, r.steps, r.summary], sort_keys=True).encode()).hexdigest()[:16]


def strings(seqs: np.ndarray) -> list[str]:
    return ["".join(life.LETTERS[x] for x in row if x != life.PAD) for row in seqs]


def final_replicator_lengths(seed: int, **params) -> np.ndarray:
    p = life._params(seed, params)
    *_, (_, seqs, _, _) = life.simulate(seed, p)
    return (seqs[life.has_motif(seqs, p["gap"])] != life.PAD).sum(1)


@pytest.fixture(scope="module")
def rollouts():
    return {seed: SIM.rollout(seed) for seed in SEEDS}


@pytest.fixture(scope="module")
def generated():
    return SIM.generate(random.Random(1), 600)


@pytest.fixture(scope="module")
def all_lines(rollouts, generated):
    return [ln for r in rollouts.values() for ln in life.lines(r)] + generated


def test_implements_the_protocol():
    assert isinstance(SIM, Simulation)
    assert SIM.sim == "life" and set(SIM.KEYS) == set(life.KEYS)
    assert isinstance(SIM.run(5), Rollout)
    assert life.METRICS == [k for k in life.KEYS if k != "step"]


def test_same_seed_same_rollout():
    a, b = life.run(5), life.run(5)
    assert a.params == b.params and a.steps == b.steps and a.summary == b.summary
    assert [ln.text for ln in life.lines(a)] == [ln.text for ln in life.lines(b)]
    assert life.random_params(random.Random(5)) == life.random_params(random.Random(5))
    assert [ln.text for ln in SIM.generate(random.Random(3), 300)] == [ln.text for ln in SIM.generate(random.Random(3), 300)]
    assert [ln.text for ln in SIM.generate(random.Random(3), 300)] != [ln.text for ln in SIM.generate(random.Random(4), 300)]
    assert life.run(1).steps != life.run(2).steps


def test_forty_seeded_runs_repeat_exactly_and_conserve():
    for seed in range(1, 41):
        a = life.run(seed)
        b = life.run(seed, **life.random_params(random.Random(seed)))
        assert a.params == b.params == life.random_params(random.Random(seed))
        assert a.steps == b.steps and a.summary == b.summary, seed
        assert len(a.steps) == life.STEPS + 1
        for s in a.steps:  # plain python numbers, so equal means identical
            assert all(type(v) in (int, float, str) for v in s.values())
        v = SIM.conserved(a)
        assert v.ok, (seed, v)


def test_the_random_stream_and_three_rollouts_are_pinned():
    # the raw PCG64 stream is what numpy guarantees across versions; everything else is built on it here
    assert np.random.PCG64(7).random_raw(3).tolist() == [11530976094092348043, 16550673365885938325, 14308875409591826786]
    assert life._uniform(np.random.PCG64(7).random_raw, 2).tolist() == [11530976094092348043 // 2048 / 2 ** 53,
                                                                    16550673365885938325 // 2048 / 2 ** 53]
    assert life._LENGTH_CDF.tolist()[:3] == [0.30000000000000004, 0.51, 0.657] and len(life._LENGTH_CDF) == 14
    assert life.random_params(random.Random(7)) == {"monomers": 10305, "inflow": 240, "poly_rate": 0.002,
                                                    "decay": 0.1, "mu": 0.02, "gap": 0}
    assert {seed: digest(life.run(seed)) for seed in DIGESTS} == DIGESTS


def test_a_fresh_interpreter_gives_the_same_rollout():
    code = ("import hashlib, json; from haishool.cosmos import life; r = life.run(7); "
            "print(hashlib.sha256(json.dumps([r.params, r.steps, r.summary], sort_keys=True).encode()).hexdigest()[:16])")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True,
                         cwd=Path(__file__).resolve().parents[1], env={**__import__("os").environ, "PYTHONHASHSEED": "123"})
    assert out.stdout.strip() == DIGESTS[7]


def test_random_params_stay_in_range():
    rng = random.Random(9)
    for _ in range(300):
        p = life.random_params(rng)
        assert set(p) == set(life.PARAM_KEYS)
        assert 5000 <= p["monomers"] <= 20000
        assert p["inflow"] == 0 or (40 <= p["inflow"] <= 400 and p["inflow"] % 4 == 0)
        assert 0.001 <= p["mu"] <= 0.05 and p["gap"] in (0, 1, 2, 3)
        assert p["poly_rate"] in (0.002, 0.005, 0.01, 0.02) and 0.1 <= p["decay"] <= 3.2


def test_bad_parameters_are_refused():
    for bad in ({"inflow": 10}, {"inflow": -4}, {"monomers": -1}, {"monomers": 10.5}, {"mu": 2}, {"mu": -0.1},
                {"poly_rate": 1.5}, {"decay": -1}, {"gap": -1}, {"gap": 13}, {"mutation": 0.1}):
        with pytest.raises(ValueError):
            life.run(1, **bad)
    assert SIM.conserved(life.run(3, monomers=0, inflow=0)).ok
    odd = life.run(3, monomers=3, inflow=4, poly_rate=1, decay=0, mu=1, gap=12)
    assert SIM.conserved(odd).ok and odd.steps[-1]["free_monomers"] + odd.steps[-1]["bound"] == 3 + 4 * life.STEPS
    assert SIM.conserved(life.run(3, monomers=20000, inflow=0, poly_rate=1.0, decay=0.5, mu=0.01, gap=3)).ok


def test_lines_are_dense_and_short(all_lines):
    assert len(all_lines) > 4000
    for ln in all_lines:
        assert is_dense(ln.text), ln.text
        assert n_tokens(ln.text) <= MAX_TOKENS and len(tokens(ln.text)) == n_tokens(ln.text), ln.text
        assert ln.topic == "life"
        if ln.kind != "record":
            assert "." not in ln.prompt and "." not in ln.answer
            assert ln.text == f"q {ln.prompt}. a {ln.answer}."
    assert {ln.kind for ln in all_lines} == {"record", "fact", "calc"}


def test_the_longest_possible_record_fits():
    # five-digit counts everywhere, a share in e notation, six-digit generations, the longest stage name
    worst = {"step": 40, "free_monomers": 18000, "bound": 18000, "polymers": 10000, "replicators": 10000,
             "longest": 16, "mean_length": 3.33, "diversity": 10000, "dominant_share": 0.000123,
             "generations": 360000, "stage": "first_replicator"}
    r = Rollout("life", 9999, {}, [worst] * (life.STEPS + 1))
    line = state_line(r, life.STEPS, life.METRICS)
    assert is_dense(line.text) and n_tokens(line.text) == 75
    params = life.params_line(Rollout("life", 9999, {"monomers": 20000, "inflow": 400, "poly_rate": 0.002,
                                                    "decay": 0.1, "mu": 0.001, "gap": 3}, []))
    assert n_tokens(params.text) <= 50


def test_every_line_kind_is_there(rollouts):
    texts = [ln.text for ln in life.lines(rollouts[7])]
    assert texts[0] == ("life seed 7 params. monomers 1 0 3 0 5. inflow 2 4 0. poly_rate 0 point 0 0 2. "
                        "decay 0 point 1. mu 0 point 0 2. gap 0.")
    assert ("life seed 7 step 3 0. free_monomers 1 6 7 8 6. bound 7 1 9. polymers 1 5 8. replicators 2 8. longest 1 5. "
            "mean_length 4 point 5 5. diversity 2. dominant_share 0 point 9 6 4. generations 3 2. stage dominated.") in texts
    for want in ("q life seed 7 step 2 6 replicators. a 2.", "q life seed 7 step 2 6 next replicators. a 4.",
                 "q life seed 7 step 3 0 stage. a dominated.", "q life seed 7 final first_replicator_step. a 2 3.",
                 "q life seed 7 final outcome. a dominated.", "q life seed 7 param mu. a 0 point 0 2."):
        assert want in texts, want
    assert sum(t.startswith("life seed 7 step ") for t in texts) == life.STEPS + 1
    assert not any(" step. " in t for t in texts)  # the step number is in the prompt, not a field
    n_metrics = len(life.METRICS)
    assert len(texts) == 1 + 41 + len(life.SUMMARY_KEYS) + len(life.PARAM_KEYS) + 41 * n_metrics + 40 * n_metrics
    assert len(set(texts)) == len(texts)


def test_generate_gives_n_lines_over_several_seeds(generated):
    assert len(generated) == 600
    seeds = {ln.text.split()[2 if ln.kind == "record" else 3] for ln in generated}
    assert len(seeds) >= 3
    assert any(ln.kind == "record" and " params. " in ln.text for ln in generated)
    assert any(" final " in ln.prompt for ln in generated if ln.kind != "record")
    assert any(" next " in ln.prompt for ln in generated if ln.kind != "record")
    assert any(" param " in ln.prompt for ln in generated if ln.kind != "record")
    assert len(SIM.generate(random.Random(2), 7)) == 7 and SIM.generate(random.Random(2), 0) == []


def test_gate_agrees_with_its_own_lines(all_lines):
    for ln in all_lines:
        if ln.kind == "record":
            continue
        v = SIM.check(ln.prompt, ln.answer)
        assert v.ok and v.expected == ln.answer and v.reason == "", (ln.text, v)


def test_gate_rejects_wrong_answers(all_lines):
    words = life.STAGES + life.OUTCOMES
    for ln in all_lines:
        if ln.kind == "record":
            continue
        x = parse_num(ln.answer)
        if x is None:  # a word: every other word is wrong
            wrongs = [w for w in words if w != ln.answer] + ["wrong", "yes"]
        else:  # a number: far off, and with its first digit changed
            first = next(i for i, w in enumerate(ln.answer.split()) if w.isdigit())
            changed = ln.answer.split()
            changed[first] = str((int(changed[first]) + 2) % 10)
            wrongs = [num(x * 2 + 7 if isinstance(x, int) else x * 2 + 1), " ".join(changed), "growth"]
        for wrong in wrongs + [""]:
            v = SIM.check(ln.prompt, wrong)
            assert not v.ok and v.expected == ln.answer and v.reason, (ln.text, wrong, v)


def test_tolerance_words_and_identities_exact_counts_within_one_numbers_within_5_percent(rollouts):
    r = rollouts[2]
    t = next(t for t, s in enumerate(r.steps) if s["replicators"] >= 100)
    n = r.steps[t]["replicators"]
    p = f"life seed 2 step {num(t)} replicators"
    assert SIM.check(p, num(n + 1)).ok and SIM.check(p, num(n - 1)).ok and SIM.check(p, num(int(n * 1.04))).ok
    assert not SIM.check(p, num(int(n * 1.2) + 2)).ok
    assert not SIM.check(p, num(n + 0.5)).ok  # a count is a whole number
    mean = r.steps[t]["mean_length"]
    q = f"life seed 2 step {num(t)} mean_length"
    assert SIM.check(q, num(mean * 1.03)).ok and not SIM.check(q, num(mean * 1.2)).ok
    assert SIM.check("life seed 2 final outcome", r.summary["outcome"]).ok
    assert not SIM.check("life seed 2 final outcome", "none" if r.summary["outcome"] != "none" else "extinct").ok
    # 0 against 1 is soup against life: a zero is exact both ways
    assert r.steps[3]["replicators"] == 0
    zero = "life seed 2 step 3 replicators"
    assert SIM.check(zero, "0").ok and not SIM.check(zero, "1").ok and not SIM.check(zero, "minus 1").ok
    seed, t1 = next((s, t) for s, x in rollouts.items() for t, st in enumerate(x.steps) if st["replicators"] == 1)
    one = f"life seed {num(seed)} step {num(t1)} replicators"
    assert SIM.check(one, "1").ok and SIM.check(one, "2").ok and not SIM.check(one, "0").ok
    # identities and table values are exact
    first = r.summary["first_replicator_step"]
    for prompt, value in (("life seed 2 final first_replicator_step", first), ("life seed 2 param gap", r.params["gap"]),
                          ("life seed 2 param monomers", r.params["monomers"]), ("life seed 2 param inflow", r.params["inflow"]),
                          ("life seed 2 step 9 step", 9), ("life seed 2 step 9 next step", 10)):
        assert SIM.check(prompt, num(value)).ok, prompt
        assert not SIM.check(prompt, num(value + 1)).ok and not SIM.check(prompt, num(value - 1)).ok, prompt
    assert SIM.check("life seed 2 param mu", num(r.params["mu"])).ok
    assert not SIM.check("life seed 2 param mu", num(r.params["mu"] * 1.03)).ok
    none = next(seed for seed in range(1, 200) if SIM.rollout(seed).summary["outcome"] == "none")
    assert SIM.check(f"life seed {num(none)} final first_replicator_step", "minus 1").ok
    assert not SIM.check(f"life seed {num(none)} final first_replicator_step", "0").ok


def test_owns_only_its_own_prompts(all_lines):
    for ln in all_lines:
        if ln.kind != "record":
            assert SIM.owns(ln.prompt), ln.prompt
    for other in OTHER_PROMPTS + (
            "life seed 7 step 3 clumps", "life seed 7 final nope", "life seed 7 param clumps", "life seed 7 final mu",
            "life seed 7 param outcome", "life seed x step 3 replicators", "life seed 7 step 3", "life seed 7",
            "life seed 0 7 step 3 replicators", "life seed 7 step 0 3 replicators", "life seed minus 7 step 3 replicators",
            "life seed 7 step 3 replicators ", "life  seed 7 step 3 replicators", "q life seed 7 step 3 replicators",
            "life seed 7 step 3 next next replicators", "life seed 7 final next replicators", "life seed 7 step 3 final outcome", ""):
        assert not SIM.owns(other), other
        v = SIM.check(other, "4")
        assert not v.ok and v.expected is None and v.reason == "not my question"
    assert SIM.owns("life seed 0 step 0 stage") and SIM.check("life seed 0 step 0 stage", "soup").ok
    assert SIM.check("life seed 7 step 9 9 replicators", "4").reason == "no such step"
    assert SIM.check(f"life seed 7 step {num(life.STEPS)} next replicators", "4").reason == "no such step"
    assert SIM.check(f"life seed 7 step {num(life.STEPS)} stage", SIM.rollout(7).steps[-1]["stage"]).ok


def test_conserved_holds_for_every_seed(rollouts):
    for r in rollouts.values():
        assert SIM.conserved(r).ok, r.seed
        for t, s in enumerate(r.steps):
            assert s["free_monomers"] + s["bound"] == r.params["monomers"] + r.params["inflow"] * t
    for seed in range(50, 80):
        assert SIM.conserved(life.run(seed)).ok
    for params in ({"mu": 0.05, "inflow": 200}, {"inflow": 0, "decay": 0.0}, {"decay": 3.2, "poly_rate": 0.002, "gap": 0}):
        for seed in range(100, 110):
            v = SIM.conserved(life.run(seed, **params))
            assert v.ok, (seed, params, v)


def test_letters_are_conserved_kind_by_kind():
    for seed in (1, 3, 11, 20, 107, 126):
        params = {"inflow": 0, "decay": 0.0} if seed > 100 else {}
        p = life._params(seed, params)
        r = life.run(seed, **params)
        start = None
        for t, (free, seqs, generations, failed) in enumerate(life.simulate(seed, p)):
            start = free.copy() if start is None else start
            assert free.dtype == np.int64 and seqs.dtype == np.uint8 and (free >= 0).all()
            assert (free + life.letter_counts(seqs).sum(0) == start + p["inflow"] // 4 * t).all(), (seed, t)
            lens = (seqs != life.PAD).sum(1)
            assert (lens >= 2).all() and ((seqs != life.PAD) == (np.arange(life.MAX_LEN)[None, :] < lens[:, None])).all()
            s = r.steps[t]
            assert (s["free_monomers"], s["polymers"], s["generations"]) == (int(free.sum()), len(seqs), generations)
            assert s["replicators"] == int(life.has_motif(seqs, p["gap"]).sum()) and s["bound"] == int(lens.sum())
            if failed and len(seqs) < life.MAX_POLYMERS:  # no copy fails while every letter is plentiful
                assert free.min() < life.MAX_LEN, (seed, t)
            if s["stage"] == "growth":
                assert failed == 0
        assert start.sum() == p["monomers"] and start.max() - start.min() <= 1


def test_conserved_catches_a_broken_rollout(rollouts):
    def broken(change) -> bool:
        r = copy.deepcopy(rollouts[1])
        change(r)
        v = SIM.conserved(r)
        return not v.ok and bool(v.reason)

    def setter(t, key, value):
        return lambda r: r.steps[t].__setitem__(key, value(r.steps[t][key]) if callable(value) else value)

    assert SIM.conserved(rollouts[1]).ok and rollouts[1].steps[20]["stage"] == "dominated"
    assert broken(setter(5, "free_monomers", lambda x: x + 1))
    assert broken(setter(5, "bound", lambda x: x - 1))
    assert broken(setter(20, "replicators", lambda x: 10 ** 6))
    assert broken(setter(20, "dominant_share", 1.5))
    assert broken(setter(20, "dominant_share", 0.001))  # less than one in diversity
    assert broken(setter(20, "diversity", 0))
    assert broken(setter(20, "generations", -1))
    assert broken(setter(20, "generations", lambda x: x + 10 ** 6))  # more copies than templates
    assert broken(setter(20, "mean_length", lambda x: x + 1))
    assert broken(setter(20, "longest", 1))
    assert broken(setter(20, "stage", "soup"))
    assert broken(setter(20, "stage", "growth"))
    assert broken(setter(20, "stage", "first_replicator"))
    assert broken(setter(20, "stage", "boiling"))
    assert broken(setter(20, "step", 21))
    assert broken(lambda r: r.summary.__setitem__("outcome", "extinct"))
    assert broken(lambda r: r.summary.__setitem__("peak_replicators", 1))
    assert broken(lambda r: r.summary.__setitem__("first_replicator_step", 0))


def test_motif_finder_matches_the_regex():
    rng = np.random.default_rng(3)
    n = 5000
    lens = rng.integers(2, life.MAX_LEN + 1, size=n)
    seqs = np.full((n, life.MAX_LEN), life.PAD, dtype=np.uint8)
    mask = np.arange(life.MAX_LEN)[None, :] < lens[:, None]
    seqs[mask] = rng.integers(0, 4, size=int(mask.sum()), dtype=np.uint8)
    for gap in range(4):
        pat = re.compile(f"ag[acgu]{{0,{gap}}}cu")
        want = np.array([bool(pat.search(s)) for s in strings(seqs)])
        assert (life.has_motif(seqs, gap) == want).all()
        assert 0.005 < want.mean() < 0.1  # rare in random strings
    counts = life.letter_counts(seqs)
    assert counts.dtype == np.int64 and (counts.sum(1) == lens).all()
    assert counts[0].tolist() == [strings(seqs[:1])[0].count(ch) for ch in life.LETTERS]


def test_the_draws_are_fair_and_never_overdraw_the_pool():
    raw = np.random.PCG64(5).random_raw
    u = life._uniform(raw, 200000)
    assert u.dtype == np.float64 and 0 <= u.min() and u.max() < 1 and abs(u.mean() - 0.5) < 0.005
    lengths = 2 + np.searchsorted(life._LENGTH_CDF, u, side="right")
    assert lengths.min() == 2 and lengths.max() == life.MAX_LEN and abs(lengths.mean() - 4.31) < 0.05
    assert abs((lengths == 2).mean() - 0.3) < 0.01 and abs((lengths == 3).mean() - 0.21) < 0.01
    free = np.array([30, 10, 5, 55], dtype=np.int64)
    for k, rounds in ((20, 4000), (80, 1000)):  # few of the pool (drawn one by one), most of it (all ordered)
        total = np.zeros(4)
        for _ in range(rounds):
            got = np.bincount(life._draw(raw, free, k), minlength=4)
            assert got.sum() == k and (got <= free).all()
            total += got
        assert np.allclose(total / rounds, free * k / 100, atol=0.2), (k, total / rounds)
    assert np.bincount(life._draw(raw, free, 100), minlength=4).tolist() == free.tolist()
    assert np.bincount(life._draw(raw, np.array([0, 7, 0, 0]), 3), minlength=4).tolist() == [0, 3, 0, 0]
    assert life._order(np.array([3, 1, 3, 1, 2])).tolist() == [1, 3, 4, 0, 2]  # ties keep their place
    seen = {tuple(life._order(raw(3)).tolist()) for _ in range(200)}
    assert len(seen) == 6
    a, b = np.random.PCG64(9).random_raw, np.random.PCG64(9).random_raw
    assert life._draw(a, free, 20).tolist() == life._draw(b, free, 20).tolist()


def test_stages_and_summary_follow_the_steps(rollouts):
    seen = set()
    for r in list(rollouts.values()) + [life.run(s) for s in range(80, 120)]:
        reps = [s["replicators"] for s in r.steps]
        first = next((t for t, n in enumerate(reps) if n), -1)
        assert r.summary["first_replicator_step"] == first
        assert r.summary["peak_replicators"] == max(reps)
        last = r.steps[-1]
        assert (r.summary["replicators"], r.summary["diversity"], r.summary["dominant_share"]) == \
            (last["replicators"], last["diversity"], last["dominant_share"])
        assert list(r.summary) == list(life.SUMMARY_KEYS) and list(r.steps[0]) == list(life.KEYS)
        assert r.summary["outcome"] in life.OUTCOMES
        assert (r.summary["outcome"] == "none") == (first < 0)
        assert (r.summary["outcome"] == "extinct") == (first >= 0 and last["replicators"] == 0)
        assert (r.summary["outcome"] == "dominated") == (last["stage"] == "dominated")
        assert r.steps[0] == {"step": 0, "free_monomers": r.params["monomers"], "bound": 0, "polymers": 0,
                              "replicators": 0, "longest": 0, "mean_length": 0.0, "diversity": 0,
                              "dominant_share": 0.0, "generations": 0, "stage": "soup"}
        for t, s in enumerate(r.steps):
            seen.add(s["stage"])
            assert s["step"] == t and s["stage"] in life.STAGES
            assert s["diversity"] <= s["replicators"] <= s["polymers"]
            assert (s["stage"] == "soup") == (t < first or first < 0)
            assert (s["stage"] == "first_replicator") == (t == first)
            assert (s["stage"] == "extinct") == (first >= 0 and t > first and s["replicators"] == 0)
            # the stage can be read off the record: dominated is the written share and count
            assert (s["stage"] == "dominated") == (first >= 0 and t > first and s["replicators"] >= life.DOMINATED_MIN
                                                   and s["dominant_share"] >= life.DOMINATED_SHARE)
            if s["stage"] == "growth":
                assert s["replicators"] >= r.steps[t - 1]["replicators"]
            if s["stage"] not in ("growth", "competition"):
                assert s["stage"] == life._stage(s["replicators"], s["dominant_share"], first if t > first else -1, 0, 0)
            if s["polymers"]:
                assert 2 <= s["mean_length"] <= s["longest"] <= life.MAX_LEN
            else:
                assert s["bound"] == 0 and s["longest"] == 0
            if t:
                copies = s["generations"] - r.steps[t - 1]["generations"]
                assert 0 <= copies <= s["replicators"]
    assert seen == set(life.STAGES)


def test_a_first_replicator_is_a_chance_event_most_vessels_see():
    firsts = [life.run(seed).summary["first_replicator_step"] for seed in range(1, 41)]
    had = [t for t in firsts if t >= 0]
    assert len(had) >= 30 and len(set(had)) >= 8  # measured 39 of 40, at steps 1 to 37
    assert 1 <= min(had) and 2 <= statistics.median(had) <= 15  # measured median 6
    outcomes = {SIM.rollout(seed).summary["outcome"] for seed in range(1, 41)} | \
        {life.run(seed).summary["outcome"] for seed in range(1000, 1040)}
    assert outcomes == set(life.OUTCOMES)


def test_small_mu_and_inflow_let_one_lineage_dominate():
    outcomes = [life.run(seed, mu=0.001, inflow=200).summary["outcome"] for seed in range(100, 130)]
    assert outcomes.count("dominated") >= 0.6 * len(outcomes), outcomes  # measured 25 of 30


def test_large_mu_keeps_diversity_high_or_kills_the_replicators():
    runs = [life.run(seed, mu=0.05, inflow=200) for seed in range(100, 130)]
    had = [r for r in runs if r.summary["first_replicator_step"] >= 0]
    spread = [r for r in had if r.summary["diversity"] >= 20 or r.summary["outcome"] == "extinct"]
    assert len(spread) >= 0.8 * len(had)  # measured 26 of 29
    assert sum(r.summary["outcome"] == "dominated" for r in runs) <= 0.2 * len(runs)  # measured 1 of 30
    calm = [life.run(seed, mu=0.001, inflow=200).summary["diversity"] for seed in range(100, 130)]
    assert statistics.median(r.summary["diversity"] for r in runs) > 5 * statistics.median(calm)


def test_no_inflow_and_no_decay_starves_the_replicators():
    runs = [life.run(seed, inflow=0, decay=0.0) for seed in range(100, 130)]
    had = [r for r in runs if r.summary["first_replicator_step"] >= 0]
    assert len(had) >= 20  # measured 28
    flat = [r for r in had if r.steps[-1]["replicators"] == r.steps[-6]["replicators"]]
    assert len(flat) >= 0.9 * len(had)  # measured 28 of 28: the replicators stopped multiplying
    stopped = [r for r in had if r.steps[-1]["generations"] == r.steps[-6]["generations"]]
    assert len(stopped) >= 0.5 * len(had)  # measured 19 of 28: no copy at all
    for r in had:
        last, before = r.steps[-1], r.steps[-2]
        # what still copies are mutants that need no letter of the missing kind: measured at most 2.2 percent
        assert last["generations"] - before["generations"] <= 0.05 * last["replicators"]
        assert last["stage"] != "growth"
        assert last["free_monomers"] < r.params["monomers"]
        assert last["free_monomers"] + last["bound"] == r.params["monomers"]
    fed = [life.run(seed, inflow=200, decay=0.0) for seed in range(100, 130)]
    fed = [r for r in fed if r.summary["first_replicator_step"] >= 0]
    assert all(r.steps[-1]["generations"] > r.steps[-6]["generations"] for r in fed)  # with inflow they go on


def test_extinction_exists_for_harsh_decay():
    outcomes = [life.run(seed, decay=3.2, poly_rate=0.002, gap=0).summary["outcome"] for seed in range(200, 260)]
    assert "extinct" in outcomes and "dominated" in outcomes and "none" in outcomes  # measured 18, 11 and 15 of 60
    assert outcomes.count("extinct") >= 6


def test_harsh_decay_selects_longer_replicators():
    harsh = np.concatenate([final_replicator_lengths(seed, decay=3.2, gap=0) for seed in range(300, 330)])
    mild = np.concatenate([final_replicator_lengths(seed, decay=0.1, gap=0) for seed in range(300, 330)])
    assert len(harsh) > 1000 and len(mild) > 1000
    assert harsh.mean() > mild.mean() + 2  # measured 12.5 against 7.7 letters


def test_growth_is_doubling_while_the_letters_last():
    ratios = []
    for seed in range(300, 360):
        r = life.run(seed, decay=0.1)
        ratios += [b["replicators"] / a["replicators"] for a, b in zip(r.steps, r.steps[1:])
                   if a["replicators"] >= 10 and b["stage"] == "growth"]
    assert len(ratios) >= 50 and 1.8 <= statistics.median(ratios) <= 2.0  # measured 1.95


def test_a_run_is_fast_and_the_vessel_is_capped():
    t0 = time.perf_counter()
    for seed in range(1, 11):
        life.run(seed)
    assert (time.perf_counter() - t0) / 10 < 0.5  # measured 0.02 s per rollout
    t0 = time.perf_counter()
    r = life.run(5, monomers=20000, inflow=400, poly_rate=0.02, decay=0.1, mu=0.05, gap=3)
    assert time.perf_counter() - t0 < 2
    assert SIM.conserved(r).ok
    t0 = time.perf_counter()
    big = life.run(1, monomers=200000, inflow=0, poly_rate=0.02, decay=0.1, mu=0.01, gap=2)
    assert time.perf_counter() - t0 < 2
    assert max(s["polymers"] for s in big.steps) == life.MAX_POLYMERS and SIM.conserved(big).ok
