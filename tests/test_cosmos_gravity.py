import ast
import copy
import hashlib
import inspect
import math
import os
import random
import subprocess
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest

from haishool.cosmos import Rollout, Simulation, close, gravity
from haishool.cosmos.gravity import (DT, EPS, KEYS, LINE_KEYS, MASS, N, N_OUT, STAGES, SUB_STEPS, SUMMARY_KEYS, T_FF,
                                     _accel, _fof, _normal, _potential, _total, canonical, evolve, initial_state,
                                     lines, measure, parse_prompt, random_params, rollout, simulation, stage)
from haishool.truth import is_dense, num, parse_num

SIM = simulation()
#: seeds whose random parameters give a cooled fast rotator (5: spin 0.27, cooling 0.5), a cooled
#: slow one (1: spin 0.1, cooling 0.48), the conservative control (4: spin 0.13, cooling 0), a
#: weakly cooled fast one (2: spin 0.38, cooling 0.13) and a cooled run that never halves its
#: radius (20: spin 0.37, cooling 0.48)
SEEDS = (5, 1, 4, 2, 20)
#: seeds whose runs (with their random parameters) must all pass the gate
GATE_SEEDS = tuple(range(1, 9))
#: seeds for the runs with hand-picked parameters
FRESH = (101, 102, 103, 104, 105)
ROOT = Path(__file__).resolve().parents[1]


def n_tokens(text: str) -> int:
    """The count of ``haishool.student.tokens`` (not imported: it needs torch)."""
    return len(text.replace(".", " . ").split())


@lru_cache(maxsize=None)
def run(seed: int, spin: float, cooling: float) -> Rollout:
    return SIM.run(seed, spin=spin, cooling=cooling)


@pytest.fixture(scope="module")
def all_lines():
    return {seed: lines(rollout(seed)) for seed in SEEDS}


def normal(seed: int, shape: tuple[int, ...], sigma: float = 1.0) -> np.ndarray:
    """Made-up test data from the module's own generator (no numpy random stream)."""
    return sigma * _normal(random.Random(seed), math.prod(shape)).reshape(shape)


def wrongs(answer: str, key: str) -> list[str]:
    """Wrong answers: another word; or the leading digit changed by 5, the sign flipped, the
    value doubled (a clump count moved by 2 or made a fraction)."""
    if answer in ("yes", "no"):
        return ["no" if answer == "yes" else "yes", "maybe", "1"]
    if answer in STAGES:
        return [STAGES[(STAGES.index(answer) + 1) % len(STAGES)], "star", "2"]
    if answer == "never":
        return ["1 point 2", "0", "no"]
    words = answer.split()
    i = next((i for i, w in enumerate(words) if w.isdigit() and w != "0"), 0)
    words[i] = str((int(words[i]) + 5) % 10)
    out = [" ".join(words), "never", "yes"]
    value = parse_num(answer)
    if value != 0:
        out.append(answer[6:] if answer.startswith("minus ") else "minus " + answer)
    if key.endswith("clumps"):
        out += [num(value + 2), num(value + 0.5)] + ([num(value - 2)] if value >= 2 else [])
    elif value != 0:
        out.append(num(2 * value, sig=3))
    else:
        out += ["1", "0 point 5"]
    return out


def state_digest(seed: int, spin: float, cooling: float, n: int) -> str:
    """The last positions and velocities of a run, bit for bit."""
    for x, v, _ in evolve(seed, spin, cooling, n=n):
        pass
    return hashlib.sha256(x.tobytes() + v.tobytes()).hexdigest()[:16]


def test_it_is_a_simulation():
    assert isinstance(SIM, Simulation) and SIM.sim == "gravity"
    assert set(LINE_KEYS) < set(KEYS) and set(KEYS) - set(LINE_KEYS) == {"mass", "momentum", "radiated"}
    assert [random_params(random.Random(s)) for s in SEEDS] == [
        {"spin": 0.27, "cooling": 0.5}, {"spin": 0.1, "cooling": 0.48}, {"spin": 0.13, "cooling": 0.0},
        {"spin": 0.38, "cooling": 0.13}, {"spin": 0.37, "cooling": 0.48}]
    assert (N, MASS, EPS) == (256, 1.0, 0.05)


def test_same_seed_gives_bit_identical_rollouts():
    a, b = run(101, 0.35, 0.4), SIM.run(101, spin=0.35, cooling=0.4)
    assert a.steps == b.steps and a.summary == b.summary and a.params == b.params
    for one, two in zip(initial_state(9, 256, 1.0, 0.2, 0.05), initial_state(9, 256, 1.0, 0.2, 0.05)):
        assert np.array_equal(one, two)
    assert state_digest(3, 0.3, 0.2, 64) == state_digest(3, 0.3, 0.2, 64)  # positions and velocities, bit for bit
    assert run(102, 0.35, 0.4).steps != a.steps
    assert run(101, 0.1, 0.6).steps != a.steps
    # the run a seed names: again from scratch, the very same states and lines
    again = SIM.run(4, **random_params(random.Random(4)))
    assert again.steps == rollout(4).steps and again.summary == rollout(4).summary
    assert again.params == rollout(4).params
    assert [ln.text for ln in lines(again)] == [ln.text for ln in lines(rollout(4))]
    # the gate's replay is cached; a caller gets a copy and cannot spoil it
    mine = rollout(5)
    assert mine is not rollout(5) and mine.steps == rollout(5).steps
    mine.steps[12]["clumps"] = 99
    mine.summary["clumps"] = 99
    assert rollout(5).steps[12]["clumps"] == 8 and SIM.check("gravity seed 5 step 1 2 clumps", "8").ok
    assert SIM.check("gravity seed 5 final clumps", "5").ok


def test_known_rollout():
    # A canary: the N-body is chaotic, so any change of the arithmetic changes the rollouts. The
    # module keeps to operations that are the same on every machine (see its docstring); if this
    # still fails somewhere, the gate there replays different runs than the ones the training
    # lines were made from: rebuild the lines there.
    assert state_digest(3, 0.3, 0.2, 64) == "ccfc24a94222cfc7"
    r = rollout(5)
    assert r.summary == {"clumps": 5, "largest_clump": 0.496, "flattening": 0.04, "min_flattening": 0.0372,
                         "spiral_ever": "yes", "collapse_time": 1.3}
    assert lines(r)[12 * 19].text == (
        "gravity seed 5 step 1 2. time 1 point 2. radius 0 point 3 6 5. flattening 0 point 5 7 6. clumps 8. "
        "largest_clump 0 point 2 6 6. spiral no. energy minus 0 point 7 2 8. angular_momentum 0 point 3 3 3. "
        "stage fragmenting.")
    digests = {seed: hashlib.sha256(repr(rollout(seed).steps).encode()).hexdigest()[:16] for seed in SEEDS}
    assert digests == {5: "498236474f224ac2", 1: "ae953707ef404787", 4: "20d2bf6855410ef0", 2: "2e63e057c12d0344",
                       20: "abf186d0bddab633"}
    assert rollout(4).summary["collapse_time"] == "never" == rollout(20).summary["collapse_time"]


def py_total(values: list[float]) -> float:
    """The order of ``_total`` in plain Python: neighbours in pairs, level by level."""
    values = list(values)
    while len(values) > 1:
        n = len(values)
        values = [values[i] + values[i + 1] for i in range(0, n - 1, 2)] + ([values[-1]] if n % 2 else [])
    return values[0] if values else 0.0


def test_total_adds_in_a_fixed_order():
    rnd = random.Random(1)
    for n in (*range(18), 100, 255, 256, 257):
        values = [rnd.uniform(-1.0, 1.0) * 10.0 ** rnd.randint(-8, 8) for _ in range(n)]
        assert float(_total(np.array(values))) == py_total(values), n
    grid = np.array([[rnd.uniform(-1.0, 1.0) * 10.0 ** rnd.randint(-8, 8) for _ in range(37)] for _ in range(5)])
    assert _total(grid).tolist() == [py_total(row) for row in grid.tolist()]
    assert _total(grid.T).tolist() == [py_total(col) for col in grid.T.tolist()]  # a strided view
    assert _total(np.ones((3, 5))).tolist() == [5.0, 5.0, 5.0] and _total(np.zeros((2, 0))).tolist() == [0.0, 0.0]
    assert float(_total(np.arange(7.0))) == 21.0
    # the order matters: these three numbers give 0 or 1 depending on it
    assert float(_total(np.array([1e16, 1.0, -1e16]))) == 0.0 != float(_total(np.array([1e16, -1e16, 1.0])))


def py_run(seed: int, n: int, mass: float, spin: float, cooling: float, eps: float, steps: int):
    """The cloud and ``steps`` leapfrog steps in plain Python floats: the same operations in the
    same order as the module, without numpy. Returns positions, velocities and the energy radiated."""
    rnd = random.Random(seed)
    eps2 = eps * eps

    def normal_numbers(count: int) -> list[float]:
        out = []
        for _ in range(count):
            u = [rnd.random() for _ in range(12)]
            g = u[0]
            for j in range(1, 12):
                g = g + u[j]
            out.append(g - 6.0)
        return out

    def weighted(a: list[list[float]]) -> list[float]:
        return [py_total([m[i] * a[i][c] for i in range(n)]) for c in range(3)]

    def potential(x: list[list[float]]) -> float:
        rows = []
        for i in range(n):
            row = []
            for j in range(n):
                dx, dy, dz = x[i][0] - x[j][0], x[i][1] - x[j][1], x[i][2] - x[j][2]
                row.append(0.0 if i == j else 1.0 / math.sqrt(dx * dx + dy * dy + dz * dz + eps2) * m[j])
            rows.append(m[i] * py_total(row))
        return -0.5 * py_total(rows)

    def accel(x: list[list[float]]) -> list[list[float]]:
        out = []
        for i in range(n):
            parts: list[list[float]] = [[], [], []]
            for j in range(n):
                d = [x[i][c] - x[j][c] for c in range(3)]
                r2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2] + eps2
                w = m[j] / (r2 * math.sqrt(r2))
                for c in range(3):
                    parts[c].append(w * d[c])
            out.append([-py_total(parts[c]) for c in range(3)])
        return out

    def kinetic(v: list[list[float]]) -> float:
        return 0.5 * py_total([m[i] * (v[i][0] * v[i][0] + v[i][1] * v[i][1] + v[i][2] * v[i][2]) for i in range(n)])

    raw = normal_numbers(12 * n)
    x = [[0.5 * raw[3 * k + c] for c in range(3)] for k in range(4 * n)]
    x = [p for p in x if p[0] * p[0] + p[1] * p[1] + p[2] * p[2] < 1.0][:n]
    m = [mass / n] * n
    shift = [s / mass for s in weighted(x)]
    x = [[p[c] - shift[c] for c in range(3)] for p in x]
    w = abs(potential(x))
    plane = py_total([m[i] * (x[i][0] * x[i][0] + x[i][1] * x[i][1]) for i in range(n)])
    omega = math.sqrt(2.0 * spin * w / plane)
    in_plane = math.sqrt(max(0.0, 1.0 / 3.0 - spin) * w / mass)
    sigma = [in_plane, in_plane, math.sqrt(w / (3.0 * mass))]
    raw = normal_numbers(3 * n)
    v = [[omega * turn + raw[3 * i + c] * sigma[c] for c, turn in enumerate((-x[i][1], x[i][0], 0.0))]
         for i in range(n)]
    shift = [s / mass for s in weighted(v)]
    v = [[q[c] - shift[c] for c in range(3)] for q in v]

    damp = 1.0 / (1.0 + gravity.COOL_RATE * cooling * DT / T_FF)
    radiated = 0.0
    a = accel(x)
    for _ in range(steps):
        v = [[v[i][c] + 0.5 * DT * a[i][c] for c in range(3)] for i in range(n)]
        x = [[x[i][c] + DT * v[i][c] for c in range(3)] for i in range(n)]
        a = accel(x)
        v = [[v[i][c] + 0.5 * DT * a[i][c] for c in range(3)] for i in range(n)]
        if cooling > 0:
            before = kinetic(v)
            for i in range(n):
                r2 = x[i][0] * x[i][0] + x[i][1] * x[i][1]
                radial = (1.0 - damp) * (v[i][0] * x[i][0] + v[i][1] * x[i][1]) / (r2 if r2 != 0.0 else 1.0)
                v[i] = [v[i][0] - radial * x[i][0], v[i][1] - radial * x[i][1], v[i][2] * damp]
            shift = [s / mass for s in weighted(v)]
            v = [[q[c] - shift[c] for c in range(3)] for q in v]
            radiated += before - kinetic(v)
    return x, v, radiated


def test_the_run_is_the_same_bits_as_plain_python_arithmetic():
    # numpy only vectorises: a rollout is fixed by IEEE 754 arithmetic in the order the module
    # writes down, and by Python's random(). 23 particles: an odd count, so every sum has leftovers.
    for seed, spin, cooling in ((7, 0.3, 0.4), (8, 0.1, 0.0)):
        ledger: dict = {}
        states = evolve(seed, spin, cooling, n=23, ledger=ledger)
        for k in range(3):
            x, v, m = next(states)
            px, pv, radiated = py_run(seed, 23, 1.0, spin, cooling, 0.05, k * SUB_STEPS)
            assert x.tolist() == px and v.tolist() == pv, (seed, k)
            assert ledger["radiated"] == radiated and (radiated > 0) == (cooling > 0 and k > 0)
            assert m.tolist() == [1.0 / 23] * 23


def test_source_uses_only_reproducible_operations():
    # no reduction whose order numpy chooses, no function that the C library rounds as it likes,
    # no numpy random stream, no Python sum() of floats (3.12 changed its rounding)
    banned = {"sum", "fsum", "mean", "std", "var", "prod", "dot", "einsum", "matmul", "inner", "linalg", "exp", "log",
              "log1p", "sin", "cos", "tan", "arctan2", "hypot", "power", "cbrt", "default_rng", "RandomState",
              "normal", "gauss", "normalvariate", "cKDTree", "float32"}
    for node in ast.walk(ast.parse(inspect.getsource(gravity))):
        if isinstance(node, ast.Attribute):
            assert node.attr not in banned, node.attr
        elif isinstance(node, ast.Name):
            assert node.id not in banned, node.id
        elif isinstance(node, ast.BinOp):
            assert not isinstance(node.op, (ast.Pow, ast.MatMult)), ast.dump(node.op)


def test_bits_do_not_depend_on_the_cpu_path():
    # numpy picks its loops (AVX2, AVX512, ...) by the CPU it runs on; with those switched off
    # the run must come out the same
    umath = getattr(getattr(np, "_core", None), "_multiarray_umath", None)
    features = getattr(umath, "__cpu_features__", {})
    switchable = [f for f in getattr(umath, "__cpu_dispatch__", []) if features.get(f)]
    if not switchable:
        pytest.skip("this numpy has no CPU features to switch off")
    code = ("import hashlib\nfrom haishool.cosmos.gravity import evolve\n"
            "for x, v, _ in evolve(3, 0.3, 0.2, n=64):\n    pass\n"
            "print(hashlib.sha256(x.tobytes() + v.tobytes()).hexdigest()[:16])")
    env = dict(os.environ, NPY_DISABLE_CPU_FEATURES=" ".join(switchable), PYTHONPATH=str(ROOT))
    done = subprocess.run([sys.executable, "-c", code], env=env, cwd=ROOT, capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    assert done.stdout.split()[-1] == state_digest(3, 0.3, 0.2, 64)


def test_rollout_shape():
    for seed in SEEDS:
        r = rollout(seed)
        assert r.sim == "gravity" and r.seed == seed and len(r.steps) == N_OUT == 41
        assert set(r.summary) == set(SUMMARY_KEYS)
        assert r.params == {**random_params(random.Random(seed)), "n": 256, "mass": 1.0, "eps": 0.05, "dt": 0.00444,
                            "t_ff": 1.11}
        for k, s in enumerate(r.steps):
            assert list(s) == list(KEYS)
            assert s["time"] == k / 10
            assert s["stage"] in STAGES and s["spiral"] in ("yes", "no")
            assert isinstance(s["clumps"], int) and s["clumps"] >= 0
            assert 0 <= s["largest_clump"] <= 1 and s["radius"] > 0 and s["flattening"] > 0
            assert (s["clumps"] == 0) == (s["largest_clump"] == 0)
            assert all(isinstance(s[key], float) for key in KEYS if key not in ("clumps", "spiral", "stage"))
        last = r.steps[-1]
        assert (r.summary["clumps"], r.summary["largest_clump"], r.summary["flattening"]) == (
            last["clumps"], last["largest_clump"], last["flattening"])
        assert r.summary["min_flattening"] == min(s["flattening"] for s in r.steps)
        assert (r.summary["spiral_ever"] == "yes") == any(s["spiral"] == "yes" for s in r.steps)
        halved = [s["time"] for s in r.steps if s["radius"] < 0.5 * r.steps[0]["radius"]]
        assert r.summary["collapse_time"] == (halved[0] if halved else "never")


def test_lines_are_dense_and_short(all_lines):
    for seed, lns in all_lines.items():
        assert len(lns) == 41 + 41 * 9 + 40 * 9 + 6 == 776
        assert len({ln.text for ln in lns}) == len(lns)
        assert sum(ln.kind == "record" for ln in lns) == 41
        for ln in lns:
            assert ln.topic == "gravity"
            assert is_dense(ln.text), ln.text
            assert n_tokens(ln.text) <= 80, ln.text
            if ln.kind == "record":
                assert ln.text.startswith(f"gravity seed {num(seed)} step ") and ln.answer == ""
                assert [field.split()[0] for field in ln.text[:-1].split(". ")[1:]] == LINE_KEYS
            else:
                assert "." not in ln.prompt and "." not in ln.answer and ln.answer
                assert ln.prompt.startswith(f"gravity seed {num(seed)} ")
                assert ln.text == f"q {ln.prompt}. a {ln.answer}."
    assert max(n_tokens(ln.text) for lns in all_lines.values() for ln in lns) <= 66
    thin = lines(rollout(5), every=4)
    assert len(thin) == 41 + 11 * 9 + 10 * 9 + 6 and set(ln.text for ln in thin) <= set(ln.text for ln in all_lines[5])


def test_lines_only_for_the_run_a_seed_names():
    # check() replays rollout(seed); lines of a run with other parameters could not be judged
    assert all(canonical(rollout(seed)) for seed in SEEDS)
    assert not canonical(run(101, 0.35, 0.4))
    with pytest.raises(ValueError, match="rollout"):
        lines(run(101, 0.35, 0.4))
    other = copy.deepcopy(rollout(5))
    other.params["eps"] = 0.1
    assert not canonical(other)
    with pytest.raises(ValueError):
        lines(other)


def test_numbers_have_at_most_three_digits(all_lines):
    for lns in all_lines.values():
        for ln in lns:
            if ln.kind != "record" and parse_num(ln.answer) is not None:
                digits = "".join(w for w in ln.answer.split(" e ")[0].split() if w.isdigit()).lstrip("0")
                assert len(digits) <= 3, ln.text
                assert num(parse_num(ln.answer), sig=3) == ln.answer  # one spelling per number


def test_gate_agrees_with_every_line(all_lines):
    for lns in all_lines.values():
        for ln in lns:
            if ln.kind != "record":
                verdict = SIM.check(ln.prompt, ln.answer)
                assert verdict.ok and verdict.expected == ln.answer, ln.text


def test_gate_rejects_wrong_answers(all_lines):
    for lns in all_lines.values():
        for ln in lns:
            if ln.kind != "record":
                for wrong in wrongs(ln.answer, ln.prompt):
                    verdict = SIM.check(ln.prompt, wrong)
                    assert not verdict.ok and verdict.expected == ln.answer, (ln.text, wrong)
    for junk in ("big", "", "yes", "0 point point 3", "0 point 3 6 5 big"):
        verdict = SIM.check("gravity seed 5 step 1 2 radius", junk)
        assert (verdict.ok, verdict.expected, verdict.reason) == (False, "0 point 3 6 5", "not a number")
    assert not SIM.check("gravity seed 5 final collapse_time", "never").ok
    assert not SIM.check("gravity seed 4 final collapse_time", "1 point 2").ok
    assert not SIM.check("gravity seed 5 step 1 2 stage", "disc").ok


def test_gate_tolerance():
    r = rollout(5)
    radius, clumps = r.steps[12]["radius"], r.steps[12]["clumps"]
    assert (radius, clumps) == (0.365, 8)
    assert SIM.check("gravity seed 5 step 1 2 radius", num(radius * 1.04, sig=3)).ok
    assert not SIM.check("gravity seed 5 step 1 2 radius", num(radius * 1.07, sig=3)).ok
    assert not SIM.check("gravity seed 5 step 1 2 radius", num(-radius, sig=3)).ok
    assert SIM.check("gravity seed 5 step 1 2 clumps", "9").ok and SIM.check("gravity seed 5 step 1 2 clumps", "7").ok
    assert not SIM.check("gravity seed 5 step 1 2 clumps", "1 0").ok
    assert not SIM.check("gravity seed 5 step 1 2 clumps", "6").ok
    # a clump count is a whole number
    assert not SIM.check("gravity seed 5 step 1 2 clumps", "8 point 5").ok
    assert not SIM.check("gravity seed 5 step 1 2 clumps", "8 point 0").ok
    # "next" is the state one step later
    assert SIM.check("gravity seed 5 step 1 2 next spiral", r.steps[13]["spiral"]).ok
    assert r.steps[12]["spiral"] != r.steps[13]["spiral"]
    beyond = SIM.check("gravity seed 5 step 4 0 next clumps", "3")
    assert not beyond.ok and beyond.expected is None and "beyond" in beyond.reason
    assert not SIM.check("gravity seed 5 step 4 1 clumps", "3").ok
    # the gate diagnostics can be asked too, though no training line does
    assert SIM.check("gravity seed 5 step 1 2 mass", "1").ok
    assert SIM.check("gravity seed 5 step 1 2 radiated", num(r.steps[12]["radiated"], sig=3)).ok


def test_owns_only_its_own_prompts(all_lines):
    for lns in all_lines.values():
        assert all(SIM.owns(ln.prompt) for ln in lns if ln.kind != "record")
    for prompt in ("turkey capital", "q spoon color", "spoon color", "carbon protons", "water molar_mass",
                   "calc 1 2 plus 7", "check 1 2 plus 7 equals 2 0", "solve 3 x plus 4 equals 1 9",
                   "nucleo seed 7 final helium_fraction", "planets seed 7 final rocky", "chem seed 7 step 3 water",
                   "life seed 7 step 3 next population", "world seed 3 era planets", "turkey borders greece",
                   # the same shape under another simulation's name
                   "nucleo seed 5 step 1 2 clumps", "planets seed 5 step 1 2 next clumps", "chem seed 5 final clumps",
                   "life seed 5 final clumps", "world seed 5 final clumps", "world seed 5 era gravity",
                   # the force called gravity belongs to round 5
                   "gravity carrier", "gravity_force m 1 0 m 2 0 r 2", "stronger gravity strong",
                   "check gravity carrier photon", "forces kind fundamental",
                   "gravity", "gravity seed 7", "gravity seed 7 step 3", "gravity seed 7 step 3 next",
                   "gravity seed 7 step 3 colour", "gravity seed 7 final stage", "gravity seed 7 final next clumps",
                   "gravity seed 7 final energy", "gravity seed 7 final radius", "gravity seed 7 step 3 m1",
                   "gravity seed 7 step 3 next next clumps", "gravity seed 7 step 3 min_flattening",
                   "gravity seed 0 7 final clumps", "gravity seed 7 step 0 3 clumps", "gravity seed x final clumps",
                   "gravity seed minus 7 final clumps", "gravity seed 7 point 0 final clumps",
                   "gravity seed 7 step minus 1 clumps", "gravity seed  7 final clumps", "gravity seed 7 final clumps ",
                   "gravity seed 7 final clumps extra", "q gravity seed 7 final clumps", "Gravity seed 7 final clumps",
                   "gravity seed 7 final clumps. a 4", ""):
        assert not SIM.owns(prompt), prompt
        verdict = SIM.check(prompt, "4")
        assert (verdict.ok, verdict.expected, verdict.reason) == (False, None, "not my question")


def test_parse_prompt():
    assert parse_prompt("gravity seed 7 step 1 2 clumps") == (7, 12, "clumps")
    assert parse_prompt("gravity seed 7 step 1 2 next clumps") == (7, 13, "clumps")
    assert parse_prompt("gravity seed 1 2 3 4 final min_flattening") == (1234, None, "min_flattening")
    assert parse_prompt("gravity seed 7 step 0 mass") == (7, 0, "mass")
    assert parse_prompt("gravity seed 0 step 0 time") == (0, 0, "time")
    assert parse_prompt("gravity seed 7 final energy") is None


def test_random_params():
    drawn = [random_params(random.Random(s)) for s in range(300)]
    assert drawn == [random_params(random.Random(s)) for s in range(300)]
    assert all(0.05 <= p["spin"] <= 0.4 for p in drawn)
    assert all(p["cooling"] == 0 or 0.1 <= p["cooling"] <= 0.6 for p in drawn)
    assert 0.1 < sum(p["cooling"] == 0 for p in drawn) / len(drawn) < 0.25
    assert len({(p["spin"], p["cooling"]) for p in drawn}) > 200


def test_near_normal_numbers():
    g = _normal(random.Random(3), 30000)
    assert g.dtype == np.float64 and g.shape == (30000,)
    assert np.array_equal(g, _normal(random.Random(3), 30000))
    assert not np.array_equal(g, _normal(random.Random(4), 30000))
    assert abs(float(_total(g)) / 30000) < 0.02 and abs(float(_total(g * g)) / 30000 - 1.0) < 0.03
    assert 3.5 < float(np.abs(g).max()) < 6.0
    assert 0.66 < float((np.abs(g) < 1.0).mean()) < 0.70  # a normal distribution has 0.683 within one sigma
    rnd = random.Random(3)
    first = [rnd.random() for _ in range(12)]
    total = first[0]
    for u in first[1:]:
        total = total + u
    assert float(g[0]) == total - 6.0


def test_initial_cloud():
    for seed, spin in ((1, 0.05), (2, 0.2), (3, 0.4)):
        x, v, m = initial_state(seed, 256, 1.0, spin, 0.05)
        assert x.shape == v.shape == (256, 3) and float(_total(m)) == 1.0
        assert x.dtype == v.dtype == m.dtype == np.float64
        assert np.abs((m[:, None] * x).sum(0)).max() < 1e-12 and np.abs((m[:, None] * v).sum(0)).max() < 1e-12
        assert np.sqrt((x * x).sum(1)).max() < 1.2
        assert 0.55 < float(np.sqrt((x * x).sum(1).mean())) < 0.75  # rms radius of the cut ball
        w = abs(_potential(x, m, 0.05 ** 2))
        lz = float((m * (x[:, 0] * v[:, 1] - x[:, 1] * v[:, 0])).sum())
        inertia = float((m * (x[:, 0] ** 2 + x[:, 1] ** 2)).sum())
        # rotational energy over |W|; the random motions of 256 particles add some Lz of their own
        assert lz > 0 and close(0.5 * lz * lz / inertia / w, spin, rel=0.3)
        kinetic = 0.5 * float((m * (v * v).sum(1)).sum())
        assert 0.8 < 2 * kinetic / w < 1.25  # virial balance (above 1 only for spin over 1/3)
    for bad in ({"cooling": -0.1}, {"spin": -0.1}, {"n": 9}, {"mass": 0.0}, {"eps": 0.0}):
        with pytest.raises(ValueError):
            SIM.run(1, **bad)


def test_forces_and_potential_of_two_bodies():
    x = np.array([[-0.5, 0.0, 0.0], [0.5, 0.0, 0.0]])
    m = np.array([0.5, 0.5])
    soft = 1.0 + 0.05 ** 2
    assert _potential(x, m, 0.05 ** 2) == pytest.approx(-0.25 / soft ** 0.5)
    assert _accel(x, m, 0.05 ** 2) == pytest.approx(np.array([[0.5 / soft ** 1.5, 0, 0], [-0.5 / soft ** 1.5, 0, 0]]))
    # the softened force is the gradient of the softened potential
    h = 1e-6
    up = _potential(x + [[h, 0, 0], [0, 0, 0]], m, 0.05 ** 2)
    down = _potential(x - [[h, 0, 0], [0, 0, 0]], m, 0.05 ** 2)
    assert -(up - down) / (2 * h) / m[0] == pytest.approx(_accel(x, m, 0.05 ** 2)[0, 0], rel=1e-6)
    assert DT == pytest.approx(T_FF / 250)
    # many bodies: the forces add up to nothing (Newton's third law)
    cloud, masses = normal(5, (50, 3), 0.3), np.full(50, 0.02)
    assert np.abs((masses[:, None] * _accel(cloud, masses, 0.05 ** 2)).sum(0)).max() < 1e-13


def test_measure_on_made_up_states():
    m = np.full(200, 1 / 200)
    still = np.zeros((200, 3))
    ball = normal(1, (200, 3), 0.3)
    round_ = measure(ball, still, m, 0.05, None, 0.0)
    assert 0.8 < round_["flattening"] < 1.25 and round_["spiral"] == "no" and round_["time"] == 0.0
    assert round_["radiated"] == 0.0 and measure(ball, still, m, 0.05, None, 0.0, 0.3)["radiated"] == 0.3
    sheet = measure(ball * [1, 1, 0.05], still, m, 0.05, None, 0.0)
    assert sheet["flattening"] < 0.1 and sheet["m2"] < 0.2 and sheet["spiral"] == "no"
    # two tight clumps on opposite sides, 120 and 80 particles: a two-fold pattern
    offset = np.concatenate([np.tile([0.3, 0.0, 0.0], (120, 1)), np.tile([-0.3, 0.0, 0.0], (80, 1))])
    pair = measure(offset + normal(2, (200, 3), 0.01), still, m, 0.05, None, 0.0)
    assert (pair["clumps"], pair["largest_clump"]) == (2, pytest.approx(0.6))
    assert pair["m2"] > 0.9 and pair["m1"] == pytest.approx(0.2, abs=0.02)
    assert pair["spiral"] == "yes" and pair["stage"] == "clumped"
    assert pair["radius"] == pytest.approx(0.24, abs=0.03)  # the big clump's distance from the centre of mass
    # the same pair puffed up along z is not a disc, so no spiral
    tall = measure(offset + normal(3, (200, 3), 0.01) + normal(4, (200, 1), 0.3) * [0, 0, 1], still, m,
                   0.05, None, 0.0)
    assert tall["flattening"] > 0.5 and tall["spiral"] == "no"
    # solid-body rotation: Lz = omega * sum m R^2, kinetic energy = omega * Lz / 2
    spin = 0.7 * np.stack([-ball[:, 1], ball[:, 0], np.zeros(200)], 1)
    turning = measure(ball, spin, m, 0.05, None, 0.0)
    centred = ball - ball.mean(0)
    inertia = float((m * (centred[:, 0] ** 2 + centred[:, 1] ** 2)).sum())
    assert turning["angular_momentum"] == pytest.approx(0.7 * inertia)
    about_origin = float((m * (ball[:, 0] * spin[:, 1] - ball[:, 1] * spin[:, 0])).sum())
    assert turning["energy"] - round_["energy"] == pytest.approx(0.5 * 0.7 * about_origin)
    assert turning["mass"] == pytest.approx(1.0) and not still.any()
    # the metrics do not care where the system is or how fast it drifts as a whole, except for
    # the energy and the momentum of the drift itself
    moved = measure(ball + [3.0, -2.0, 1.0], spin + [0.5, 0.0, 0.0], m, 0.05, None, 0.0)
    for key in ("radius", "flattening", "angular_momentum"):
        assert moved[key] == pytest.approx(turning[key], rel=1e-9)
    assert all(moved[key] == turning[key] for key in ("clumps", "spiral", "stage"))
    assert moved["momentum"] == pytest.approx(turning["momentum"] + 0.5, abs=0.02)
    assert moved["energy"] > turning["energy"]


def test_friends_of_friends():
    groups = np.concatenate([normal(6, (8, 3), 0.005), normal(7, (5, 3), 0.005) + [1, 0, 0],
                             normal(8, (4, 3), 0.005) + [0, 1, 0], [[0, 0, 2.0], [0, 0, 3.0], [0, 0, 4.0]]])
    m = np.full(20, 0.05)
    assert _fof(groups, m, 0.05) == (2, pytest.approx(0.4))  # the group of 4 and the singles do not count
    assert _fof(groups[13:], m[13:], 0.05) == (0, 0.0)
    assert _fof(groups, m, 5.0) == (1, pytest.approx(1.0))
    # friends of friends: a chain links up although its ends are far apart; the order does not matter
    chain = np.array([[0.04 * i, 0.0, 0.0] for i in range(12)])
    assert _fof(chain, np.full(12, 1 / 12), 0.05) == (1, pytest.approx(1.0))
    assert _fof(chain, np.full(12, 1 / 12), 0.039) == (0, 0.0)
    shuffled = list(range(20))
    random.Random(1).shuffle(shuffled)
    assert _fof(groups[shuffled], m, 0.05) == (2, pytest.approx(0.4))


def test_stage_rules():
    assert stage(0.6, 1.0, 0, 0.0, 0.64) == "cloud"
    assert stage(0.6, 1.0, 3, 0.05, 0.64) == "cloud"  # a few tiny groups are not fragments yet
    assert stage(0.4, 0.8, 1, 0.05, 0.64) == "collapsing"
    assert stage(0.4, 0.4, 1, 0.3, 0.64) == "disc"
    assert stage(0.4, 0.4, 3, 0.3, 0.64) == "fragmenting"
    assert stage(0.4, 0.8, 2, 0.1, 0.64) == "fragmenting"
    assert stage(0.1, 0.05, 3, 0.7, 0.64) == "clumped"
    assert stage(0.6, 1.0, 1, 0.5, 0.64) == "clumped"


def test_gate_holds_on_every_seed():
    slow = os.environ.get("HAISHOOL_SLOW")  # all 40 seeds of the review, each run twice: 2 to 3 minutes
    for seed in (range(1, 41) if slow else GATE_SEEDS):
        r = rollout(seed)
        verdict = SIM.conserved(r)
        assert verdict.ok, (seed, verdict.reason)
        first, last = r.steps[0], r.steps[-1]
        assert first["stage"] == "cloud" and first["radiated"] == 0 and 0.55 < first["radius"] < 0.75
        if r.params["cooling"] > 0:
            # cooling takes the support away: the cloud flattens, binds tighter and breaks up
            assert r.summary["min_flattening"] < 0.5 and last["stage"] in ("fragmenting", "clumped")
            assert last["energy"] < 2 * first["energy"] < 0 and last["radiated"] > abs(first["energy"])
            assert r.summary["largest_clump"] > 0.4
        else:
            assert r.summary["min_flattening"] > 0.6 and last["stage"] == "cloud"
            assert r.summary["collapse_time"] == "never" and r.summary["spiral_ever"] == "no"
            assert r.summary["largest_clump"] < 0.1 and all(s["radiated"] == 0 for s in r.steps)
        if slow:
            again = SIM.run(seed, **random_params(random.Random(seed)))
            assert again.steps == r.steps and again.summary == r.summary


def test_conservative_run_keeps_energy_and_angular_momentum():
    for seed, spin in ((101, 0.1), (102, 0.4)):
        r = run(seed, spin, 0.0)
        assert SIM.conserved(r).ok
        e0, lz0 = r.steps[0]["energy"], r.steps[0]["angular_momentum"]
        assert e0 < 0 < lz0
        for s in r.steps:
            assert close(s["energy"], e0, rel=0.02) and close(s["angular_momentum"], lz0, rel=0.01)
            assert s["momentum"] < 1e-9 and s["mass"] == 1.0 and s["radiated"] == 0.0
    # at full precision the leapfrog does far better than the gate asks
    full = [measure(x, v, m, 0.05, None, 0.0) for x, v, m in evolve(101, 0.1, 0.0)]
    e0, lz0 = full[0]["energy"], full[0]["angular_momentum"]
    assert max(abs(f["energy"] - e0) for f in full) < 1e-4 * abs(e0)
    assert max(abs(f["angular_momentum"] - lz0) for f in full) < 1e-12 * lz0
    assert max(f["momentum"] for f in full) < 1e-12
    assert all(f["mass"] == full[0]["mass"] for f in full)


def test_cooling_only_removes_energy_and_keeps_angular_momentum():
    for r in (rollout(5), rollout(1), run(101, 0.35, 0.4), run(101, 0.1, 0.6)):
        assert r.params["cooling"] > 0 and SIM.conserved(r).ok
        energy = [s["energy"] for s in r.steps]
        radiated = [s["radiated"] for s in r.steps]
        assert all(b <= a for a, b in zip(energy, energy[1:]))  # monotonic, even without the gate's tolerance
        assert radiated[0] == 0 and all(b > a for a, b in zip(radiated, radiated[1:]))
        assert energy[-1] < 2 * energy[0] < 0  # more than twice as tightly bound at the end
        # the ledger: what the cloud lost is what the cooling took, within 1 % (the gate allows 2 %)
        assert all(abs(e + q - energy[0]) < 0.01 * abs(e) for e, q in zip(energy, radiated))
        assert all(close(s["angular_momentum"], r.steps[0]["angular_momentum"], rel=0.01) for s in r.steps)
        assert all(s["momentum"] < 1e-9 and s["mass"] == 1.0 for s in r.steps)
    ledger: dict = {}
    full = [measure(x, v, m, 0.05, None, 0.0, ledger["radiated"]) for x, v, m in evolve(5, 0.27, 0.5, ledger=ledger)]
    assert all(b["energy"] < a["energy"] for a, b in zip(full, full[1:]))
    e0, lz0 = full[0]["energy"], full[0]["angular_momentum"]
    assert max(abs(f["angular_momentum"] - lz0) for f in full) < 1e-12 * lz0
    assert max(abs(f["energy"] + f["radiated"] - e0) / abs(f["energy"]) for f in full) < 0.005
    assert full[-1]["radiated"] > 5 * abs(e0)  # most of the final binding energy went through the ledger


def test_conserved_rejects_broken_rollouts():
    def broken(r: Rollout, t: int, key: str, value: float) -> Rollout:
        bad = copy.deepcopy(r)
        bad.steps[t][key] = value
        return bad

    still, cooled = run(101, 0.1, 0.0), rollout(5)
    assert SIM.conserved(still).ok and SIM.conserved(cooled).ok
    for r in (still, cooled):
        assert "mass" in SIM.conserved(broken(r, 20, "mass", 0.99)).reason
        assert "momentum" in SIM.conserved(broken(r, 20, "momentum", 0.01)).reason
        lz0 = r.steps[0]["angular_momentum"]
        assert "angular momentum" in SIM.conserved(broken(r, 20, "angular_momentum", lz0 * 1.02)).reason
        assert SIM.conserved(broken(r, 20, "angular_momentum", lz0 * 1.005)).ok
        for bad in (broken(r, 20, "mass", 0.99), broken(r, 20, "angular_momentum", lz0 * 1.02)):
            verdict = SIM.conserved(bad)
            assert not verdict.ok and verdict.expected is None and "step 20" in verdict.reason
    e0 = still.steps[0]["energy"]
    assert "energy" in SIM.conserved(broken(still, 20, "energy", e0 * 1.03)).reason
    assert "energy" in SIM.conserved(broken(still, 20, "energy", e0 * 0.97)).reason
    assert SIM.conserved(broken(still, 20, "energy", e0 * 1.01)).ok
    assert "without cooling" in SIM.conserved(broken(still, 20, "radiated", 0.001)).reason
    # with cooling the ledger must close: energy + radiated stays at the first energy
    e20, q20 = cooled.steps[20]["energy"], cooled.steps[20]["radiated"]
    assert "energy changed" in SIM.conserved(broken(cooled, 20, "radiated", q20 + 0.04 * abs(e20))).reason
    assert "energy changed" in SIM.conserved(broken(cooled, 20, "energy", e20 * 1.04)).reason
    assert SIM.conserved(broken(cooled, 20, "radiated", q20 + 0.005 * abs(e20))).ok
    # the cooling only takes: radiated never falls
    fallen = SIM.conserved(broken(cooled, 20, "radiated", cooled.steps[19]["radiated"] - 0.01 * abs(e20)))
    assert not fallen.ok and "fell" in fallen.reason
    # 0.98 of a negative energy is less negative: a rise of 2 %
    risen = SIM.conserved(broken(cooled, 20, "energy", cooled.steps[19]["energy"] * 0.98))
    assert not risen.ok and "rose" in risen.reason
    # an energy that stands still while the ledger books a loss is caught by the ledger
    stuck = SIM.conserved(broken(cooled, 20, "energy", cooled.steps[19]["energy"]))
    assert not stuck.ok and "energy changed at step 20" in stuck.reason


def test_fast_rotator_with_cooling_makes_a_disc_with_a_spiral_and_clumps():
    runs = [run(seed, 0.35, 0.4) for seed in FRESH]
    assert all(r.summary["min_flattening"] < 0.2 for r in runs)
    assert all(r.summary["spiral_ever"] == "yes" for r in runs)
    assert all(r.summary["clumps"] >= 3 for r in runs)
    assert all(r.steps[0]["stage"] == "cloud" and r.steps[-1]["stage"] in ("fragmenting", "clumped") for r in runs)
    # the disc is real: thin, and wide for its mass because the angular momentum is still there
    assert all(r.steps[-1]["flattening"] < 0.1 and r.steps[-1]["radius"] > 0.15 for r in runs)


def test_slow_rotator_with_strong_cooling_ends_in_clumps():
    runs = [run(seed, 0.1, 0.6) for seed in FRESH]
    assert sum(r.summary["clumps"] >= 2 for r in runs) >= 4
    assert all(max(s["clumps"] for s in r.steps) >= 3 for r in runs)  # every run fragments on the way
    assert all(r.summary["largest_clump"] > 0.7 and r.steps[-1]["stage"] == "clumped" for r in runs)
    assert all(r.summary["collapse_time"] != "never" and r.summary["collapse_time"] <= 1.5 for r in runs)
    # less angular momentum: every slow rotator ends tighter than every fast one
    assert max(r.steps[-1]["radius"] for r in runs) < min(run(seed, 0.35, 0.4).steps[-1]["radius"] for seed in FRESH)


def test_without_cooling_no_disc_forms():
    for spin in (0.1, 0.2):
        for seed in FRESH[:2]:
            r = run(seed, spin, 0.0)
            assert r.summary["min_flattening"] > 0.6
            assert r.summary["spiral_ever"] == "no" and r.summary["collapse_time"] == "never"
            assert all(s["stage"] in ("cloud", "collapsing") for s in r.steps)
            assert r.summary["largest_clump"] < 0.1
    # above spin 1/3 the cloud turns oblate by rotation alone, but still no disc
    for seed in FRESH[:2]:
        r = run(seed, 0.4, 0.0)
        assert 0.5 < r.summary["min_flattening"] < 0.9 and r.summary["spiral_ever"] == "no"
        assert all(s["stage"] != "disc" for s in r.steps)
