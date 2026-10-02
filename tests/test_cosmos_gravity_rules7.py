"""Level 2 under rules 7: the corrections of the round-7 review (``haishool/cosmos/gravity.py``,
section "Rules 7"). Round 6 itself is held by ``tests/test_cosmos_gravity.py`` and
``tests/test_round6_frozen.py``; nothing here changes what a call without ``rules`` gives."""
import copy
import hashlib
import inspect
import math
import os
import random
import subprocess
import sys
import time
from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest

from haishool.cosmos import Rollout, Simulation, chem, close, gravity, life, nucleo, planets
from haishool.cosmos.gravity import (DT, EPS, KEYS, KEYS7, LESSONS, LINE_KEYS, LINE_KEYS7, MIN_MEMBERS7, N,
                                     N_OUT, N_OUT7, PARAM_KEYS7, PARAM_QUESTIONS7, PATTERNS7, RULES, STAGES7,
                                     SUB_STEPS, SUMMARY_KEYS7, T_FF, T_FF_MYR, _accel, _accel7, _drag7, _normal,
                                     _potential, bound_groups, canonical, evolve, initial_state, lesson_gate, lines,
                                     lines7, measure, measure7, parse_prompt, parse_prompt7, pattern7,
                                     random_params, rollout, simulation, stage7)
from haishool.evo import LessonGate
from haishool.truth import is_dense, num, parse_num

SIM = simulation()
#: the 40 seeds of the plausibility targets, each with ``random_params(rng, rules=7)``
SEEDS40 = tuple(range(1, 41))
#: seeds whose lines are checked one by one: a cooled middle rotator (5: spin 0.19, cooling 0.5), a
#: cooled slow one (1: 0.06, 0.48), the uncooled control (4: 0.09, 0) and a cooled fast one (23: 0.28, 0.55)
LINE_SEEDS = (5, 1, 4, 23)
ROOT = Path(__file__).resolve().parents[1]


def n_tokens(text: str) -> int:
    """The count of ``haishool.student.tokens`` (not imported: it needs torch)."""
    return len(text.replace(".", " . ").split())


@lru_cache(maxsize=None)
def roll(seed: int) -> Rollout:
    """The rules-7 run a seed names (do not change what comes back: it is shared)."""
    return rollout(seed, rules=7)


@pytest.fixture(scope="module")
def all_lines():
    return {seed: lines(roll(seed)) for seed in LINE_SEEDS}


@pytest.fixture(scope="module")
def runs40():
    return {seed: roll(seed) for seed in SEEDS40}


def normal(seed: int, shape: tuple[int, ...], sigma: float = 1.0) -> np.ndarray:
    """Made-up test data from the module's own generator (no numpy random stream)."""
    return sigma * _normal(random.Random(seed), math.prod(shape)).reshape(shape)


def state_digest(seed: int, spin: float, cooling: float, n: int) -> str:
    """The last positions and velocities of a rules-7 run, bit for bit."""
    for x, v, _ in evolve(seed, spin, cooling, n=n, rules=7):
        pass
    return hashlib.sha256(x.tobytes() + v.tobytes()).hexdigest()[:16]


def corr(a: list[float], b: list[float]) -> float:
    return float(np.corrcoef(np.array(a, dtype=float), np.array(b, dtype=float))[0, 1])


def wrongs(answer: str) -> list[str]:
    """Wrong answers: another word; or the leading digit changed by 5, the sign flipped, the value doubled."""
    if answer in ("yes", "no"):
        return ["no" if answer == "yes" else "yes", "maybe", "1"]
    if answer in STAGES7:
        return [STAGES7[(STAGES7.index(answer) + 1) % len(STAGES7)], "clumped", "2"]
    if parse_num(answer) is None:  # never, none, pair, lopsided bar, ...
        other = "bar" if answer != "bar" else "pair"
        return ["1 point 2", "0", "no", other, answer + " " + other]
    words = answer.split()
    i = next((i for i, w in enumerate(words) if w.isdigit() and w != "0"), 0)
    words[i] = str((int(words[i]) + 5) % 10)
    out = [" ".join(words), "never", "none", "yes"]
    value = parse_num(answer)
    if value != 0:
        out += [answer[6:] if answer.startswith("minus ") else "minus " + answer, num(2 * value, sig=3)]
    else:
        out += ["1", "0 point 5"]
    return out


# ---------------------------------------------------------------------------------------------
# the default stays round 6; how the corrected level is called
# ---------------------------------------------------------------------------------------------

def test_the_default_is_still_round_6():
    assert [random_params(random.Random(s)) for s in (5, 1, 4)] == [
        {"spin": 0.27, "cooling": 0.5}, {"spin": 0.1, "cooling": 0.48}, {"spin": 0.13, "cooling": 0.0}]
    assert random_params(random.Random(9)) == random_params(random.Random(9), rules=6)
    small = SIM.run(3, spin=0.3, cooling=0.2, n=64)
    assert "rules" not in small.params and len(small.steps) == N_OUT == 41 and list(small.steps[0]) == list(KEYS)
    assert small.steps == SIM.run(3, spin=0.3, cooling=0.2, n=64, rules=6).steps
    assert small.steps == gravity.run(3, spin=0.3, cooling=0.2, n=64).steps
    assert LINE_KEYS == ["time", "radius", "flattening", "clumps", "largest_clump", "spiral", "energy",
                         "angular_momentum", "stage"]
    assert inspect.signature(SIM.run).parameters["rules"].default == 6
    assert inspect.signature(random_params).parameters["rules"].default == 6
    assert inspect.signature(evolve).parameters["rules"].default == 6
    assert inspect.signature(measure).parameters["rules"].default == 6
    for bad in (5, 8, 0, "7"):
        with pytest.raises(ValueError, match="rules"):
            SIM.run(3, n=64, rules=bad)
        with pytest.raises(ValueError, match="rules"):
            random_params(random.Random(1), rules=bad)
        with pytest.raises(ValueError, match="rules"):
            rollout(3, rules=bad)
    # the level's own methods refuse them too (rules 5 was once taken for round 6 without a word)
    for bad in (5, 8, 0, -7):
        with pytest.raises(ValueError, match="rules"):
            SIM.random_params(random.Random(1), rules=bad)
        with pytest.raises(ValueError, match="rules"):
            SIM.rollout(3, rules=bad)
        with pytest.raises(ValueError, match="rules"):
            simulation(rules=7).random_params(random.Random(1), rules=bad)
        with pytest.raises(ValueError, match="rules"):
            simulation(rules=7).rollout(3, rules=bad)
    for seed in range(1, 21):  # with rules=6 said or not: the same parameters, and nothing says "rules"
        drawn = random_params(random.Random(seed))
        assert drawn == random_params(random.Random(seed), rules=6) == SIM.random_params(random.Random(seed), rules=6)
        assert set(drawn) == {"spin", "cooling"}
        a, b = gravity.run(seed, n=64, **drawn), gravity.run(seed, rules=6, n=64, **drawn)
        assert (a.params, a.steps, a.summary) == (b.params, b.steps, b.summary) and repr(a.steps) == repr(b.steps)
        assert "rules" not in a.params and "rules" not in b.params
    # the round-6 run of a seed is another run than its rules-7 run, and still has its own lines
    old, new = rollout(4), roll(4)
    assert "rules" not in old.params and new.params["rules"] == 7
    assert old.params["spin"] == 0.13 and new.params["spin"] == 0.09
    assert len(lines(old)) == 776 and not {ln.text for ln in lines(old)} & {ln.text for ln in lines(new)}
    assert not any("rules" in ln.text.split() for ln in lines(old))
    assert "Rules 7" in gravity.__doc__


def test_it_is_a_simulation_with_lessons():
    assert isinstance(SIM, Simulation) and SIM.sim == gravity.SIM == "gravity"
    for name in ("simulation", "random_params", "run", "conserved", "lines", "check", "owns", "lesson_gate"):
        assert callable(getattr(gravity, name)), name
    assert lesson_gate() is LESSONS and isinstance(LESSONS, LessonGate)
    assert LESSONS.sim == "gravity" and LESSONS.topic == "predict_gravity" and LESSONS.rules is RULES
    assert len(RULES) >= 8
    assert set(LINE_KEYS7) < set(KEYS7) and set(KEYS7) - set(LINE_KEYS7) == {"angular_momentum", "mass", "momentum"}
    assert "spiral" not in KEYS7 and "collapse_time" not in SUMMARY_KEYS7 and "spiral_ever" not in SUMMARY_KEYS7
    assert set(PARAM_QUESTIONS7) < set(PARAM_KEYS7)
    r = gravity.run(5, rules=7, **random_params(random.Random(5), rules=7))
    assert r.steps == roll(5).steps and r.summary == roll(5).summary and r.params == roll(5).params
    assert gravity.conserved(r).ok
    # the level bound to rules 7: the same runs without saying rules=7 at every call
    sim7 = simulation(rules=7)
    assert isinstance(sim7, Simulation) and sim7.rules == 7 and sim7.KEYS is KEYS7
    assert SIM.rules == 6 and SIM.KEYS is KEYS
    assert sim7.random_params(random.Random(5)) == random_params(random.Random(5), rules=7)
    assert SIM.random_params(random.Random(5)) == random_params(random.Random(5))
    assert sim7.rollout(5).params == roll(5).params and SIM.rollout(5).params == rollout(5).params
    small = sim7.run(3, spin=0.2, cooling=0.3, n=64)
    assert small.params["rules"] == 7 and len(small.steps) == N_OUT7
    assert small.steps == SIM.run(3, spin=0.2, cooling=0.3, n=64, rules=7).steps
    assert small.steps == gravity.run(3, rules=7, spin=0.2, cooling=0.3, n=64).steps
    assert sim7.conserved(small).ok and sim7.owns("gravity seed 5 step 1 2 clumps")
    assert sim7.check("gravity seed 5 rules 7 final star_time", "1 point 3").ok
    with pytest.raises(ValueError, match="rules"):
        simulation(rules=8)


def test_random_params_under_rules_7():
    drawn = [random_params(random.Random(s), rules=7) for s in range(300)]
    assert drawn == [random_params(random.Random(s), rules=7) for s in range(300)]
    assert all(set(p) == {"spin", "cooling"} for p in drawn)
    assert all(0.02 <= p["spin"] <= 0.3 for p in drawn) and max(p["spin"] for p in drawn) < 1 / 3
    assert min(p["spin"] for p in drawn) <= 0.03 and max(p["spin"] for p in drawn) >= 0.29
    assert all(p["cooling"] == 0 or 0.1 <= p["cooling"] <= 0.6 for p in drawn)
    assert 0.1 < sum(p["cooling"] == 0 for p in drawn) / len(drawn) < 0.25
    # the same draws in the same order as round 6: only the spin range differs
    assert [p["cooling"] for p in drawn] == [random_params(random.Random(s))["cooling"] for s in range(300)]
    assert [random_params(random.Random(s), rules=7) for s in LINE_SEEDS] == [
        {"spin": 0.19, "cooling": 0.5}, {"spin": 0.06, "cooling": 0.48}, {"spin": 0.09, "cooling": 0.0},
        {"spin": 0.28, "cooling": 0.55}]


def test_bad_parameters_are_refused():
    # above spin 1/3 no start is in virial balance: round 6 clamps, rules 7 refuse
    with pytest.raises(ValueError, match="1/3"):
        SIM.run(1, spin=0.34, cooling=0.3, n=64, rules=7)
    assert len(SIM.run(1, spin=0.34, cooling=0.3, n=64).steps) == 41
    for bad in ({"cooling": -0.1}, {"spin": -0.1}, {"n": 9}, {"mass": 0.0}, {"eps": 0.0}):
        with pytest.raises(ValueError):
            SIM.run(1, rules=7, **bad)


# ---------------------------------------------------------------------------------------------
# determinism
# ---------------------------------------------------------------------------------------------

def test_same_seed_gives_bit_identical_rollouts():
    a = SIM.run(23, rules=7, **random_params(random.Random(23), rules=7))
    b = SIM.run(23, rules=7, **random_params(random.Random(23), rules=7))
    assert a.steps == b.steps == roll(23).steps and a.summary == b.summary and a.params == b.params
    assert [ln.text for ln in lines(a)] == [ln.text for ln in lines(roll(23))]
    assert state_digest(3, 0.3, 0.2, 64) == state_digest(3, 0.3, 0.2, 64)  # positions and velocities, bit for bit
    assert roll(5).steps != roll(23).steps
    # the gate's replay is cached; a caller gets a copy and cannot spoil it
    mine = rollout(5, rules=7)
    assert mine is not rollout(5, rules=7)
    mine.steps[12]["clumps"] = 99
    mine.summary["star_time"] = 9.9
    assert rollout(5, rules=7).steps[12]["clumps"] == 3 and SIM.check("gravity seed 5 rules 7 step 1 2 clumps", "3").ok
    assert SIM.check("gravity seed 5 rules 7 final star_time", "1 point 3").ok


def test_bits_do_not_depend_on_the_cpu_path():
    # as in round 6: with numpy's AVX loops switched off the rules-7 run must come out the same
    umath = getattr(getattr(np, "_core", None), "_multiarray_umath", None)
    features = getattr(umath, "__cpu_features__", {})
    switchable = [f for f in getattr(umath, "__cpu_dispatch__", []) if features.get(f)]
    if not switchable:
        pytest.skip("this numpy has no CPU features to switch off")
    code = ("import hashlib\nfrom haishool.cosmos.gravity import evolve\n"
            "for x, v, _ in evolve(3, 0.3, 0.2, n=64, rules=7):\n    pass\n"
            "print(hashlib.sha256(x.tobytes() + v.tobytes()).hexdigest()[:16])")
    env = dict(os.environ, NPY_DISABLE_CPU_FEATURES=" ".join(switchable), PYTHONPATH=str(ROOT))
    done = subprocess.run([sys.executable, "-c", code], env=env, cwd=ROOT, capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    assert done.stdout.split()[-1] == state_digest(3, 0.3, 0.2, 64)


def test_known_rollout():
    # the canary of rules 7, like test_known_rollout of round 6: if it fails on another machine or
    # numpy, the gate there replays other runs than the lines were made from; rebuild the lines there
    assert state_digest(3, 0.3, 0.2, 64) == "ee9e593756bc03f5"
    r = roll(5)
    assert r.params == {"spin": 0.19, "cooling": 0.5, "n": 256, "mass": 1.0, "eps": 0.05, "dt": 0.00444,
                        "t_ff": 1.11, "rules": 7, "cooling_time": 0.2, "bar_unstable": "yes", "t_ff_myr": 0.428,
                        "angular_momentum_z": 0.275, "centrifugal_radius": 0.0756, "radius_start": 0.642}
    assert r.summary == {"stage": "multiple", "clumps": 2, "star": 0.496, "companions": 1, "disc_mass": 0.273,
                         "disc_radius": 0.212, "flattening": 0.43, "min_flattening": 0.356, "star_time": 1.3,
                         "patterns": "none", "angular_momentum_z": 0.275}
    digests = {seed: hashlib.sha256(repr(roll(seed).steps).encode()).hexdigest()[:16] for seed in LINE_SEEDS}
    assert digests == {5: "af5eef7002045ae7", 1: "5f56215afc4a628a", 4: "8ac08e0d9d89eab2", 23: "94e6eb38c0795cca"}
    assert roll(4).summary["star_time"] == "never" and roll(23).summary["patterns"] == "pair lopsided bar"


def test_the_buffered_force_is_the_round_6_force():
    # the same operations in the same order: the same bits, on a cloud and on a collapsed state
    eps2 = EPS * EPS
    for x, _, m in (initial_state(5, 256, 1.0, 0.27, 0.05), initial_state(9, 23, 1.0, 0.1, 0.05)):
        n = len(m)
        buf = [np.empty((n, n)) for _ in range(6)]
        assert np.array_equal(_accel7(x, m, eps2, buf), _accel(x, m, eps2))
        assert np.array_equal(buf[0], x[:, 0][:, None] - x[:, 0][None, :])  # what the drag reads afterwards
        assert np.array_equal(buf[3], gravity._pairs(x, eps2)[3])
        tight = 0.02 * x
        assert np.array_equal(_accel7(tight, m, eps2, buf), _accel(tight, m, eps2))
    # through a whole run: without cooling the two rules differ in nothing but the force routine
    old = evolve(11, 0.15, 0.0, n=64)
    new = evolve(11, 0.15, 0.0, n=64, rules=7)
    for k in range(N_OUT):
        (x6, v6, _), (x7, v7, _) = next(old), next(new)
        assert np.array_equal(x6, x7) and np.array_equal(v6, v7), k


def test_the_shared_distances_are_the_same_bits():
    # measure7 computes the squared distances once for the clump finder and both potential energies;
    # they must be the numbers the round-6 routines compute themselves
    for seed, n, sigma in ((5, 256, 0.3), (9, 23, 0.02), (4, 2, 0.5), (7, 31, 0.001)):
        x = normal(seed, (n, 3), sigma)
        m = np.full(n, 1.0 / n)
        d2 = gravity._square_distances(x)
        assert np.array_equal(d2, gravity._pairs(x, 0.0)[3]) and np.array_equal(d2 + EPS * EPS, gravity._pairs(x, EPS * EPS)[3])
        assert gravity._potential_d2(d2, m, EPS * EPS) == _potential(x, m, EPS * EPS)
        part = np.arange(0, n, 2)
        assert gravity._potential_d2(d2[np.ix_(part, part)], m[part], EPS * EPS) == _potential(x[part], m[part], EPS * EPS)
        v = normal(seed + 1, (n, 3), 0.01)
        assert [g.tolist() for g in bound_groups(x, v, m, EPS)] == [g.tolist() for g in bound_groups(x, v, m, EPS, d2)]
    # and on a state of a real run, where a star and a disc exist
    for k, (x, v, m) in enumerate(evolve(23, 0.28, 0.55, rules=7)):
        if k == 40:
            d2 = gravity._square_distances(x)
            assert gravity._potential_d2(d2, m, EPS * EPS) == _potential(x, m, EPS * EPS)
            groups = bound_groups(x, v, m, EPS, d2)
            assert len(groups) >= 1 and [g.tolist() for g in groups] == [g.tolist() for g in bound_groups(x, v, m, EPS)]
            break


def py_total(values: list[float]) -> float:
    """The order of ``gravity._total`` in plain Python: neighbours in pairs, level by level."""
    values = list(values)
    while len(values) > 1:
        n = len(values)
        values = [values[i] + values[i + 1] for i in range(0, n - 1, 2)] + ([values[-1]] if n % 2 else [])
    return values[0] if values else 0.0


def py_run7(seed: int, n: int, spin: float, cooling: float, eps: float, steps: int):
    """``steps`` leapfrog steps with the pairwise drag in plain Python floats: the same operations
    in the same order as the module, without numpy. Returns positions, velocities, energy radiated."""
    x0, v0, m0 = initial_state(seed, n, 1.0, spin, eps)
    x, v, m = x0.tolist(), v0.tolist(), m0.tolist()
    eps2 = eps * eps
    reach2 = gravity.H7 * gravity.H7 + eps2
    f0 = gravity.COOL_RATE7 * cooling * (DT / T_FF)

    def accel(x):
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

    def kinetic(v):
        return 0.5 * py_total([m[i] * (v[i][0] * v[i][0] + v[i][1] * v[i][1] + v[i][2] * v[i][2]) for i in range(n)])

    radiated = 0.0
    a = accel(x)
    for _ in range(steps):
        v = [[v[i][c] + 0.5 * DT * a[i][c] for c in range(3)] for i in range(n)]
        x = [[x[i][c] + DT * v[i][c] for c in range(3)] for i in range(n)]
        a = accel(x)
        v = [[v[i][c] + 0.5 * DT * a[i][c] for c in range(3)] for i in range(n)]
        pairs = []
        for i in range(n):  # the pairs i < j, row by row
            for j in range(i + 1, n):
                d = [x[i][c] - x[j][c] for c in range(3)]
                if d[0] * d[0] + d[1] * d[1] + d[2] * d[2] + eps2 < reach2:
                    closing = (v[i][0] - v[j][0]) * d[0] + (v[i][1] - v[j][1]) * d[1] + (v[i][2] - v[j][2]) * d[2]
                    if closing < 0.0:
                        pairs.append((i, j, d, closing))
        if cooling > 0 and pairs:
            count = [0] * n
            for i, j, _, _ in pairs:
                count[i] += 1
                count[j] += 1
            before = kinetic(v)
            gained = [[0.0, 0.0, 0.0] for _ in range(n)]  # what a particle gets as the second of a pair
            given = [[0.0, 0.0, 0.0] for _ in range(n)]   # and as the first
            for i, j, d, closing in pairs:
                cap = max(1.0, f0 * max(count[i], count[j]))
                s = 0.5 * f0 / cap * closing / (d[0] * d[0] + d[1] * d[1] + d[2] * d[2])
                for c in range(3):
                    gained[j][c] += s * d[c]
                    given[i][c] += s * d[c]
            v = [[v[i][c] + (gained[i][c] - given[i][c]) for c in range(3)] for i in range(n)]
            radiated += before - kinetic(v)
    return x, v, radiated


def test_the_drag_is_the_same_bits_as_plain_python_arithmetic():
    # numpy only vectorises: nonzero walks the pairs row by row and bincount adds in that order
    for seed, spin, cooling in ((7, 0.3, 0.4), (8, 0.1, 0.6)):
        ledger: dict = {}
        states = evolve(seed, spin, cooling, n=31, ledger=ledger, rules=7)
        for k in range(3):
            x, v, m = next(states)
            px, pv, radiated = py_run7(seed, 31, spin, cooling, 0.05, k * SUB_STEPS)
            assert x.tolist() == px and v.tolist() == pv, (seed, k)
            assert ledger["radiated"] == radiated and (radiated > 0) == (k > 0)


def test_one_pair_loses_the_drag_factor_of_its_closing_speed():
    # two particles 0.1 apart: 3 towards each other along the line, 1 across it
    x = np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]])
    m = np.array([0.5, 0.5])
    f0 = gravity.drag_factor(0.5, DT / T_FF)
    assert f0 == pytest.approx(10 * 0.5 / 250) and f0 == gravity.COOL_RATE7 * 0.5 * (DT / T_FF)
    buf, pick = [np.empty((2, 2)) for _ in range(6)], np.empty((2, 2), dtype=bool)
    reach2 = gravity.H7 * gravity.H7 + EPS * EPS
    for speed in (3.0, -3.0):  # approaching, then receding
        v = np.array([[0.5 * speed, 0.5, 0.2], [-0.5 * speed, -0.5, -0.2]])
        _accel7(x, m, EPS * EPS, buf)
        lost = _drag7(v, m, buf, pick, reach2, f0)
        if speed > 0:
            assert v[0, 0] - v[1, 0] == pytest.approx(speed * (1 - f0)) and lost > 0
            assert lost == pytest.approx(0.5 * 0.25 * speed * speed * (1 - (1 - f0) * (1 - f0)))  # reduced mass 1/4
        else:
            assert v[0, 0] - v[1, 0] == speed and lost == 0.0  # only approaching pairs are damped
        assert v[:, 1].tolist() == [0.5, -0.5] and v[:, 2].tolist() == [0.2, -0.2]  # nothing across the line
        assert (v[0] + v[1]).tolist() == [0.0, 0.0, 0.0]
    # beyond the reach nothing happens
    far = np.array([[0.0, 0.0, 0.0], [0.16, 0.0, 0.0]])
    v = np.array([[1.5, 0.0, 0.0], [-1.5, 0.0, 0.0]])
    _accel7(far, m, EPS * EPS, buf)
    assert _drag7(v, m, buf, pick, reach2, f0) == 0.0 and v[0, 0] == 1.5


def test_the_drag_in_a_crowd():
    # n particles falling towards their common centre, all within reach of each other: v = -5 x
    def infall(n: int, f0: float) -> np.ndarray:
        x = normal(n, (n, 3), 0.015)
        x -= x.mean(0)
        assert float(np.sqrt((x * x).sum(1)).max()) < 0.07  # every pair is closer than H7
        v, m = -5.0 * x, np.full(n, 1.0 / n)
        buf, pick = [np.empty((n, n)) for _ in range(6)], np.empty((n, n), dtype=bool)
        _accel7(x, m, EPS * EPS, buf)
        lost = _drag7(v, m, buf, pick, gravity.H7 * gravity.H7 + EPS * EPS, f0)
        assert lost > 0 and np.abs(v.sum(0)).max() < 1e-13 and np.abs(np.cross(x, v).sum(0)).max() < 1e-13
        return (v * x).sum(1) / (-5.0 * (x * x).sum(1))  # what is left of each particle's inward speed

    f0 = gravity.drag_factor(0.6, DT / T_FF)
    assert f0 == pytest.approx(0.024) and 41 < 1 / f0 < 42
    # few neighbours: a particle in c pairs is braked c times as fast as a single pair
    assert infall(20, f0) == pytest.approx(1 - 0.5 * f0 * 20, abs=1e-9)
    # from 1 / f0 neighbours on the cap holds: half the inflow is gone in one step, whatever the cooling,
    # and nothing is ever turned around (never more than a full stop)
    for n, f in ((60, f0), (60, 0.5), (200, f0), (120, 4.0)):
        left = infall(n, f)
        assert 0.45 < left.min() and left.max() < 0.65, (n, f)
    assert np.array_equal(infall(60, f0), infall(60, 0.5))


def test_the_drag_keeps_momentum_and_angular_momentum_and_only_takes_energy():
    ledger: dict = {}
    first = None
    radiated, energy = [], []
    for x, v, m in evolve(23, 0.28, 0.55, ledger=ledger, rules=7):
        p = (m[:, None] * v).sum(0)
        ang = (m[:, None] * np.cross(x, v)).sum(0)
        e = 0.5 * float((m * (v * v).sum(1)).sum()) + _potential(x, m, EPS * EPS)
        first = first or (ang, e)
        # all three components, with no axis told and no reset: to rounding
        assert np.abs(p).max() < 1e-13 and np.abs(ang - first[0]).max() < 1e-13
        assert abs(e + ledger["radiated"] - first[1]) < 0.005 * abs(e)
        radiated.append(ledger["radiated"])
        energy.append(e)
    assert radiated[0] == 0 and all(b > a for a, b in zip(radiated, radiated[1:]))
    assert all(b < a for a, b in zip(energy, energy[1:]))
    assert len(energy) == N_OUT7 == 61


# ---------------------------------------------------------------------------------------------
# the rollout and its lines
# ---------------------------------------------------------------------------------------------

def test_rollout_shape():
    for seed in LINE_SEEDS:
        r = roll(seed)
        assert r.sim == "gravity" and r.seed == seed and len(r.steps) == N_OUT7 == 61
        assert list(r.summary) == list(SUMMARY_KEYS7)
        p = r.params
        assert {k: p[k] for k in ("spin", "cooling")} == random_params(random.Random(seed), rules=7)
        assert (p["n"], p["mass"], p["eps"], p["dt"], p["t_ff"], p["rules"]) == (256, 1.0, 0.05, 0.00444, 1.11, 7)
        assert p["t_ff_myr"] == T_FF_MYR == 0.428
        assert p["bar_unstable"] == gravity.bar_unstable(p["spin"]) == ("yes" if p["spin"] > 0.14 else "no")
        assert p["cooling_time"] == ("never" if p["cooling"] == 0 else float(f"{1 / (10 * p['cooling']):.3g}"))
        first = r.steps[0]
        assert p["angular_momentum_z"] == first["angular_momentum"] and p["radius_start"] == first["radius"]
        assert p["centrifugal_radius"] == float(f"{gravity.centrifugal_radius(first['angular_momentum'], 1.0):.3g}")
        for k, s in enumerate(r.steps):
            assert list(s) == list(KEYS7)
            assert s["time"] == k / 10
            assert s["stage"] in STAGES7 and s["pattern"] in PATTERNS7
            assert s["stage"] == stage7(s["radius"], s["flattening"], s["clumps"], s["largest_clump"], first["radius"])
            assert isinstance(s["clumps"], int) and s["clumps"] >= 0
            assert (s["clumps"] == 0) == (s["largest_clump"] == 0)
            assert s["clumps"] == 0 or s["largest_clump"] >= MIN_MEMBERS7 / 256 - 1e-4
            assert 0 <= s["largest_clump"] <= 1 and 0 <= s["disc_mass"] <= 1
            assert s["largest_clump"] + s["disc_mass"] <= 1.002
            assert s["radius"] > 0 and s["flattening"] > 0 and s["disc_radius"] >= 0
            assert s["toomre_q"] == "none" or (s["toomre_q"] > 0 and s["flattening"] < 0.5)
            assert s["flattening"] < 0.5 or s["toomre_q"] == "none"
            assert s["pattern"] != "pair" or s["clumps"] >= 2
            # what holds the cloud up: the two shares add up to half the virial ratio
            assert s["random"] == pytest.approx(s["support"] / 2 - s["rotation"], abs=0.006)
            assert all(isinstance(s[key], float) for key in KEYS7
                       if key not in ("clumps", "pattern", "stage", "toomre_q"))
            assert all(float(f"{s[key]:.3g}") == s[key] for key in KEYS7 if isinstance(s[key], float))
        last = r.steps[-1]
        assert (r.summary["stage"], r.summary["clumps"], r.summary["star"], r.summary["disc_mass"]) == (
            last["stage"], last["clumps"], last["largest_clump"], last["disc_mass"])
        assert (r.summary["disc_radius"], r.summary["flattening"]) == (last["disc_radius"], last["flattening"])
        assert r.summary["companions"] == max(0, last["clumps"] - 1)
        assert r.summary["min_flattening"] == min(s["flattening"] for s in r.steps)
        starred = [s["time"] for s in r.steps if s["largest_clump"] >= 0.25]
        assert r.summary["star_time"] == (starred[0] if starred else "never")
        seen = {s["pattern"] for s in r.steps}
        assert r.summary["patterns"] == (" ".join(w for w in ("pair", "lopsided", "bar") if w in seen) or "none")
        assert r.summary["angular_momentum_z"] == last["angular_momentum"] == first["angular_momentum"]


def test_lines_are_dense_and_short(all_lines):
    for seed, lns in all_lines.items():
        head = f"gravity seed {num(seed)} rules 7 "
        assert len(lns) == 1 + 5 + 61 + 61 * 16 + 60 * 16 + 11 == 2014
        assert len({ln.text for ln in lns}) == len(lns)
        assert sum(ln.kind == "record" for ln in lns) == 62
        for ln in lns:
            assert ln.topic == "gravity"
            assert is_dense(ln.text), ln.text
            assert n_tokens(ln.text) <= 128, ln.text
            assert ln.prompt.startswith(head), ln.text  # no line can be taken for the round-6 run
            if ln.kind == "record":
                assert ln.answer == ""
                fields = [field.split()[0] for field in ln.text[:-1].split(". ")[1:]]
                assert fields == (list(PARAM_KEYS7) if ln.text.startswith(head + "params.") else LINE_KEYS7)
            else:
                assert "." not in ln.prompt and "." not in ln.answer and ln.answer
                assert ln.text == f"q {ln.prompt}. a {ln.answer}."
        assert lns[0].text.startswith(head + "params. spin ") and lns[6].text.startswith(head + "step 0. time 0. ")
        assert [ln.prompt for ln in lns[1:6]] == [head + "params " + key for key in PARAM_QUESTIONS7]
        assert [ln.prompt for ln in lns[-11:]] == [head + "final " + key for key in SUMMARY_KEYS7]
    assert max(n_tokens(ln.text) for lns in all_lines.values() for ln in lns) <= 112
    thin = lines(roll(5), every=4)
    assert len(thin) == 1 + 5 + 61 + 16 * 16 + 15 * 16 + 11
    assert set(ln.text for ln in thin) <= set(ln.text for ln in all_lines[5])
    assert all_lines[5][0].text == (
        "gravity seed 5 rules 7 params. spin 0 point 1 9. cooling 0 point 5. cooling_time 0 point 2. "
        "bar_unstable yes. particles 2 5 6. softening 0 point 0 5. t_ff 1 point 1 1. t_ff_myr 0 point 4 2 8. "
        "angular_momentum_z 0 point 2 7 5. centrifugal_radius 0 point 0 7 5 6. radius_start 0 point 6 4 2.")
    assert all_lines[5][6 + 12 * 33].text == (
        "gravity seed 5 rules 7 step 1 2. time 1 point 2. radius 0 point 2 5 5. flattening 0 point 6 6 2. clumps 3. "
        "largest_clump 0 point 0 7 4 2. disc_mass 0 point 6 8 8. disc_radius 0 point 2 7 3. pattern none. "
        "energy minus 0 point 9 0 9. radiated 0 point 5 5 5. support 0 point 6 3 1. rotation 0 point 1 1 6. "
        "random 0 point 1 9 9. jeans_number 6 point 0 8. toomre_q none. stage fragmenting.")
    texts = {ln.text for ln in all_lines[5]}
    for text in ("q gravity seed 5 rules 7 params spin. a 0 point 1 9.",
                 "q gravity seed 5 rules 7 step 1 2 clumps. a 3.",
                 "q gravity seed 5 rules 7 step 1 2 next stage. a star.",
                 "q gravity seed 5 rules 7 final star_time. a 1 point 3.",
                 "q gravity seed 5 rules 7 final companions. a 1."):
        assert text in texts, text
    assert "q gravity seed 4 rules 7 final star_time. a never." in {ln.text for ln in all_lines[4]}
    assert "q gravity seed 4 rules 7 params cooling_time. a never." in {ln.text for ln in all_lines[4]}


def test_the_longest_possible_seed_still_fits(monkeypatch):
    lns = lines(roll(5))
    extra = n_tokens("gravity seed 9 9 9 9 rules 7 step 6 0.") - n_tokens("gravity seed 5 rules 7 step 0.")
    assert max(n_tokens(ln.text) for ln in lns) + extra <= 128
    # and should a state ever need more than 128 tokens, lines() says so instead of writing it
    assert gravity.MAX_TOKENS7 == 128
    monkeypatch.setattr(gravity, "MAX_TOKENS7", 100)
    with pytest.raises(ValueError, match="longer than 100 tokens"):
        lines(roll(5))


def test_numbers_have_at_most_three_digits(all_lines):
    for lns in all_lines.values():
        for ln in lns:
            if ln.kind != "record" and parse_num(ln.answer) is not None:
                digits = "".join(w for w in ln.answer.split(" e ")[0].split() if w.isdigit()).lstrip("0")
                assert len(digits) <= 3, ln.text
                assert num(parse_num(ln.answer), sig=3) == ln.answer  # one spelling per number


def test_lines_only_for_the_run_a_seed_names():
    assert all(canonical(roll(seed)) for seed in LINE_SEEDS) and canonical(rollout(5))
    r = roll(5)
    assert [ln.text for ln in lines(r)] == [ln.text for ln in lines(r, rules=7)] == [ln.text for ln in lines7(r)]
    with pytest.raises(ValueError, match="rules"):
        lines(r, rules=6)
    with pytest.raises(ValueError, match="rules"):
        lines(rollout(5), rules=7)
    with pytest.raises(ValueError):
        lines7(rollout(5))
    other = SIM.run(101, spin=0.25, cooling=0.4, n=64, rules=7)  # not the parameters seed 101 names
    assert not canonical(other)
    with pytest.raises(ValueError, match="rollout"):
        lines(other)
    # the round-6 parameters of the seed under rules 7 are not its rules-7 run either
    crossed = copy.deepcopy(r)
    crossed.params["spin"] = 0.27
    assert not canonical(crossed)
    with pytest.raises(ValueError):
        lines(crossed)
    unknown = copy.deepcopy(r)
    unknown.params["rules"] = 8
    assert not canonical(unknown)


# ---------------------------------------------------------------------------------------------
# the gate: check and owns
# ---------------------------------------------------------------------------------------------

def test_gate_agrees_with_every_line(all_lines):
    for lns in all_lines.values():
        for ln in lns:
            if ln.kind != "record":
                for gate in (SIM.check, gravity.check):
                    verdict = gate(ln.prompt, ln.answer)
                    assert verdict.ok and verdict.expected == ln.answer, ln.text


def test_gate_rejects_wrong_answers(all_lines):
    for lns in all_lines.values():
        for ln in lns:
            if ln.kind != "record":
                for wrong in wrongs(ln.answer):
                    verdict = SIM.check(ln.prompt, wrong)
                    assert not verdict.ok and verdict.expected == ln.answer, (ln.text, wrong)
    for junk in ("big", "", "yes", "0 point point 3", "0 point 2 5 5 big"):
        verdict = SIM.check("gravity seed 5 rules 7 step 1 2 radius", junk)
        assert (verdict.ok, verdict.expected, verdict.reason) == (False, "0 point 2 5 5", "not a number")
    assert not SIM.check("gravity seed 5 rules 7 final star_time", "never").ok
    assert not SIM.check("gravity seed 4 rules 7 final star_time", "1 point 2").ok
    assert not SIM.check("gravity seed 5 rules 7 step 1 2 toomre_q", "1 point 2").ok
    assert not SIM.check("gravity seed 5 rules 7 step 1 2 stage", "disc").ok


def test_gate_tolerance():
    r = roll(5)
    s = r.steps[12]
    assert (s["radius"], s["clumps"], s["time"]) == (0.255, 3, 1.2)
    # measured numbers: within 5 percent
    assert SIM.check("gravity seed 5 rules 7 step 1 2 radius", num(0.255 * 1.04, sig=3)).ok
    assert not SIM.check("gravity seed 5 rules 7 step 1 2 radius", num(0.255 * 1.07, sig=3)).ok
    assert not SIM.check("gravity seed 5 rules 7 step 1 2 radius", "minus 0 point 2 5 5").ok
    # the clump count: exactly (round 6 allowed one more or less)
    assert SIM.check("gravity seed 5 rules 7 step 1 2 clumps", "3").ok
    for wrong in ("2", "4", "3 point 0", "3 point 1"):
        assert not SIM.check("gravity seed 5 rules 7 step 1 2 clumps", wrong).ok, wrong
    assert SIM.check("gravity seed 5 rules 7 final companions", "1").ok
    assert not SIM.check("gravity seed 5 rules 7 final companions", "2").ok
    # times are multiples of 0.1: exactly (round 6 took 1.25 for 1.3)
    assert SIM.check("gravity seed 5 rules 7 final star_time", "1 point 3").ok
    for wrong in ("1 point 2", "1 point 4", "1 point 2 5", "1 point 3 5"):
        assert not SIM.check("gravity seed 5 rules 7 final star_time", wrong).ok, wrong
    assert SIM.check("gravity seed 5 rules 7 step 1 2 time", "1 point 2").ok
    assert not SIM.check("gravity seed 5 rules 7 step 1 2 time", "1 point 2 5").ok
    assert SIM.check("gravity seed 5 rules 7 step 3 0 time", "3").ok
    # the params: exactly, every field of the record can be asked
    assert SIM.check("gravity seed 5 rules 7 params spin", "0 point 1 9").ok
    assert not SIM.check("gravity seed 5 rules 7 params spin", "0 point 1 9 5").ok
    assert not SIM.check("gravity seed 5 rules 7 params spin", "0 point 2").ok
    assert SIM.check("gravity seed 5 rules 7 params particles", "2 5 6").ok
    assert not SIM.check("gravity seed 5 rules 7 params particles", "2 5 0").ok
    assert SIM.check("gravity seed 5 rules 7 params softening", "0 point 0 5").ok
    assert SIM.check("gravity seed 5 rules 7 params t_ff_myr", "0 point 4 2 8").ok
    assert SIM.check("gravity seed 5 rules 7 params bar_unstable", "yes").ok
    # "next" is the state one step later
    assert r.steps[12]["stage"] == "fragmenting" and r.steps[13]["stage"] == "star"
    assert SIM.check("gravity seed 5 rules 7 step 1 2 next stage", "star").ok
    beyond = SIM.check("gravity seed 5 rules 7 step 6 0 next clumps", "2")
    assert not beyond.ok and beyond.expected is None and "beyond" in beyond.reason
    assert not SIM.check("gravity seed 5 rules 7 step 6 1 clumps", "2").ok
    assert SIM.check("gravity seed 5 rules 7 step 6 0 clumps", "2").ok
    # the gate diagnostics can be asked too, though no training line does
    assert SIM.check("gravity seed 5 rules 7 step 1 2 mass", "1").ok
    assert SIM.check("gravity seed 5 rules 7 step 1 2 angular_momentum", "0 point 2 7 5").ok
    # the same seed without "rules 7" is the round-6 run, with its own numbers and tolerance
    assert rollout(5).steps[12]["clumps"] == 8
    assert SIM.check("gravity seed 5 step 1 2 clumps", "8").ok and SIM.check("gravity seed 5 step 1 2 clumps", "9").ok
    assert not SIM.check("gravity seed 5 step 1 2 clumps", "3").ok


def test_owns_only_its_own_prompts(all_lines):
    for lns in all_lines.values():
        assert all(SIM.owns(ln.prompt) and gravity.owns(ln.prompt) for ln in lns if ln.kind != "record")
    others = [
        "turkey capital", "carbon protons", "calc 1 2 plus 7", "check 1 2 plus 7 equals 2 0",
        "nucleo seed 7 final helium_fraction", "planets seed 7 final rocky", "chem seed 7 step 3 water",
        "life seed 7 step 3 next population", "world seed 3 era planets", "stars seed 3 params efficiency",
        "cells seed 3 step 2 genomes", "signals seed 7 say predator near",
        # the same shape under another level's name
        "nucleo seed 5 rules 7 step 1 2 clumps", "planets seed 5 rules 7 final star_time",
        "world7 seed 5 rules 7 final star", "stars seed 5 rules 7 params spin",
        # round-6 words under rules 7, rules-7 words under round 6, other rules
        "gravity seed 5 rules 7 step 1 2 spiral", "gravity seed 5 rules 7 final collapse_time",
        "gravity seed 5 rules 7 final spiral_ever", "gravity seed 5 step 1 2 pattern", "gravity seed 5 final star_time",
        "gravity seed 5 step 1 2 toomre_q", "gravity seed 5 params spin", "gravity seed 5 rules 6 step 1 2 clumps",
        "gravity seed 5 rules 8 step 1 2 clumps", "gravity seed 5 rules 7 7 step 1 2 clumps",
        # malformed
        "gravity seed 5 rules 7", "gravity seed 5 rules 7 step 3", "gravity seed 5 rules 7 step 3 next",
        "gravity seed 5 rules 7 step 3 colour", "gravity seed 5 rules 7 final radius",
        "gravity seed 5 rules 7 final time", "gravity seed 5 rules 7 params n", "gravity seed 5 rules 7 params rules",
        "gravity seed 5 rules 7 params next spin", "gravity seed 5 rules 7 final next star",
        "gravity seed 5 rules 7 step 3 m1", "gravity seed 5 rules 7 step 3 sigma",
        "gravity seed 0 5 rules 7 final star", "gravity seed 5 rules 7 step 0 3 clumps",
        "gravity seed x rules 7 final star", "gravity seed minus 5 rules 7 final star",
        "gravity rules 7 seed 5 final star", "gravity seed 5 rules 7 final star ",
        "gravity seed 5 rules 7 final star extra", "q gravity seed 5 rules 7 final star",
        "gravity seed 5 rules 7 final star. a 1",
        "gravity seed 5 rules 7  final star", ""]
    for prompt in others:
        assert not SIM.owns(prompt) and not gravity.owns(prompt), prompt
        for gate in (SIM.check, gravity.check):
            verdict = gate(prompt, "4")
            assert (verdict.ok, verdict.expected, verdict.reason) == (False, None, "not my question"), prompt
    # the other levels of round 6 do not take a rules-7 gravity prompt for theirs
    for level in (nucleo, planets, chem, life):
        assert not level.simulation().owns("gravity seed 5 rules 7 final star")
    # a lesson belongs to the module's gate (and to LESSONS), not to the story gate of a seed
    lesson = "gravity predict bar_unstable spin 0 point 2"
    assert gravity.owns(lesson) and LESSONS.owns(lesson) and not SIM.owns(lesson)
    assert gravity.check(lesson, "yes").ok and not gravity.check(lesson, "no").ok


def test_parse_prompt7():
    assert parse_prompt7("gravity seed 7 rules 7 step 1 2 clumps") == (7, 12, "clumps", "step")
    assert parse_prompt7("gravity seed 7 rules 7 step 1 2 next clumps") == (7, 13, "clumps", "step")
    assert parse_prompt7("gravity seed 1 2 3 4 rules 7 final star_time") == (1234, None, "star_time", "final")
    assert parse_prompt7("gravity seed 7 rules 7 params spin") == (7, None, "spin", "params")
    assert parse_prompt7("gravity seed 7 rules 7 step 0 mass") == (7, 0, "mass", "step")
    assert parse_prompt7("gravity seed 0 rules 7 step 6 0 time") == (0, 60, "time", "step")
    assert parse_prompt7("gravity seed 7 step 1 2 clumps") is None
    assert parse_prompt7("gravity seed 7 rules 7 final energy") is None
    # and the round-6 parser does not take a rules-7 prompt
    assert parse_prompt("gravity seed 7 rules 7 step 1 2 clumps") is None
    assert parse_prompt("gravity seed 7 step 1 2 clumps") == (7, 12, "clumps")


def test_conserved_rejects_broken_rollouts():
    def broken(r: Rollout, t: int, key: str, value: float) -> Rollout:
        bad = copy.deepcopy(r)
        bad.steps[t][key] = value
        return bad

    still, cooled = roll(4), roll(5)
    assert SIM.conserved(still).ok and SIM.conserved(cooled).ok
    for r in (still, cooled):
        assert "mass" in SIM.conserved(broken(r, 20, "mass", 0.99)).reason
        assert "momentum" in SIM.conserved(broken(r, 20, "momentum", 0.01)).reason
        lz0 = r.steps[0]["angular_momentum"]
        assert "angular momentum" in SIM.conserved(broken(r, 20, "angular_momentum", lz0 * 1.02)).reason
    e0 = still.steps[0]["energy"]
    assert "energy" in SIM.conserved(broken(still, 20, "energy", e0 * 1.03)).reason
    assert "without cooling" in SIM.conserved(broken(still, 20, "radiated", 0.001)).reason
    e20, q20 = cooled.steps[20]["energy"], cooled.steps[20]["radiated"]
    assert "energy changed" in SIM.conserved(broken(cooled, 20, "radiated", q20 + 0.04 * abs(e20))).reason
    assert "energy changed" in SIM.conserved(broken(cooled, 20, "energy", e20 * 1.04)).reason
    fallen = SIM.conserved(broken(cooled, 20, "radiated", cooled.steps[19]["radiated"] - 0.01 * abs(e20)))
    assert not fallen.ok and "fell" in fallen.reason
    risen = SIM.conserved(broken(cooled, 20, "energy", cooled.steps[19]["energy"] * 0.98))
    assert not risen.ok and "rose" in risen.reason


# ---------------------------------------------------------------------------------------------
# plausibility: the corrected behaviour, measured over 40 seeds
# ---------------------------------------------------------------------------------------------

def test_gate_holds_on_40_seeds(runs40):
    for seed, r in runs40.items():
        verdict = SIM.conserved(r)
        assert verdict.ok, (seed, verdict.reason)
        assert canonical(r)
        first = r.steps[0]
        # the ledger closes far better than the gate asks (2 percent), even after the rounding to 3 digits
        e0 = first["energy"]
        assert all(abs(s["energy"] + s["radiated"] - e0) <= 0.01 * max(abs(e0), abs(s["energy"]))
                   for s in r.steps), seed
        # no axis is told and nothing is reset, yet Lz and the momentum stay: one value for the whole run
        assert len({s["angular_momentum"] for s in r.steps}) == 1 and max(s["momentum"] for s in r.steps) < 1e-12
        assert all(s["mass"] == 1.0 for s in r.steps)
        assert all(b["radiated"] >= a["radiated"] for a, b in zip(r.steps, r.steps[1:]))


def test_every_line_of_the_40_seeds_is_dense_short_and_accepted(runs40):
    longest = 0
    for seed, r in runs40.items():
        lns = lines(r)
        assert len(lns) == 2014 and len({ln.text for ln in lns}) == 2014, seed
        for ln in lns:
            assert is_dense(ln.text), ln.text
            longest = max(longest, n_tokens(ln.text))
            if ln.kind != "record":
                verdict = SIM.check(ln.prompt, ln.answer)
                assert verdict.ok and verdict.expected == ln.answer and SIM.owns(ln.prompt), ln.text
    assert 100 <= longest <= 112  # of 128; a seed of four digits adds 2


def test_every_start_is_in_virial_balance(runs40):
    for seed, r in runs40.items():
        first = r.steps[0]
        assert 0.85 < first["support"] < 1.15, seed  # round 6 started at up to 1.13 above spin 1/3
        assert first["stage"] == "cloud" and first["clumps"] == 0 and first["radiated"] == 0
        assert first["pattern"] == "none" and first["toomre_q"] == "none" and first["disc_mass"] > 0.95
        assert 0.55 < first["radius"] < 0.75 and 0.8 < first["flattening"] < 1.2
        # rotation is the share the spin asked for; the random motions of 256 particles add some of their own
        assert abs(first["rotation"] - r.params["spin"]) < 0.05
        # the inner half of the cloud is denser than a uniform sphere and falls in about 0.75 t_ff
        inner = gravity.free_fall_time(gravity.sphere_density(0.5, first["radius"])) / T_FF
        assert 0.6 < inner < 0.9


def test_without_cooling_nothing_happens(runs40):
    still = [r for r in runs40.values() if r.params["cooling"] == 0]
    assert len(still) == 7
    for r in still:
        assert r.summary["star_time"] == "never" and r.summary["stage"] == "cloud" and r.summary["patterns"] == "none"
        assert r.summary["clumps"] == r.summary["companions"] == 0 and r.summary["star"] == 0
        assert r.params["cooling_time"] == "never"
        # round 6 found "clumps" in a third of the uncooled states; a bound clump never forms
        assert all(s["clumps"] == 0 and s["stage"] in ("cloud", "contracting") for s in r.steps)
        assert all(s["pattern"] == "none" and s["toomre_q"] == "none" and s["radiated"] == 0 for s in r.steps)
        assert r.summary["min_flattening"] > 0.6
        assert all(0.8 < s["support"] < 1.25 for s in r.steps)
        assert all(close(s["energy"], r.steps[0]["energy"], rel=0.005) for s in r.steps)


def test_an_uncooled_cloud_spreads_but_does_not_evaporate():
    # the docstring once called the mass that leaves the cloud radius evaporation; it leaves within one
    # free-fall time, long before encounters could do it, and next to none of it is unbound
    inside, energies = [], None
    for k, (x, v, m) in enumerate(evolve(4, 0.09, 0.0, rules=7)):
        if k == 0:
            d = x[:, None, :] - x[None, :, :]
            inv = 1.0 / np.sqrt((d * d).sum(-1) + EPS * EPS)
            np.fill_diagonal(inv, 0.0)
            energies = 0.5 * (v * v).sum(1) - (inv * m).sum(1)  # per unit mass, of each particle
        if k in (0, 10, 60):
            inside.append(float(m[(x * x).sum(1) < 1.0].sum()))
    assert float((energies > 0).mean()) < 0.03
    assert inside[0] > 0.97 and 0.8 < inside[1] < 0.92 and 0.7 < inside[2] < 0.92
    assert roll(4).params["cooling"] == 0 and roll(4).params["spin"] == 0.09


def test_with_cooling_a_star_forms_and_the_rest_settles_around_it(runs40):
    cooled = [r for r in runs40.values() if r.params["cooling"] > 0]
    assert len(cooled) == 33
    for r in cooled:
        first, last = r.steps[0], r.steps[-1]
        # star_time is a number for every cooled run (round 6: collapse_time "never" for some that had collapsed)
        assert r.summary["star_time"] != "never" and 1.0 <= r.summary["star_time"] <= 2.5, r.seed
        assert last["stage"] in ("star", "multiple") and 0.3 <= r.summary["star"] <= 0.8
        assert 0.1 <= r.summary["disc_mass"] <= 0.5 and 0.15 <= r.summary["disc_radius"] <= 0.7
        assert r.summary["star"] + r.summary["disc_mass"] <= 1.002
        assert last["energy"] < 3 * first["energy"] < 0 and last["radiated"] > 3 * abs(first["energy"])
        assert all(b["energy"] <= a["energy"] for a, b in zip(r.steps, r.steps[1:]))
        # a cloud first; the state that first shows a star is the one star_time names
        stages = [s["stage"] for s in r.steps]
        k = next(i for i, stage in enumerate(stages) if stage in ("star", "multiple"))
        assert stages[0] == "cloud" and r.steps[k]["time"] == r.summary["star_time"]
        # support is lost before the star appears
        assert min(s["support"] for s in r.steps[:k + 1]) < 0.9
    assert sum(r.summary["clumps"] == 1 for r in cooled) >= 20  # most end as one star


def test_outcomes_follow_the_parameters(runs40):
    cooled = [r for r in runs40.values() if r.params["cooling"] > 0]
    spin = [r.params["spin"] for r in cooled]
    cooling = [r.params["cooling"] for r in cooled]
    # trends, not numbers to predict: more spin, a lighter star, a wider disc, a later star
    assert corr(spin, [r.summary["star"] for r in cooled]) < -0.8
    assert corr(spin, [r.summary["disc_radius"] for r in cooled]) > 0.4
    assert corr(spin, [r.summary["star_time"] for r in cooled]) > 0.3
    # more cooling, a flatter system, and no later star (the first proposal of the review, a drag of
    # reach 0.25 and rate 60, gave a later star with more cooling: it held the cloud up as a viscosity)
    assert corr(cooling, [r.summary["min_flattening"] for r in cooled]) < -0.4
    assert corr(cooling, [r.summary["star_time"] for r in cooled]) < 0.1
    slow = [r for r in cooled if r.params["spin"] <= 0.1]
    fast = [r for r in cooled if r.params["spin"] >= 0.2]
    assert len(slow) >= 8 and len(fast) >= 8
    assert min(r.summary["star"] for r in slow) > max(r.summary["star"] for r in fast)
    assert all(r.summary["star"] > 0.5 for r in slow) and sum(r.summary["clumps"] == 1 for r in slow) >= len(slow) - 1
    # the bar criterion: fast rotators are the ones that end as several stars
    assert all(r.params["bar_unstable"] == "yes" for r in fast)
    assert sum(r.summary["clumps"] > 1 for r in fast) > sum(r.summary["clumps"] > 1 for r in slow)


def test_the_clump_count_is_steadier_than_in_round_6(runs40):
    cooled = [r for r in runs40.values() if r.params["cooling"] > 0]
    changes = [sum(a["clumps"] != b["clumps"] for a, b in zip(r.steps, r.steps[1:])) for r in cooled]
    jumps = [sum(abs(a["clumps"] - b["clumps"]) > 1 for a, b in zip(r.steps, r.steps[1:])) for r in cooled]
    # round 6: 22.8 changes in 40 steps, 7.0 of them by 2 or more; here in 60 steps
    assert sum(changes) / len(cooled) < 9 and sum(jumps) / len(cooled) < 1
    assert max(s["clumps"] for r in cooled for s in r.steps) <= 5
    # every counted clump has at least 14 particles: 0.082 solar masses for a cloud of 1.5, a star by mass
    smallest = min(s["largest_clump"] for r in cooled for s in r.steps if s["clumps"])
    assert smallest >= 0.0546 and smallest * gravity.CLOUD_MASS_MSUN > 0.08


def test_patterns_are_named_for_what_they_are(runs40):
    words = [s["pattern"] for r in runs40.values() for s in r.steps]
    assert set(words) == set(PATTERNS7)  # each word occurs, and none dominates the story
    assert words.count("none") > 0.9 * len(words)
    for r in runs40.values():
        for s in r.steps:
            assert s["pattern"] != "pair" or s["clumps"] >= 2
    # the stability number is only printed where something flat exists, and such discs are not far below 1
    q = [s["toomre_q"] for r in runs40.values() for s in r.steps if s["toomre_q"] != "none"]
    assert len(q) > 100 and min(q) > 0.5


def test_the_units_label_and_why_the_discs_are_wide(runs40):
    # 1.5 solar masses within 0.1 pc: about 5000 hydrogen molecules per cm^3 (2.8 hydrogen masses each,
    # for the helium), and a free-fall time of 0.428 million years
    rho = gravity.sphere_density(gravity.CLOUD_MASS_MSUN * gravity.MSUN_G, gravity.CLOUD_RADIUS_PC * gravity.PC_CM)
    assert 4500 < rho / (2.8 * 1.6735e-24) < 5500 and T_FF_MYR == 0.428
    assert 950 < EPS * gravity.CLOUD_RADIUS_PC * 206265 < 1100  # the softening in au
    assert 5.5 < gravity.CLOUD_MASS_MSUN / N / 9.546e-4 < 6.5  # one particle in Jupiter masses
    # the slowest cooled rotators carry the angular momentum of a disc of some 200 to 500 au, yet the loose
    # mass within one cloud radius lies ten times as far out: disc_radius is not set by the spin alone
    slow = [r for r in runs40.values() if r.params["cooling"] > 0 and r.params["spin"] <= 0.06]
    assert len(slow) >= 3
    for r in slow:
        assert r.params["centrifugal_radius"] < 0.03 and r.summary["disc_radius"] > 0.25
        assert r.summary["disc_radius"] > 10 * r.params["centrifugal_radius"]


def test_a_run_takes_under_two_seconds():
    # seed 1 is a slow rotator with strong cooling, the slowest kind (a star of some 150 particles inside
    # the softening length gives over 10000 pairs to damp at every step): 1.6 to 1.95 s on a machine busy
    # with other work, against 0.7 to 0.8 s for an uncooled run. The least of up to eight tries, the usual
    # estimate of what a run itself costs, so that a busy moment of the machine does not fail it.
    # On a CPU with performance and efficiency cores the system may move a process that has been
    # computing for minutes (as this one has, by now) to an efficiency core, where a run takes about
    # twice as long (i9-13900K: seed 1 in 1.44 to 1.59 s held to the performance cores, 3.3 to 3.75 s
    # held to the efficiency cores). The 2 s are a claim about the code on a performance core, so when
    # this process is too slow the tries are repeated in a fresh one.
    params = random_params(random.Random(1), rules=7)
    best = math.inf
    for _ in range(4):
        start = time.perf_counter()
        SIM.run(1, rules=7, **params)
        best = min(best, time.perf_counter() - start)
        if best < 2.0:
            break
    if best >= 2.0:
        code = ("import random, time\nfrom haishool.cosmos import gravity\n"
                "p = gravity.random_params(random.Random(1), rules=7)\nbest = 1e9\n"
                "for _ in range(4):\n    t = time.perf_counter()\n    gravity.run(1, rules=7, **p)\n"
                "    best = min(best, time.perf_counter() - t)\n    if best < 2.0:\n        break\nprint(best)")
        env = dict(os.environ, PYTHONPATH=str(ROOT))
        done = subprocess.run([sys.executable, "-c", code], env=env, cwd=ROOT, capture_output=True, text=True)
        assert done.returncode == 0, done.stderr
        best = min(best, float(done.stdout.split()[-1]))
    assert best < 2.0


# ---------------------------------------------------------------------------------------------
# the measurements on made-up states
# ---------------------------------------------------------------------------------------------

def test_bound_groups():
    m = np.full(100, 0.01)
    x = np.concatenate([normal(1, (30, 3), 0.005), normal(2, (20, 3), 0.005) + [1, 0, 0],
                        normal(3, (20, 3), 0.005) + [0, 1, 0], normal(4, (10, 3), 0.005) + [0, 0, 1],
                        normal(5, (20, 3), 0.3) + [3, 3, 3]])
    v = np.zeros((100, 3))
    v[50:70] = normal(6, (20, 3), 3.0)  # the third group flies apart: not bound
    groups = bound_groups(x, v, m, 0.05)
    assert [g.tolist() for g in groups] == [list(range(30)), list(range(30, 50))]  # 10 are too few, 20 hot ones unbound
    # a group that moves as a whole is as bound as one at rest
    v[30:50] += [5.0, 0.0, 0.0]
    assert [len(g) for g in bound_groups(x, v, m, 0.05)] == [30, 20]
    # ties: the group with the lowest particle index first; the order of the particles does not matter
    v[:] = 0.0
    assert [int(g[0]) for g in bound_groups(x, v, m, 0.05)] == [0, 30, 50]
    shuffled = list(range(100))
    random.Random(1).shuffle(shuffled)
    assert sorted(len(g) for g in bound_groups(x[shuffled], v[shuffled], m, 0.05)) == [20, 20, 30]
    # exactly 14 count, 13 do not
    assert [len(g) for g in bound_groups(x[:14], v[:14], m[:14], 0.05)] == [14]
    assert bound_groups(x[:13], v[:13], m[:13], 0.05) == []
    # the linking length is fixed: a chain with links longer than eps is no group
    chain = np.array([[0.06 * i, 0.0, 0.0] for i in range(20)])
    assert bound_groups(chain, np.zeros((20, 3)), np.full(20, 0.05), 0.05) == []
    assert [len(g) for g in bound_groups(chain * 0.5, np.zeros((20, 3)), np.full(20, 0.05), 0.05)] == [20]


def test_measure7_on_made_up_states():
    m = np.full(250, 1 / 250)
    still = np.zeros((250, 3))
    ball = normal(1, (250, 3), 0.3)
    round_ = measure7(ball, still, m, 0.05, None, 0.0)
    assert round_ == measure(ball, still, m, 0.05, None, 0.0, rules=7)
    assert 0.8 < round_["flattening"] < 1.25 and round_["clumps"] == 0 and round_["largest_clump"] == 0
    assert round_["pattern"] == "none" and round_["toomre_q"] == "none" and round_["stage"] == "cloud"
    assert round_["support"] == 0 and round_["rotation"] == 0 and round_["random"] == 0  # nothing moves
    assert round_["disc_mass"] > 0.95 and 0.2 < round_["disc_radius"] < 0.5
    assert measure7(ball, still, m, 0.05, None, 0.0, 0.3)["radiated"] == 0.3
    # a star of 150 particles with a flat disc of 100 around it (5 rings of 20, too far apart to link up,
    # turning like a rigid body); the whole thing sits off the origin and drifts
    star = normal(3, (150, 3), 0.005)
    turn = np.array([2 * math.pi * k / 20 for k in range(20)])
    rings = np.concatenate([np.stack([r * np.cos(turn + r), r * np.sin(turn + r), np.zeros(20)], 1)
                            for r in (0.28, 0.34, 0.40, 0.46, 0.52)])
    rings[:, 2] = 0.01 * normal(2, (100,))
    orbit = 3.0 * np.stack([-rings[:, 1], rings[:, 0], np.zeros(100)], 1)
    x = np.concatenate([star, rings]) + [2.0, -1.0, 0.5]
    v = np.concatenate([np.zeros((150, 3)), orbit]) + [0.3, 0.0, 0.0]
    disc = measure7(x, v, m, 0.05, 0.6, 1.0)
    assert (disc["clumps"], disc["largest_clump"], disc["stage"]) == (1, pytest.approx(0.6), "star")
    assert disc["disc_mass"] == pytest.approx(0.4) and disc["disc_radius"] == pytest.approx(0.4, abs=0.01)
    assert disc["flattening"] < 0.1 and disc["pattern"] == "none"  # even rings have no mode
    assert disc["m1"] < 0.1 and disc["m2"] < 0.1 and disc["pattern_particles"] == 90  # the outer tenth dropped
    # cold rings in circular motion: next to no random speed, so Q is next to 0 (as unstable as can be)
    assert 0 <= disc["toomre_q"] < 0.05 and disc["q_kappa"] == pytest.approx(math.sqrt(2) * 3, rel=1e-3)
    assert disc["toomre_q"] == gravity.toomre_q(disc["q_sigma"], disc["q_kappa"], disc["q_surface_density"])
    assert disc["q_surface_density"] == pytest.approx(0.4 / (math.pi * 3.75 * disc["disc_radius"] ** 2))
    assert disc["angular_momentum"] > 0 and disc["momentum"] == pytest.approx(0.3)
    assert disc["jeans_number"] > 0  # of the 100 outside the star
    # the metrics do not care where the system is or how fast it drifts as a whole
    moved = measure7(x + [3.0, -2.0, 1.0], v + [0.5, 0.0, 0.0], m, 0.05, 0.6, 1.0)
    for key in ("radius", "flattening", "disc_mass", "disc_radius", "angular_momentum", "jeans_number"):
        assert moved[key] == pytest.approx(disc[key], rel=1e-9, abs=1e-12), key
    assert all(moved[key] == disc[key] for key in ("clumps", "pattern", "stage"))
    # the same mass laid out as a long flat slab through the star is a bar, on one side of it lopsided
    slab = np.array([[-0.57 + 0.06 * i, -0.12 + 0.06 * j, 0.0] for i in range(20) for j in range(5)])
    bar = measure7(np.concatenate([star, slab]), still, m, 0.05, 0.6, 1.0)
    assert (bar["clumps"], bar["pattern"]) == (1, "bar") and bar["m2"] > 2 * bar["m1"]
    assert bar["pattern_particles"] == 89  # the two of the slab that touch the star belong to it
    patch = np.array([[0.23 + 0.06 * i, -0.27 + 0.06 * j, 0.0] for i in range(10) for j in range(10)])
    side = measure7(np.concatenate([star, patch]), still, m, 0.05, 0.6, 1.0)
    assert (side["clumps"], side["pattern"]) == (1, "lopsided") and side["m1"] > side["m2"] > 0.3
    # stood on its end the slab is not flat, and nothing is called a pattern
    assert measure7(np.concatenate([star, slab[:, [2, 1, 0]]]), still, m, 0.05, 0.6, 1.0)["pattern"] == "none"
    # two bound clumps, 150 and 100: a pair, and the centre is the big one
    two = np.concatenate([normal(4, (150, 3), 0.005) + [0.2, 0, 0], normal(5, (100, 3), 0.005) - [0.3, 0, 0]])
    pair = measure7(two, still, m, 0.05, 0.6, 1.0)
    assert (pair["clumps"], pair["largest_clump"], pair["second_clump"]) == (2, pytest.approx(0.6), pytest.approx(0.4))
    assert pair["pattern"] == "pair" and pair["stage"] == "multiple"
    assert pair["disc_mass"] == 0 and pair["disc_radius"] == 0
    assert pair["jeans_number"] == 0 and pair["toomre_q"] == "none"  # nothing is left outside the clumps
    # round 6 called the same state a spiral
    assert measure(two, still, m, 0.05, 0.6, 1.0)["spiral"] == "yes"


def test_stage7_rules():
    assert stage7(0.6, 1.0, 0, 0.0, 0.64) == "cloud"
    assert stage7(0.4, 0.8, 0, 0.0, 0.64) == "contracting"
    assert stage7(0.4, 0.4, 0, 0.0, 0.64) == "disc"
    assert stage7(0.6, 1.0, 1, 0.06, 0.64) == "fragmenting"
    assert stage7(0.4, 0.4, 3, 0.2, 0.64) == "fragmenting"
    assert stage7(0.1, 0.9, 1, 0.25, 0.64) == "star"
    assert stage7(0.1, 0.3, 1, 0.7, 0.64) == "star"
    assert stage7(0.3, 0.4, 2, 0.3, 0.64) == "multiple"
    assert stage7(0.3, 0.4, 2, 0.24, 0.64) == "fragmenting"
    assert stage7(0.479, 1.0, 0, 0.0, 0.64) == "contracting" and stage7(0.481, 1.0, 0, 0.0, 0.64) == "cloud"
    # the largest clump counts only when there is a clump
    assert stage7(0.6, 1.0, 0, 0.5, 0.64) == "cloud"


def test_pattern7_rules():
    assert pattern7(0.1, 0, 1.0, 0.0, 0.0) == "pair"  # a second clump with a tenth of the mass, whatever else
    assert pattern7(0.09, 100, 0.3, 0.0, 0.0) == "none"
    assert pattern7(0.0, 39, 0.3, 0.9, 0.5) == "none" and pattern7(0.0, 40, 0.3, 0.9, 0.5) == "lopsided"
    assert pattern7(0.0, 100, 0.5, 0.9, 0.5) == "none" and pattern7(0.0, 100, 0.49, 0.5, 0.9) == "bar"
    # Rayleigh: an amplitude counts when A * A * n > 6.9, so 0.27 counts for 100 points and 0.26 does not
    assert pattern7(0.0, 100, 0.3, 0.27, 0.1) == "lopsided" and pattern7(0.0, 100, 0.3, 0.26, 0.1) == "none"
    assert pattern7(0.0, 100, 0.3, 0.1, 0.27) == "bar" and pattern7(0.0, 100, 0.3, 0.1, 0.26) == "none"
    assert pattern7(0.0, 200, 0.3, 0.19, 0.1) == "lopsided"  # more points, a smaller amplitude is enough
    assert pattern7(0.0, 100, 0.3, 0.5, 0.5) == "lopsided" and pattern7(0.0, 100, 0.3, 0.26, 0.5) == "bar"
    # the fixed 0.2 of round 6 would have called noise a pattern: 50 points scatter to 0.3 by chance
    assert pattern7(0.0, 50, 0.3, 0.1, 0.3) == "none"


# ---------------------------------------------------------------------------------------------
# the named rules and the lessons
# ---------------------------------------------------------------------------------------------

def test_named_rules():
    # the time unit: a uniform sphere of mass 1 and radius 1
    assert gravity.free_fall_time(3 / (4 * math.pi)) == pytest.approx(T_FF, rel=1e-15)
    assert gravity.free_fall_time(gravity.sphere_density(1.0, 1.0)) == pytest.approx(math.pi / math.sqrt(8))
    assert gravity.free_fall_time(4.0) == pytest.approx(gravity.free_fall_time(1.0) / 2)
    # the label: 1.5 solar masses within 0.1 parsec fall together in 0.428 million years
    rho = 1.5 * 1.989e33 / (4 / 3 * math.pi * (0.1 * 3.0857e18) ** 3)
    assert 2.3e-20 < rho < 2.5e-20
    assert math.sqrt(3 * math.pi / (32 * 6.674e-8 * rho)) / 3.156e13 == pytest.approx(T_FF_MYR, abs=5e-4)
    assert T_FF_MYR == 0.428 and (gravity.CLOUD_MASS_MSUN, gravity.CLOUD_RADIUS_PC) == (1.5, 0.1)
    assert 950 < EPS * 0.1 * 206265 < 1100  # the softening is about 1000 au
    assert 0.08 < MIN_MEMBERS7 / N * gravity.CLOUD_MASS_MSUN < 0.09  # a clump is at least 0.082 solar masses
    # the force is the gradient of the potential, both by the simulation's own routines
    for m1, m2, d, s in ((0.5, 0.5, 1.0, 0.05), (0.004, 0.9, 0.03, 0.05), (1, 1, 2, 0.2)):
        r2 = d * d + s * s
        assert gravity.softened_force(m1, m2, d, s) == pytest.approx(m1 * m2 * d / r2 ** 1.5, rel=1e-13)
        assert gravity.softened_potential(m1, m2, d, s) == pytest.approx(-m1 * m2 / math.sqrt(r2), rel=1e-13)
        h = 1e-6 * d
        slope = (gravity.softened_potential(m1, m2, d + h, s) - gravity.softened_potential(m1, m2, d - h, s)) / (2 * h)
        assert slope == pytest.approx(gravity.softened_force(m1, m2, d, s), rel=1e-6)
    # the start: rotation takes its share, the plane keeps the rest of |W| / 3
    assert gravity.spin_rate(0.2, 0.6, 0.3) == pytest.approx(math.sqrt(2 * 0.2 * 0.6 / 0.3))
    assert 0.5 * 0.3 * gravity.spin_rate(0.2, 0.6, 0.3) ** 2 == pytest.approx(0.2 * 0.6)
    assert gravity.plane_speed(0.2, 0.6, 1.0) == pytest.approx(math.sqrt((1 / 3 - 0.2) * 0.6))
    assert gravity.plane_speed(1 / 3, 0.6, 1.0) == 0 and gravity.plane_speed(0.4, 0.6, 1.0) == 0
    assert gravity.virial_ratio(0.3, 0.6) == 1.0
    assert gravity.rotation_share(0.3, 0.25, 0.6) == pytest.approx(0.5 * 0.09 / 0.25 / 0.6)
    assert gravity.random_share(0.3, 0.3, 0.25, 0.6) == pytest.approx(0.5 - 0.3)
    assert gravity.rotation_share(0.3, 0.25, 0.6) + gravity.random_share(0.3, 0.3, 0.25, 0.6) == pytest.approx(
        gravity.virial_ratio(0.3, 0.6) / 2)
    assert gravity.jeans_mass(1.0, 1.0) == pytest.approx(math.pi ** 2.5 / 6)
    assert gravity.jeans_mass(2.0, 4.0) == pytest.approx(gravity.jeans_mass(1.0, 1.0) * 8 / 2)
    assert gravity.jeans_number(2.0, 0.5, 3.0) == pytest.approx(2.0 / gravity.jeans_mass(0.5, 3.0))
    assert gravity.toomre_q(0.5, 2.0, 1.0) == pytest.approx(1 / 3.36)
    assert gravity.bar_unstable(0.14) == "no" and gravity.bar_unstable(0.15) == "yes"
    assert gravity.centrifugal_radius(0.3, 0.5) == pytest.approx(0.18)
    assert gravity.cooling_time(0.1) == pytest.approx(1.0) and gravity.cooling_time(0.6) == pytest.approx(1 / 6)
    assert gravity.drag_factor(0.3, 0.004) == pytest.approx(0.012) and gravity.drag_factor(0.0, 0.004) == 0
    # one step takes dt / cooling_time of the closing speed
    assert gravity.drag_factor(0.3, 0.004) == pytest.approx(0.004 / gravity.cooling_time(0.3))
    # so cooling_time is an e-folding time: after it a lone pair still closes at 1 / e of its speed, not at 0
    for cooling in (0.1, 0.3, 0.6):
        steps = round(gravity.cooling_time(cooling) * T_FF / DT)
        assert (1 - gravity.drag_factor(cooling, DT / T_FF)) ** steps == pytest.approx(math.exp(-1), rel=0.03)
    # Rayleigh's limit 6.9 is a chance of e^-6.9, 1 in 1000 (to 1 percent), for one mode of a round distribution
    assert math.exp(-gravity.RAYLEIGH) == pytest.approx(0.001, rel=0.01)
    # a uniform sphere in virial balance (sigma^2 = G M / 5 R per axis) holds 1.9 Jeans masses, not 1
    balanced = gravity.jeans_number(1.0, math.sqrt(1 / 5), gravity.sphere_density(1.0, 1.0))
    assert balanced == pytest.approx(1.87, abs=0.01)
    # the label of the units: about 5000 hydrogen molecules per cm^3, one particle about 6 Jupiter masses
    rho_core = gravity.sphere_density(gravity.CLOUD_MASS_MSUN * gravity.MSUN_G, gravity.CLOUD_RADIUS_PC * gravity.PC_CM)
    assert 4500 < rho_core / (2.8 * 1.6726e-24) < 5500
    assert 5.5 < gravity.CLOUD_MASS_MSUN / N / 9.546e-4 < 6.5


def test_the_start_uses_the_named_rules():
    for seed, spin in ((1, 0.05), (2, 0.2), (3, 0.3)):
        x, v, m = initial_state(seed, 256, 1.0, spin, 0.05)
        w = abs(_potential(x, m, 0.05 * 0.05))
        inertia = float((m * (x[:, 0] ** 2 + x[:, 1] ** 2)).sum())
        omega = gravity.spin_rate(spin, w, inertia)
        speed = gravity.plane_speed(spin, w, 1.0)
        # take the rotation out: what is left along x has the plane speed as its spread
        rest = v[:, 0] + omega * x[:, 1]
        assert float(np.sqrt((rest * rest).mean())) == pytest.approx(speed, rel=0.12)
        full = measure7(x, v, m, 0.05, None, 0.0)
        assert full["support"] == gravity.virial_ratio(0.5 * float(gravity._total(m * gravity._square(v))), w)
        assert full["rotation"] == pytest.approx(spin, abs=0.03) and 0.9 < full["support"] < 1.1
        assert full["jeans_number"] == gravity.jeans_number(1.0, full["sigma"], full["density"])
        assert full["density"] == gravity.sphere_density(0.5, full["radius"])


def test_lessons_are_accepted_and_true_of_the_simulation():
    lessons = LESSONS.generate(random.Random(1), 400)
    assert len(lessons) == 400 and len({ln.text for ln in lessons}) > 350
    assert [ln.text for ln in lessons] == [ln.text for ln in LESSONS.generate(random.Random(1), 400)]
    assert {ln.prompt.split()[2] for ln in lessons} == set(RULES)  # every rule is drawn
    for ln in lessons:
        assert is_dense(ln.text) and n_tokens(ln.text) <= 128, ln.text
        assert ln.topic == "predict_gravity" and ln.kind == "calc" and "." not in ln.prompt and "." not in ln.answer
        for gate in (LESSONS, gravity):
            verdict = gate.check(ln.prompt, ln.answer)
            assert verdict.ok and verdict.expected == ln.answer, ln.text
            assert gate.owns(ln.prompt)
        assert not SIM.owns(ln.prompt)
        # the answer is what the simulation's own function returns for the inputs as printed
        name, values = LESSONS.parse(ln.prompt)
        truth = RULES[name].compute(*values)
        assert ln.answer == (truth if isinstance(truth, str) else num(truth)), ln.text
        assert LESSONS.split_key(ln.prompt).startswith(f"gravity predict {name} ")
        wrong = "no" if ln.answer != "no" else "yes"
        assert not LESSONS.check(ln.prompt, wrong).ok
        if parse_num(ln.answer) is not None and parse_num(ln.answer) != 0:
            assert not LESSONS.check(ln.prompt, num(parse_num(ln.answer) * 1.01)).ok, ln.text
    words = {(ln.prompt.split()[2], ln.answer) for ln in lessons if parse_num(ln.answer) is None}
    assert {("bar_unstable", "yes"), ("bar_unstable", "no"), ("stage", "star"), ("stage", "fragmenting"),
            ("pattern", "pair"), ("pattern", "none")} <= words
    # the compute functions are the very functions the simulation calls
    assert {name: rule.compute for name, rule in RULES.items()} == {
        "free_fall_time": gravity.free_fall_time, "sphere_density": gravity.sphere_density,
        "softened_force": gravity.softened_force, "softened_potential": gravity.softened_potential,
        "spin_rate": gravity.spin_rate, "plane_speed": gravity.plane_speed, "virial_ratio": gravity.virial_ratio,
        "rotation_share": gravity.rotation_share, "random_share": gravity.random_share,
        "jeans_mass": gravity.jeans_mass, "jeans_number": gravity.jeans_number, "toomre_q": gravity.toomre_q,
        "bar_unstable": gravity.bar_unstable, "centrifugal_radius": gravity.centrifugal_radius,
        "cooling_time": gravity.cooling_time, "drag_factor": gravity.drag_factor, "stage": stage7,
        "pattern": pattern7}
    called = {
        "initial_state": ("spin_rate(", "plane_speed("),
        "_evolve7": ("drag_factor(",),
        "measure7": ("sphere_density(", "virial_ratio(", "rotation_share(", "random_share(", "jeans_number(",
                     "toomre_q(", "pattern7(", "stage7("),
        "jeans_number": ("jeans_mass(",),
        "softened_force": ("_accel(",), "softened_potential": ("_potential(",),
    }
    for where, names in called.items():
        source = inspect.getsource(getattr(gravity, where))
        assert all(name in source for name in names), where
    run7 = inspect.getsource(gravity.Gravity._run7)
    assert all(name in run7 for name in ("stage7(", "bar_unstable(", "centrifugal_radius(", "cooling_time("))
    assert "free_fall_time(" in inspect.getsource(gravity).split("T_FF_MYR = ")[1][:120]


def test_lessons_agree_with_a_real_state():
    ledger: dict = {}
    states = evolve(23, 0.28, 0.55, ledger=ledger, rules=7)
    full = None
    for k in range(41):
        x, v, m = next(states)
        full = measure7(x, v, m, EPS, 0.689, k * 0.1, ledger["radiated"])
        kinetic, w = 0.5 * float(gravity._total(m * gravity._square(v))), abs(_potential(x, m, EPS * EPS))
        assert full["support"] == gravity.virial_ratio(kinetic, w)
        assert full["support"] / 2 == pytest.approx(full["rotation"] + full["random"], rel=1e-12)
        assert full["pattern"] == pattern7(full["second_clump"], full["pattern_particles"], full["pattern_flattening"],
                                           full["m1"], full["m2"])
        assert full["stage"] == stage7(full["radius"], full["flattening"], full["clumps"], full["largest_clump"], 0.689)
        if full["toomre_q"] != "none":
            assert full["toomre_q"] == gravity.toomre_q(full["q_sigma"], full["q_kappa"], full["q_surface_density"])
            assert full["flattening"] < 0.5
    assert full["clumps"] >= 1 and full["largest_clump"] >= 0.25  # by 4 t_ff the star is there


def test_lesson_records_and_other_prompts():
    records = LESSONS.records()
    assert len(records) == len(RULES) == 18
    for rec in records:
        assert rec.kind == "record" and is_dense(rec.text) and n_tokens(rec.text) <= 128, rec.text
    assert set(LESSONS.KEYS) == set(RULES)
    for prompt in ("gravity predict cooling_vertical vertical_velocity 1 mean_vertical_velocity 0 cooling 0 point 3 "
                   "dt_tff 0 point 0 1",  # the round-6 lesson of haishool.cosmos.predict keeps its own gate
                   "planets predict solids_remaining remaining 1 efficiency 1 orbit 1 star_mass 1 dt 1",
                   "stars predict lifetime mass 1", "gravity predict spiral m2 0 point 3",
                   "gravity predict bar_unstable spin 0 point 5",  # outside the range of the rule
                   "gravity predict bar_unstable", "gravity predict bar_unstable spin 0 point 2 extra 1",
                   "gravity seed 5 rules 7 final star", "nucleo predict free_fall_time density 1", ""):
        assert not LESSONS.owns(prompt), prompt
        assert LESSONS.check(prompt, "1").reason == "not my question"
    assert not gravity.owns("gravity predict cooling_vertical vertical_velocity 1 mean_vertical_velocity 0 cooling "
                            "0 point 3 dt_tff 0 point 0 1")
    # word rules judge words, number rules numbers at 4 significant digits
    assert LESSONS.check("gravity predict free_fall_time density 1", num(math.sqrt(3 * math.pi / 32))).ok
    assert LESSONS.check("gravity predict cooling_time cooling 0 point 2 5", "0 point 4").ok
    assert LESSONS.check("gravity predict drag_factor cooling 0 point 5 dt_tff 0 point 0 0 4", "0 point 0 2").ok
    assert LESSONS.check("gravity predict stage radius 0 point 3 flattening 0 point 4 clumps 2 largest_clump 0 point 3 "
                         "radius_start 0 point 6 4", "multiple").ok
    assert LESSONS.check("gravity predict pattern second_clump 0 particles 1 0 0 flattening 0 point 3 m1 0 point 1 "
                         "m2 0 point 2 7", "bar").ok
    assert not LESSONS.check("gravity predict pattern second_clump 0 particles 1 0 0 flattening 0 point 3 m1 0 point 1 "
                             "m2 0 point 2 7", "spiral").ok
