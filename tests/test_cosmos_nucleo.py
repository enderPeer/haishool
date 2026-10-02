import copy
import itertools
import math
import random
import time

import numpy as np
import pytest

from haishool.cosmos import Rollout, Simulation, close
from haishool.cosmos import nucleo as N
from haishool.truth import Gate, is_dense, num, parse_num

SEEDS = [1, 2, 3, 42, 777, 4242, 9998]
#: every seed :func:`haishool.cosmos.seeds` and :func:`nucleo.generate` can draw
ALL_SEEDS = range(1, 9999)
N_PARAMS = len(N.RANGES)
LINES_PER_ROLLOUT = 1 + N_PARAMS + 30 + 30 * len(N.KEYS) + 29 * len(N.KEYS) + len(N.FINAL_KEYS)


def ntokens(text: str) -> int:
    return len(text.replace(".", " . ").split())


def freeze_ratio(params: dict) -> float:
    """n/p at freeze-out, from the parameters alone."""
    kt = 8.617333262e-11 * params["t1"] / math.sqrt(params["t_freeze"]) * params["expansion_factor"] ** (1 / 3)
    return math.exp(-1.29333236 / kt)


@pytest.fixture(scope="module")
def sim():
    return N.simulation()


@pytest.fixture(scope="module")
def all_lines():
    return {s: N.lines(N.rollout(s)) for s in SEEDS}


@pytest.fixture(scope="module")
def every_rollout():
    return [N.rollout(s) for s in ALL_SEEDS]


def test_is_a_simulation_and_a_gate(sim):
    assert isinstance(sim, Simulation) and isinstance(sim, Gate)
    assert sim.sim == sim.topic == "nucleo" and set(sim.KEYS) == set(N.KEYS)
    assert isinstance(sim.run(5), Rollout)
    assert sim.records() == []


def test_same_seed_same_rollout():
    a = N.run(7, **N.random_params(random.Random(7)))
    b = N.run(7, **N.random_params(random.Random(7)))
    assert a.params == b.params and a.steps == b.steps and a.summary == b.summary
    assert [ln.text for ln in N.lines(N.rollout(7))] == [ln.text for ln in N.lines(N.rollout(7))]
    assert N.random_params(random.Random(11)) == N.random_params(random.Random(11))
    assert N.random_params(random.Random(11)) != N.random_params(random.Random(12))
    assert N.rollout(7).steps != N.rollout(8).steps


def test_forty_seeds_conserved_and_bit_identical(sim):
    for s in range(1, 41):
        p = N.random_params(random.Random(s))
        a, b = N.run(s, **p), N.run(s, **p)
        assert a.steps == b.steps and a.summary == b.summary and a.params == b.params
        # equal floats could still differ in the sign of zero; the printed form may not
        assert [repr(v) for st in a.steps for v in st.values()] == [repr(v) for st in b.steps for v in st.values()]
        assert a.steps == N.rollout(s).steps and a.summary == N.rollout(s).summary
        v = sim.conserved(a)
        assert v.ok, (s, v)
        assert max(abs(math.fsum(N._fractions(st)) - 1) for st in a.steps) < 1e-12


def test_the_seed_is_only_a_label_for_run():
    p = N.random_params(random.Random(5))
    assert N.run(5, **p).steps == N.run(99, **p).steps
    assert N.run(5).steps == N.run(99).steps and N.run(5).params == N.DEFAULTS


def test_random_params_stay_in_range():
    for s in range(200):
        p = N.random_params(random.Random(s))
        assert set(p) == set(N.RANGES)
        for k, (lo, hi, _) in N.RANGES.items():
            assert lo <= p[k] <= hi, (k, p[k])
            assert p[k] == float(f"{p[k]:.3g}")


def test_time_grid_is_the_written_out_log_grid():
    assert list(N.TIMES) == [float(f"{0.1 * 30000 ** (i / 29):.3g}") for i in range(30)]
    assert N.TIMES[0] == 0.1 and N.TIMES[-1] == 3000.0 and len(N.TIMES) == 30
    assert all(a < b for a, b in zip(N.TIMES, N.TIMES[1:]))


def test_lines_are_dense_and_short(all_lines):
    kinds = set()
    for s, lines in all_lines.items():
        assert len(lines) == LINES_PER_ROLLOUT == 634
        assert len({ln.text for ln in lines}) == len(lines)
        for ln in lines:
            assert is_dense(ln.text), ln.text
            assert ntokens(ln.text) <= 80, ln.text
            assert ln.topic == "nucleo"
            words = ln.prompt.split()
            if ln.kind != "record":
                assert "." not in ln.prompt and "." not in ln.answer
                assert ln.text == f"q {ln.prompt}. a {ln.answer}."
            kinds.add((ln.kind, next((w for w in ("next", "final", "param", "params.") if w in words), "step")))
    assert kinds == {("record", "params."), ("record", "step"), ("fact", "param"), ("fact", "step"),
                     ("calc", "next"), ("calc", "final")}


def test_a_seven_digit_seed_still_fits():
    lines = N.lines(N.rollout(1234567))
    assert max(ntokens(ln.text) for ln in lines) <= 72
    assert all(is_dense(ln.text) for ln in lines)
    assert all(N.check(ln.prompt, ln.answer).ok for ln in lines if ln.kind != "record")


def test_state_line_has_three_digits_and_the_chosen_keys(all_lines):
    rec = [ln for ln in all_lines[1] if ln.kind == "record" and " step " in ln.text]
    assert len(rec) == 30
    assert rec[0].text.startswith("nucleo seed 1 step 0. time 0 point 1. temperature ")
    assert all(f" {k} " in ln.text for ln in rec for k in N.STATE_KEYS)
    assert all(" helium3 " not in ln.text and " lithium " not in ln.text for ln in rec)
    r = N.rollout(1)
    assert r.value(0.26250331) == "0 point 2 6 3" and r.value(1223.0) == "1 2 2 3"
    for ln in all_lines[1]:
        if ln.kind != "record" and ln.answer[0].isdigit():
            digits = "".join(w for w in ln.answer.split(" e ")[0].split() if w.isdigit()).lstrip("0")
            assert len(digits) <= 4, ln.text  # three significant digits; whole numbers below 1e6 in full


def test_known_lines(all_lines):
    texts = {ln.text for ln in all_lines[1]}
    for want in [
        "nucleo seed 1 params. eta_factor 0 point 4 0 9. tau_n 9 3 6. t_freeze 1 point 4 4. expansion_factor 0 point 8 3 5.",
        "q nucleo seed 1 param eta_factor. a 0 point 4 0 9.",
        "q nucleo seed 1 param tau_n. a 9 3 6.",
        "nucleo seed 1 step 0. time 0 point 1. temperature 3 point 4 6 e 1 0. n_p_ratio 0 point 6 4 8. "
        "free_neutrons 0 point 3 9 3. hydrogen 0 point 6 0 7. helium 0. deuterium 0. stage plasma.",
        "nucleo seed 1 step 2 2. time 2 4 9. temperature 6 point 9 4 e 8. n_p_ratio 0 point 1 1 3. "
        "free_neutrons 0 point 0 1 7 8. hydrogen 0 point 8 1 4. helium 0 point 1 6 8. "
        "deuterium 8 point 4 9 e minus 5. stage bottleneck.",
        "q nucleo seed 1 step 0 n_p_ratio. a 0 point 6 4 8.",
        "q nucleo seed 1 step 9 stage. a freeze_out.",
        "q nucleo seed 1 step 2 2 helium. a 0 point 1 6 8.",
        "q nucleo seed 1 step 2 2 lithium. a 6 point 7 9 e minus 1 1.",
        "q nucleo seed 1 step 2 2 next helium. a 0 point 2 0 1.",
        "q nucleo seed 1 step 2 2 next stage. a helium_forming.",
        "q nucleo seed 1 step 2 6 free_neutrons. a 0.",
        "q nucleo seed 1 final helium. a 0 point 2 0 2.",
        "q nucleo seed 1 final hydrogen. a 0 point 7 9 8.",
        "q nucleo seed 1 final deuterium. a 1 point 0 5 e minus 4.",
        "q nucleo seed 1 final lithium. a 8 point 3 6 e minus 1 1.",
        "q nucleo seed 1 final bottleneck_time. a 1 9 8.",
        "q nucleo seed 1 final freeze_time. a 1 point 9 4.",
        "q nucleo seed 1 final peak_n_p_ratio. a 0 point 6 4 8.",
    ]:
        assert want in texts, want


def test_gate_agrees_with_every_line(sim, all_lines):
    for lines in all_lines.values():
        for ln in lines:
            if ln.kind == "record":
                continue
            assert sim.owns(ln.prompt), ln.text
            v = sim.check(ln.prompt, ln.answer)
            assert v.ok, (ln.text, v)
            assert v.expected == ln.answer


def _wrong(answer: str) -> str:
    """Change the first digit (so the number moves by at least ten percent), or the word."""
    words = answer.split()
    if words[0].isdigit():
        words[0] = "8" if words[0] == "9" else str(int(words[0]) + 1)
        return " ".join(words)
    return "plasma" if answer != "plasma" else "done"


def test_gate_rejects_a_changed_digit_or_word(sim, all_lines):
    for lines in all_lines.values():
        for ln in lines:
            if ln.kind == "record":
                continue
            v = sim.check(ln.prompt, _wrong(ln.answer))
            assert not v.ok, (ln.text, _wrong(ln.answer), v)
            assert v.expected == ln.answer
            if " e " in ln.answer:  # a wrong exponent, or its sign lost
                mantissa, exp = ln.answer.split(" e ")
                flipped = exp.removeprefix("minus ") if exp.startswith("minus ") else "minus " + exp
                assert not sim.check(ln.prompt, f"{mantissa} e {flipped}").ok, ln.text
            if ln.answer not in N.STAGES and ln.answer != "0":
                assert not sim.check(ln.prompt, "minus " + ln.answer).ok, ln.text
                assert not sim.check(ln.prompt, "0").ok, ln.text
    assert not sim.check("nucleo seed 1 step 3 helium", "abc").ok
    assert sim.check("nucleo seed 1 step 3 helium", "abc").expected == "0"
    assert not sim.check("nucleo seed 1 step 3 stage", "0").ok
    assert sim.check("nucleo seed 1 step 3 stage", "0").expected == "plasma"


def test_five_percent_tolerance_no_floor():
    v = N.rollout(1).summary["lithium"]
    assert N.check("nucleo seed 1 final lithium", num(v * 1.04, sig=4)).ok
    assert not N.check("nucleo seed 1 final lithium", num(v * 1.06, sig=4)).ok
    assert not N.check("nucleo seed 1 final lithium", "0").ok
    assert N.check("nucleo seed 1 step 0 helium", "0 point 0").ok
    assert not N.check("nucleo seed 1 step 0 helium", "1 e minus 1 2").ok


def test_an_answer_that_is_not_a_finite_number_is_wrong():
    for prompt in ["nucleo seed 1 final helium", "nucleo seed 1 step 0 helium", "nucleo seed 1 step 5 next temperature",
                   "nucleo seed 1 param tau_n"]:
        for answer in ["1 e 9 9 9", "minus 1 e 9 9 9", " ".join("9" * 400), "", "e 5", "1 point", "point", "minus",
                       "1 point 2 point 3", "nan", "inf", "yes"]:
            v = N.check(prompt, answer)
            assert not v.ok and v.expected is not None, (prompt, answer, v)


def test_parameters_are_judged_exactly():
    r = N.rollout(1)
    for k in N.RANGES:
        assert N.check(f"nucleo seed 1 param {k}", r.value(r.params[k])).ok
        assert parse_num(r.value(r.params[k])) == r.params[k]
        off = N.check(f"nucleo seed 1 param {k}", num(r.params[k] * 1.01, sig=4))
        assert not off.ok and off.expected == r.value(r.params[k]) and off.reason == "wrong parameter"
    assert N.check("nucleo seed 1 param tau_n", "9 3 6 point 0").ok
    assert not N.owns("nucleo seed 1 param t1") and not N.owns("nucleo seed 1 params")
    assert N.params_line(r).kind == "record" and len(N.param_lines(r)) == 4


def test_owns(sim):
    assert sim.owns("nucleo seed 7 step 1 5 helium")
    assert sim.owns("nucleo seed 7 step 1 5 next stage")
    assert sim.owns("nucleo seed 1 2 3 4 final bottleneck_time")
    assert sim.owns("nucleo seed 0 param eta_factor")
    assert sim.owns("q nucleo seed 7 step 0 time.")
    for other in ["turkey capital", "q spoon color", "gravity seed 7 step 2 0 clumps", "carbon protons",
                  "calc 1 2 plus 7", "water molar_mass", "nucleo seed 7 step 3 colour", "nucleo seed x step 3 helium",
                  "nucleo seed 7", "nucleo seed 7 final", "nucleo seed 7 step helium", "nucleo seed 7 step 3 next",
                  "nucleo 7 step 3 helium", "nucleo seed 7 final time", "", "nucleo", "nucleo seed",
                  "gravity seed 7 final clumps", "chem seed 4 2 step 1 2 ch4", "chem seed 4 2 final water_fraction",
                  "life seed 7 step 2 6 replicators", "life seed 7 param mu", "world seed 3 era planets",
                  "planets seed 7 step 3 helium", "gravity seed 7 step 3 helium", "nucleo seed 7 step 3 helium extra",
                  "nucleo seed 7 step 3 next next helium", "nucleo seed minus 7 step 3 helium",
                  "nucleo seed 7 point 5 step 3 helium", "nucleo seed 0 7 step 3 helium", "nucleo seed 7 step 0 3 helium",
                  "nucleo seed 7 final stage", "nucleo seed 7 step 3 bottleneck_time", "nucleo seed 7 param helium",
                  "check nucleo seed 7 step 3 helium", "judge nucleo seed 7 step 3 helium answer 0"]:
        assert not sim.owns(other), other
        assert sim.check(other, "1") == N.Verdict(False, None, "not my question")


def test_check_handles_missing_steps(sim):
    assert not sim.check("nucleo seed 7 step 9 9 helium", "0").ok
    assert not sim.check("nucleo seed 7 step 3 0 helium", "0").ok
    assert not sim.check("nucleo seed 7 step 2 9 next helium", "0").ok
    assert sim.check("nucleo seed 7 step 2 9 next helium", "0").expected is None
    assert sim.check("nucleo seed 7 step 2 8 next helium", N.rollout(7).value(N.rollout(7).steps[29]["helium"])).ok


def test_the_gate_keeps_its_own_rollout():
    r = N.rollout(5)
    truth = r.value(r.steps[25]["helium"])
    r.steps[25]["helium"] = 0.9
    r.summary["helium"] = 0.9
    r.params["tau_n"] = 1.0
    assert N.rollout(5).steps[25]["helium"] != 0.9 and N.rollout(5).params["tau_n"] != 1.0
    assert not N.check("nucleo seed 5 step 2 5 helium", "0 point 9").ok
    assert N.check("nucleo seed 5 step 2 5 helium", truth).ok
    assert N.rollout(5) is not N.rollout(5) and N.rollout(5).steps == N.rollout(5).steps
    assert type(N.rollout(5)) is N.NucleoRollout


def test_conserved_for_defaults_and_every_seed(sim, every_rollout):
    assert sim.conserved(N.run(1)).ok
    for r in every_rollout:
        v = sim.conserved(r)
        assert v.ok, (r.seed, v)


def test_conserved_over_the_corners_of_the_parameter_box(sim):
    names = list(N.RANGES)
    for corner in itertools.product(*[(N.RANGES[k][0], N.RANGES[k][1]) for k in names]):
        r = N.run(0, **dict(zip(names, corner)))
        assert sim.conserved(r).ok, corner
        assert 0.10 <= r.summary["helium"] <= 0.50
        assert 100 <= r.summary["bottleneck_time"] <= 250
        stages = [s["stage"] for s in r.steps]
        assert stages.count("plasma") >= 3 and stages.count("helium_forming") >= 1 and stages.count("done") >= 5


def test_plausibility_targets_of_the_default_run():
    r = N.run(1)
    y = r.summary["helium"]
    assert 0.20 <= y <= 0.32
    assert close(y, 0.2625, rel=0.001)
    assert r.value(y) == "0 point 2 6 3"
    assert 1 / 6 <= freeze_ratio(r.params) <= 1 / 5
    assert 1 / 8 <= r.summary["bottleneck_n_p_ratio"] <= 1 / 6
    assert 100 <= r.summary["bottleneck_time"] <= 300 and close(r.summary["bottleneck_time"], 156.25, rel=1e-6)
    assert r.summary["freeze_time"] == 1.2
    assert r.summary["deuterium"] == 2.5e-5 and r.summary["helium3"] == 1e-5 and r.summary["lithium"] == 5e-10
    assert r.summary["helium"] + r.summary["hydrogen"] > 0.999
    assert [s["stage"] for s in r.steps] == (["plasma"] * 7 + ["freeze_out"] + ["decay"] * 13 + ["bottleneck"]
                                             + ["helium_forming"] * 2 + ["done"] * 6)
    frozen = next(s for s in r.steps if s["stage"] == "freeze_out")
    assert frozen["time"] == 1.2 and close(frozen["n_p_ratio"], freeze_ratio(r.params), rel=1e-12)


def test_plausibility_targets_of_every_seed(every_rollout):
    low, high = 1.0, 0.0
    for r in every_rollout:
        stages = [s["stage"] for s in r.steps]
        assert [N.STAGES.index(x) for x in stages] == sorted(N.STAGES.index(x) for x in stages), r.seed
        assert stages.count("plasma") >= 3 and stages.count("freeze_out") == 1 and stages.count("decay") >= 1
        assert stages.count("bottleneck") == 1 and stages.count("helium_forming") >= 1 and stages.count("done") >= 5
        y = r.summary["helium"]
        low, high = min(low, y), max(high, y)
        assert 0.10 <= y <= 0.50, (r.seed, y)
        assert 0.9995 <= y + r.summary["hydrogen"] < 1
        assert 100 <= r.summary["bottleneck_time"] <= 250
        assert 0.57 <= r.summary["peak_n_p_ratio"] <= 0.68 and r.summary["peak_n_p_ratio"] == r.steps[0]["n_p_ratio"]
        assert r.steps[-1]["free_neutrons"] == 0
    assert low < 0.15 and high > 0.45  # the box really is that wide


def test_conserved_rejects_tampering(sim):
    r = N.run(1)
    bad = copy.deepcopy(r)
    bad.steps[-1]["helium"] += 1e-6
    assert "nucleons" in sim.conserved(bad).reason
    bad = copy.deepcopy(r)
    bad.steps[5]["temperature"] = bad.steps[4]["temperature"]
    assert "temperature" in sim.conserved(bad).reason
    bad = copy.deepcopy(r)  # nucleons kept, but a proton turned back into a neutron
    bad.steps[-1]["hydrogen"] -= 1e-6
    bad.steps[-1]["free_neutrons"] += 1e-6
    assert "protons decreased" in sim.conserved(bad).reason
    bad = copy.deepcopy(r)
    bad.steps[3]["stage"] = "done"
    assert "stage" in sim.conserved(bad).reason
    bad = copy.deepcopy(r)
    bad.summary["helium"] = 0.5
    assert "summary" in sim.conserved(bad).reason
    bad = copy.deepcopy(r)
    bad.summary["peak_n_p_ratio"] = 0.1
    assert "peak_n_p_ratio" in sim.conserved(bad).reason
    bad = copy.deepcopy(r)
    del bad.summary["freeze_time"]
    assert "summary lacks" in sim.conserved(bad).reason
    bad = copy.deepcopy(r)  # a bottleneck that left fewer neutrons than the helium holds
    bad.summary["bottleneck_n_p_ratio"] = 0.1
    assert "needs more neutrons" in sim.conserved(bad).reason
    bad = copy.deepcopy(r)
    bad.summary["bottleneck_n_p_ratio"] = 0.3
    assert "less the neutrons that decay" in sim.conserved(bad).reason
    bad = copy.deepcopy(r)
    bad.steps = bad.steps[:22]
    assert "end done" in sim.conserved(bad).reason
    bad = copy.deepcopy(r)
    bad.steps[4]["helium"] = math.nan
    assert not sim.conserved(bad).ok
    bad = copy.deepcopy(r)
    del bad.steps[4]["lithium"]
    assert "lacks" in sim.conserved(bad).reason
    assert not sim.conserved(Rollout("gravity", 1, {}, [{"clumps": 1}])).ok
    assert not sim.conserved(Rollout("nucleo", 1, {}, [])).ok


def test_conserved_allows_other_binding_times(sim):
    for tau_bind in (5.0, 30.0, 100.0):
        r = N.run(1, tau_bind=tau_bind)
        assert sim.conserved(r).ok, tau_bind
    assert N.run(1, tau_bind=100.0).summary["helium"] < N.run(1, tau_bind=5.0).summary["helium"]


def test_physics_invariants():
    for s in SEEDS:
        r = N.rollout(s)
        t1 = r.params["t1"] / r.params["expansion_factor"] ** 0.5
        times = [st["time"] for st in r.steps]
        assert times == list(N.TIMES)
        for st in r.steps:
            assert close(st["temperature"], t1 / st["time"] ** 0.5, rel=1e-12)
        temps = [st["temperature"] for st in r.steps]
        assert all(a > b for a, b in zip(temps, temps[1:]))
        plasma = [st for st in r.steps if st["stage"] == "plasma"]
        for st in plasma:  # weak equilibrium: n/p = exp(-Q / kT)
            assert close(st["n_p_ratio"], math.exp(-1.29333236 / (8.617333262e-11 * st["temperature"])), rel=1e-12)
            assert st["time"] < r.summary["freeze_time"]
        decay = [st for st in r.steps if st["stage"] in ("freeze_out", "decay")]
        for a, b in zip(decay, decay[1:]):  # free decay with the neutron lifetime
            assert close(b["free_neutrons"] / a["free_neutrons"], math.exp(-(b["time"] - a["time"]) / r.params["tau_n"]), rel=1e-9)
        assert close(r.summary["freeze_time"], r.params["t_freeze"] * r.params["expansion_factor"] ** (-5 / 3), rel=1e-12)
        before = [st for st in r.steps if st["time"] < r.summary["bottleneck_time"]]
        after = [st for st in r.steps if st["time"] >= r.summary["bottleneck_time"]]
        assert all(st["helium"] == 0 and st["deuterium"] == 0 and st["helium3"] == 0 and st["lithium"] == 0 for st in before)
        assert all(st["free_neutrons"] + st["hydrogen"] == 1 for st in before)
        assert all(st["helium"] > 0 and st["deuterium"] > 0 for st in after)
        assert after[0]["stage"] == "bottleneck"
        assert r.steps[-1]["free_neutrons"] == 0
        assert r.summary["peak_n_p_ratio"] == r.steps[0]["n_p_ratio"]
        y, ratio = r.summary["helium"], r.summary["bottleneck_n_p_ratio"]
        assert close(y, 2 * ratio / (1 + ratio), rel=0.05) and y < 2 * ratio / (1 + ratio)
        assert close(r.summary["deuterium"], 2.5e-5 * r.params["eta_factor"] ** -1.6, rel=1e-9)
        assert close(r.summary["lithium"], 5e-10 * r.params["eta_factor"] ** 2, rel=1e-9)
        assert close(r.summary["helium3"], 1e-5 * r.params["eta_factor"] ** -0.6, rel=1e-9)


def test_parameters_pull_the_right_way():
    base = N.run(1).summary
    fast, slow = N.run(1, expansion_factor=1.4).summary, N.run(1, expansion_factor=0.7).summary
    assert slow["helium"] < base["helium"] < fast["helium"]
    assert fast["bottleneck_time"] < base["bottleneck_time"] < slow["bottleneck_time"]
    assert fast["freeze_time"] < base["freeze_time"] < slow["freeze_time"]
    dense, thin = N.run(1, eta_factor=3).summary, N.run(1, eta_factor=0.3).summary
    assert dense["deuterium"] < base["deuterium"] < thin["deuterium"]
    assert dense["helium3"] < base["helium3"] < thin["helium3"]
    assert dense["lithium"] > base["lithium"] > thin["lithium"]
    assert dense["bottleneck_time"] < base["bottleneck_time"] < thin["bottleneck_time"]
    assert thin["helium"] < base["helium"] < dense["helium"] < base["helium"] + 0.01
    assert N.run(1, tau_n=960).summary["helium"] > base["helium"] > N.run(1, tau_n=800).summary["helium"]
    assert N.run(1, t_freeze=2).summary["helium"] < base["helium"] < N.run(1, t_freeze=0.5).summary["helium"]
    # one more kind of neutrino speeds the expansion up by about 8 percent: helium rises by about 0.013
    assert 0.008 < N.run(1, expansion_factor=1.08).summary["helium"] - base["helium"] < 0.02


def test_run_refuses_parameters_outside_the_model():
    with pytest.raises(TypeError):
        N.run(1, gravity=3)
    for bad in [{"tau_n": -1}, {"tau_n": 0}, {"eta_factor": math.nan}, {"t1": math.inf},
                {"eta_factor": 1e15}, {"t_freeze": 500.0}]:
        with pytest.raises(ValueError):
            N.run(1, **bad)


def test_lines_want_the_seed_rollout():
    with pytest.raises(ValueError):
        N.lines(N.run(7))
    with pytest.raises(ValueError):
        N.lines(N.run(7, **N.random_params(random.Random(8))))
    with pytest.raises(ValueError):
        N.lines(N.rollout(7), horizon=2)
    with pytest.raises(ValueError):
        N.lines(N.run(-3, **N.random_params(random.Random(-3))))
    with pytest.raises(ValueError):
        N.lines(Rollout("gravity", 7, {}, []))
    changed = N.rollout(7)
    changed.steps[25]["helium"] = 0.9
    with pytest.raises(ValueError):
        N.lines(changed)
    for seed in (-1, 1.5, "7", True, None):
        with pytest.raises(ValueError):
            N.rollout(seed)
    assert len(N.lines(N.rollout(7), keys=["helium", "stage"])) == 1 + N_PARAMS + 30 + 30 * 2 + 29 * 2 + len(N.FINAL_KEYS)


def test_lines_round_trip_the_numbers():
    r = N.rollout(3)
    for ln in N.lines(r):
        if ln.kind == "record":
            continue
        words = ln.prompt.split()
        key = words[-1]
        if "final" in words:
            v = r.summary[key]
        elif "param" in words:
            v = r.params[key]
        else:
            end = words.index("next") if "next" in words else len(words) - 1
            step = parse_num(words[words.index("step") + 1:end])
            v = r.steps[step + ("next" in words)][key]
        if isinstance(v, str):
            assert ln.answer == v
        else:
            assert close(parse_num(ln.answer), v, rel=0.006, abs_=0.0)


def test_generate_is_seeded_questions_the_gate_accepts(sim):
    a = sim.generate(random.Random(3), 150)
    b = sim.generate(random.Random(3), 150)
    assert [ln.text for ln in a] == [ln.text for ln in b] and len(a) == 150
    assert [ln.text for ln in a] != [ln.text for ln in sim.generate(random.Random(4), 150)]
    assert len({ln.prompt for ln in a}) == 150
    assert len({ln.prompt.split(" step ")[0].split(" final ")[0].split(" param ")[0] for ln in a}) == 8  # 20 a seed
    for ln in a:
        assert ln.kind != "record" and ln.topic == "nucleo"
        assert is_dense(ln.text) and ntokens(ln.text) <= 80
        assert sim.owns(ln.prompt)
        v = sim.check(ln.prompt, ln.answer)
        assert v.ok and v.expected == ln.answer
        assert not sim.check(ln.prompt, _wrong(ln.answer)).ok
    assert sim.generate(random.Random(3), 0) == [] and len(sim.generate(random.Random(3), 1)) == 1
    assert len(N.generate(random.Random(1), 2500)) == 2500


def test_a_freeze_out_on_the_grid_is_exact():
    """Seed 9353 has expansion_factor 1 and t_freeze 1.2, a grid time: no rounding decides the stage."""
    r = N.rollout(9353)
    assert r.params["expansion_factor"] == 1.0 and r.params["t_freeze"] == 1.2 and r.summary["freeze_time"] == 1.2
    at = N.TIMES.index(1.2)
    assert r.steps[at]["stage"] == "freeze_out" and r.steps[at - 1]["stage"] == "plasma"
    assert close(r.steps[at]["n_p_ratio"], freeze_ratio(r.params), rel=1e-12)


def test_no_decision_hangs_on_the_last_bit(every_rollout):
    """What may differ between machines is the last bit of exp, log and pow (about 1e-16). Over
    every seed that can be drawn, no stage threshold and no printed third digit is closer than
    1e-12 to flipping, except where the value is exact by construction."""
    grid = np.array(N.TIMES)
    values: list[float] = []
    exact_ties = 0
    for r in every_rollout:
        eta = r.params["eta_factor"]
        for edge in (r.summary["freeze_time"], r.summary["bottleneck_time"]):
            gap = np.abs(grid / edge - 1)
            exact_ties += int((gap == 0).sum())
            assert gap[gap > 0].min() > 1e-9, r.seed
        if r.summary["freeze_time"] in N.TIMES:
            assert r.params["expansion_factor"] == 1.0 and r.params["t_freeze"] in N.TIMES
        n_d = r.summary["bottleneck_n_p_ratio"] / (1 + r.summary["bottleneck_n_p_ratio"])
        tau_eff = 1 / (1 / r.params["tau_bind"] + 1 / r.params["tau_n"])
        for st in r.steps:
            if st["time"] >= r.summary["bottleneck_time"]:
                n = n_d * math.exp(-(st["time"] - r.summary["bottleneck_time"]) / tau_eff)
                assert abs(n / 1e-6 - 1) > 1e-9 and abs(n / 1e-10 - 1) > 1e-9, r.seed
                assert (st["free_neutrons"] == 0) == (n < 1e-10)
            if st["free_neutrons"] == 0 and st["stage"] == "done":
                assert st["lithium"] == 5e-10 * (eta * eta)  # a plain product, the same on every machine
            else:
                values.append(st["lithium"])
            values += [st[k] for k in ("n_p_ratio", "free_neutrons", "hydrogen", "helium", "deuterium", "helium3")]
        assert r.summary["lithium"] == 5e-10 * (eta * eta)
        values += [v for k, v in r.summary.items() if k != "lithium"]
        rng = random.Random(r.seed)  # the raw draws behind the three-digit parameters
        for key, (lo, hi, scale) in N.RANGES.items():
            values.append(math.exp(rng.uniform(math.log(lo), math.log(hi))) if scale == "log" else rng.uniform(lo, hi))
    assert exact_ties >= 1
    x = np.array(values, dtype=np.float64)
    x = x[x > 0]
    mantissa = x / 10.0 ** (np.floor(np.log10(x)) - 2)  # 100 .. 1000: the third digit is the last printed
    edge = np.abs(mantissa - np.floor(mantissa) - 0.5) / mantissa
    assert len(x) > 1_000_000 and edge.min() > 1e-12


def test_the_module_needs_no_numpy():
    assert not hasattr(N, "np") and not hasattr(N, "numpy")
    assert all(type(v) in (float, str) for st in N.run(1).steps for v in st.values())
    assert all(type(v) is float for v in N.run(1).summary.values())


def test_constants_are_marked_as_remembered():
    for name, row in N.CONSTANTS.items():
        assert "from memory" in row["notes"], name
    assert close(N.CONSTANTS["q_np_mev"]["value"], 939.56542 - 938.27209, rel=1e-4)  # m_n - m_p in MeV
    assert close(N.CONSTANTS["k_mev_per_k"]["value"], 1.380649e-23 / 1.602176634e-19 * 1e-6, rel=1e-9)


def test_a_run_is_fast():
    t0 = time.perf_counter()
    for s in range(50):
        N.run(s, **N.random_params(random.Random(s)))
    assert (time.perf_counter() - t0) / 50 < 0.01
