import pytest

from haishool.science_routes import EXAMPLES, registry, route


@pytest.mark.parametrize("question,prompt,topic", [
    ("12plus7", "calc 1 2 plus 7", "maths"),
    ("what is 12 plus 7", "calc 1 2 plus 7", "maths"),
    ("-2.5 + 1.25", "calc minus 2 point 5 plus 1 point 2 5", "maths"),
    ("calculate (2 + 3) * 4", "calc open 2 plus 3 close times 4", "maths"),
    ("12 / 4", "calc 1 2 over 4", "maths"),
    ("What is 12 divided by 4?", "calc 1 2 over 4", "maths"),
    ("What is 12 multiplied by 7?", "calc 1 2 times 7", "maths"),
    ("What is 2 to the power of 3?", "calc 2 power 3", "maths"),
    ("What is 12 times 7?", "calc 1 2 times 7", "maths"),
    ("2 ^ 3", "calc 2 power 3", "maths"),
    ("solve 3x+4=19", "solve 3 x plus 4 equals 1 9", "maths"),
    ("47 times 6 with steps", "calc 4 7 times 6 steps", "maths"),
    ("47 plus 38 show steps", "calc 4 7 plus 3 8 steps", "maths"),
    ("How many protons does carbon have?", "carbon protons", "elements"),
    ("carbon proton count", "carbon protons", "elements"),
    ("Atomic number of Fe", "iron number", "elements"),
    ("What is the symbol for carbon?", "carbon symbol", "elements"),
    ("number of neutrons in Fe", "iron neutrons", "elements"),
    ("Fe atomic mass", "iron mass", "elements"),
    ("what is the molar mass of water", "molar_mass water", "substances"),
    ("molar mass of H2O", "molar_mass water", "substances"),
    ("chemical formula of water", "water formula", "substances"),
    ("what is the chemical formula of water", "water formula", "substances"),
    ("Mass percent of oxygen in H2O", "mass_percent water oxygen", "substances"),
    ("Balance H2 + O2 -> H2O", "balance h 2 plus o 2 gives h 2 o 1", "reactions"),
    ("balance hydrogen + oxygen = water", "balance h 2 plus o 2 gives h 2 o 1", "reactions"),
    ("friction mu=0.3 n=50 N", "friction mu 0 point 3 n 5 0", "forces"),
    ("weight mass=5 kg on moon", "weight m 5 on moon", "forces"),
    ("pressure area=2 m2 force=100 N", "pressure f 1 0 0 area 2", "forces"),
    ("calculate pressure with force=100 N and area=2 m^2", "pressure f 1 0 0 area 2", "forces"),
    ("spring force k=200 N/m x=0.05 m", "spring_force k 2 0 0 x 0 point 0 5", "forces"),
    ("gravitational force mass1=10 kg mass2=20 kg distance=2 m", "gravity_force m 1 0 m 2 0 r 2", "forces"),
])
def test_natural_questions_route_to_real_gates(question, prompt, topic):
    result = route(question)
    assert result is not None and result.error is None
    assert result.prompt == prompt
    assert result.topic == topic
    assert result.gate.owns(prompt)
    verdict = result.gate.check(prompt, "")
    assert verdict.expected is not None
    assert result.gate.check(prompt, verdict.expected).ok
    assert not hasattr(result, "answer") and not hasattr(result, "expected")


@pytest.mark.parametrize("prompt", [
    "calc 1 2 plus 7", "calc 4 7 times 6 steps", "pressure f 1 0 0 area 2",
    "life predict expected_mutations length 8 mu 0 point 2",
    "world predict handoff cooling metallicity 0 point 0 1",
    "carbon protons", "molar_mass h 2 o 1",
])
def test_raw_dense_numbers_and_empty_answer_prefix_preserved(prompt):
    for question in (prompt, "q " + prompt + ". a", "q " + prompt + "."):
        result = route(question)
        assert result is not None and result.error is None
        assert result.prompt == prompt


@pytest.mark.parametrize("question", [
    "q calc 1 2 plus 7. a 1 9.", "carbon protons. a 6",
    "12 / 0", "What is 12 divided by 0?", "What is 12 squared?", "What is the square root of 9?",
    "9 ^ 999", "solve x*x=4", "12 minus 7 steps",
    "friction mu=0.3", "weight mass=5 g", "weight mass=5 N", "weight m 5 on pluto",
    "pressure force=100 N area=2 cm2", "spring force k=200 N/m x=5 cm",
    "pressure f=1e999 area=2", "calculate gravitational force", "what is the force on 5 kg",
    "molar mass of unobtainium", "protons in unobtainium", "balance H2 -> CO2",
    "gravity seed 7 final clumps", "what is world seed 5 final outcome",
    "life predict expected_mutations length 8 mu 5",
    "calc 2 plus 2 answer 4", "calculate __import__('os')",
])
def test_unsupported_science_and_answer_injection_fail_closed(question):
    result = route(question)
    assert result is not None
    assert result.error
    assert result.gate is None
    assert not result.prompt


@pytest.mark.parametrize("question", [
    "What is gravity?", "what is friction", "what color is iron", "iron color",
    "what is physics", "what is chemistry",
    "what is the capital of Japan", "Japan capital", "who is Beethoven", "turkey borders greece",
])
def test_ordinary_fact_questions_remain_old_route(question):
    assert route(question) is None


def test_preview_examples_and_registry():
    for example in EXAMPLES:
        result = route(example)
        assert result and not result.error, (example, result)
    assert {gate.topic for gate in registry()} == {
        "maths", "elements", "substances", "reactions", "forces",
        "predict_nucleo", "predict_gravity", "predict_planets", "predict_chem", "predict_life", "predict_world"}
    assert registry() is registry()


def test_no_numeric_precision_loss_and_bounded_input():
    result = route("0.123456789 + 1")
    assert result and result.prompt == "calc 0 point 1 2 3 4 5 6 7 8 9 plus 1"
    assert route("9" * 1100 + "+1").error
    assert route(None).error
    assert route("") is None


def test_gate_parser_exceptions_cannot_escape_validation(monkeypatch):
    from haishool import science_routes
    class BrokenGate:
        topic = "maths"
        def owns(self, prompt):
            raise RuntimeError("malformed input parser failure")
    monkeypatch.setattr(science_routes, "gates", lambda: (BrokenGate(),))
    assert science_routes._validated("calc 1 plus 2") is None
