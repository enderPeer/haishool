from types import SimpleNamespace

import pytest

from haishool.student import tokens
from haishool.truth import Verdict, maths, reactions
from haishool.v5_app import display_dense, make_app, make_handler


PROMPT = "calc 1 2 plus 7"


def routed(prompt=PROMPT, gate=None, error=None):
    return SimpleNamespace(prompt=prompt, gate=gate or maths.gate(), error=error,
                           topic="maths", label="Arithmetic")


def make_test_handler(output="2 0", route=None, context=256, vocab=None):
    route = route or routed()
    calls = []

    def ask_model(prompt, room):
        calls.append(("model", prompt, room))
        return output

    def legacy(question):
        calls.append(("legacy", question))
        return {"answer": "Legacy answer", "query": "q japan capital. a", "dense": "tokyo"}

    words = set(["<eos>", *tokens(f"q {route.prompt}. a")]) if vocab is None else vocab
    return make_handler(ask_model, legacy, words, context, lambda question: route), calls


def test_wrong_model_answer_is_not_replaced_by_reference():
    ask, calls = make_test_handler()
    result = ask("what is 12 plus 7")
    assert result["route"] == "science"
    assert result["dense"] == "2 0" and result["answer"] == "20"
    assert result["verification"]["status"] == "rejected"
    assert result["verification"]["ok"] is False
    assert result["verification"]["expected"] == "1 9"
    assert result["verification"]["expected_display"] == "19"
    assert calls == [("model", f"q {PROMPT}. a", 256 - len(tokens(f"q {PROMPT}. a")) - 1)]


def test_correct_answer_and_empty_answer_remain_distinct():
    ask, _ = make_test_handler("1 9")
    assert ask("calculate")["verification"]["status"] == "accepted"
    ask, _ = make_test_handler("")
    result = ask("calculate")
    assert result["dense"] == ""
    assert result["verification"]["status"] == "rejected"
    assert result["answer"] == "The model returned no answer."


def test_unknown_vocabulary_never_gets_silently_dropped():
    ask, calls = make_test_handler(vocab={"<eos>", "q", "a", ".", "calc", "1", "2", "7"})
    result = ask("calculate")
    assert result["route"] == "unsupported" and "plus" in result["answer"]
    assert result["verification"]["ok"] is None and calls == []


def test_context_overflow_stops_before_model_inference():
    prompt_size = len(tokens(f"q {PROMPT}. a")) + 1
    ask, calls = make_test_handler(context=prompt_size + 7)
    result = ask("calculate")
    assert result["route"] == "unsupported" and "context" in result["answer"]
    assert calls == []


def test_unsupported_science_does_not_fall_through_to_old_facts():
    ask, calls = make_test_handler(route=routed("calc 1 over 0", error="Division by zero is unsupported."))
    result = ask("what is 1 divided by zero")
    assert result["route"] == "unsupported"
    assert result["verification"]["expected"] is None and calls == []


@pytest.mark.parametrize("nested", [False, True])
def test_gate_errors_are_unavailable_not_failed_model_answers(nested):
    class Broken:
        def check(self, prompt, answer):
            if nested:
                return Verdict(False, None, "maths gate error ValueError: invalid state")
            raise ValueError("invalid state")

    ask, _ = make_test_handler(route=routed(gate=Broken()))
    result = ask("calculate")
    assert result["dense"] == "2 0"
    assert result["verification"]["status"] == "unavailable"
    assert result["verification"]["ok"] is None
    assert result["verification"]["expected"] is None


def test_nonfinite_answer_is_rejected_even_by_a_permissive_gate():
    class Permissive:
        def check(self, prompt, answer):
            return Verdict(True, "1")

    answer = "1 e 9 9 9 9"
    ask, _ = make_test_handler(answer, routed(gate=Permissive()))
    result = ask("calculate")
    assert result["dense"] == result["answer"] == answer
    assert result["verification"]["status"] == "rejected"
    assert result["verification"]["expected_display"] == "1"


def test_ordinary_questions_preserve_legacy_response():
    calls = []
    def legacy(question):
        calls.append(question)
        return {"answer": "Tokyo", "query": "q japan capital. a", "dense": "tokyo"}
    def model(*args):
        pytest.fail("legacy route must not call science model")
    ask = make_handler(model, legacy, set(), router=lambda question: None)
    result = ask("  what is the capital of japan  ")
    assert calls == ["what is the capital of japan"]
    assert result["answer"] == "Tokyo" and result["dense"] == "tokyo"
    assert result["route"] == "legacy" and result["verification"]["status"] == "unavailable"
    result = ask("x" * 301)
    assert result["route"] == "unsupported" and len(calls) == 1


@pytest.mark.parametrize("question", ["", "   ", "x" * 1201])
def test_invalid_input_is_stopped_before_routing(question):
    def forbidden(*args): pytest.fail("must validate length before callbacks")
    ask = make_handler(forbidden, forbidden, set(), router=forbidden)
    assert ask(question)["route"] == "unsupported"


@pytest.mark.parametrize("text,expected", [
    ("minus 1 2 point 5", "-12.5"),
    ("1 point 2 e minus 3", "1.2e-3"),
    ("1 e 9 9 9 9", "1 e 9 9 9 9"),
    ("7 plus 8 is 1 5 write 5 carry 1", "7 plus 8 is 15 write 5 carry 1"),
    ("electron_proton_nucleus", "electron proton nucleus"),
])
def test_dense_rendering_preserves_numbers_and_words(text, expected):
    assert display_dense(text) == expected


def test_balancing_coefficients_are_not_concatenated_into_one_number():
    prompt = "balance h 2 plus o 2 gives h 2 o 1"
    assert display_dense("2 1 2", prompt) == "2, 1, 2"
    ask, _ = make_test_handler("2 1 2", routed(prompt, reactions.gate()))
    result = ask("balance hydrogen and oxygen")
    assert result["answer"] == "2, 1, 2" and result["dense"] == "2 1 2"
    assert result["verification"]["expected_display"] == "2, 1, 2"
    assert result["verification"]["status"] == "accepted"


@pytest.mark.parametrize("question,prompt,output,display", [
    ("what is 12 plus 7", "calc 1 2 plus 7", "1 9", "19"),
    ("how many protons does carbon have", "carbon protons", "6", "6"),
    ("balance H2 + O2 -> H2O", "balance h 2 plus o 2 gives h 2 o 1", "2 1 2", "2, 1, 2"),
])
def test_real_router_reaches_science_model_before_legacy(question, prompt, output, display):
    calls = []
    def model(full_prompt, room):
        calls.append(full_prompt)
        return output
    def legacy(question):
        pytest.fail("a supported science question must not reach old fact fallback")
    full_prompt = f"q {prompt}. a"
    ask = make_handler(model, legacy, {"<eos>", *tokens(full_prompt)})
    result = ask(question)
    assert calls == [full_prompt]
    assert result["route"] == "science"
    assert result["dense"] == output and result["answer"] == display
    assert result["verification"]["status"] == "accepted"


def test_fastapi_accepts_question_json_and_validates_body():
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    pytest.importorskip("haishool.science_routes")
    from fastapi.testclient import TestClient

    received = []
    def handler(question):
        received.append(question)
        return {"answer": "19", "dense": "1 9", "route": "science"}
    app = make_app(handler, {"version": "v5", "science_ready": True}, "<h1>Preview</h1>")
    with TestClient(app) as client:
        response = client.post("/api/ask", json={"question": "12 + 7"})
        assert response.status_code == 200
        assert response.json()["dense"] == "1 9"
        assert received == ["12 + 7"]
        for invalid in ({}, {"question": ""}, {"question": "x" * 1201}):
            assert client.post("/api/ask", json=invalid).status_code == 422
        assert client.get("/api/info").json()["science_ready"] is True
        assert client.get("/api/examples").json()["examples"]
        home = client.get("/")
        assert home.status_code == 200 and home.headers["cache-control"] == "no-store"
