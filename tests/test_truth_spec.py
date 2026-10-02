import pytest

from haishool.truth import Line, dedupe, is_dense, num, parse_num, split_line
from haishool.truth.formula import dense, parse, parse_dense, same


@pytest.mark.parametrize("x,text", [
    (1838, "1 8 3 8"), (-3.5, "minus 3 point 5"), (0.25, "0 point 2 5"), (6.674e-11, "6 point 6 7 4 e minus 1 1"),
    (2.0, "2"), (0, "0"), (1e6, "1 e 6"), (55.845, "5 5 point 8 4"),
])
def test_num(x, text):
    assert num(x) == text


def test_num_significant_digits():
    assert num(18.015, sig=5) == "1 8 point 0 1 5"
    assert num(55.845, sig=5) == "5 5 point 8 4 5"


@pytest.mark.parametrize("text,x", [
    ("1 8 point 0 1 5", 18.015), ("minus 4", -4), ("6 point 6 7 4 e minus 1 1", 6.674e-11), ("abc", None), ("", None),
])
def test_parse_num(text, x):
    assert parse_num(text) == x


def test_round_trip():
    for x in [0, 7, -12, 1838, 0.5, 18.015, 299792458, 6.674e-11, -0.001]:
        back = parse_num(num(x, sig=6))
        assert back == pytest.approx(x, rel=1e-5)


def test_lines_and_density():
    q = Line("carbon protons", "6", "elements")
    assert q.text == "q carbon protons. a 6."
    assert split_line(q.text) == ("carbon protons", "6")
    r = Line("carbon element. number 6. symbol c", topic="elements", kind="record")
    assert r.text.endswith("symbol c.") and split_line(r.text) is None
    with pytest.raises(ValueError):
        Line("carbon  protons", "6")
    with pytest.raises(ValueError):
        Line("carbon protons", "6.0")
    assert not is_dense("Carbon protons.") and is_dense("q x. a 1 point 5.")
    assert len(dedupe([q, q, r])) == 2


def test_formulas():
    assert parse("Ca(OH)2") == {"Ca": 1, "O": 2, "H": 2}
    assert parse("C6H12O6") == {"C": 6, "H": 12, "O": 6}
    assert dense("Fe2O3") == "fe 2 o 3"
    assert parse_dense("ca 1 o 2 h 2") == {"Ca": 1, "O": 2, "H": 2}
    assert same(parse("H2O"), parse_dense("h 2 o"))
    assert parse_dense("hello") is None
    with pytest.raises(ValueError):
        parse("h2o")
