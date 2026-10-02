"""Round 5, topic ``elements``: the 118 chemical elements as a truth gate.

The table is built ONCE from the ``mendeleev`` package and written to
``data/truth-v5/elements.jsonl`` (one JSON object per element); the gate itself reads only the
jsonl, so the training machine needs no mendeleev:

    python -m haishool.truth.elements build            # writes data/truth-v5/elements.jsonl
    python -m haishool.truth.elements sample --n 30    # prints a few training lines

Values are looked up or computed. Where a value goes beyond a plain mendeleev lookup the row
carries a ``notes`` field (not trained) so a checker can target it; the few values that correct
the source from memory (:data:`_CORRECTIONS`, :data:`_NO_BULK`) say "from memory" there:

* ``mass_number`` and ``neutrons`` come from the isotope table: the most abundant isotope. The
  34 elements with no measured abundance (technetium, promethium, polonium to actinium,
  neptunium onwards) get the isotope of the source's bracket atomic weight ([98], [145]),
  unless another isotope is longer-lived beyond doubt: its half-life is at least twice its own
  uncertainty and exceeds the bracket isotope's by more than both uncertainties (meitnerium
  278, roentgenium 282, moscovium 290). A tentative isotope with a half-life of 680 +- 540 ms
  (oganesson 295) never wins. ``Element.mass_number`` is not used: it returns the first listed
  isotope when no abundance is known (technetium 83, polonium 186).
* ``mass`` is the standard atomic weight to 5 significant digits (decimal ties round up:
  50.9415 -> 50.942). The 34 elements without one get their mass number, stored as an integer;
  an integer mass is compared exactly and never enters a heavier / lighter question.
* ``radioactive`` is ``yes`` when no isotope is stable; only bismuth differs from mendeleev's
  own flag (bismuth 209 has a measured half-life of 2e19 years).
* ``category`` is mendeleev's series, except that lanthanum to lutetium are ``lanthanide`` and
  actinium to lawrencium ``actinide`` (15 each, as IUPAC counts them; the source calls
  lutetium and lawrencium transition metals). ``group`` is the group number; the f-block
  elements have none and say ``lanthanide`` / ``actinide`` instead (cerium to lutetium,
  thorium to lawrencium: lanthanum and actinium are d-block, group 3).
* ``melting_k`` / ``boiling_k`` are at 1 atm. The allotrope elements (phosphorus, sulfur,
  selenium, tin) take them from the one allotrope row that is neither a solid-solid transition
  nor a sublimation point (white phosphorus, whose density the source also gives). Carbon and
  arsenic do not melt at 1 atm: they have ``sublimation_k`` and neither of the two.
  ``state`` at 293 K follows from these values.
* astatine, francium and fermium onwards were never made in weighable amounts: the source's
  densities and melting points for them are predictions, so ``density_g_cm3``, ``melting_k``,
  ``boiling_k`` and ``state`` are left out. A list or count over a key covers only the
  elements that have it (97 elements have a state).
* ``density_g_cm3`` of a gas is at 25 C and 1 atm (hydrogen 8.2e-05; at 0 C it would be 9 %
  higher); ``electronegativity`` is the source's handbook Pauling value (newer tables print
  revised values for some heavy elements: tungsten 1.7 here, 2.36 there).
* ``valence`` holds the source's main oxidation states, all of them (chlorine up to plus 7).
  ``valence_known`` (not a key, never trained) holds every state the source lists; a wrong
  value in a check line always contains a state outside it, so no line denies a real state.
* ``discovered`` is ``ancient`` when mendeleev gives no year and says "known to the ancients".

Record lines (kind ``record``): every element is written as TWO lines, because nearly every
full row would exceed 80 tokens. The first carries the nucleus and the table position, the
second the physical properties; a key the table has no value for is left out:

    iron element. number 2 6. symbol fe. protons 2 6. electrons 2 6. neutrons 3 0. mass_number 5 6.
      mass 5 5 point 8 4 5. group 8. period 4. block d. category transition_metal. metal metal.
      radioactive no.
    iron element_more. state solid. melting_k 1 8 1 1 point 2. boiling_k 3 1 3 4 point 2.
      density_g_cm3 7 point 8 7. electronegativity 1 point 8 3. econf ar 3d6 4s2.
      valence plus 2 plus 3. discovered ancient.

Question lines, and the share of each in ``generate()``. The shares are of draws: the table
holds about 2 100 facts, 4 600 value checks and 90 lists and counts in all, so beyond a few
thousand lines the mix leans towards the computed kinds. About 139 000 distinct lines exist;
asked for more, ``generate()`` warns and returns what there is:

    fact   34 %  q iron protons. a 2 6.                       one key of one element
    calc    4 %  q iron neutrons. a 3 0.                      rules: mass_number minus protons,
                 q iron electrons. a 2 6.                     electrons equal protons
    calc    4 %  q neutrons mass_number 5 6 protons 2 6. a 3 0.   the rule as arithmetic
    calc    6 %  q element number 2 6. a iron.                reverse lookups
                 q element symbol fe. a iron.
                 q element group 8 period 4. a iron.
    calc    4 %  q element after iron. a cobalt.              neighbours by atomic number
                 q element before iron. a manganese.
    calc    8 %  q heavier iron copper. a copper.             by standard atomic weight
                 q lighter iron copper. a iron.
    calc    7 %  q higher melting_k iron copper. a iron.      by number, melting_k, boiling_k,
                 q lower density_g_cm3 iron copper. a iron.   density_g_cm3 or electronegativity
    calc    3 %  q elements group 1 8. a helium neon argon krypton xenon radon oganesson.
                 q elements period 1. a hydrogen helium.      ordered by number, at most 32 names
    calc    3 %  q count elements category noble_gas. a 7.
    yesno  20 %  q check iron protons 2 7. a no.              yes and no in turn; a wrong value is
                 q check helium category noble_gas. a yes.    another element's or one digit off
    yesno   7 %  q check iron heavier_than copper. a no.
                 q check iron lighter_than copper. a yes.

A comparison is only asked, and only judged, when the two values differ by more than the
tolerance. Nobelium's symbol is ``no``, the same token as the answer of a check line
(``q nobelium symbol. a no.``).

Numbers are digit tokens (:func:`haishool.truth.num`): integers exact, table floats with 5
significant digits. ``check()`` compares floats (temperatures, density, electronegativity)
with a relative tolerance of 0.5 % and atomic weights with 0.05 %, so a neighbour's weight
never passes; integers, names and lists word for word, oxidation states as sets. A number
must be written the way :func:`num` writes it: no leading zeros, no ``2 6 point 0`` for 26.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import warnings
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

from haishool.truth import Line, Verdict, is_finite, num, parse_num

TOPIC = "elements"
DEFAULT_TABLE = Path(__file__).resolve().parents[2] / "data" / "truth-v5" / "elements.jsonl"

#: key -> meaning and unit (the model only sees the key names)
KEYS: dict[str, str] = {
    "number": "atomic number",
    "symbol": "chemical symbol, lowercase",
    "protons": "protons in the nucleus (equal to the atomic number)",
    "electrons": "electrons of the neutral atom (equal to the protons)",
    "neutrons": "neutrons of the most abundant isotope, or of the bracket isotope (mass_number minus protons)",
    "mass_number": "mass number of the most abundant isotope, or of the longest-lived (bracket) isotope",
    "mass": "standard atomic weight in g/mol, 5 significant digits; the mass number where there is no standard weight",
    "group": "group 1 to 18; lanthanide / actinide for the f-block elements, which have no group number",
    "period": "period 1 to 7",
    "block": "s, p, d or f",
    "category": "alkali_metal, alkaline_earth_metal, transition_metal, poor_metal, metalloid, nonmetal, halogen, "
                "noble_gas, lanthanide (lanthanum to lutetium) or actinide (actinium to lawrencium)",
    "metal": "metal, metalloid or nonmetal (from the category)",
    "radioactive": "yes when the element has no stable isotope, else no",
    "state": "solid, liquid or gas at 293 K and 1 atm; left out where no melting or boiling point is measured",
    "melting_k": "melting point in kelvin at 1 atm",
    "boiling_k": "boiling point in kelvin at 1 atm",
    "sublimation_k": "sublimation point in kelvin at 1 atm, for the elements that do not melt there (carbon, arsenic)",
    "density_g_cm3": "density in g/cm3; gases at 25 C (298 K) and 1 atm",
    "electronegativity": "pauling electronegativity (the source's handbook values)",
    "econf": "electron configuration, noble-gas core first (ar 3d6 4s2)",
    "valence": "main oxidation states as signed numbers (plus 2 plus 3)",
    "discovered": "year of discovery or first isolation as the source gives it, or ancient",
}
#: keys of the first record line (``<name> element.``) and of the second (``<name> element_more.``)
RECORD_KEYS = ("number", "symbol", "protons", "electrons", "neutrons", "mass_number", "mass", "group",
               "period", "block", "category", "metal", "radioactive")
RECORD_MORE_KEYS = ("state", "melting_k", "boiling_k", "sublimation_k", "density_g_cm3", "electronegativity",
                    "econf", "valence", "discovered")
#: keys asked as plain facts; ``neutrons`` and ``electrons`` follow from rules and are ``calc``
RULE_KEYS = ("neutrons", "electrons")
FACT_KEYS = tuple(k for k in KEYS if k not in RULE_KEYS)
#: keys whose values group the elements, for ``elements <key> <value>`` lists and counts
LIST_KEYS = ("group", "period", "block", "category", "state", "metal", "radioactive")
#: keys two elements are compared on in ``higher <key> a b`` / ``lower <key> a b``
ORDER_KEYS = ("number", "melting_k", "boiling_k", "density_g_cm3", "electronegativity")
#: a list answer names at most this many elements (period 6 and 7 have 32)
MAX_LIST = 32
#: integers are compared exactly, floats within a relative tolerance
INT_KEYS = frozenset({"number", "protons", "electrons", "neutrons", "mass_number", "period"})
FLOAT_KEYS = frozenset({"mass", "melting_k", "boiling_k", "sublimation_k", "density_g_cm3", "electronegativity"})
TOLERANCE = 0.005
#: atomic weights are compared ten times tighter: cobalt's weight must not pass as nickel's
MASS_TOLERANCE = 0.0005
#: ``generate()`` drops a kind of question after this many draws in a row gave nothing new
GIVE_UP = 400

_SERIES = {"Nonmetals": "nonmetal", "Noble gases": "noble_gas", "Alkali metals": "alkali_metal",
           "Alkaline earth metals": "alkaline_earth_metal", "Metalloids": "metalloid", "Halogens": "halogen",
           "Poor metals": "poor_metal", "Transition metals": "transition_metal", "Lanthanides": "lanthanide",
           "Actinides": "actinide"}
_NONMETAL = {"nonmetal", "halogen", "noble_gas"}
_SECONDS = {"ysec": 1e-24, "zsec": 1e-21, "asec": 1e-18, "psec": 1e-12, "nsec": 1e-9, "usec": 1e-6,
            "msec": 1e-3, "sec": 1.0, "minute": 60.0, "hour": 3600.0, "day": 86400.0, "year": 31557600.0,
            "kyear": 31557600.0 * 1e3, "Myear": 31557600.0 * 1e6, "Gyear": 31557600.0 * 1e9,
            "Tyear": 31557600.0 * 1e12, "Pyear": 31557600.0 * 1e15, "Eyear": 31557600.0 * 1e18,
            "Zyear": 31557600.0 * 1e21, "Yyear": 31557600.0 * 1e24}
ROOM_K = 293.0
#: atomic numbers never made in weighable amounts (astatine, francium, fermium onwards): every
#: bulk property the source lists for them is a prediction. Which elements these are is from memory.
_NO_BULK = frozenset({85, 87}) | frozenset(range(100, 119))
#: name -> (key, value, note): values of the source corrected from memory, each noted on its row
_CORRECTIONS: dict[str, tuple[str, object, str]] = {
    "neodymium": ("discovered", 1885, "discovered 1885 from memory (von welsbach split didymium into praseodymium "
                                      "and neodymium in 1885); mendeleev says 1925"),
    "yttrium": ("discovered", 1794, "discovered 1794 from memory (gadolin's analysis of yttria); mendeleev says 1789"),
    "radon": ("discovered", 1900, "discovered 1900 from memory (dorn); mendeleev says 1898"),
    "nihonium": ("discovered", 2004, "discovered 2004 from memory (riken's first atom; reported from dubna decay "
                                     "chains in 2003); mendeleev's 2015 is the year iupac recognised it"),
    "darmstadtium": ("econf", "rn 5f14 6d8 7s2", "econf from memory (the calculated ground state); mendeleev copies "
                                                 "platinum's pattern, 6d9 7s1"),
    "roentgenium": ("econf", "rn 5f14 6d9 7s2", "econf from memory (the calculated ground state); mendeleev copies "
                                                "gold's pattern, 6d10 7s1"),
}


# ----------------------------------------------------------------------------- building the table

def _sig5(x: float) -> float:
    """``x`` to 5 significant digits, decimal ties rounded up (50.9415 -> 50.942, 4273.15 -> 4273.2).
    Float noise is stripped first (286.34999999999997 is 286.35)."""
    d = Decimal(f"{x:.10g}")
    return float(d.quantize(Decimal(1).scaleb(d.adjusted() - 4), rounding=ROUND_HALF_UP))


def _econf(text: str) -> str:
    """``[Ar] 3d6 4s2`` -> ``ar 3d6 4s2``; a bare orbital (``[He] 2s``) gets its count (``he 2s1``)."""
    out = []
    for tok in text.split():
        if tok.startswith("["):
            out.append(tok.strip("[]").lower())
        else:
            out.append(tok if tok[-1].isdigit() else tok + "1")
    return " ".join(out)


def _isotope_numbers(isotopes: list, bracket: int) -> tuple[int | None, bool, bool, str]:
    """(mass number, has a standard weight, radioactive, note) from an element's isotope rows;
    ``bracket`` is the source's atomic weight as a whole number."""
    measured = [i for i in isotopes if i.abundance is not None]
    stable = [i for i in measured if not i.is_radioactive]
    if measured:
        best = max(measured, key=lambda i: (i.abundance, -i.mass_number))
        return best.mass_number, True, not stable, ""
    why = "no standard atomic weight"

    def seconds(i) -> float:
        return i.half_life * _SECONDS[i.half_life_unit]

    def spread(i) -> float:
        return (i.half_life_uncertainty or 0.0) * _SECONDS[i.half_life_unit]

    timed = sorted((i for i in isotopes if i.half_life is not None and i.half_life_unit in _SECONDS),
                   key=lambda i: (-seconds(i), i.mass_number))
    if not timed:
        return None, False, True, "no isotope half-life in the table"
    if bracket not in {i.mass_number for i in isotopes}:
        return timed[0].mass_number, False, True, f"mass_number and mass are the longest-lived isotope: {why}"
    if timed[0].mass_number == bracket:
        return bracket, False, True, f"mass_number and mass are the longest-lived isotope: {why}"
    kept = f"mass_number and mass are the isotope of the source's bracket weight [{bracket}]: {why}"
    own = next((i for i in timed if i.mass_number == bracket), None)
    sure = next((i for i in timed if i.half_life_uncertainty is not None and seconds(i) >= 2 * spread(i)), None)
    listed = f"{timed[0].mass_number} is listed with a longer half-life"
    if own is None:
        return bracket, False, True, f"{kept}; it has no half-life in the isotope table ({listed})"
    if sure is not None and seconds(sure) - spread(sure) > seconds(own) + spread(own):
        note = (f"mass_number and mass are the longest-lived isotope with a half-life of at least twice its "
                f"uncertainty: {why}; the source's bracket weight is [{bracket}]")
        if sure is not timed[0]:
            note += f", and {listed} that is not known that well"
        return sure.mass_number, False, True, note
    return bracket, False, True, f"{kept}; {listed}, but not beyond the uncertainties"


def _phase_points(e) -> tuple[float | None, float | None, float | None, list[str]]:
    """(melting, boiling, sublimation point, notes) at 1 atm from the element and its allotrope rows."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # allotropes: the element-level value is None then
        melting, boiling = e.melting_point, e.boiling_point
    rows = [pt for pt in e.phase_transitions if not pt.is_transition]
    sublimes = [pt for pt in rows if pt.is_sublimation_point]
    rows = [pt for pt in rows if not pt.is_sublimation_point]
    if sublimes and not rows:  # carbon, arsenic: the listed melting point is a triple point under pressure
        points = sorted({pt.boiling_point for pt in sublimes if pt.boiling_point is not None})
        if len(points) == 1:
            return None, None, points[0], ["sublimes at 1 atm: the source's boiling point is the sublimation point, "
                                           "a melting point exists only under pressure"]
        return None, None, None, []
    notes = []
    for attr in ("melting_point", "boiling_point"):
        if (melting if attr == "melting_point" else boiling) is not None:
            continue
        values = sorted({getattr(pt, attr) for pt in rows if getattr(pt, attr) is not None})
        if len(values) != 1:
            continue
        which = " ".join(sorted({pt.allotrope for pt in rows if pt.allotrope}))
        key = "melting_k" if attr == "melting_point" else "boiling_k"
        notes.append(f"{key} from the {which} allotrope row (the only one that is neither a solid-solid transition "
                     f"nor a sublimation point)")
        if attr == "melting_point":
            melting = values[0]
        else:
            boiling = values[0]
    return melting, boiling, None, notes


def _state(melting: float | None, boiling: float | None, sublimation: float | None) -> tuple[str | None, str]:
    """(state at 293 K, how it was derived when not from the melting point)."""
    if melting is not None:
        if melting > ROOM_K:
            return "solid", ""
        return (None, "") if boiling is None else (("liquid" if boiling > ROOM_K else "gas"), "")
    if boiling is not None and boiling <= ROOM_K:
        return "gas", "state from the boiling point: no melting point at 1 atm"  # helium
    if sublimation is not None and sublimation > ROOM_K:
        return "solid", "state from the sublimation point"
    return None, ""


def build_row(e) -> dict:
    """One table row from a mendeleev ``Element`` (only called by ``build``)."""
    z = e.atomic_number
    name = e.name.lower().replace(" ", "_")
    mass_number, has_weight, radioactive, note = _isotope_numbers(e.isotopes, int(round(e.atomic_weight)))
    notes = [note] if note else []
    if mass_number is None:
        mass_number = e.mass_number
    if radioactive != bool(e.is_radioactive):
        notes.append("radioactive from the isotope table (no stable isotope); mendeleev's element flag disagrees")
    category = _SERIES[e.series]
    family = "lanthanide" if 57 <= z <= 71 else "actinide" if 89 <= z <= 103 else None
    if family and category != family:
        notes.append(f"category {family} by atomic number (iupac counts 15); mendeleev's series says {category}")
        category = family
    row: dict = {
        "name": name, "symbol": e.symbol.lower(), "number": z,
        "protons": z, "electrons": z, "neutrons": mass_number - z,
        "mass_number": mass_number,
        "mass": _sig5(e.atomic_weight) if has_weight else mass_number,
        "group": e.group_id if e.group_id is not None else ("lanthanide" if e.period == 6 else "actinide"),
        "period": e.period, "block": e.block, "category": category,
        "metal": "nonmetal" if category in _NONMETAL else "metalloid" if category == "metalloid" else "metal",
        "radioactive": "yes" if radioactive else "no",
    }
    if z in _NO_BULK:
        notes.append("state, melting_k, boiling_k and density_g_cm3 left out: never made in weighable amounts, "
                     "the source's values are predictions (which elements: from memory)")
    else:
        melting, boiling, sublimation, how = _phase_points(e)
        notes += how
        state, how_state = _state(melting, boiling, sublimation)
        if state:
            row["state"] = state
            if how_state:
                notes.append(how_state)
        for key, value in (("melting_k", melting), ("boiling_k", boiling), ("sublimation_k", sublimation)):
            if value is not None:
                row[key] = _sig5(value)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            density = e.density
        if density is not None:
            row["density_g_cm3"] = _sig5(density)
    if e.en_pauling is not None:
        row["electronegativity"] = _sig5(e.en_pauling)
    if e.econf:
        row["econf"] = _econf(e.econf)
        if z >= 104:
            notes.append("econf is a calculated prediction")
    valence = e.oxidation_states()
    if valence:
        row["valence"] = [int(v) for v in valence]
        row["valence_known"] = sorted({int(v) for v in e.oxidation_states("all")} | set(row["valence"]))
    if e.discovery_year is not None:
        row["discovered"] = int(e.discovery_year)
    elif e.discoverers and "ancient" in e.discoverers.lower():
        row["discovered"] = "ancient"
    if name in _CORRECTIONS:
        key, value, why = _CORRECTIONS[name]
        row[key] = value
        notes.append(why)
    if notes:
        row["notes"] = "; ".join(notes)
    return row


def build(out: Path = DEFAULT_TABLE) -> list[dict]:
    from mendeleev import get_all_elements  # only the build needs mendeleev

    rows = [build_row(e) for e in get_all_elements()]
    rows.sort(key=lambda r: r["number"])
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    return rows


def load_table(path: Path = DEFAULT_TABLE) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        rows = [json.loads(ln) for ln in f if ln.strip()]
    return sorted(rows, key=lambda r: r["number"])


# ----------------------------------------------------------------------------- dense values

def _signs(values: list[int]) -> str:
    return " ".join(("minus " if x < 0 else "plus ") + num(abs(x)) for x in values)


def dense_value(row: dict, key: str) -> str:
    """A row value as dense tokens: ``55.845`` -> ``5 5 point 8 4 5``, ``[2, 3]`` -> ``plus 2 plus 3``."""
    v = row[key]
    if key == "valence":
        return _signs(v)
    if isinstance(v, bool):
        raise TypeError(key)
    if isinstance(v, int):
        return num(v)
    if isinstance(v, float):
        return num(v, sig=5)
    return str(v)


def _number(text: str) -> float | int | None:
    """The number in ``text`` when it is written the way :func:`num` writes one: finite, no
    leading zero (``0 2 6``), no ``point`` without a digit after it. Otherwise ``None``."""
    words = text.split()
    value = parse_num(words)
    if not is_finite(value):
        return None
    body = words[1:] if words[0] == "minus" else words
    if len(body) > 1 and body[0] == "0" and body[1] != "point":
        return None
    if any(w == "point" and not (i + 1 < len(body) and body[i + 1].isdigit()) for i, w in enumerate(body)):
        return None
    return value


def _signed(text: str) -> frozenset[int] | None:
    """``plus 2 minus 1`` -> {2, -1}; ``None`` if the text is not a list of signed whole numbers,
    each written once and without leading zeros."""
    words = text.split()
    out, i = [], 0
    while i < len(words):
        if words[i] not in ("plus", "minus"):
            return None
        j = i + 1
        while j < len(words) and words[j] not in ("plus", "minus"):
            j += 1
        n = parse_num(words[i + 1:j])
        if not isinstance(n, int) or num(n).split() != words[i + 1:j]:
            return None
        out.append(-n if words[i] == "minus" else n)
        i = j
    return frozenset(out) if out and len(set(out)) == len(out) else None


def _digit_off(rng: random.Random, text: str, leading: bool = False) -> str:
    """One digit changed; a leading digit never becomes 0. With ``leading`` only the first two
    significant digits are touched, so a float moves by far more than the tolerance."""
    words = text.split()
    mantissa = words.index("e") if "e" in words else len(words)
    digits = [i for i, w in enumerate(words[:mantissa]) if w.isdigit()]
    if not digits:
        return text
    significant = [i for i in digits if words[i] != "0"] or digits
    first = significant[0]
    if leading:
        digits = [first] + [i for i in digits if i > first][:1]
    i = rng.choice(digits)
    choices = [d for d in "0123456789" if d != words[i] and not (i == first and d == "0")]
    words[i] = rng.choice(choices)
    return " ".join(words)


def _apart(key: str, x: float, y: float) -> bool:
    """True when two table values differ by more than the tolerance the gate judges ``key`` with."""
    if isinstance(x, int) and isinstance(y, int):
        return x != y
    return abs(x - y) > (MASS_TOLERANCE if key == "mass" else TOLERANCE) * max(abs(x), abs(y))


# ----------------------------------------------------------------------------- the gate

class ElementsGate:
    topic = TOPIC
    KEYS = KEYS

    def __init__(self, table: Path = DEFAULT_TABLE) -> None:
        self.rows = load_table(table)
        self.by_name = {r["name"]: r for r in self.rows}
        self.by_symbol = {r["symbol"]: r for r in self.rows}
        self.by_number = {r["number"]: r for r in self.rows}
        #: the elements with a standard atomic weight; only these are compared by mass
        self.weighed = [r for r in self.rows if isinstance(r["mass"], float)]
        self.groups: dict[tuple[str, str], list[str]] = {}
        for r in self.rows:
            for key in LIST_KEYS:
                if key in r:
                    self.groups.setdefault((key, dense_value(r, key)), []).append(r["name"])
        self.lists = [(k, v) for (k, v), names in self.groups.items() if len(names) <= MAX_LIST]
        #: (group, period) -> the one element there; the f-block has no group number
        self.by_position = {(r["group"], r["period"]): r for r in self.rows if isinstance(r["group"], int)}

    # ----- records

    def records(self) -> list[Line]:
        out = []
        for r in self.rows:
            for suffix, keys in (("element", RECORD_KEYS), ("element_more", RECORD_MORE_KEYS)):
                fields = " ".join(f"{k} {dense_value(r, k)}." for k in keys if k in r)
                out.append(Line(f"{r['name']} {suffix}. {fields}", topic=TOPIC, kind="record",
                                meta={"element": r["name"]}))
        return out

    # ----- questions

    def generate(self, rng: random.Random, n: int) -> list[Line]:
        """``n`` distinct question lines. A kind whose pool is used up is dropped and the others
        fill in; when every pool is used up before ``n`` the lines are returned with a warning."""
        makers = ((0.34, self._fact), (0.04, self._rule), (0.04, self._sum), (0.06, self._reverse),
                  (0.04, self._neighbour), (0.08, self._compare), (0.07, self._order), (0.03, self._list),
                  (0.03, self._count), (0.20, self._check_value), (0.07, self._check_heavier))
        out: dict[str, Line] = {}
        live = list(range(len(makers)))
        misses = [0] * len(makers)
        checks = {"yes": 0, "no": 0}  # value checks alternate, so yes and no stay level
        while len(out) < n and live:
            r = rng.random() * sum(makers[i][0] for i in live)
            pick = live[-1]
            for i in live:
                r -= makers[i][0]
                if r < 0:
                    pick = i
                    break
            make = makers[pick][1]
            if make == self._check_value:
                line = self._check_value(rng, yes=checks["yes"] <= checks["no"])
            else:
                line = make(rng)
            if line is None or line.text in out:
                misses[pick] += 1
                if misses[pick] >= GIVE_UP:
                    live.remove(pick)
                continue
            misses[pick] = 0
            out[line.text] = line
            if make == self._check_value:
                checks[line.answer] += 1
        if len(out) < n:
            warnings.warn(f"elements: only {len(out)} of the {n} lines asked for exist", RuntimeWarning, stacklevel=2)
        return list(out.values())

    def _fact(self, rng: random.Random) -> Line:
        row = rng.choice(self.rows)
        key = rng.choice([k for k in FACT_KEYS if k in row])
        return Line(f"{row['name']} {key}", dense_value(row, key), TOPIC, "fact", {"element": row["name"], "key": key})

    def _rule(self, rng: random.Random) -> Line:
        row = rng.choice(self.rows)
        key = rng.choice(RULE_KEYS)
        return Line(f"{row['name']} {key}", dense_value(row, key), TOPIC, "calc", {"element": row["name"], "key": key})

    def _sum(self, rng: random.Random) -> Line:
        """The neutron rule with the numbers in the prompt: any nucleus, not only the table's isotope."""
        protons = rng.randint(1, len(self.rows))
        mass_number = rng.randint(protons, min(3 * protons, 300))
        return Line(f"neutrons mass_number {num(mass_number)} protons {num(protons)}", num(mass_number - protons),
                    TOPIC, "calc", {"key": "neutrons"})

    def _reverse(self, rng: random.Random) -> Line | None:
        row = rng.choice(self.rows)
        key = rng.choice(("number", "symbol", "group"))
        if key == "group":
            if not isinstance(row["group"], int):
                return None
            prompt = f"element group {num(row['group'])} period {num(row['period'])}"
        else:
            prompt = f"element {key} {dense_value(row, key)}"
        return Line(prompt, row["name"], TOPIC, "calc", {"element": row["name"], "key": key})

    def _neighbour(self, rng: random.Random) -> Line | None:
        row = rng.choice(self.rows)
        where = rng.choice(("after", "before"))
        other = self.by_number.get(row["number"] + (1 if where == "after" else -1))
        if other is None:
            return None
        return Line(f"element {where} {row['name']}", other["name"], TOPIC, "calc", {"element": row["name"]})

    def _pair(self, rng: random.Random) -> tuple[dict, dict] | None:
        a, b = rng.sample(self.weighed, 2)
        return (a, b) if _apart("mass", a["mass"], b["mass"]) else None

    def _compare(self, rng: random.Random) -> Line | None:
        pair = self._pair(rng)
        if pair is None:
            return None
        a, b = pair
        which = rng.choice(("heavier", "lighter"))
        winner = max(a, b, key=lambda r: r["mass"]) if which == "heavier" else min(a, b, key=lambda r: r["mass"])
        return Line(f"{which} {a['name']} {b['name']}", winner["name"], TOPIC, "calc", {"a": a["name"], "b": b["name"]})

    def _order(self, rng: random.Random) -> Line | None:
        key = rng.choice(ORDER_KEYS)
        a, b = rng.sample(self.rows, 2)
        if key not in a or key not in b or not _apart(key, a[key], b[key]):
            return None
        which = rng.choice(("higher", "lower"))
        winner = max(a, b, key=lambda r: r[key]) if which == "higher" else min(a, b, key=lambda r: r[key])
        return Line(f"{which} {key} {a['name']} {b['name']}", winner["name"], TOPIC, "calc",
                    {"a": a["name"], "b": b["name"], "key": key})

    def _list(self, rng: random.Random) -> Line:
        key, value = rng.choice(self.lists)
        return Line(f"elements {key} {value}", " ".join(self.groups[(key, value)]), TOPIC, "calc", {"key": key})

    def _count(self, rng: random.Random) -> Line:
        key, value = rng.choice(list(self.groups))
        return Line(f"count elements {key} {value}", num(len(self.groups[(key, value)])), TOPIC, "calc", {"key": key})

    def _check_value(self, rng: random.Random, yes: bool | None = None) -> Line | None:
        row = rng.choice(self.rows)
        key = rng.choice([k for k in KEYS if k in row])
        if yes is None:
            yes = rng.random() < 0.5
        if yes:
            return Line(f"check {row['name']} {key} {dense_value(row, key)}", "yes", TOPIC, "yesno",
                        {"element": row["name"], "key": key})
        wrong = self._wrong(rng, row, key)
        if wrong is None:
            return None
        return Line(f"check {row['name']} {key} {wrong}", "no", TOPIC, "yesno", {"element": row["name"], "key": key})

    def _wrong(self, rng: random.Random, row: dict, key: str) -> str | None:
        """A plausible wrong value: a neighbouring element's, any other element's, or one digit off.
        Wrong oxidation states always contain a state the element is not known to have."""
        truth = dense_value(row, key)
        near = [self.by_number[n] for n in range(row["number"] - 3, row["number"] + 4)
                if n in self.by_number and n != row["number"]]
        known = set(row.get("valence_known", ())) | set(row.get("valence", ()))
        numeric = key in INT_KEYS or key in FLOAT_KEYS
        for _ in range(20):
            pick = rng.random() if numeric or key == "valence" else 0.5 + rng.random() / 2
            if pick < 0.5 and key == "valence":
                unknown = [s for s in range(-4, 9) if s != 0 and s not in known]
                if not unknown:
                    continue
                states = list(row[key])
                states[rng.randrange(len(states))] = rng.choice(unknown)
                cand = _signs(sorted(set(states)))
            elif pick < 0.5:
                cand = _digit_off(rng, truth, leading=isinstance(row[key], float))
            else:
                other = rng.choice(near if pick < 0.75 else self.rows)
                if key not in other:
                    continue
                if key == "valence" and set(other[key]) <= known:
                    continue
                cand = dense_value(other, key)
            if not self._agree(key, truth, cand):
                return cand
        return None

    def _check_heavier(self, rng: random.Random) -> Line | None:
        pair = self._pair(rng)
        if pair is None:
            return None
        a, b = pair
        rel = rng.choice(("heavier_than", "lighter_than"))
        yes = (a["mass"] > b["mass"]) == (rel == "heavier_than")
        return Line(f"check {a['name']} {rel} {b['name']}", "yes" if yes else "no", TOPIC, "yesno",
                    {"a": a["name"], "b": b["name"]})

    # ----- judging

    def owns(self, prompt: str) -> bool:
        w = prompt.split()
        if len(w) < 2:
            return False
        if w[0] in self.by_name:
            return len(w) == 2 and w[1] in KEYS
        if w[0] == "element":
            return len(w) >= 3 and w[1] in ("number", "symbol", "after", "before", "group")
        if w[0] == "elements":
            return len(w) >= 3 and w[1] in LIST_KEYS
        if w[0] == "count":
            return len(w) >= 4 and w[1] == "elements" and w[2] in LIST_KEYS
        if w[0] in ("heavier", "lighter"):
            return len(w) == 3 and w[1] in self.by_name and w[2] in self.by_name
        if w[0] in ("higher", "lower"):
            return len(w) == 4 and w[1] in ORDER_KEYS and w[2] in self.by_name and w[3] in self.by_name
        if w[0] == "neutrons":
            return len(w) >= 5 and w[1] == "mass_number" and "protons" in w[3:-1]
        if w[0] == "check":
            return len(w) >= 4 and w[1] in self.by_name and (w[2] in KEYS or w[2] in ("heavier_than", "lighter_than"))
        return False

    def _weights(self, a: str, b: str) -> tuple[dict, dict] | None:
        """Two elements that can be compared by mass: both have a standard atomic weight, and
        the weights differ by more than the tolerance."""
        x, y = self.by_name.get(a), self.by_name.get(b)
        if x is None or y is None or not (isinstance(x["mass"], float) and isinstance(y["mass"], float)):
            return None
        return (x, y) if _apart("mass", x["mass"], y["mass"]) else None

    def expected(self, prompt: str) -> tuple[str, str] | None:
        """(the right answer, how to compare it: a key name, ``name``, ``list``, ``count`` or
        ``yesno``), or ``None`` when the prompt is not this gate's or the table has no answer."""
        w = prompt.split()
        if not self.owns(" ".join(w)):
            return None
        if w[0] in self.by_name:
            row = self.by_name[w[0]]
            return (dense_value(row, w[1]), w[1]) if w[1] in row else None
        if w[0] == "element":
            if w[1] == "number":
                n = _number(" ".join(w[2:]))
                row = self.by_number.get(n) if isinstance(n, int) else None
            elif w[1] == "symbol":
                row = self.by_symbol.get(w[2]) if len(w) == 3 else None
            elif w[1] == "group":
                at = w.index("period") if "period" in w else len(w)
                group, period = _number(" ".join(w[2:at])), _number(" ".join(w[at + 1:]))
                both = isinstance(group, int) and isinstance(period, int)
                row = self.by_position.get((group, period)) if both else None
            else:
                base = self.by_name.get(w[2]) if len(w) == 3 else None
                row = base and self.by_number.get(base["number"] + (1 if w[1] == "after" else -1))
            return (row["name"], "name") if row else None
        if w[0] == "elements":
            names = self.groups.get((w[1], " ".join(w[2:])))
            return (" ".join(names), "list") if names and len(names) <= MAX_LIST else None
        if w[0] == "count":
            names = self.groups.get((w[2], " ".join(w[3:])))
            return (num(len(names)), "count") if names else None
        if w[0] in ("heavier", "lighter"):
            pair = self._weights(w[1], w[2])
            if pair is None:
                return None
            winner = max(pair, key=lambda r: r["mass"]) if w[0] == "heavier" else min(pair, key=lambda r: r["mass"])
            return winner["name"], "name"
        if w[0] in ("higher", "lower"):
            key, a, b = w[1], self.by_name[w[2]], self.by_name[w[3]]
            if key not in a or key not in b or not _apart(key, a[key], b[key]):
                return None
            winner = max(a, b, key=lambda r: r[key]) if w[0] == "higher" else min(a, b, key=lambda r: r[key])
            return winner["name"], "name"
        if w[0] == "neutrons":
            at = w.index("protons", 3)
            mass_number, protons = _number(" ".join(w[2:at])), _number(" ".join(w[at + 1:]))
            if not (isinstance(mass_number, int) and isinstance(protons, int)) or not 1 <= protons <= mass_number:
                return None
            return num(mass_number - protons), "count"
        row = self.by_name[w[1]]
        if w[2] in ("heavier_than", "lighter_than"):
            pair = self._weights(w[1], w[3]) if len(w) == 4 else None
            if pair is None:
                return None
            yes = (pair[0]["mass"] > pair[1]["mass"]) == (w[2] == "heavier_than")
            return ("yes" if yes else "no"), "yesno"
        if w[2] not in row:
            return None
        return ("yes" if self._agree(w[2], dense_value(row, w[2]), " ".join(w[3:])) else "no"), "yesno"

    def _agree(self, key: str, expected: str, answer: str) -> bool:
        answer = " ".join(answer.split())
        if key in FLOAT_KEYS:
            x, y = _number(expected), _number(answer)
            if x is None or y is None:
                return False
            if key == "mass" and isinstance(x, int):  # a mass number stands in for the weight: exact
                return answer == expected
            return abs(x - y) <= (MASS_TOLERANCE if key == "mass" else TOLERANCE) * max(abs(x), abs(y))
        if key == "valence":
            return _signed(answer) is not None and _signed(answer) == _signed(expected)
        return answer == expected  # whole numbers have one dense form, so words decide

    def check(self, prompt: str, answer: str) -> Verdict:
        prompt = " ".join(prompt.split())
        if not self.owns(prompt):
            return Verdict(False, None, "not my question")
        found = self.expected(prompt)
        if found is None:
            return Verdict(False, None, "no answer in the table")
        expected, key = found
        if self._agree(key, expected, answer):
            return Verdict(True, expected)
        return Verdict(False, expected, f"expected {expected}")


def gate(table: Path = DEFAULT_TABLE) -> ElementsGate:
    return ElementsGate(table)


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="write the table from the mendeleev package")
    b.add_argument("--out", type=Path, default=DEFAULT_TABLE)
    s = sub.add_parser("sample", help="print training lines")
    s.add_argument("--table", type=Path, default=DEFAULT_TABLE)
    s.add_argument("--n", type=int, default=20)
    s.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    if args.cmd == "build":
        rows = build(args.out)
        print(f"{len(rows)} elements -> {args.out}")
    else:
        g = gate(args.table)
        for ln in g.records()[:2] + g.generate(random.Random(args.seed), args.n):
            print(ln.text)


if __name__ == "__main__":
    main()
