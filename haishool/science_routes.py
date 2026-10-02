"""Conservative text adapters to the existing v5 gates; no answers live here.

``None`` means the old fact routes may handle the question. A Route with ``error``
means a science intent was recognized but cannot be represented safely: callers
must display that error instead of falling through to nearest-record retrieval.
Units are SI when omitted. Explicit units must match the requested operand; this
adapter does not silently convert grams, centimetres, or other units.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Any


@dataclass(frozen=True)
class Route:
    prompt: str
    topic: str
    label: str
    gate: Any = None
    error: str | None = None


EXAMPLES = [
    "12 + 7", "47 times 6 steps", "solve 3x + 4 = 19",
    "How many protons does carbon have?", "Atomic number of Fe",
    "Molar mass of water", "Mass percent of oxygen in H2O",
    "Balance H2 + O2 -> H2O", "Friction mu=0.3 n=50 N",
    "Weight mass=5 kg on moon",
    "life predict expected_mutations length 8 mu 0 point 0 2",
]
MAX_CHARS = 1000
MAX_PROMPT_TOKENS = 192
_NUM = r"[+-]?(?:\d+(?:\.\d+)?|\.\d+)(?:[eE][+-]?\d+)?"
_SCIENCE = re.compile(r"\b(?:calc|calculate|arithmetic|solve|balance|reaction|chemical|chemistry|physics|"
                      r"protons?|neutrons?|electrons?|atomic|molar|mass_percent|molar_mass|"
                      r"predict|nucleo|centripetal|buoyancy|acceleration|coulomb_force|spring_force)\b")


@lru_cache(maxsize=1)
def gates():
    from haishool.truth.loop import all_gates
    from haishool.cosmos import predict
    return tuple(all_gates()) + tuple(predict.gate(t) for t in predict.TOPICS)


def registry():
    """Cached gate registry for preview answer checking (no legacy replays)."""
    return gates()


def _gate(topic):
    return next(g for g in gates() if g.topic == topic)


def _error(message="This science question needs a supported form and all required inputs.", topic="science"):
    return Route("", topic, "Unsupported science question", None, message)


def _validated(prompt: str, topic: str | None = None) -> Route | None:
    if len(prompt.split()) > MAX_PROMPT_TOKENS or not re.fullmatch(r"[a-z0-9_ ]+", prompt):
        return None
    for gate in gates():
        if topic and gate.topic != topic:
            continue
        try:
            owned = gate.owns(prompt)
            valid = owned and gate.check(prompt, "").expected is not None
        except Exception:
            # Gate boundaries are untrusted parser input. Unsupported input must
            # produce a route error, never an exception in the preview handler.
            valid = False
        if valid:
            # Some table gates intentionally own malformed or unknown parameter
            # forms. Only expose questions the gate can actually judge. The gold
            # answer is neither stored in Route nor passed to the student.
            label = "Input-conditioned simulation" if gate.topic.startswith("predict_") else gate.topic.capitalize()
            if gate.topic == "forces":
                label = "Forces (SI inputs)"
            return Route(prompt, gate.topic, label, gate)
    return None


def _number(text: str) -> str:
    """Spell numeric literals exactly, without float rounding or merging digits."""
    match = re.fullmatch(r"([+-]?)(\d*(?:\.\d+)?)(?:[eE]([+-]?\d+))?", text)
    if not match or not match[2] or len(text) > 60:
        raise ValueError("invalid or oversized numeric literal")
    sign, body, exponent = match.groups()
    whole, dot, fraction = body.partition(".")
    whole = whole.lstrip("0") or "0"
    value = ("minus " if sign == "-" else "") + " ".join(whole)
    if dot:
        value += " point " + " ".join(fraction)
    if exponent is not None:
        value += " e " + _number(exponent)
    return value


def _math(text: str) -> Route | None:
    lower = text.lower()
    # Normalize supported verbal operators before deciding whether this is a
    # calculation. Otherwise "12 divided by 0" misses the numeric-intent test
    # and can incorrectly fall through to old fact retrieval.
    lower = lower.replace("multiplied by", "times").replace("divided by", "over").replace("to the power of", "power")
    explicit = bool(re.match(r"^(?:calc(?:ulate)?|solve|compare|check|arithmetic)\b", lower))
    numeric = bool(re.search(r"\d", lower) and re.search(r"[+*/^=()]|(?<=\d)\s*-|plus|minus|times|over|power|percent|\b(?:root|squared|cubed)\b", lower))
    if not explicit and not numeric:
        return None
    lower = re.sub(r"^calculate\s+", "calc ", lower)
    lower = re.sub(r"^arithmetic\s+", "calc ", lower)
    head = "calc"
    if m := re.match(r"^(calc|solve|compare|check)\b\s*", lower):
        head, lower = m[1], lower[m.end():]
    lower = re.sub(r"\s+(?:show(?:ing)? (?:the )?(?:work|steps)|with steps|step by step)$", " steps", lower)
    # Numeric signs are separate tokens here so 12-7 is subtraction, not two
    # adjacent signed numbers. A unary '-' becomes the gate's unary minus.
    lex = re.findall(r"\d+(?:\.\d+)?|\.\d+|[a-z_]+|[^\s]", lower)
    allowed = {"plus", "minus", "times", "over", "power", "open", "close", "root", "percent",
               "of", "x", "equals", "with", "point", "steps"}
    symbols = {"+": "plus", "-": "minus", "*": "times", "/": "over", "^": "power",
               "(": "open", ")": "close", "=": "equals", "%": "percent"}
    out = []
    for token in lex:
        if re.fullmatch(r"\d+(?:\.\d+)?|\.\d+", token):
            out.append(_number(token))
        elif token in symbols:
            out.append(symbols[token])
        elif token in allowed:
            out.append(token)
        else:
            return _error("Use arithmetic numbers/operators or a linear equation in x; this expression is unsupported.", "maths")
    return _validated(head + " " + " ".join(out), "maths") or _error(
        "The arithmetic gate cannot judge this expression. Check division by zero, equation form, or supported steps (integer addition/multiplication).", "maths")


def _element_name(text: str) -> str | None:
    aliases = {"aluminum": "aluminium", "sulfur": "sulfur", "cesium": "caesium"}
    value = aliases.get(text.lower().strip(), text.lower().strip())
    gate = _gate("elements")
    if value in gate.by_name:
        return value
    if value in gate.by_symbol:
        return gate.by_symbol[value]["name"]
    return None


def _clean_subject(text):
    text = re.sub(r"\b(?:how many|what is|what are|tell me|does|do|the|of|for|in|have|has|an|a|element)\b", " ", text,
                  flags=re.IGNORECASE)
    return " ".join(text.replace("'s", " ").split()).strip()


def _element(text: str) -> Route | None:
    properties = {"number of protons": "protons", "number of neutrons": "neutrons", "number of electrons": "electrons",
                  "atomic number": "number", "atomic mass": "mass", "atomic weight": "mass",
                  "mass number": "mass_number", "proton count": "protons", "neutron count": "neutrons",
                  "electron count": "electrons", "chemical symbol": "symbol",
                  "protons": "protons", "proton": "protons", "neutrons": "neutrons", "neutron": "neutrons",
                  "electrons": "electrons", "electron": "electrons", "symbol": "symbol", "mass": "mass"}
    for phrase, key in properties.items():
        match = re.search(r"\b" + phrase + r"\b", text, re.IGNORECASE)
        if match:
            subject = _clean_subject(text[:match.start()] + " " + text[match.end():])
            name = _element_name(subject)
            if name:
                return _validated(f"{name} {key}", "elements") or _error("This element property is unavailable.", "elements")
            return _error("Name one known element (for example carbon or Fe) and its requested property.", "elements")
    return None


def _subject(text: str):
    """A known substance name or validated formula, never a guessed composition."""
    from haishool.truth.formula import dense, parse
    gate = _gate("substances")
    name = text.lower().strip().replace(" ", "_")
    if name in gate.rows:
        return name
    candidates = [r for r in gate.pure if r["formula"].lower() == text.lower().strip()
                  and r["class"] != "ice" and not r["unit"]]
    if candidates:
        return candidates[0]["name"]
    try:
        if parse(text):
            return dense(text)
    except (ValueError, KeyError):
        pass
    return None


def _substance(text: str) -> Route | None:
    lower = text.lower()
    if re.search(r"\bmass[_ ]percent(?:age)?\b", lower):
        if m := re.fullmatch(r"mass[_ ]percent(?:age)?(?: of)? (.+?) (?:in|of) (.+)", text, re.IGNORECASE):
            element, subject = _element_name(m[1]), _subject(m[2])
            if element and subject:
                return _validated(f"mass_percent {subject} {element}", "substances") or _error("Cannot calculate that mass percentage.", "substances")
        return _error("Use 'mass percent of oxygen in water' with an element and a substance.", "substances")
    if m := re.search(r"\b(molar[_ ]mass|(?:chemical )?formula)\b", text, re.IGNORECASE):
        key = "molar_mass" if m[1].lower().startswith("molar") else "formula"
        subject_text = _clean_subject(text[:m.start()] + " " + text[m.end():])
        subject = _subject(subject_text)
        if subject:
            prompt = f"molar_mass {subject}" if key == "molar_mass" else f"{subject} formula"
            return _validated(prompt, "substances") or _error("No substance formula is available for those inputs.", "substances")
        return _error("Name a known substance or a chemical formula, for example water or H2O.", "substances")
    return None


def _reaction(text: str) -> Route | None:
    if not re.match(r"^(?:balance|balancing)\b", text, re.IGNORECASE):
        return None
    from haishool.truth.formula import dense, parse
    equation = re.sub(r"^(?:balance|balancing)\s+(?:the\s+)?(?:(?:reaction|equation)\s*:?\s*)?", "", text, flags=re.IGNORECASE)
    sides = re.split(r"\s*(?:->|→|=|\bgives\b|\byields\b)\s*", equation, flags=re.IGNORECASE)
    if len(sides) != 2:
        return _error("Give both sides: balance H2 + O2 -> H2O.", "reactions")
    converted = []
    for side in sides:
        species = re.split(r"\s*(?:\+|\bplus\b)\s*", side, flags=re.IGNORECASE)
        terms = []
        for item in species:
            subject = _subject(item)
            if not subject:
                return _error("An equation species is unknown; supply valid neutral chemical formulas without coefficients.", "reactions")
            row = _gate("substances").rows.get(subject)
            formula = row["formula"] if row and not row["mixture"] else item
            try:
                if not parse(formula):
                    raise ValueError("empty formula")
                terms.append(dense(formula))
            except (ValueError, KeyError):
                return _error("This species is not a supported chemical formula.", "reactions")
        converted.append(" plus ".join(terms))
    return _validated("balance " + " gives ".join(converted), "reactions") or _error(
        "The reaction gate cannot find a unique positive balance for this equation.", "reactions")


_FORCE_NAMES = {"gravitational force": "gravity_force", "gravity force": "gravity_force",
                "coulomb force": "coulomb_force", "spring force": "spring_force", "magnetic force": "magnetic_force",
                "net force": "net_force", "orbital speed": "orbital_speed", "escape speed": "escape_speed",
                "surface gravity": "surface_gravity"}
_PARAMS = {"m": ("mass", "mass1", "mass2", "m1", "m2"), "r": ("radius", "distance"),
           "charge": ("charge1", "charge2", "q1", "q2"), "k": ("spring_constant",), "x": ("stretch",),
           "mu": ("coefficient",), "n": ("normal_force",), "f": ("force", "force1", "force2", "f1", "f2"),
           "area": (), "rho": ("density",), "vol": ("volume",), "v": ("speed", "velocity"),
           "c": ("coefficient",), "b": ("field",), "mdot": ("mass_flow",), "ve": ("exhaust_speed",), "d": ()}
_UNITS = {"m": {"kg"}, "r": {"m"}, "d": {"m"}, "charge": {"c"}, "k": {"n/m"}, "x": {"m"},
          "mu": set(), "n": {"n", "newton", "newtons"}, "f": {"n", "newton", "newtons"},
          "area": {"m2", "m^2", "m²"}, "rho": {"kg/m3", "kg/m^3", "kg/m³"},
          "vol": {"m3", "m^3", "m³"}, "v": {"m/s"}, "c": set(), "b": {"t", "tesla"},
          "mdot": {"kg/s"}, "ve": {"m/s"}}


def _forces(text: str) -> Route | None:
    from haishool.truth.forces import CALC
    lower = text.lower()
    names = {**{key: key for key in CALC}, **_FORCE_NAMES}
    name = next((name for name in sorted(names, key=len, reverse=True)
                 if re.match(re.escape(name) + r"(?:\b|\s)", lower)), None)
    if name is None:
        return None
    verb, body = names[name], lower[len(name):].strip()
    # Bare old concepts ("what is gravity", "what is friction") retain the
    # old fact route; once operands/calculation intent are present, fail closed.
    if not body:
        return _error("Provide the named inputs in SI units: " + verb + " " + " ".join(CALC[verb]) + ".", "forces")
    suffix = ""
    if body.endswith(" steps"):
        body, suffix = body[:-6].strip(), " steps"
    tail = ""
    if verb == "weight" and (m := re.search(r"\bon ([a-z]+)$", body)):
        tail, body = " on " + m[1], body[:m.start()].strip()
    if verb == "net_force" and (m := re.search(r"\b(same|opposite)$", body)):
        tail, body = " " + m[1], body[:m.start()].strip()
    aliases = {alias: key for key in CALC[verb] for alias in (key, *_PARAMS[key])}
    pattern = re.compile(r"(?<![a-z0-9_])(" + "|".join(sorted(map(re.escape, aliases), key=len, reverse=True))
                         + r")\s*(?:=|:)?\s*(" + _NUM + r")", re.IGNORECASE)
    matches = list(pattern.finditer(body))
    clean = lambda s: re.sub(r"\b(?:with|and)\b", "", s).strip(" ,;")
    if not matches or clean(body[:matches[0].start()]):
        return _error("Supply named parameters, for example friction mu=0.3 n=50 N. Values without units use SI.", "forces")
    values = {}
    for index, match in enumerate(matches):
        key = aliases[match[1]]
        unit = clean(body[match.end():matches[index + 1].start() if index + 1 < len(matches) else len(body)])
        if unit and unit not in _UNITS[key]:
            return _error(f"Unsupported unit '{unit}' for {key}; use the SI unit for this parameter. No unit conversion was applied.", "forces")
        values.setdefault(key, []).append(_number(match[2]))
    output = [verb]
    for key in CALC[verb]:
        if not values.get(key):
            return _error("Missing required parameters: " + " ".join(CALC[verb]) + ".", "forces")
        output.extend((key, values[key].pop(0)))
    if any(values.values()):
        return _error("A parameter was supplied more times than this formula accepts.", "forces")
    return _validated(" ".join(output) + tail + suffix, "forces") or _error(
        "The forces gate cannot judge these parameter values or this planet/direction.", "forces")


def route(question: str) -> Route | None:
    if not isinstance(question, str):
        return _error("Enter a text question.")
    text = " ".join(question.strip().split())
    if not text:
        return None
    if len(text) > MAX_CHARS:
        return _error("The question is too long; ask one calculation or science question at a time.")
    if re.search(r"\.\s*a\b", text, re.IGNORECASE):
        match = re.fullmatch(r"(.+?)\.\s*a\s*", text, re.IGNORECASE)
        if not match:
            return _error("Supply only the question, without a supplied '. a' answer.")
        text = match[1]
    text = re.sub(r"^q\s+", "", text, flags=re.IGNORECASE).rstrip("?!.").strip()
    lower = text.lower()
    if re.search(r"\b(?:world|nucleo|gravity|planets|chem|life)\s+seed\b", lower):
        return _error("Seed-only rollout queries are unavailable here; use an explicit-input predict question.")
    if direct := _validated(lower):
        return direct
    # Remove common question scaffolding, retaining case for chemical formulas.
    natural = re.sub(r"^(?:please\s+)?(?:what is|what are|calculate|compute|tell me)\s+(?:the\s+)?", "", text,
                     flags=re.IGNORECASE)
    if direct := _validated(natural.lower()):
        return direct
    if natural.lower() in {"physics", "chemistry"} and re.match(r"^(?:what is|tell me)\b", lower):
        return None
    if natural.lower() in {"gravity", "friction", "weight", "pressure", "drag", "lift", "tension", "normal force"}:
        return _error("A calculation requires named parameters in SI units.", "forces") if re.match(r"^(?:calculate|compute)\b", lower) else None
    try:
        # Force unit syntax and reaction '+' signs must be handled before maths.
        for parser in (_reaction, _forces, _substance, _element, _math):
            if result := parser(natural):
                return result
    except (ValueError, ArithmeticError, KeyError, IndexError):
        return _error("These inputs are malformed or outside the supported science grammar.")
    if _SCIENCE.search(lower) or re.search(r"\b(?:mass[_ ]percent|molar[_ ]mass|gravity_force|net_force)\b", lower):
        return _error()
    if re.search(r"\bforce\b", lower) and (re.search(r"\d", lower) or re.search(r"\b(?:between|calculate|how much)\b", lower)):
        return _error("Choose a supported force formula and provide its named SI parameters.", "forces")
    return None
