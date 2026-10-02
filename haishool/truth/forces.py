"""Round 5, topic ``forces``: the known forces and what they arise from.

Worked curriculum: ``gate().generate_worked(rng, n)`` appends ``steps`` to calculation
prompts, retaining the original answer-only questions. Answers state the formula,
substitute the prompt's operands and constants, then give the gate-computed result.
Every step is checked; changing the work while preserving the result fails. The
separate :data:`WORKED_MAX_TOKENS` budget allows longer astronomical calculations.

Four fundamental interactions (gravity, electromagnetism, strong, weak) and the effective
forces that arise from them: electrostatic, magnetic, friction (static and kinetic), normal,
tension, drag, lift, buoyancy, spring, the pressure force, surface tension, capillary, tidal,
radiation pressure, nuclear binding (the residual strong force), van der Waals and the four
bonds (hydrogen, ionic, covalent, metallic), thrust and weight; centripetal, which is a *role*
any force can play, not a kind of its own; and the two inertial forces of a rotating frame
(centrifugal, coriolis). Every row of :data:`TABLE` names what the force acts on, how far it
reaches, what it holds together, who stated its law, its formula as words, its constant and
everyday examples. Values that come from memory rather than from a calculation (ranges, bond
lengths, years, names) say so in the row's ``notes``, which are written to the jsonl but never
trained. The table is written to ``data/truth-v5/forces.jsonl`` by::

    python -m haishool.truth.forces build
    python -m haishool.truth.forces sample --n 30 --seed 1

Record lines (kind ``record``), one per row, keys in :data:`ROW_KEYS` order, a key left out
where the table has nothing sure to say (no carrier for an effective force, no formula for the
weak force, no range for the centripetal role):

    strong force. kind fundamental. carrier gluon. acts_on colour_charge. range_m 1 e minus 1 5.
        relative_strength 1. holds_together protons neutrons. described_by gell_mann.
        discovered 1 9 7 3. unit newton. examples proton neutron atomic_nucleus.
        attractive attractive. direction between_quarks. acts_at_distance yes.

A row whose line would pass 80 tokens is split in two, ``<name> force.`` with the identity
keys (kind ... discovered) and ``<name> force_more.`` with the physics keys (formula ...
acts_at_distance). No row of the current table needs it (the longest, gravity, is 73 tokens),
but :func:`record_lines` applies the rule to any row.

Question lines from :meth:`ForcesGate.generate`, mixed as below for ``n`` up to about 1000.
The table-bound kinds are finite (365 facts, 562 comparisons, 230 lists) and are asked without
repetition, so a larger run holds the whole table once and calc and yesno fill the rest:

    fact 35 %     q gravity carrier. a graviton_hypothetical.
                  q strong range_m. a 1 e minus 1 5.
                  q gravity formula. a big_g m_1 m_2 over r squared.
    compare 8 %   q stronger gravity strong. a strong.
                  q longer_range strong weak. a strong.
                  q longer_range friction strong. a friction.
                  q longer_range gravity electromagnetism. a same.
    list 4 %      q forces kind fundamental. a gravity electromagnetism strong weak.
                  q forces described_by newton. a gravity normal tension drag tidal centripetal thrust weight.
    calc 38 %     q gravity_force m 1 0 m 2 0 r 2. a 3 point 3 3 7 e minus 9.
                  q coulomb_force charge 1 e minus 6 charge 2 e minus 6 r 0 point 1. a 1 point 7 9 8.
                  q weight m 5. a 4 9 point 0 5.                 q weight m 5 on moon. a 8 point 1.
                  q spring_force k 2 0 0 x 0 point 0 5. a 1 0.   q friction mu 0 point 3 n 5 0. a 1 5.
                  q pressure f 1 0 0 area 2. a 5 0.              q acceleration f 1 0 m 2. a 5.
                  q net_force f 1 0 f 2 0 opposite. a 1 0.       q net_force f 1 0 f 2 0 same. a 3 0.
                  q buoyancy rho 1 0 0 0 vol 0 point 0 0 2. a 1 9 point 6 2.
                  q centripetal m 2 v 3 r 1 point 5. a 1 2.
                  q orbital_speed m 5 point 9 7 e 2 4 r 6 point 7 7 e 6. a 7 6 7 2.
                  q escape_speed m 5 point 9 7 e 2 4 r 6 point 3 7 e 6. a 1 1 1 8 0.
                  q surface_gravity m 5 point 9 7 e 2 4 r 6 point 3 7 e 6. a 9 point 8 1 9.
                  q drag rho 1 point 2 v 1 0 c 1 area 2. a 1 2 0.
                  q drag rho 1 0 0 0 v 2 3 c 0 point 2 5 area 1. a 6 6 1 2 5.
                  q lift rho 1 point 2 v 5 0 c 0 point 5 area 2 0. a 1 5 0 0 0.
                  q magnetic_force charge 2 v 1 0 b 0 point 5. a 1 0.
                  q thrust mdot 2 5 0 ve 3 0 0 0. a 7 5 0 0 0 0.
                  q tidal m 7 point 3 5 e 2 2 m 1 r 6 point 3 7 e 6 d 3 point 8 4 e 8. a 1 point 1 0 4 e minus 6.
    yesno 15 %    q check gravity carrier photon. a no.
                  q check weight m 5 equals 4 9 point 0 5. a yes.
                  q check stronger gravity strong equals gravity. a no.

Calculations are SI: G 6.674e-11, Coulomb k 8.988e9, g 9.81 on earth, 1.62 on the moon, 3.73
on mars, 24.79 on jupiter. A whole-number result below a million is written in full (``6 6 1
2 5``), every other result with 4 significant digits. :data:`VERBS` names each verb's
operands and :data:`OPERANDS` their units: ``pressure`` gives f / area in pascal,
``acceleration`` f / m, ``net_force`` the sum of two collinear forces (``same``) or their
difference (``opposite``), ``orbital_speed`` sqrt(G m / r), ``escape_speed`` sqrt(2 G m / r),
``surface_gravity`` G m / r^2, ``drag`` and ``lift`` half rho v^2 c area, ``magnetic_force``
charge v b, ``thrust`` mdot ve, ``tidal`` 2 G m m r / d^3. A negative ``coulomb_force`` means
the charges attract; only a charge may be negative. Orbit prompts are generated below a tenth
of the speed of light, and a speed at or above light is not judged at all, because Newton's
formula does not hold there.

``longer_range`` compares orders of magnitude only: an infinite range beats every finite one,
``contact`` counts as the spacing of atoms (1e-10 m, far more than the 1e-15 m of the nuclear
forces), and two ranges closer than a factor of 100 (one bond length against another, contact
against a bond, the strong force against nuclear binding) are not compared at all. Equal words
(``infinite``, ``contact``) answer ``same``. ``stronger`` compares the four fundamental
interactions, the only rows with a relative strength.

Yes and no answers are kept in balance within every kind of check (fact, calc, compare,
list). A wrong value in a fact check is never part of the truth: for a list it shares no
member with the true list; for who stated the law, what the force binds and its formula it
comes from a force of another family (an effective force is never tested against the values
of its own interaction, and the centripetal role and the inertial forces belong to every
family); a wrong range differs by a factor of 100 or more, and ``infinite`` is not offered
against a bond length; a wrong year is none the row's notes name; ``both`` is only called
wrong by ``neither``. ``acts_on``, ``examples`` and ``direction`` are not checked at all:
gravity acts on everything, an example of one force is often an example of its parent, and
a direction in words is too loose to call another one wrong.

``check`` judges a number by its printed digits: the answer must round to the gold answer
at every digit the gold shows, and at 4 significant digits at least for a calculation (``4 9
point 0 5 1`` passes for ``4 9 point 0 5``, ``4 9 point 0 6`` and ``4 9`` do not; ``2 point
8 2 e minus 1 0`` needs its three digits, the round ``1 e minus 1 5`` only one). Numbers
with a leading zero, ``minus 0`` and anything infinite are refused. Lists are compared as
sets without repeats, formulas and words exactly, years exactly. ``check <force> <key>
<value>`` with a value that is only part of a list (``check gravity described_by newton``)
is not judged, since it is neither the whole truth nor wrong. An owned prompt the table
cannot answer (``weak formula``, ``weight m 5 on pluto``) gets ``Verdict(False, None, "cannot
judge: ...")``, never an exception.

``owns``: ``<force> <key>`` with a key of :data:`ROW_KEYS`, a calc verb followed by one of
its operands, ``forces <key> ...``, ``stronger`` / ``longer_range`` followed by force names,
or ``check`` before one of those. A prompt that merely starts with a force name (``gravity
seed 7 step 2 0 clumps`` of round 6, ``gravity field`` and ``friction is_a`` of rounds 3 and
4) is not a forces question. :data:`SHARED` lists the ``<name> kind`` questions round 3
already answers in its own words (``gravity kind fundamental_force``); they are neither asked
nor owned here.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
from pathlib import Path

from haishool.truth import DIGITS, NUMBER_WORDS, Line, Verdict, is_dense, is_finite, num, parse_num
from haishool.truth.worked import check_steps

TOPIC = "forces"
MAX_TOKENS = 80
# Worked formula/substitution lessons have a separate explicit context requirement.
# Use a training context of at least 192 for the full gravitational/tidal traces.
WORKED_MAX_TOKENS = 160
#: a calc answer is judged at this many significant digits at least
CALC_DIGITS = 4
#: a list answer names at most this many forces
MAX_LIST = 16

#: codata values rounded to 4 digits (checked in the tests)
G = 6.674e-11
K_COULOMB = 8.988e9
LIGHT_SPEED = 2.9979e8
#: generated orbit prompts stay below this speed in m/s (a tenth of light), where newton holds
NEWTONIAN_SPEED = 3e7
#: surface gravity in m/s^2: earth is standard gravity 9.80665 to 3 digits; the others are nasa
#: fact sheet values (from memory) and agree with G m / r^2 of BODIES within 0.1 % (tested)
SURFACE_G = {"earth": 9.81, "moon": 1.62, "mars": 3.73, "jupiter": 24.79}
#: mass in kg and radius in m (nasa fact sheets, 3 digits, from memory; jupiter's radius is the
#: equatorial one its surface gravity is quoted for). Only ever written into a prompt as
#: operands; the gate computes from the prompt, so a slip here cannot make an answer wrong
BODIES = {"earth": (5.97e24, 6.37e6), "moon": (7.35e22, 1.74e6), "mars": (6.42e23, 3.39e6),
          "jupiter": (1.9e27, 7.15e7), "sun": (1.99e30, 6.96e8), "venus": (4.87e24, 6.05e6),
          "mercury": (3.3e23, 2.44e6), "saturn": (5.68e26, 5.82e7), "uranus": (8.68e25, 2.54e7),
          "neptune": (1.02e26, 2.46e7)}
#: heights above the surface in m for orbit prompts (0 twice: the surface is the common case)
ALTITUDES = (0, 0, 1e5, 2e5, 3e5, 4e5, 5e5, 1e6, 2e6, 5e6, 1e7, 3.58e7)
#: operands for tidal prompts: mass of the pulling body in kg, radius of the pulled body in m,
#: distance of their centres in m (moon on earth, sun on earth, earth on moon, jupiter on io;
#: from memory, operands only)
TIDES = ((7.35e22, 6.37e6, 3.84e8), (1.99e30, 6.37e6, 1.5e11), (5.97e24, 1.74e6, 3.84e8), (1.9e27, 1.82e6, 4.22e8))
#: what ``contact`` counts as when ranges are compared: touching bodies act across the spacing
#: of their atoms, about 1e-10 m
CONTACT_RANGE_M = 1e-10
#: two ranges are only compared when they differ by this factor or more
RANGE_FACTOR = 100

#: the keys of a row -> what they mean, in record order
ROW_KEYS: dict[str, str] = {
    "kind": "fundamental (one of the four interactions), effective (arises from them), role (centripetal: what "
            "any force does on a circle) or inertial (felt only in a rotating frame)",
    "arises_from": "for an effective force the fundamental interaction(s) behind it; any for a role",
    "carrier": "exchange particle of a fundamental interaction (left out for every other row)",
    "acts_on": "what the force acts on",
    "range_m": "how far it reaches in metres, or infinite, or contact (touching bodies only, about 1 e minus 1 0 m "
               "between their atoms); for a bond its length in the example the notes name",
    "relative_strength": "conventional strength of a fundamental interaction at about 1 e minus 1 5 m with strong 1 "
                         "(textbook convention, no unit)",
    "holds_together": "what it binds (left out when it binds nothing)",
    "described_by": "who stated its law",
    "discovered": "year its law was stated or first measured, or an era (ancient)",
    "formula": "the force in newton as words: factors side by side, over for division, plus, half, squared, cubed, "
               "pow 7; big_g is the gravitational constant and g the surface gravity, k_e coulomb's constant and k "
               "a spring constant, c the speed of light, c_d c_l c_6 the drag, lift and dispersion coefficients, "
               "m_1 m_2 and charge_1 charge_2 the two bodies",
    "constant": "the constant of its law as a number: big_g in m3 kg-1 s-2 (gravity, tidal), k_e in n m2 c-2 "
                "(electrostatic, ionic_bond), mu_0 in n a-2 (magnetic), g in m s-2 (weight, buoyancy), c in m s-1 "
                "(radiation_pressure), the fine structure constant, which has no unit (electromagnetism), the fermi "
                "constant in gev-2, which is not si (weak)",
    "unit": "si unit of the force",
    "examples": "everyday examples",
    "attractive": "attractive (pulls toward its source), repulsive (pushes away), both, or neither",
    "direction": "where it points",
    "acts_at_distance": "yes for a field force or a bond, no for a contact force",
}
#: calc operand -> what it is, with its si unit
OPERANDS: dict[str, str] = {
    "m": "mass in kg", "r": "distance between centres, or radius, in m", "d": "distance between centres in m",
    "charge": "electric charge in coulomb (may be negative)", "k": "spring constant in n per m",
    "x": "stretch in m", "mu": "friction coefficient, no unit", "n": "normal force in newton",
    "f": "force in newton", "area": "area in m2", "rho": "density of the fluid in kg per m3",
    "vol": "displaced volume in m3", "v": "speed in m per s", "c": "drag or lift coefficient, no unit",
    "b": "magnetic field in tesla", "mdot": "mass flow in kg per s", "ve": "exhaust speed in m per s",
}
#: calc verb -> operand names it needs, in the order they are written
CALC: dict[str, tuple[str, ...]] = {
    "gravity_force": ("m", "m", "r"), "coulomb_force": ("charge", "charge", "r"), "weight": ("m",),
    "spring_force": ("k", "x"), "friction": ("mu", "n"), "pressure": ("f", "area"), "acceleration": ("f", "m"),
    "net_force": ("f", "f"), "buoyancy": ("rho", "vol"), "centripetal": ("m", "v", "r"),
    "orbital_speed": ("m", "r"), "escape_speed": ("m", "r"), "drag": ("rho", "v", "c", "area"),
    "magnetic_force": ("charge", "v", "b"), "surface_gravity": ("m", "r"), "lift": ("rho", "v", "c", "area"),
    "thrust": ("mdot", "ve"), "tidal": ("m", "m", "r", "d"),
}
# These are calculation formulas, including quantities that are not themselves forces.
# Substitution retains the operand's original digits; g/big_g/k_e are expanded constants.
WORKED_FORMULAS = {
    "gravity_force": "big_g times m_1 times m_2 over r squared",
    "coulomb_force": "k_e times charge_1 times charge_2 over r squared",
    "weight": "m times g", "spring_force": "k times x", "friction": "mu times n",
    "pressure": "f over area", "acceleration": "f over m",
    "net_force": "f_1 plus f_2",
    "buoyancy": "rho times g times vol", "centripetal": "m times v squared over r",
    "orbital_speed": "sqrt open big_g times m over r close",
    "escape_speed": "sqrt open 2 times big_g times m over r close",
    "surface_gravity": "big_g times m over r squared",
    "drag": "half times rho times v squared times c times area",
    "lift": "half times rho times v squared times c times area",
    "magnetic_force": "charge times v times b", "thrust": "mdot times ve",
    "tidal": "2 times big_g times m_1 times m_2 times r over d cubed",
}
#: the question forms -> what they ask and the unit of the answer
VERBS: dict[str, str] = {
    "steps": "append to a calculation: formula, exact operand substitution, then rounded result",
    "forces": "forces <key> <value>: every force whose <key> names <value>, in table order; none when there is none",
    "stronger": "stronger <a> <b>: the fundamental interaction with the larger relative_strength",
    "longer_range": "longer_range <a> <b>: the force that reaches farther by a factor of 100 or more; same for "
                    "equal words",
    "check": "check <force> <key> <value> or check <question> equals <value>: yes or no",
    "gravity_force": "m m r: big G m m / r^2 in newton",
    "coulomb_force": "charge charge r: k_e charge charge / r^2 in newton, negative when the charges attract",
    "weight": "m, then on earth, moon, mars or jupiter (earth when left out): m g in newton",
    "spring_force": "k x: k x in newton",
    "friction": "mu n: mu n in newton",
    "pressure": "f area: f / area in pascal",
    "acceleration": "f m: f / m in m per s2",
    "net_force": "f f, then same or opposite: sum or difference of two collinear forces in newton",
    "buoyancy": "rho vol: rho g vol in newton with g 9.81",
    "centripetal": "m v r: m v^2 / r in newton",
    "orbital_speed": "m r: sqrt(big G m / r) in m per s",
    "escape_speed": "m r: sqrt(2 big G m / r) in m per s",
    "drag": "rho v c area: half rho v^2 c area in newton",
    "magnetic_force": "charge v b: charge v b in newton",
    "surface_gravity": "m r: big G m / r^2 in m per s2",
    "lift": "rho v c area: half rho v^2 c area in newton",
    "thrust": "mdot ve: mdot ve in newton",
    "tidal": "m m r d: 2 big G m m r / d^3 in newton, the stretch on the second mass at r from the centre of a "
             "body whose centre is d from the first mass",
}
KEYS: dict[str, str] = {**ROW_KEYS, **VERBS}

#: keys of the first record line when a row is split
FIRST_KEYS = ("kind", "arises_from", "carrier", "acts_on", "range_m", "relative_strength", "holds_together",
              "described_by", "discovered")
#: keys whose values are lists, compared as sets
SET_KEYS = ("arises_from", "carrier", "acts_on", "holds_together", "described_by", "examples")
#: keys whose values may be numbers, judged by their printed digits
NUMERIC_KEYS = ("range_m", "relative_strength", "constant")
#: keys a ``forces <key> <value>`` list may ask about
LIST_KEYS = ("kind", "arises_from", "carrier", "acts_on", "range_m", "holds_together", "described_by", "unit",
             "examples", "attractive", "direction", "acts_at_distance")
#: keys no yes/no check is made of: a value of another row is too often true here as well
UNCHECKED_KEYS = ("acts_on", "examples", "direction")
#: keys whose wrong values in a yes/no check come from a force of another family only
FAMILY_KEYS = ("holds_together", "described_by", "formula")
#: wrong values for a yes/no check when no other row offers one
FALLBACK_WRONG = {"unit": ["joule", "pascal", "watt"]}
#: wrong values of ``attractive``: both contains attractive and repulsive, so neither is offered for it
WRONG_PULL = {"attractive": ["repulsive", "neither", "both"], "repulsive": ["attractive", "neither", "both"],
              "both": ["neither"], "neither": ["attractive", "repulsive", "both"]}
#: the only operands that may be negative
SIGNED = ("charge",)
#: ``<name> <key>`` questions that round 3 already answers in its own words (``gravity kind
#: fundamental_force``, ``friction kind force``, ``weight kind concept``): neither asked nor
#: owned here. The tests check this against the round 1-3 records.
SHARED: dict[str, tuple[str, ...]] = {name: ("kind",) for name in (
    "gravity", "friction", "weight", "ionic_bond", "covalent_bond", "metallic_bond")}

_STRENGTH_NOTE = ("relative_strength is the textbook convention at about 1e-15 m: strong 1, electromagnetism 1e-2, "
                  "weak 1e-6, gravity 1e-39 (some books give weak 1e-13 and gravity 1e-38)")

#: keys whose values are remembered textbook knowledge, not computed; every row's notes name
#: the ones it carries so a checker can target them (constants are checked against codata in the tests)
MEMORY_KEYS = ("carrier", "acts_on", "range_m", "relative_strength", "holds_together", "described_by", "discovered",
               "formula", "examples", "attractive", "direction")


def _row(name: str, **fields) -> dict:
    fields["name"] = name
    memory = [k for k in MEMORY_KEYS if k in fields and fields[k] not in ("infinite", "contact")]
    fields["notes"] = fields.get("notes", "") + "; from memory, not computed: " + " ".join(memory)
    return fields


#: the table; list values are lists, numbers are numbers, everything else a dense word or phrase
TABLE: list[dict] = [
    _row("gravity", kind="fundamental", carrier="graviton_hypothetical", acts_on=["mass_energy"], range_m="infinite",
         relative_strength=1e-39, holds_together=["planets", "stars", "galaxies"], described_by=["newton", "einstein"],
         discovered=1687, formula="big_g m_1 m_2 over r squared", constant=G, unit="newton",
         examples=["falling_apple", "ocean_tides", "moon_orbit"], attractive="attractive", direction="toward_mass",
         acts_at_distance="yes",
         notes=_STRENGTH_NOTE + "; newton principia 1687, einstein general relativity 1915; the graviton is not observed"),
    _row("electromagnetism", kind="fundamental", carrier="photon", acts_on=["electric_charge"], range_m="infinite",
         relative_strength=1e-2, holds_together=["atoms", "molecules"], described_by=["maxwell", "faraday"],
         discovered=1865, formula="charge e_field plus charge v b", constant=7.2974e-3, unit="newton",
         examples=["light", "radio", "electric_motor"], attractive="both", direction="along_field",
         acts_at_distance="yes",
         notes=_STRENGTH_NOTE + "; constant is the fine structure constant alpha = 1/137.036, dimensionless (checked "
                                "against codata in the tests); formula is the lorentz force; maxwell's equations "
                                "1865, faraday induction 1831, oersted 1820"),
    _row("strong", kind="fundamental", carrier="gluon", acts_on=["colour_charge"], range_m=1e-15, relative_strength=1,
         holds_together=["protons", "neutrons"], described_by=["gell_mann"], discovered=1973, unit="newton",
         examples=["proton", "neutron", "atomic_nucleus"], attractive="attractive", direction="between_quarks",
         acts_at_distance="yes",
         notes=_STRENGTH_NOTE + "; range is the confinement scale, an order of magnitude; it binds quarks into "
                                "protons and neutrons; quantum chromodynamics 1973 (fritzsch, gell_mann, leutwyler; "
                                "asymptotic freedom gross, wilczek, politzer), quarks 1964"),
    _row("weak", kind="fundamental", carrier=["w_boson", "z_boson"], acts_on=["quarks_leptons"], range_m=1e-18,
         relative_strength=1e-6, described_by=["fermi", "glashow", "weinberg", "salam"],
         discovered=1933, constant=1.1664e-5, unit="newton", examples=["beta_decay", "sun_fusion", "neutrino_scattering"],
         attractive="neither", direction="none", acts_at_distance="yes",
         notes=_STRENGTH_NOTE + "; constant is the fermi constant G_F / (hbar c)^3 in gev^-2, not si (checked against "
                                "codata in the tests); range 1e-18 m is the usual round figure, an order of magnitude "
                                "(hbar / m_w c is 2.5e-18 m); it changes particle flavour rather than pulling or "
                                "pushing and binds nothing; fermi theory 1933/1934, electroweak 1967/1968, w and z "
                                "bosons found 1983"),
    _row("electrostatic", kind="effective", arises_from=["electromagnetism"], acts_on=["electric_charge"],
         range_m="infinite", holds_together=["atoms", "ionic_crystals"], described_by=["coulomb"],
         discovered=1785, formula="k_e charge_1 charge_2 over r squared", constant=K_COULOMB, unit="newton",
         examples=["static_cling", "balloon_on_hair", "lightning"], attractive="both", direction="along_line_of_charges",
         acts_at_distance="yes",
         notes="the static limit of electromagnetism; coulomb's law 1785; k_e = 1 / (4 pi epsilon_0)"),
    _row("magnetic", kind="effective", arises_from=["electromagnetism"],
         acts_on=["moving_charge", "currents", "magnets"], range_m="infinite",
         described_by=["oersted", "ampere", "lorentz"], discovered=1820, formula="charge v b", constant=1.2566e-6,
         unit="newton", examples=["compass", "fridge_magnet", "electric_motor"], attractive="both",
         direction="perpendicular_to_velocity_and_field", acts_at_distance="yes",
         notes="lodestones known in antiquity; the law dates from 1820, when oersted linked current and magnetism and "
               "ampere and biot and savart measured the force; lorentz force 1895; constant is mu_0 (checked against "
               "codata in the tests)"),
    _row("friction", kind="effective", arises_from=["electromagnetism"], acts_on=["surfaces"],
         range_m="contact", described_by=["amontons", "coulomb"], discovered=1699, formula="mu n", unit="newton",
         examples=["walking", "brakes", "striking_a_match"], attractive="neither", direction="opposes_motion",
         acts_at_distance="no",
         notes="amontons' laws 1699, coulomb 1781, leonardo da vinci earlier; mu is the friction coefficient, n the "
               "normal force; static and kinetic friction have their own rows"),
    _row("static_friction", kind="effective", arises_from=["electromagnetism"], acts_on=["surfaces"],
         range_m="contact", described_by=["amontons", "coulomb"], formula="mu_s n", unit="newton",
         examples=["parked_car", "box_at_rest", "walking"], attractive="neither",
         direction="opposes_impending_motion", acts_at_distance="no",
         notes="mu_s n is the maximum before sliding starts; mu_s is usually larger than mu_k"),
    _row("kinetic_friction", kind="effective", arises_from=["electromagnetism"], acts_on=["surfaces"],
         range_m="contact", described_by=["coulomb"], formula="mu_k n", unit="newton",
         examples=["sliding_box", "skidding_tyre", "sledge"], attractive="neither", direction="opposes_motion",
         acts_at_distance="no",
         notes="coulomb 1781: roughly independent of speed and of contact area"),
    _row("normal", kind="effective", arises_from=["electromagnetism"], acts_on=["surfaces"],
         range_m="contact", described_by=["newton"], unit="newton",
         examples=["book_on_table", "standing_on_floor", "chair"], attractive="repulsive",
         direction="perpendicular_to_surface", acts_at_distance="no",
         notes="electron repulsion between touching surfaces; no formula: it equals m g only on level ground with "
               "nothing else pushing"),
    _row("tension", kind="effective", arises_from=["electromagnetism"],
         acts_on=["ropes", "strings", "cables"], range_m="contact", described_by=["newton"], unit="newton",
         examples=["rope", "crane_cable", "guitar_string"], attractive="attractive",
         direction="along_rope", acts_at_distance="no",
         notes="the bonds of the rope pull; no formula: it equals m g only for a mass hanging still"),
    _row("drag", kind="effective", arises_from=["electromagnetism"], acts_on=["bodies_in_fluids"],
         range_m="contact", described_by=["newton", "stokes"], formula="half rho v squared c_d area", unit="newton",
         examples=["air_resistance", "parachute", "swimming"], attractive="neither", direction="opposes_motion",
         acts_at_distance="no",
         notes="quadratic drag for fast flow; stokes' law 6 pi eta r v for slow flow 1851; newton on resistance 1687"),
    _row("lift", kind="effective", arises_from=["electromagnetism"], acts_on=["wings", "bodies_in_fluids"],
         range_m="contact", described_by=["bernoulli", "kutta", "joukowski"], formula="half rho v squared c_l area",
         unit="newton", examples=["airplane_wing", "helicopter_rotor", "sail"], attractive="neither",
         direction="perpendicular_to_flow", acts_at_distance="no",
         notes="a pressure difference across the wing; bernoulli 1738, kutta joukowski theorem about 1906"),
    _row("buoyancy", kind="effective", arises_from=["gravity", "electromagnetism"],
         acts_on=["bodies_in_fluids"], range_m="contact", described_by=["archimedes"], discovered="ancient",
         formula="rho g vol", constant=9.81, unit="newton", examples=["ship", "hot_air_balloon", "iceberg"],
         attractive="neither", direction="upward", acts_at_distance="no",
         notes="gravity sets up the pressure gradient, the push itself is electromagnetic; archimedes about 250 bc; "
               "rho is the fluid density, vol the displaced volume, g standard gravity"),
    _row("spring", kind="effective", arises_from=["electromagnetism"], acts_on=["springs", "elastic_solids"],
         range_m="contact", described_by=["hooke"], discovered=1678, formula="k x", unit="newton",
         examples=["trampoline", "mattress", "car_suspension"], attractive="both", direction="toward_rest_length",
         acts_at_distance="no",
         notes="hooke's law published 1678 (found 1660); k is the spring constant, x the stretch"),
    _row("pressure_force", kind="effective", arises_from=["electromagnetism"], acts_on=["surfaces", "fluids"],
         range_m="contact", described_by=["pascal", "bernoulli"], discovered=1653, formula="pressure area",
         unit="newton", examples=["tyre", "balloon", "hydraulic_press"], attractive="repulsive",
         direction="perpendicular_to_surface", acts_at_distance="no",
         notes="molecular collisions; the force a pressure puts on an area, in newton, while pressure itself is in "
               "pascal (the calc verb pressure); pascal's law 1653 (published 1663), torricelli 1643"),
    _row("surface_tension", kind="effective", arises_from=["electromagnetism"], acts_on=["liquid_surfaces"],
         range_m="contact", holds_together=["droplets", "bubbles"], described_by=["young", "laplace"], discovered=1805,
         formula="gamma l", unit="newton", examples=["water_strider", "raindrop", "soap_bubble"],
         attractive="attractive", direction="along_surface", acts_at_distance="no",
         notes="cohesion between molecules; gamma is the surface tension of the liquid (water at 20 c 0.0728 n/m), "
               "l the length of the edge; young 1805, laplace 1806"),
    _row("capillary", kind="effective", arises_from=["electromagnetism"], acts_on=["liquids", "narrow_tubes"],
         range_m="contact", described_by=["jurin", "young", "laplace"], discovered=1718, formula="2 pi r gamma cos theta",
         unit="newton", examples=["paper_towel", "candle_wick", "plant_xylem"], attractive="both", direction="along_wall",
         acts_at_distance="no",
         notes="adhesion to the wall plus surface tension; rise height 2 gamma cos theta over rho g r (jurin 1718); "
               "water rises, mercury is pushed down"),
    _row("tidal", kind="effective", arises_from=["gravity"], acts_on=["extended_bodies"], range_m="infinite",
         described_by=["newton"], discovered=1687, formula="2 big_g m_1 m_2 r over d cubed", constant=G, unit="newton",
         examples=["ocean_tides", "io_volcanoes", "roche_limit"], attractive="both", direction="along_line_to_mass",
         acts_at_distance="yes",
         notes="the difference of gravity across a body of size r at distance d; stretches along the line, squeezes "
               "across it; newton explained the tides 1687"),
    _row("radiation_pressure", kind="effective", arises_from=["electromagnetism"], acts_on=["surfaces", "dust"],
         range_m="infinite", described_by=["maxwell", "lebedev"], discovered=1901, formula="power over c",
         constant=2.9979e8, unit="newton", examples=["comet_tail", "solar_sail", "laser_cooling"],
         attractive="repulsive", direction="away_from_source", acts_at_distance="yes",
         notes="momentum of light: absorbed power over c, twice that when reflected; predicted by maxwell 1873, "
               "measured by lebedev 1901 and nichols and hull 1901 (the year is the measurement); constant is c"),
    _row("nuclear_binding", kind="effective", arises_from=["strong"], acts_on=["protons", "neutrons"],
         range_m=1.41e-15, holds_together=["nuclei"], described_by=["yukawa"], discovered=1935,
         unit="newton", examples=["atomic_nucleus", "nuclear_fusion", "nuclear_fission"], attractive="both",
         direction="toward_nucleon", acts_at_distance="yes",
         notes="the residual strong force between nucleons, carried by pions in yukawa's theory 1935; range is the "
               "pion compton wavelength hbar / (m_pi c) = 1.41 fm (computed in the tests); attractive at 1 to 2 fm, "
               "repulsive core below 0.7 fm"),
    _row("van_der_waals", kind="effective", arises_from=["electromagnetism"], acts_on=["atoms", "molecules"],
         range_m=1e-9, holds_together=["molecular_solids", "liquids"], described_by=["van_der_waals", "london"],
         discovered=1873, formula="6 c_6 over r pow 7", unit="newton",
         examples=["gecko_feet", "graphite_layers", "condensation"],
         attractive="attractive", direction="toward_molecule", acts_at_distance="yes",
         notes="fluctuating dipoles; the potential is minus c_6 over r^6, so the force 6 c_6 over r^7; felt up to "
               "about 1 nm, an order of magnitude; van der waals equation 1873, london dispersion 1930; repulsive at "
               "very short range"),
    _row("hydrogen_bond", kind="effective", arises_from=["electromagnetism"], acts_on=["polar_molecules"],
         range_m=2.76e-10, holds_together=["water", "ice", "dna_strands", "proteins"],
         described_by=["latimer", "rodebush", "pauling"], discovered=1920, unit="newton",
         examples=["water", "ice", "dna_double_helix"], attractive="attractive", direction="donor_to_acceptor",
         acts_at_distance="yes",
         notes="hydrogen bound to n, o or f attracted to a lone pair; range is the donor to acceptor (o to o) "
               "distance in ice 2.76e-10 m, the h to o part of it is about 1.8e-10 m; latimer and rodebush 1920, "
               "pauling 1939"),
    _row("ionic_bond", kind="effective", arises_from=["electromagnetism"], acts_on=["ions"], range_m=2.82e-10,
         holds_together=["salt_crystals"], described_by=["coulomb", "kossel", "born"], discovered=1916,
         formula="k_e charge_1 charge_2 over r squared", constant=K_COULOMB, unit="newton",
         examples=["table_salt", "magnesium_oxide", "calcium_fluoride"], attractive="attractive",
         direction="toward_opposite_ion", acts_at_distance="yes",
         notes="electrostatic attraction of opposite ions; range is the na to cl distance in rock salt 2.82e-10 m "
               "(half the lattice constant 5.64e-10 m); kossel 1916, born lande lattice energy 1918"),
    _row("covalent_bond", kind="effective", arises_from=["electromagnetism"], acts_on=["atoms"],
         range_m=1.54e-10, holds_together=["molecules", "diamond"], described_by=["lewis", "heitler", "london", "pauling"],
         discovered=1916, unit="newton", examples=["hydrogen_molecule", "water_molecule", "diamond"],
         attractive="attractive", direction="between_nuclei", acts_at_distance="yes",
         notes="shared electron pairs, a quantum effect of electromagnetism; range is the c to c bond length in "
               "diamond 1.54e-10 m (lattice constant 3.567e-10 m times sqrt(3) / 4); lewis 1916, heitler and london "
               "1927, pauling 1939"),
    _row("metallic_bond", kind="effective", arises_from=["electromagnetism"], acts_on=["metal_atoms", "electrons"],
         range_m=2.56e-10, holds_together=["metals"], described_by=["drude", "sommerfeld"], discovered=1900, unit="newton",
         examples=["copper_wire", "iron_beam", "gold_ring"], attractive="attractive", direction="toward_electron_sea",
         acts_at_distance="yes",
         notes="ions in a sea of free electrons; range is the cu to cu distance 2.56e-10 m (lattice constant "
               "3.615e-10 m over sqrt(2)); drude 1900, sommerfeld 1927"),
    _row("centripetal", kind="role", arises_from=["any"], acts_on=["circling_bodies"],
         described_by=["huygens", "newton"], discovered=1673, formula="m v squared over r", unit="newton",
         examples=["moon_orbit", "car_turning", "spin_dryer"], attractive="attractive", direction="toward_centre",
         notes="not a kind of force but the role of whatever force keeps a body on a circle: gravity for an orbit, "
               "friction for a turning car, tension for a sling; huygens published m v^2 / r in horologium "
               "oscillatorium 1673 (derived 1659), newton named it in 1687"),
    _row("thrust", kind="effective", arises_from=["electromagnetism"], acts_on=["rockets", "jets", "propellers"],
         range_m="contact", described_by=["newton", "tsiolkovsky"], formula="m_dot v_e", unit="newton",
         examples=["rocket_engine", "jet_engine", "propeller"], attractive="neither", direction="opposite_to_exhaust",
         acts_at_distance="no",
         notes="reaction to mass thrown backwards (newton's third law); the push of the gas is electromagnetic; "
               "m_dot is the mass flow, v_e the exhaust speed; tsiolkovsky rocket equation 1903"),
    _row("weight", kind="effective", arises_from=["gravity"], acts_on=["bodies_near_a_planet"],
         described_by=["newton"], discovered=1687, formula="m g", constant=9.81, unit="newton",
         examples=["bathroom_scale", "falling_apple", "heavy_suitcase"], attractive="attractive",
         direction="toward_planet_centre", acts_at_distance="yes",
         notes="gravity on a body near a planet's surface; g is standard gravity on earth 9.80665 m/s2 to 3 digits "
               "(checked against codata in the tests); a mass in kg is not a weight in newton; newton 1687"),
    _row("centrifugal", kind="inertial", acts_on=["bodies_in_rotating_frames"], described_by=["huygens"],
         discovered=1673, formula="m v squared over r", unit="newton", examples=["carousel", "centrifuge", "salad_spinner"],
         direction="away_from_axis",
         notes="felt only in a rotating frame, where it balances the centripetal force, hence the same formula "
               "(m omega^2 r with v = omega r); huygens named it, theorems published 1673, de vi centrifuga "
               "written 1659 and printed 1703"),
    _row("coriolis", kind="inertial", acts_on=["moving_bodies_in_rotating_frames"], described_by=["coriolis"],
         discovered=1835, formula="2 m omega v", unit="newton",
         examples=["hurricane_rotation", "trade_winds", "foucault_pendulum"],
         direction="perpendicular_to_velocity_and_axis",
         notes="felt only in a rotating frame; 2 m omega v is its size for motion perpendicular to the axis, omega "
               "the frame's angular speed; coriolis 1835"),
]

#: target share of each question kind in generate()
PROPORTIONS = {"fact": 0.35, "compare": 0.08, "list": 0.04, "calc": 0.38, "yesno": 0.15}
#: share of each kind of check among the yes/no lines while all last
CHECK_SHARES = {"fact": 0.5, "calc": 0.3, "compare": 0.125, "list": 0.075}
#: a kind of check is closed after this many failed attempts in a row (its true lines are used up)
CHECK_PATIENCE = 200
LINE_KIND = {"fact": "fact", "compare": "fact", "list": "fact", "calc": "calc", "yesno": "yesno"}

_WORD = re.compile(r"^[a-z0-9_]+$")
_NUMBERISH = DIGITS | NUMBER_WORDS
_TIDY = re.compile(r"^(minus )?(0|[1-9]( [0-9])*)( point [0-9]( [0-9])*)?( e (minus )?[1-9]( [0-9])*)?$")
_YEAR = re.compile(r"\b[0-9]{4}\b")


def n_tokens(line: str) -> int:
    """Token count as ``haishool.student.tokens`` counts it (``.`` is its own token)."""
    return len(line.replace(".", " . ").split())


def dense_value(value) -> str:
    """A table value as dense words: numbers through :func:`num` with up to 5 digits, lists joined."""
    if isinstance(value, bool):
        raise TypeError("table values are not bools")
    if isinstance(value, (int, float)):
        return num(value, sig=5)
    if isinstance(value, list):
        return " ".join(value)
    return value


def record_lines(row: dict) -> list[str]:
    """``gravity force. kind fundamental. ...``, or two lines when one would pass 80 tokens."""
    fields = [(k, dense_value(row[k])) for k in ROW_KEYS if k in row]

    def join(head: str, some: list[tuple[str, str]]) -> str:
        return head + " " + " ".join(f"{k} {v}." for k, v in some)

    one = join(f"{row['name']} force.", fields)
    if n_tokens(one) <= MAX_TOKENS:
        return [one]
    return [join(f"{row['name']} force.", [f for f in fields if f[0] in FIRST_KEYS]),
            join(f"{row['name']} force_more.", [f for f in fields if f[0] not in FIRST_KEYS])]


def sig4(x: float) -> float:
    """``x`` rounded to 4 significant digits, so a calc answer stays short."""
    return float(f"{x:.4g}")


def calc_answer(x: float) -> str:
    """A calc result as digits: a whole number below a million in full, anything else with 4
    significant digits.

    >>> calc_answer(66125.0), calc_answer(6484.41), calc_answer(15.000000000000002)
    ('6 6 1 2 5', '6 4 8 4', '1 5')
    """
    whole = float(f"{x:.12g}")
    if whole.is_integer() and abs(whole) < 1e6:
        return num(int(whole))
    return num(sig4(x))


def tidy_number(tokens: str) -> bool:
    """True for a number written the way :func:`num` writes it: no leading zero, no ``minus 0``."""
    x = parse_num(tokens)
    return bool(_TIDY.match(tokens)) and is_finite(x) and not (x == 0 and "minus" in tokens)


def _printed_digits(tokens: str) -> int:
    """How many significant digits a number shows; every digit of a whole number counts."""
    mantissa = "".join(w for w in tokens.split(" e ")[0].split() if w in DIGITS)
    return max(1, len(mantissa.lstrip("0")))


def same_number(gold: str, answer: str, digits: int = 1) -> bool:
    """True when ``answer`` rounds to ``gold`` at every digit ``gold`` shows, and at ``digits``
    significant digits at least. Infinite and untidy numbers never pass.

    >>> same_number("4 9 point 0 5", "4 9 point 0 5 1", 4), same_number("4 9 point 0 5", "4 9 point 0 6", 4)
    (True, False)
    >>> same_number("2 0 0", "2 0 1", 4), same_number("1 e minus 1 5", "1 point 4 e minus 1 5")
    (False, True)
    """
    a, b = parse_num(gold), parse_num(answer)
    if a is None or b is None or not tidy_number(answer) or not math.isfinite(a):
        return False
    d = max(digits, _printed_digits(gold)) - 1
    return f"{float(a):.{d}e}" == f"{float(b):.{d}e}"


def matches(expected: str, answer: str, mode: str) -> bool:
    """``mode`` is ``num`` (a table number), ``calc`` (a calc result), ``set`` (order free, no
    repeats) or ``exact``."""
    answer = " ".join(answer.split())
    if mode in ("num", "calc"):
        return same_number(expected, answer, CALC_DIGITS if mode == "calc" else 1)
    if mode == "set":
        words = answer.split()
        return len(words) == len(set(words)) and set(words) == set(expected.split())
    return expected == answer


def _range_value(text: str) -> float | None:
    if text == "infinite":
        return math.inf
    if text == "contact":
        return CONTACT_RANGE_M
    x = parse_num(text)
    return None if x is None else float(x)


def range_order(a: str, b: str) -> int | None:
    """1 when range ``a`` is longer than ``b``, -1 when shorter, 0 for the same value, ``None``
    when both are finite and closer than :data:`RANGE_FACTOR` (not comparable)."""
    if a == b:
        return 0
    x, y = _range_value(a), _range_value(b)
    if x is None or y is None or x <= 0 or y <= 0:
        return None
    if not (math.isinf(x) or math.isinf(y)) and max(x, y) < RANGE_FACTOR * min(x, y):
        return None
    return 1 if x > y else -1


def _operands(words: list[str]) -> tuple[dict[str, list[float]], list[str]]:
    """``m 1 0 m 2 0 r 2`` -> ``({"m": [10, 20], "r": [2]}, [])``; bare words become flags (``on moon``)."""
    params: dict[str, list[float]] = {}
    flags: list[str] = []
    cur: str | None = None
    buf: list[str] = []

    def flush() -> None:
        if cur is None:
            return
        if buf:
            x = parse_num(buf)
            if not is_finite(x):
                raise ValueError(f"bad number for {cur}: {' '.join(buf)}")
            params.setdefault(cur, []).append(float(x))
        else:
            flags.append(cur)

    for w in words:
        if w in _NUMBERISH and cur is not None:
            buf.append(w)
        else:
            flush()
            cur, buf = w, []
    flush()
    return params, flags


def compute(verb: str, words: list[str]) -> float:
    """The SI value of a calc prompt's operands; raises ``ValueError`` (or an ``ArithmeticError``
    for a division by zero or an overflow) when it cannot be computed."""
    if verb not in CALC:
        raise ValueError(f"unknown calc {verb}")
    params, flags = _operands(words)
    need = CALC[verb]
    for name in dict.fromkeys(need):
        if len(params.get(name, [])) != need.count(name):
            raise ValueError(f"{verb} needs {' '.join(need)}")
    if set(params) - set(need):
        raise ValueError(f"{verb} needs {' '.join(need)}")
    for name, values in params.items():
        if name not in SIGNED and min(values) < 0:
            raise ValueError(f"{name} cannot be negative")
    if verb == "weight":
        body = "earth"
        if flags:
            if len(flags) != 2 or flags[0] != "on" or flags[1] not in SURFACE_G:
                raise ValueError("weight takes on earth, moon, mars or jupiter")
            body = flags[1]
        return params["m"][0] * SURFACE_G[body]
    if verb == "net_force":
        if flags not in (["same"], ["opposite"]):
            raise ValueError("net_force needs same or opposite")
        f1, f2 = params["f"]
        return f1 + f2 if flags == ["same"] else abs(f1 - f2)
    if flags:
        raise ValueError(f"{verb} takes no extra words")
    p = {k: v[0] for k, v in params.items()}
    if verb == "gravity_force":
        return G * params["m"][0] * params["m"][1] / p["r"] ** 2
    if verb == "coulomb_force":
        return K_COULOMB * params["charge"][0] * params["charge"][1] / p["r"] ** 2
    if verb == "spring_force":
        return p["k"] * p["x"]
    if verb == "friction":
        return p["mu"] * p["n"]
    if verb == "pressure":
        return p["f"] / p["area"]
    if verb == "acceleration":
        return p["f"] / p["m"]
    if verb == "buoyancy":
        return p["rho"] * SURFACE_G["earth"] * p["vol"]
    if verb == "centripetal":
        return p["m"] * p["v"] ** 2 / p["r"]
    if verb in ("orbital_speed", "escape_speed"):
        speed = math.sqrt((2 if verb == "escape_speed" else 1) * G * p["m"] / p["r"])
        if speed >= LIGHT_SPEED:
            raise ValueError("faster than light, newton's formula does not hold there")
        return speed
    if verb == "surface_gravity":
        return G * p["m"] / p["r"] ** 2
    if verb in ("drag", "lift"):
        return 0.5 * p["rho"] * p["v"] ** 2 * p["c"] * p["area"]
    if verb == "thrust":
        return p["mdot"] * p["ve"]
    if verb == "tidal":
        return 2 * G * params["m"][0] * params["m"][1] * p["r"] / p["d"] ** 3
    return p["charge"] * p["v"] * p["b"]  # magnetic_force


def worked_answer(verb: str, words: list[str]) -> str:
    """A deterministic formula -> substitution -> result trace of a valid SI calculation.

    No intermediate rounding is applied. The final result uses the original gate's
    precision. Exact canonical steps are required, just as for maths column methods.
    """
    result = compute(verb, words)  # validate counts, signs, flags, domain and denominators
    if not math.isfinite(result):
        raise ValueError("not finite")
    params, flags = _operands(words)
    literals: dict[str, list[str]] = {}
    i = 0
    while i < len(words):
        key = words[i]
        i += 1
        start = i
        while i < len(words) and words[i] in _NUMBERISH:
            i += 1
        if i > start:
            literals.setdefault(key, []).append(" ".join(words[start:i]))
    substitutions = {}
    for key, values in literals.items():
        for index, value in enumerate(values, 1):
            name = f"{key}_{index}" if len(params[key]) > 1 else key
            substitutions[name] = value
    body = flags[-1] if verb == "weight" and flags else "earth"
    substitutions.update(big_g=num(G), k_e=num(K_COULOMB), g=num(SURFACE_G[body]), half="0 point 5")
    formula = WORKED_FORMULAS[verb]
    if verb == "net_force" and flags == ["opposite"]:
        formula = "abs open f_1 minus f_2 close"
    substituted = " ".join(substitutions.get(word, word) for word in formula.split())
    return f"formula {formula} then substitute {substituted} then result {calc_answer(result)}"


def _draw(rng: random.Random, weights: dict[str, float]) -> str:
    r = rng.random() * sum(weights.values())
    last = ""
    for k, w in weights.items():
        r -= w
        last = k
        if r < 0:
            return k
    return last


class ForcesGate:
    """The forces gate: table lookups, comparisons, lists, SI calculations and yes/no checks."""

    topic = TOPIC
    KEYS = KEYS

    def __init__(self, table: list[dict] = TABLE) -> None:
        self.table = table
        self.rows = {r["name"]: r for r in table}
        self.names = [r["name"] for r in table]
        self._validate()
        fundamental = {n for n in self.names if self.rows[n]["kind"] == "fundamental"}
        #: the interactions a row belongs to; a role and an inertial force belong to all of them,
        #: so their values are never offered as wrong for another force, nor another's for them
        self.family = {r["name"]: frozenset(fundamental if r["kind"] in ("role", "inertial")
                                            else r.get("arises_from", [r["name"]])) for r in table}
        self.facts = [(r["name"], k) for r in table for k in ROW_KEYS if k in r and k not in SHARED.get(r["name"], ())]
        self.check_facts = [f for f in self.facts if f[1] not in UNCHECKED_KEYS]
        strength = [n for n in self.names if "relative_strength" in self.rows[n]]
        ranges = [n for n in self.names if "range_m" in self.rows[n]]
        #: every comparison the table can answer, both orders
        self.comparisons = {verb: [p for p in (f"{verb} {a} {b}" for a in pool for b in pool if a != b)
                                   if self.expected(p)[0] is not None]
                            for verb, pool in (("stronger", strength), ("longer_range", ranges))}
        self.list_pairs: list[tuple[str, str]] = []
        for key in LIST_KEYS:
            values: list[str] = []
            for row in table:
                for tok in dense_value(row[key]).split() if key in row else ():
                    if tok not in values and tok not in _NUMBERISH:
                        values.append(tok)
            self.list_pairs += [(key, v) for v in values if 1 <= len(self._matching(key, v)) <= MAX_LIST]

    def _validate(self) -> None:
        if len(self.rows) != len(self.table):
            raise ValueError("duplicate force names")
        fundamental = [r["name"] for r in self.table if r["kind"] == "fundamental"]
        for row in self.table:
            for key, value in row.items():
                if key == "notes":
                    continue
                if key != "name" and key not in ROW_KEYS:
                    raise ValueError(f"{row['name']}: unknown key {key}")
                for tok in dense_value(value).split():
                    if not _WORD.match(tok):
                        raise ValueError(f"{row['name']} {key}: {tok!r} is not a dense word")
            if row["kind"] not in ("fundamental", "effective", "role", "inertial"):
                raise ValueError(f"{row['name']}: bad kind")
            if row["kind"] == "effective" and not set(row["arises_from"]) <= set(fundamental):
                raise ValueError(f"{row['name']}: arises_from must name fundamental interactions")
            if "carrier" in row and row["kind"] != "fundamental":
                raise ValueError(f"{row['name']}: only a fundamental interaction has a carrier")
            for line in record_lines(row):
                if not is_dense(line) or n_tokens(line) > MAX_TOKENS:
                    raise ValueError(f"{row['name']}: record line not dense or too long: {line}")

    # -- the table ---------------------------------------------------------------------------

    def value(self, name: str, key: str) -> str | None:
        row = self.rows.get(name)
        return dense_value(row[key]) if row and key in row else None

    def _mode(self, key: str, expected: str) -> str:
        if key in NUMERIC_KEYS and parse_num(expected) is not None:
            return "num"
        return "set" if key in SET_KEYS else "exact"

    def _matching(self, key: str, value: str) -> list[str]:
        return [r["name"] for r in self.table if key in r and value in dense_value(r[key]).split()]

    def related(self, a: str, b: str) -> bool:
        """True when two forces share a fundamental interaction (friction and magnetic, gravity and tidal)."""
        return bool(self.family[a] & self.family[b])

    def records(self) -> list[Line]:
        return [Line(line, topic=TOPIC, kind="record", meta={"name": row["name"]})
                for row in self.table for line in record_lines(row)]

    # -- judging -----------------------------------------------------------------------------

    def owns(self, prompt: str) -> bool:
        words = prompt.split()
        if words and words[0] == "check":
            words = words[1:]
            if "equals" in words:
                words = words[:words.index("equals")]
            elif len(words) > 2 and words[0] in self.rows and words[1] in ROW_KEYS:
                words = words[:2]
        return self._is_question(words)

    def _is_question(self, words: list[str]) -> bool:
        """True for the forms this gate asks: ``<force> <key>``, a calc verb with an operand,
        ``forces <key> ...``, a comparison of force names."""
        if len(words) < 2:
            return False
        head, rest = words[0], words[1:]
        if head in ("stronger", "longer_range"):
            return all(w in self.rows for w in rest)
        if head == "forces":
            return rest[0] in ROW_KEYS
        if head in self.rows and len(rest) == 1 and rest[0] in ROW_KEYS:
            return rest[0] not in SHARED.get(head, ())
        return head in CALC and rest[0] in CALC[head]

    def expected(self, prompt: str) -> tuple[str | None, str, str]:
        """(gold answer, compare mode, reason) for an owned prompt; the answer is ``None`` when the
        prompt cannot be judged (the reason says why)."""
        words = prompt.split()
        if not words:
            return None, "", "empty prompt"
        head, rest = words[0], words[1:]
        if head == "check":
            return self._expected_check(rest)
        if head in CALC and rest[-1:] == ["steps"]:
            try:
                return worked_answer(head, rest[:-1]), "steps", ""
            except (ValueError, ArithmeticError) as error:
                return None, "", f"cannot judge: {error}"
        if head in ("stronger", "longer_range"):
            if len(rest) != 2 or any(n not in self.rows for n in rest):
                return None, "", f"cannot judge: {head} needs two force names"
            key = "relative_strength" if head == "stronger" else "range_m"
            vals = [self.value(n, key) for n in rest]
            if any(v is None for v in vals):
                return None, "", f"cannot judge: no {key} for " + " ".join(n for n, v in zip(rest, vals) if v is None)
            if head == "longer_range":
                order = range_order(vals[0], vals[1])
            else:
                a, b = (parse_num(v) for v in vals)
                order = None if a is None or b is None else (a > b) - (a < b)
            if order is None:
                return None, "", f"cannot judge: the two {key} values are too close to compare"
            return (rest[0] if order > 0 else rest[1] if order < 0 else "same"), "exact", ""
        if head == "forces":
            if len(rest) != 2 or rest[0] not in LIST_KEYS or rest[1] in _NUMBERISH:
                return None, "", "cannot judge: forces needs <key> <value>, the value a word"
            names = self._matching(rest[0], rest[1])
            return (" ".join(names) if names else "none"), "set", ""
        if head in self.rows and len(rest) == 1 and rest[0] in ROW_KEYS:
            if rest[0] in SHARED.get(head, ()):
                return None, "", "not my question"
            gold = self.value(head, rest[0])
            if gold is None:
                return None, "", f"cannot judge: no {rest[0]} for {head}"
            return gold, self._mode(rest[0], gold), ""
        if head in CALC:
            try:
                x = compute(head, rest)
                if not math.isfinite(x):
                    raise ValueError("not finite")
                return calc_answer(x), "calc", ""
            except ValueError as e:
                return None, "", f"cannot judge: {e}"
            except ArithmeticError as e:  # a division by zero, or an operand so large that the result overflows
                return None, "", f"cannot judge: {type(e).__name__} in {head}"
        if head in self.rows:
            return None, "", f"cannot judge: {head} needs a key from ROW_KEYS"
        return None, "", "not my question"

    def _expected_check(self, rest: list[str]) -> tuple[str | None, str, str]:
        if "equals" in rest:
            i = rest.index("equals")
            inner, given, stated = rest[:i], rest[i + 1:], False
        elif len(rest) >= 2 and rest[0] in self.rows and rest[1] in ROW_KEYS:
            inner, given, stated = rest[:2], rest[2:], True
        else:
            return None, "", "cannot judge: check needs <force> <key> <value> or <question> equals <value>"
        if not inner or inner[0] == "check":
            return None, "", "cannot judge: check needs a question"
        gold, mode, reason = self.expected(" ".join(inner))
        if gold is None:
            return None, "", reason
        if not given:
            return None, "", "cannot judge: no value to check"
        if matches(gold, " ".join(given), mode):
            return "yes", "exact", ""
        if stated and mode == "set" and set(given) < set(gold.split()):
            return None, "", "cannot judge: part of the list, neither the whole truth nor wrong"
        return "no", "exact", ""

    def check(self, prompt: str, answer: str) -> Verdict:
        if not self.owns(prompt):
            return Verdict(False, None, "not my question")
        gold, mode, reason = self.expected(prompt)
        if gold is None:
            return Verdict(False, None, reason)
        if mode == "steps":
            return check_steps(gold, answer)
        if matches(gold, answer, mode):
            return Verdict(True, gold, "")
        return Verdict(False, gold, f"expected {gold}")

    # -- generating --------------------------------------------------------------------------

    def generate_worked(self, rng: random.Random, n: int, max_tokens: int = WORKED_MAX_TOKENS) -> list[Line]:
        """Separate opt-in worked curriculum; preserves the legacy generator and its mix.

        All 18 calculation verbs are sampled. Set max_tokens=96 for a short-context
        subset; the default keeps long tidal/astronomical substitutions as well.
        """
        out, seen = [], set()
        for _ in range(100 * n + 1000):
            if len(out) == n:
                return out
            base = self._calc_prompt(rng, rng.choice(list(CALC)))
            prompt = base + " steps"
            answer, _, _ = self.expected(prompt)
            if answer is None or prompt in seen:
                continue
            line = Line(prompt, answer, self.topic, "calc", {"form": "steps", "base_prompt": base})
            if n_tokens(line.text) > max_tokens:
                continue
            seen.add(prompt)
            out.append(line)
        raise ValueError(f"cannot generate {n} distinct worked forces lines within {max_tokens} tokens")

    def table_prompts(self) -> dict[str, list[str]]:
        """Every question the table itself can ask, in table order: all facts, all ordered
        comparison pairs it can judge, all lists of at most :data:`MAX_LIST` names."""
        return {"fact": [f"{name} {key}" for name, key in self.facts],
                "stronger": list(self.comparisons["stronger"]), "longer_range": list(self.comparisons["longer_range"]),
                "list": [f"forces {k} {v}" for k, v in self.list_pairs]}

    def generate(self, rng: random.Random, n: int) -> list[Line]:
        """``n`` distinct lines in the :data:`PROPORTIONS`. The table-bound kinds (fact, compare,
        list) are drawn without repetition, so a large run asks every fact exactly once; when
        one is used up the others fill in. Comparisons are half ``stronger``, half
        ``longer_range`` while both last. Yes and no stay within one line of each other in
        every kind of check; a kind whose true lines are used up is closed."""
        pools = self.table_prompts()
        for pool in pools.values():
            rng.shuffle(pool)
        out: list[Line] = []
        seen: set[str] = set()
        weights = dict(PROPORTIONS)
        checks = dict(CHECK_SHARES)
        lead = dict.fromkeys(CHECK_SHARES, 0)  # yes answers minus no answers so far, per kind of check
        dry = dict.fromkeys(CHECK_SHARES, 0)
        sub: str | None = None
        misses = 0
        attempts = 0
        while len(out) < n and weights and attempts < 60 * n + 1000:
            attempts += 1
            if sub is None:
                sub, misses = _draw(rng, weights), 0
            if sub == "yesno":
                of = _draw(rng, checks)
                want = lead[of] < 0 or (lead[of] == 0 and rng.random() < 0.5)
                line = self._yesno(rng, of, want)
                if line is None or line.text in seen or (line.answer == "yes") != want:
                    dry[of] += 1
                    if dry[of] > CHECK_PATIENCE:
                        del checks[of]
                        if not checks:
                            del weights[sub]
                            sub = None
                    continue
                dry[of] = 0
                lead[of] += 1 if want else -1
            else:
                if sub == "calc":
                    line = self._line(self._calc_prompt(rng, rng.choice(list(CALC))), sub)
                else:
                    left = [p for p in (("stronger", "longer_range") if sub == "compare" else (sub,)) if pools[p]]
                    if not left:
                        del weights[sub]
                        sub = None
                        continue
                    line = self._line(pools[rng.choice(left)].pop(), sub)
                if line is None or line.text in seen:
                    misses += 1
                    if misses > 300:
                        del weights[sub]
                        sub = None
                    continue
            seen.add(line.text)
            out.append(line)
            sub = None
        return out

    def _line(self, prompt: str, sub: str) -> Line | None:
        gold, _, _ = self.expected(prompt)
        if gold is None:
            return None
        return Line(prompt, gold, TOPIC, LINE_KIND[sub], self._meta(prompt, sub))

    @staticmethod
    def _meta(prompt: str, sub: str) -> dict:
        w = prompt.split()
        if sub == "fact":
            return {"sub": sub, "name": w[0], "key": w[1]}
        if sub == "list":
            return {"sub": sub, "key": w[1], "value": w[2]}
        return {"sub": sub, "verb": w[0]}

    def _prompt(self, rng: random.Random, sub: str) -> str:
        """One random prompt of a kind (with repetition; used inside yes/no checks)."""
        if sub == "compare":
            return rng.choice(self.comparisons[rng.choice([v for v, pool in self.comparisons.items() if pool])])
        if sub == "list":
            key, value = rng.choice(self.list_pairs)
            return f"forces {key} {value}"
        return self._calc_prompt(rng, rng.choice(list(CALC)))

    def _calc_prompt(self, rng: random.Random, verb: str) -> str:
        def mass() -> str:
            return num(rng.randint(1, 99) * rng.choice((1, 1, 1, 10, 100, 1000)))

        def big() -> float:
            return rng.randint(10, 99) / 10 * 10 ** rng.randint(22, 30)

        def body_mass_radius(aloft: bool = True) -> tuple[str, str]:
            if rng.random() < 0.4:
                m, r = BODIES[rng.choice(list(BODIES))]
                return num(m), num(r + (rng.choice(ALTITUDES) if aloft else 0))
            while True:
                m = big()
                r = rng.randint(10, 99) / 10 * 10 ** rng.randint(5, 8)
                if 2 * G * m / r < NEWTONIAN_SPEED ** 2:
                    return num(m), num(r)

        def wing(coefficients: tuple[float, ...], areas: tuple[float, ...], top: int) -> str:
            rho = rng.choice((1.2, 1.2, 1000, 1.0, 1025))
            return (f"{verb} rho {num(rho)} v {num(rng.randint(1, top))} c {num(rng.choice(coefficients))} "
                    f"area {num(rng.choice(areas))}")

        if verb == "gravity_force":
            if rng.random() < 0.25:
                m, r = body_mass_radius()
                return f"gravity_force m {m} m {mass()} r {r}"
            r = rng.choice((0.5, 1, 1.5, 2, 3, 4, 5, 10, 20, 50, 100))
            return f"gravity_force m {mass()} m {mass()} r {num(r)}"
        if verb == "coulomb_force":
            def charge() -> str:
                sign = -1 if rng.random() < 0.2 else 1
                return num(sign * rng.randint(1, 9) * rng.choice((1e-6, 1e-6, 1e-9)))
            r = rng.choice((0.01, 0.02, 0.03, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.5, 1, 2, 5, 10))
            return f"coulomb_force charge {charge()} charge {charge()} r {num(r)}"
        if verb == "weight":
            m = num(rng.choice((rng.randint(1, 1000), rng.randint(1, 200) / 2, rng.randint(1, 999) / 10)))
            body = rng.choice(("", "", "earth", "moon", "mars", "jupiter"))
            return f"weight m {m}" + (f" on {body}" if body else "")
        if verb == "spring_force":
            k = rng.randint(1, 200) * rng.choice((1, 5, 10))
            return f"spring_force k {num(k)} x {num(rng.randint(1, 100) / 100)}"
        if verb == "friction":
            mu = rng.randint(1, 100) / 100
            return f"friction mu {num(mu)} n {num(rng.randint(1, 100) * rng.choice((1, 1, 5, 10)))}"
        if verb == "pressure":
            area = rng.randint(1, 100) * rng.choice((0.01, 0.1, 1))
            return f"pressure f {num(rng.randint(1, 100) * rng.choice((1, 1, 10, 100)))} area {num(area)}"
        if verb == "acceleration":
            return f"acceleration f {num(rng.randint(1, 100) * rng.choice((1, 1, 10)))} m {num(rng.randint(1, 100))}"
        if verb == "net_force":
            f1, f2 = (num(rng.randint(1, 100) * rng.choice((1, 1, 10))) for _ in range(2))
            return f"net_force f {f1} f {f2} {rng.choice(('same', 'opposite'))}"
        if verb == "buoyancy":
            rho = rng.choice((1000, 1000, 1025, 13600, 1.2, rng.randint(5, 140) * 10))
            vol = rng.randint(1, 100) * rng.choice((0.001, 0.01, 0.1))
            return f"buoyancy rho {num(rho)} vol {num(vol)}"
        if verb == "centripetal":
            r = rng.choice((0.5, 1, 1.5, 2, 2.5, 3, 4, 5, 10, 20, 50, 100))
            return f"centripetal m {num(rng.randint(1, 50))} v {num(rng.randint(1, 30))} r {num(r)}"
        if verb in ("orbital_speed", "escape_speed"):
            m, r = body_mass_radius()
            return f"{verb} m {m} r {r}"
        if verb == "surface_gravity":
            m, r = body_mass_radius(aloft=False)
            return f"surface_gravity m {m} r {r}"
        if verb == "drag":
            return wing((0.1, 0.25, 0.3, 0.5, 1, 1.2), (0.01, 0.1, 0.25, 0.5, 1, 2, 5, 10), 60)
        if verb == "lift":
            return wing((0.2, 0.3, 0.5, 0.8, 1, 1.2, 1.5), (0.5, 1, 2, 5, 10, 20, 50, 100), 100)
        if verb == "thrust":
            return f"thrust mdot {num(rng.randint(1, 500) * rng.choice((1, 1, 10)))} ve {num(rng.randint(2, 45) * 100)}"
        if verb == "tidal":
            if rng.random() < 0.4:
                m, r, d = rng.choice(TIDES)
            else:
                exp = rng.randint(5, 7)
                m, r = big(), rng.randint(10, 99) / 10 * 10 ** exp
                d = rng.randint(10, 99) / 10 * 10 ** (exp + rng.randint(1, 4))
            return f"tidal m {num(m)} m {mass()} r {num(r)} d {num(d)}"
        q = rng.choice((rng.randint(1, 9), rng.randint(1, 9) * 1e-6, rng.randint(1, 9) * 1e-3))
        b = rng.choice((0.01, 0.05, 0.1, 0.2, 0.5, 1, 2))
        return f"magnetic_force charge {num(q)} v {num(rng.randint(1, 100) * rng.choice((1, 1, 10, 100)))} b {num(b)}"

    def _yesno(self, rng: random.Random, of: str, want: bool) -> Line | None:
        """One check of kind ``of``: a fact check (``check gravity carrier photon``) or a question
        check (``check weight m 5 equals 4 9 point 0 5``), true when ``want`` is. ``meta["inner"]``
        is the question it restates, so a build can keep sealed questions out of the checks."""
        if of == "fact":
            name, key = rng.choice(self.check_facts)
            value = self.value(name, key) if want else self._wrong_fact(rng, name, key)
            inner = f"{name} {key}"
            prompt = f"check {inner} {value}"
            meta = {"of": of, "name": name, "key": key}
        else:
            inner = self._prompt(rng, of)
            gold, mode, _ = self.expected(inner)
            if gold is None:
                return None
            value = gold if want else self._wrong_answer(rng, inner, gold, mode, of)
            prompt = f"check {inner} equals {value}"
            meta = {"of": of, "verb": inner.split()[0]}
        if not value:
            return None
        answer, _, _ = self.expected(prompt)
        if answer is None:
            return None
        return Line(prompt, answer, TOPIC, "yesno", dict(meta, sub="yesno", inner=inner))

    def wrong_values(self, name: str, key: str) -> list[str]:
        """Every value a yes/no check may offer as plainly wrong for ``name``'s ``key`` (see the
        module docstring for what counts); a constant is varied by a factor instead."""
        row = self.rows[name]
        gold = dense_value(row[key])
        if key == "attractive":
            return list(WRONG_PULL[gold])
        if gold == "any" or key in UNCHECKED_KEYS or key == "constant":
            return []
        if key == "discovered" and isinstance(row[key], int):
            named = {int(y) for y in _YEAR.findall(row["notes"])}
            return [num(row[key] + d) for d in (-100, -50, -10, -1, 1, 10, 50, 100) if row[key] + d not in named]
        #: a bond or the van der waals force is electromagnetic, whose own reach is infinite, so
        #: infinite is not offered as wrong for a length of atomic scale
        atomic = key == "range_m" and parse_num(gold) is not None and range_order(gold, "contact") is None
        #: what is true of the force itself or, for a family key, of any of its relatives
        kin = [dense_value(o[key]) for o in self.table
               if key in o and (o is row or (key in FAMILY_KEYS and self.related(name, o["name"])))]
        taken = {tok for v in kin for tok in v.split()}
        out: list[str] = []
        for other in self.table:
            if other is row or key not in other:
                continue
            v = dense_value(other[key])
            if v in out or v == "any" or (atomic and v == "infinite"):
                continue
            if key == "range_m":
                wrong = range_order(gold, v) not in (None, 0)
            elif key in SET_KEYS:
                wrong = not taken & set(v.split())
            else:
                wrong = v not in kin and not matches(gold, v, self._mode(key, gold))
            if wrong:
                out.append(v)
        return out + [v for v in FALLBACK_WRONG.get(key, []) if v not in out and v != gold]

    def _wrong_fact(self, rng: random.Random, name: str, key: str) -> str | None:
        if key == "constant":
            return num(self.rows[name][key] * rng.choice((10, 100, 0.1, 0.01, 2, 0.5)), sig=5)
        others = self.wrong_values(name, key)
        return rng.choice(others) if others else None

    def _wrong_answer(self, rng: random.Random, inner: str, gold: str, mode: str, of: str) -> str | None:
        if mode == "calc":
            x = parse_num(gold)
            if x == 0:
                return "1"
            return calc_answer(x * rng.choice((0.5, 0.8, 0.9, 1.1, 1.25, 1.5, 2, 10, 0.1)))
        if of == "compare":  # the other force of the pair, or same
            return rng.choice([w for w in inner.split()[1:] + ["same"] if w != gold])
        names = gold.split()
        if rng.random() < 0.5 and len(names) > 1:
            dropped = rng.choice(names)
            return " ".join(n for n in names if n != dropped)
        extra = rng.choice([n for n in self.names if n not in names] or ["none"])
        return " ".join(names + [extra]) if gold != "none" else extra


_GATE: ForcesGate | None = None


def gate() -> ForcesGate:
    """The one configured forces gate."""
    global _GATE
    if _GATE is None:
        _GATE = ForcesGate()
    return _GATE


def build(out: Path) -> dict:
    """Write the table (with notes and record lines) as json lines."""
    g = gate()
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="\n") as f:
        for row in g.table:
            f.write(json.dumps(dict(row, lines=record_lines(row))) + "\n")
    records = g.records()
    return {"rows": len(g.table), "record_lines": len(records), "split_rows": len(records) - len(g.table),
            "keys": len(ROW_KEYS), "calc_verbs": len(CALC), "list_pairs": len(g.list_pairs), "out": str(out)}


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--out", type=Path, default=Path("data/truth-v5/forces.jsonl"))
    s = sub.add_parser("sample")
    s.add_argument("--n", type=int, default=30)
    s.add_argument("--seed", type=int, default=1)
    s.add_argument("--records", action="store_true")
    args = ap.parse_args()
    if args.cmd == "build":
        print(json.dumps(build(args.out), indent=1))
    else:
        g = gate()
        for ln in (g.records() if args.records else g.generate(random.Random(args.seed), args.n)):
            print(ln.text)


if __name__ == "__main__":
    main()
