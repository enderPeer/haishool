"""Level 6 of the toy universe: the chain, from the first second to the first replicator.

The five levels below each simulate one step of the story. This level runs them one after the
other for one seed, each feeding the next through a documented *hand-off*: a formula that turns
numbers of one level into the parameters of the next. A hand-off is not a simulation and
claims no physics beyond its formula; it is what makes the five toys one timeline. The steps of a
world rollout are its **eras**:

    nucleo        level 1 with the seed's own parameters (``nucleo.rollout(seed)``): the hydrogen
                  and helium the first hour leaves, and the light traces (1 - hydrogen - helium)
    enrichment    hand-off: ``generations`` (0 to 3) of earlier stars turn hydrogen into helium
                  and metals; the result is the make-up of the cloud
    cloud         hand-off: the cloud's ``spin`` (from the seed) and its ``cooling`` (from the metals)
    disc          level 2 with the seed, that spin and that cooling: how the cloud ends
    star          hand-off: the largest clump becomes the star; its light, frost line and
                  habitable zone; the solids of its disc (from the metals)
    planets       level 3 with the seed, that star and that disc
    chemistry_k   level 4 for the k-th rocky planet in the habitable zone (innermost first), seed
                  ``seed + k``: the surface cools to the planet's temperature
    life_k        level 5 for planet k if it has liquid water and organic molecules, seed
                  ``seed + k``

The six eras from ``nucleo`` to ``planets`` are always there; a world has one ``chemistry_k`` per
habitable rocky planet and at most one ``life_k`` per ``chemistry_k``.

The hand-offs. Every one after the enrichment takes its input as the lines print it (3 significant
digits), so each can be recomputed from the lines alone, and each is also a question of its own
(``q world handoff ...``) that needs no seed:

* **enrichment** (:func:`enrich`). The cloud starts as the nucleo run ends: hydrogen X, helium Y,
  traces 1 - X - Y. Each generation of stars burns 3 % of the hydrogen the cloud then holds:
  2 % of it becomes helium (:data:`HELIUM_YIELD`), 1 % becomes metals (:data:`METAL_YIELD`),
  shared out by the fixed yields of :data:`YIELDS`: oxygen 0.44, carbon 0.18, neon 0.10, iron 0.10,
  nitrogen 0.06, magnesium 0.06, silicon 0.06 of the metal mass. Nothing else changes, so the
  mass fractions still add up to 1. ``metallicity`` is the sum of the seven metals: with
  X = 0.75 it is 0, 0.0075, 0.0148, 0.0218 after 0, 1, 2, 3 generations (the sun has 0.014).
* **cooling** = min(0.6, 0.1 + 20 * metallicity), to 2 decimals: gas without metals cools badly
  (0.1), gas of solar make-up well (0.4). ``spin`` is drawn from the seed, 0.05 to 0.4.
* **star_mass** = 1.5 solar masses * largest_clump, to 3 digits (the cloud is taken to weigh
  :data:`CLOUD_MASS` = 1.5 solar masses). **fusion** is yes from 0.08 solar masses on; a lighter
  clump does not shine: its luminosity, frost line and habitable zone are 0 and no planets form.
* **luminosity** = star_mass ^ 3.5, **frost_line** = 2.7 au * sqrt(luminosity), habitable zone
  from 0.95 to 1.7 au * sqrt(luminosity): the rules of level 3.
* **disc_mass** = 10000 earth masses * metallicity (a disc of gas and dust of 10000 earth masses,
  0.03 solar masses, whose metals are its solids). Without metals there are no solids: level 3
  is not run and the world has no planets. ``t_gas`` is drawn from the seed, 1 to 10 Myr.
* **temperature** of a planet = 278.3 K * luminosity^(1/4) / sqrt(orbit in au) + 33 K, rounded to
  a whole kelvin: a black body at that distance plus a greenhouse of 33 K. The luminosity is the
  one of the printed star mass, not the printed luminosity. **mix** is ``ocean`` when that
  allows liquid water (273 to 373 K), else ``rocky``. Level 4 then cools
  :data:`CHEM_ATOMS` atoms of that mix from 4000 K to the planet's temperature in 30 steps.
  ``water`` is its final count of h2o molecules; ``organic`` counts the final molecules with at
  least one carbon-hydrogen bond (:func:`organic_molecules`).
* **monomers** = 700 * organic: the soup level 5 starts from on a planet with ``mix ocean``,
  water and at least one organic molecule. Its other parameters are those of seed ``seed + k``.

Rounding. A hand-off that is a product or a sum of printed numbers (``cooling``, ``star_mass``,
``disc_mass`` and the ``metallicity`` question) is worked out in exact decimals and a 5 rounds up,
the school rule: 1.5 * 0.453 is 0.6795 and gives 0.68, 0.1 + 20 * 0.00725 is 0.245 and gives
0.25. In binary floats such a tie falls to either side by rounding noise (1.5 * 0.453 comes out
as 0.67949...), and 15 of the first 40 worlds have a largest clump that makes one, so the rule
would have had a last digit that cannot be predicted. The hand-offs with a square root
(``luminosity``, ``frost_line``, the habitable zone, ``temperature``) round the float64 result:
their exact values are irrational but for a few inputs, so they have no ties to speak of.

``outcome`` names the first link that is missing, in this order: ``no_metals``, ``no_star``,
``no_planets``, ``no_habitable``, ``no_water``, ``no_organics``, ``sterile`` (no replicator ever
arose), ``extinct`` (none is left on any planet); otherwise ``dominated`` when the planet with
the most replicators at the end (the inner one of two with equally many) is dominated by one
lineage, else ``life``. The ``why`` question lists every link that held, in the order ``metals
star planets habitable water organic replicators survivors`` (``nothing`` when none did).

What is real and what is toy. Real in kind: stars make the metals, more with every generation;
metals let a gas cool; rock and ice are made of metals, so a cloud without them grows no
planets; a planet's temperature falls with the square root of its distance and rises with the
fourth root of the star's light; life as we know it wants liquid water and carbon chemistry.
Toy: every number of every hand-off. The yields are round shares in roughly the solar order;
"generations" of stars do not exist as separate things; the cooling law, the cloud mass, the
disc scale and the monomers per organic molecule are chosen so that the levels meet inside the
ranges they were built for; the black-body temperature ignores albedo and the greenhouse is the
earth's for every planet; level 4's ``organic`` key (chains of more than two carbons) is not used
because the ocean mix never makes one (0 of 30 runs), so this level counts carbon-hydrogen bonds
itself; a "planet with life" is a vessel of level 5 that ends with replicators. The traces of
deuterium, helium-3 and lithium are carried along unchanged (real stars burn deuterium). Each
level's own toy parts are listed in its module. A world is a story the rules tell, not a forecast.

Three seams between the toys show in the numbers and are no findings. (1) The first hour is
level 1 with every seed's own parameters, so the worlds are not regions of one universe: their
hydrogen runs from 0.53 to 0.87 (ours: 0.75). (2) Level 3 calls a planet habitable out to
1.7 au * sqrt(luminosity), an edge a real planet only holds under a thick greenhouse of carbon
dioxide; with the earth's 33 K the surface freezes beyond 1.34 au * sqrt(luminosity). So 85 of
the 190 planets "in the zone" are frozen and get the ``rocky`` mix (level 4 has no ice), and
``no_water`` is the commonest end of a world with metals. (3) The "star" is the largest clump
of 256 particles in units tied to no real cloud; calling the cloud 1.5 solar masses is what
turns a mass fraction into a star, and the rest of the cloud is not followed: the disc of
level 3 is set by the metallicity alone, not by what level 2 left in orbit.

Parameters. :func:`random_params` draws ``generations`` (0 to 3), ``spin`` (0.05 to 0.4) and
``t_gas`` (1 to 10 Myr) from the seed. The sizes are fixed (:data:`DEFAULTS`): ``particles`` 256
(level 2's own), ``atoms`` 20000, ``cloud_mass`` 1.5. ``run(seed, particles=128, atoms=8000)`` is
a faster, coarser world; :func:`check` and :func:`lines` only know the world a seed names
(:func:`rollout`), with the default sizes. A world takes about 2 s, most of it level 2.

Lines (numbers as digits, measured numbers to 3 significant digits; seed 3 verbatim, each one
line in the data):

    world seed 3 params. generations 1. spin 0 point 2 6. t_gas 2 point 1 7.
    world seed 3 timeline. eras nucleo enrichment cloud disc star planets chemistry_1 life_1. outcome life.
    world seed 3 era nucleo. hydrogen 0 point 6 6 3. helium 0 point 3 3 6. traces 1 point 2 4 e minus 4.
    world seed 3 era enrichment. generations 1. hydrogen 0 point 6 4 3. helium 0 point 3 5.
        metallicity 0 point 0 0 6 6 3. carbon 0 point 0 0 1 1 9. oxygen 0 point 0 0 2 9 2. iron 6 point 6 3 e minus 4.
    world seed 3 era cloud. spin 0 point 2 6. cooling 0 point 2 3.
    world seed 3 era disc. flattening 0 point 1 6 3. spiral no. clumps 2. largest_clump 0 point 5 9. collapse_time 1 point 3.
    world seed 3 era star. fusion yes. star_mass 0 point 8 8 5. luminosity 0 point 6 5 2. frost_line 2 point 1 8.
        hz_inner 0 point 7 6 7. hz_outer 1 point 3 7. disc_mass 6 6 point 3. t_gas 2 point 1 7.
    world seed 3 era planets. planets 7. rocky 3. ice 4. gas 0. habitable 1. largest_mass 2 1 point 3.
    world seed 3 era chemistry_1. orbit 0 point 8 8 3. mass 1 point 3 6. temperature 2 9 9. mix ocean.
        water 5 5 3 6. organic 2 0.
    world seed 3 era life_1. monomers 1 4 0 0 0. first_replicator_step 7. peak_replicators 2 7 0 8.
        replicators 2 7 0 8. diversity 1 7. outcome coexist.
    q world seed 3 era planets habitable. a 1.
    q world seed 3 era enrichment silicon. a 3 point 9 8 e minus 4.   (four metals are asked but not in the record)
    q world seed 3 final planets_with_life. a 1.
    q world seed 3 param generations. a 1.
    q world seed 3 timeline. a nucleo enrichment cloud disc star planets chemistry_1 life_1.
    q world seed 3 why life. a metals star planets habitable water organic replicators survivors.
    q world handoff cooling metallicity 0 point 0 0 6 6 3. a 0 point 2 3.
    q world handoff temperature star_mass 0 point 8 8 5 orbit 0 point 8 8 3. a 2 9 9.
    q world handoff monomers organic 2 0. a 1 4 0 0 0.
    world handoff cooling. from metallicity. cooling_primordial 0 point 1. cooling_per_metal 2 0. cooling_max 0 point 6.
    world yields. oxygen 0 point 4 4. carbon 0 point 1 8. neon 0 point 1. iron 0 point 1. nitrogen 0 point 0 6. ...
    q world constant cloud_mass. a 1 point 5.
    q world yield oxygen. a 0 point 4 4.

The hand-off question of a world carries the numbers its eras print, so its answer is the number
the next era prints. One exception: ``handoff metallicity`` starts from the printed hydrogen,
the era from the unrounded one, so the two may differ in the last digit.

Gates. :func:`conserved` holds a world rollout to: every level's own ``conserved`` on its own
rollout; the timeline (six fixed eras, then ``chemistry_k``, then ``life_k``); the mass ledger
(the cloud starts as the nucleo run ends, the mass fractions sum to 1 before and after the
enrichment within 1e-12, the hydrogen burned is the helium and the metals made, the metals are
shared by the yields); every hand-off formula; the planets (level 3 ran exactly when there is a
star and solids, with that star and disc; ice and gas planets lie beyond the frost line unless a
logged merger carried them in, habitable ones are rocky and inside it); one chemistry era per
habitable planet with the temperature, mix, water and organic count of its own level-4 run; a
life era exactly where there is liquid water and an organic molecule; and a summary that
follows from the eras. :func:`check` parses the prompt forms above, replays the chain of the seed
and compares: words, parameters, table values and the counts of the chain (:data:`EXACT_KEYS`)
exactly; counts measured in a level (:data:`MEASURED_KEYS`) as whole numbers within 1 or 5 %,
zero exactly; other numbers within 5 %. A hand-off or table prompt is judged from the prompt
alone.

Plausibility targets, with the values measured on this code over seeds 1 to 300
(``tests/test_cosmos_world.py`` holds each of them on seeds 1 to 40):

    every world       the gate holds; every line is dense, at most 80   300 of 300; 65 to 91 lines a world,
                      tokens and passes ``check``                        73.5 on average, the longest 61 tokens
                      a rerun gives the same eras and levels            40 of 40 (seeds 1 to 40)
    mass              the fractions sum to 1 before and after the       within 1e-12 (the gate's ledger)
                      enrichment; hydrogen burned = helium + metals
    no metals         generations 0: no solids, no planets, no rock     80 of 80 end ``no_metals``
                      with water
    metals            2 generations give about the sun's 0.014          0.0104 to 0.017 (1 generation:
                                                                        0.0053 to 0.0086, 3: 0.0154 to 0.0252)
    the cloud         cooling inside level 2's 0.1 to 0.6               0.1 without metals, else 0.21 to 0.6
    the star          shines; with metals inside level 3's 0.5 to       0.603 to 1.29 (without metals, where
                      1.5 solar masses                                  level 3 does not run, 0.375 to 1.02)
    the disc          solids inside level 3's 10 to 300 earth masses    53.1 to 252
    planets           2 to 10 around a star with metals                 4 to 8
                      gas giants need solids: more after more           0, 0.25 and 1.03 a world after 1, 2
                      generations, none after one                       and 3 generations
                      ice and gas planets beyond the frost line,        all (the gate); of the 220 worlds with
                      habitable ones rocky and inside it                metals 34 have none, 182 one, 4 two
    surfaces          between the edges of the zone, 246 to 319 K       247 to 318 K; liquid water on 105 of 190
                      an ocean makes water and organic molecules,       ocean: 5507 to 5620 h2o, 8 to 25
                      a frozen rock next to none                        organic; rocky: 23 to 46 h2o, 0 or 1
    life              the soup inside level 5's 5000 to 20000           5600 to 17500 monomers
                      at least one world in twenty ends with life,      101 of 300: 37 of 73 after 1
                      none without metals                               generation, 30 of 73 after 2, 34 of 74
                                                                        after 3; 0 of 80 after none
    cost              a world in a few seconds                          1.3 to 3.4 s on seeds 1 to 40 (1.5 s on
                                                                        average in a quiet spell of a machine
                                                                        shared with other work, 2.7 s in a
                                                                        busy one), nine tenths of it level 2

The outcomes of the 300: no_water 81, no_metals 80, dominated 62, life 39, no_habitable 34,
sterile 3, extinct 1. Of the 105 vessels 62 end dominated, 39 coexist, 3 never see a replicator
and 1 goes extinct. A two-fold pattern shows on the way to the disc in 157 worlds. ``no_star``,
``no_planets`` and ``no_organics`` do not occur: a clump of level 2 is always heavy enough, a
disc with solids always grows planets, the ocean mix always makes an organic molecule.
``no_star`` needs a lighter cloud (``cloud_mass``); the other two are kept so that every link
of the chain has its word.

Reproducibility. The chain adds no randomness of its own beyond three draws of
``random.Random(seed)`` (``randint`` and ``uniform``, unchanged since Python 3.2 as far as
memory serves; ``test_known_parameters`` is their canary). What the chain itself computes is the same on every
machine: exact decimals, or + - * / and sqrt on float64, which IEEE 754 fixes to the last bit;
``math.fsum`` for the sums of the ledger; no ``exp``, ``log`` or power and no numpy sum or
sort in anything a line prints. The organic molecules are counted as connected components, a
number that does not depend on how scipy labels them; the habitable planets are put in the
order of their orbits by Python's stable sort; every tie has a rule (a 5 rounds up; of two
planets the inner one). Everything else is each level's (see their modules): level 2 is
chaotic and bit-exact by construction; level 1 uses ``exp`` and ``log``, level 3 powers and
``expm1``, whose last bit may differ between C libraries or numpy builds, which matters only
when a number sits exactly on an edge (a rounding, a threshold, a random draw): rare, not
impossible; levels 4 and 5 draw from the raw PCG64 stream. The numbers here were written with
Python 3.11 and numpy 2.4: ``test_known_world`` and ``test_known_frozen_world`` pin two whole
chains, and if they fail on the training machine the lines must be rebuilt there.
"""

from __future__ import annotations

import copy
import math
import random
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from decimal import ROUND_HALF_UP, Context, Decimal
from functools import lru_cache

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from haishool.cosmos import Rollout, chem, close, gravity, life, nucleo, planets, seeds, summary_lines
from haishool.truth import Line, Verdict, num, parse_num

SIM = "world"

#: a black body at 1 au from the sun, in K: 5772 K * sqrt(6.957e8 m / (2 * 1.495978707e11 m)) = 278.3
BLACK_BODY = round(5772.0 * math.sqrt(6.957e8 / (2.0 * 1.495978707e11)), 1)

#: every constant of the hand-offs, with where it comes from. None is measured for this purpose:
#: a checker can target each row. The code reads ``value`` only.
CONSTANTS: dict[str, dict[str, float | str]] = {
    "helium_yield": {"value": 0.02, "notes": "toy: share of the cloud's hydrogen one generation of stars turns into "
                                             "helium; chosen here"},
    "metal_yield": {"value": 0.01, "notes": "toy: share of the cloud's hydrogen one generation turns into metals; "
                                            "chosen so that two generations give about the sun's 0.014 (from memory)"},
    "cooling_primordial": {"value": 0.1, "notes": "toy: level 2's weakest cooling, for a gas without metals; chosen here"},
    "cooling_per_metal": {"value": 20.0, "notes": "toy: cooling gained per unit metallicity; chosen so that solar "
                                                  "metallicity gives 0.4, the middle of level 2's range"},
    "cooling_max": {"value": 0.6, "notes": "the top of level 2's cooling range"},
    "cloud_mass": {"value": 1.5, "notes": "toy: solar masses of the cloud of level 2 (whose own units are not tied to "
                                          "any cloud); chosen so that the stars fall into level 3's range"},
    "min_star": {"value": 0.08, "notes": "solar masses from which hydrogen burns, about 0.08 (from memory)"},
    "luminosity_power": {"value": 3.5, "notes": "L = M^3.5 for stars like the sun: level 3's rule (from memory there)"},
    "frost_au": {"value": planets.FROST_AU, "notes": "frost line of the sun in au: level 3's constant"},
    "hz_inner_au": {"value": planets.HABITABLE[0], "notes": "inner edge of the sun's habitable zone: level 3's constant"},
    "hz_outer_au": {"value": planets.HABITABLE[1], "notes": "outer edge of the sun's habitable zone: level 3's constant"},
    "disc_scale": {"value": 10000.0, "notes": "toy: earth masses of gas and dust in the disc (0.03 solar masses at "
                                              "333000 earth masses each, from memory); its metals are the solids"},
    "black_body_k": {"value": BLACK_BODY, "notes": "278.3: T_sun * sqrt(R_sun / (2 au)) with 5772 K, 6.957e8 m and "
                                                   "1.495978707e11 m (IAU nominal values, from memory), computed above "
                                                   "and rounded to 0.1 K: a black body at 1 au from the sun"},
    "greenhouse_k": {"value": 33.0, "notes": "the earth's greenhouse warming, about 33 K (from memory); toy: added to "
                                             "a black body without albedo, the same for every planet"},
    "water_min_k": {"value": 273, "notes": "water freezes at 273 K (1 bar)"},
    "water_max_k": {"value": 373, "notes": "water boils at 373 K (1 bar)"},
    "chem_start_k": {"value": 4000, "notes": "level 4's plasma temperature: the surface starts as atoms"},
    "chem_steps": {"value": 30, "notes": "toy: cooling steps of a surface, the middle of level 4's range"},
    "monomers_per_organic": {"value": 700, "notes": "toy: monomers of level 5 per organic molecule of level 4; chosen so "
                                                    "that the 8 to 25 organic molecules of an ocean give 5600 to "
                                                    "17500 monomers, inside level 5's range of 5000 to 20000"},
}
#: share of the metal mass each generation makes of every metal (sums to 1)
YIELDS: dict[str, float] = {"oxygen": 0.44, "carbon": 0.18, "neon": 0.10, "iron": 0.10, "nitrogen": 0.06,
                            "magnesium": 0.06, "silicon": 0.06}
#: where the tables without a ``notes`` column come from, for a checker
NOTES: dict[str, str] = {
    "yields": "toy: round shares chosen here, in roughly the order of the sun's metals by mass (oxygen 0.43, "
              "carbon 0.18, iron 0.10, neon 0.09, nitrogen, silicon and magnesium about 0.05 each; from memory); "
              "real yields differ from star to star and sulphur and the rest are left out",
    "outcomes": "the words are this module's own; they follow from the eras by outcome_of()",
}

HELIUM_YIELD = float(CONSTANTS["helium_yield"]["value"])
METAL_YIELD = float(CONSTANTS["metal_yield"]["value"])
COOLING_PRIMORDIAL = float(CONSTANTS["cooling_primordial"]["value"])
COOLING_PER_METAL = float(CONSTANTS["cooling_per_metal"]["value"])
COOLING_MAX = float(CONSTANTS["cooling_max"]["value"])
CLOUD_MASS = float(CONSTANTS["cloud_mass"]["value"])
MIN_STAR = float(CONSTANTS["min_star"]["value"])
DISC_SCALE = float(CONSTANTS["disc_scale"]["value"])
GREENHOUSE = float(CONSTANTS["greenhouse_k"]["value"])
WATER_MIN_K, WATER_MAX_K = int(CONSTANTS["water_min_k"]["value"]), int(CONSTANTS["water_max_k"]["value"])
CHEM_START_K, CHEM_STEPS = int(CONSTANTS["chem_start_k"]["value"]), int(CONSTANTS["chem_steps"]["value"])
MONOMERS_PER_ORGANIC = int(CONSTANTS["monomers_per_organic"]["value"])
MAX_GENERATIONS = 3
#: atoms of a planet's surface in level 4
CHEM_ATOMS = 20000

#: the sizes of the sub-simulations and the cloud's mass; ``random_params`` draws the rest
DEFAULTS: dict[str, float | int] = {"particles": gravity.N, "atoms": CHEM_ATOMS, "cloud_mass": CLOUD_MASS}
PARAM_KEYS = ("generations", "spin", "t_gas")
SPECIES = ("hydrogen", "helium", "traces", *YIELDS)

FIXED_ERAS = ("nucleo", "enrichment", "cloud", "disc", "star", "planets")
#: era -> the metrics of its record line, in order (``chemistry`` and ``life`` carry a planet number)
ERAS: dict[str, tuple[str, ...]] = {
    "nucleo": ("hydrogen", "helium", "traces"),
    "enrichment": ("generations", "hydrogen", "helium", "metallicity", "carbon", "oxygen", "iron"),
    "cloud": ("spin", "cooling"),
    "disc": ("flattening", "spiral", "clumps", "largest_clump", "collapse_time"),
    "star": ("fusion", "star_mass", "luminosity", "frost_line", "hz_inner", "hz_outer", "disc_mass", "t_gas"),
    "planets": ("planets", "rocky", "ice", "gas", "habitable", "largest_mass"),
    "chemistry": ("orbit", "mass", "temperature", "mix", "water", "organic"),
    "life": ("monomers", "first_replicator_step", "peak_replicators", "replicators", "diversity", "outcome"),
}
#: era -> every metric that can be asked (the enrichment era has four more than its line shows)
ERA_KEYS: dict[str, tuple[str, ...]] = {**ERAS, "enrichment": (*ERAS["enrichment"], "nitrogen", "neon", "magnesium",
                                                               "silicon")}
NUMBERED_ERAS = ("chemistry", "life")
SUMMARY_KEYS = ("hydrogen", "helium", "metallicity", "clumps", "planets", "habitable", "planets_with_water",
                "planets_with_life", "eras", "outcome")
CONDITIONS = ("metals", "star", "planets", "habitable", "water", "organic", "replicators", "survivors")
#: the outcome when a condition is the first that does not hold
MISSING = dict(zip(CONDITIONS, ("no_metals", "no_star", "no_planets", "no_habitable", "no_water", "no_organics",
                                "sterile", "extinct")))
OUTCOMES = (*MISSING.values(), "life", "dominated")
#: counts of the chain and identities: compared exactly
EXACT_KEYS = frozenset({"generations", "clumps", "planets", "rocky", "ice", "gas", "habitable", "planets_with_water",
                        "planets_with_life", "eras", "first_replicator_step"})
#: counts measured inside level 4 or 5: a whole number within 1 or 5 percent, zero exactly
MEASURED_KEYS = frozenset({"water", "organic", "monomers", "peak_replicators", "replicators", "diversity"})

KEYS: dict[str, str] = {
    "generations": "generations of earlier stars that enriched the cloud, 0 to 3 (parameter)",
    "spin": "rotational energy of the cloud over its potential energy, 0.05 to 0.4 (parameter; level 2's spin)",
    "t_gas": "time at which the disc gas is gone, Myr, 1 to 10 (parameter; level 3's t_gas)",
    "hydrogen": "mass fraction of hydrogen: after the first hour (era nucleo), of the cloud (era enrichment, final)",
    "helium": "mass fraction of helium-4: after the first hour (era nucleo), of the cloud (era enrichment, final)",
    "traces": "mass fraction of deuterium, helium-3 and lithium-7 together: 1 - hydrogen - helium of era nucleo",
    "metallicity": "mass fraction of the cloud in metals, the sum of the seven below",
    **{el: f"mass fraction of {el} in the cloud, {share} of the metallicity" for el, share in YIELDS.items()},
    "cooling": "level 2's cooling of the cloud: min(0.6, 0.1 + 20 * metallicity), to 2 decimals, a 5 rounding up",
    "flattening": "flattening of the cloud's last state in level 2 (1 sphere, small disc)",
    "spiral": "yes when any state of the level-2 run showed a two-fold pattern (its spiral_ever)",
    "clumps": "clumps in the last state of the level-2 run (count)",
    "largest_clump": "mass fraction of the biggest clump in the last state: the later star",
    "collapse_time": "first time, in free-fall times, the cloud's half-mass radius is below half its start, or never",
    "fusion": "yes when the star reaches 0.08 solar masses and shines, else no",
    "star_mass": "mass of the star, solar masses: 1.5 * largest_clump, to 3 digits, a 5 rounding up",
    "luminosity": "light of the star, solar luminosities: star_mass ^ 3.5 (0 without fusion)",
    "frost_line": "frost line, au: 2.7 * sqrt(luminosity)",
    "hz_inner": "inner edge of the habitable zone, au: 0.95 * sqrt(luminosity)",
    "hz_outer": "outer edge of the habitable zone, au: 1.7 * sqrt(luminosity)",
    "disc_mass": "solids in the disc, earth masses: 10000 * metallicity",
    "planets": "planets at the end of the level-3 run (count; 0 when it did not run)",
    "rocky": "rocky planets (count)",
    "ice": "ice giants (count)",
    "gas": "gas giants (count)",
    "habitable": "rocky planets inside the habitable zone (count); each gets a chemistry era",
    "largest_mass": "mass of the heaviest planet, earth masses",
    "orbit": "orbit of the planet, au",
    "mass": "mass of the planet, earth masses",
    "temperature": "surface temperature in K: 278.3 * luminosity^(1/4) / sqrt(orbit) + 33, a whole number",
    "mix": "element mix of the surface in level 4: ocean when 273 <= temperature <= 373, else rocky",
    "water": "h2o molecules at the end of the level-4 run (count)",
    "organic": "molecules with at least one carbon-hydrogen bond at the end of the level-4 run (count)",
    "monomers": "free monomers level 5 starts with: 700 * organic",
    "first_replicator_step": "step of the level-5 run at which the first replicator appeared, minus 1 if none did",
    "peak_replicators": "most replicators at any step of the level-5 run",
    "replicators": "replicators at the end of the level-5 run",
    "diversity": "distinct replicator sequences at the end of the level-5 run",
    "outcome": "of a life era: level 5's none, coexist, dominated or extinct; final: the first missing link "
               "(no_metals, no_star, no_planets, no_habitable, no_water, no_organics, sterile, extinct) or life / dominated",
    "planets_with_water": "habitable planets with mix ocean and at least one h2o molecule (count)",
    "planets_with_life": "planets whose level-5 run ends with replicators (count)",
    "eras": "number of eras of the world, 6 plus one per chemistry and life era; in the timeline line: their names",
    "why": "the links of the chain that held, of: " + " ".join(CONDITIONS) + " (nothing when none did)",
    "handoff": "a hand-off formula asked with its inputs: " + ", ".join(
        ("metallicity", "cooling", "star_mass", "fusion", "luminosity", "frost_line", "disc_mass", "temperature",
         "mix", "monomers")),
    "constant": "a constant of the hand-offs, by name (see CONSTANTS)",
    "yield": "share of the metal mass a generation of stars makes of one metal",
}


@dataclass
class WorldRollout(Rollout):
    """A :class:`Rollout` whose steps are the eras, with the rollout of every level that ran."""
    #: ``nucleo``, ``gravity``, ``planets`` (when it ran), ``chemistry_1`` ..., ``life_1`` ...
    levels: dict[str, Rollout] = field(default_factory=dict)
    #: mass fractions at full precision before and after the enrichment: the gate's ledger
    primordial: dict[str, float] = field(default_factory=dict)
    cloud: dict[str, float] = field(default_factory=dict)

    def value(self, text: float | int | str) -> str:
        return dense_value(text)

    def era(self, name: str) -> dict | None:
        return next((s for s in self.steps if s["era"] == name), None)

    def numbered(self, base: str) -> list[dict]:
        """The ``chemistry_k`` or ``life_k`` eras, in order."""
        return [s for s in self.steps if s["era"].startswith(base + "_")]


def _sig3(x: float) -> float:
    """A measured number to 3 significant digits (the float as it is, correctly rounded)."""
    return float(f"{x:.3g}")


#: exact decimal arithmetic for the hand-offs, whatever the caller's decimal context is set to:
#: 60 digits hold every product of two or three printed numbers
_EXACT = Context(prec=60, rounding=ROUND_HALF_UP)


def _dec(x: float | int) -> Decimal:
    """A number as the decimal it prints as: ``0.673`` is 673 thousandths here, not the nearest
    binary fraction (``repr`` is the shortest text that gives the float back)."""
    return Decimal(x) if isinstance(x, int) else Decimal(repr(float(x)))


def _half_up(d: Decimal, sig: int = 3) -> float:
    """``d`` to ``sig`` significant digits, a 5 rounding up: the school rule, on exact decimals.

    >>> _half_up(Decimal("1.0095")), _half_up(Decimal("0.5865")), _half_up(Decimal("0.99951")), _half_up(Decimal(0))
    (1.01, 0.587, 1.0, 0.0)
    """
    if d == 0:
        return 0.0
    return float(d.quantize(Decimal((0, (1,), d.adjusted() - sig + 1)), rounding=ROUND_HALF_UP, context=_EXACT))


def dense_value(value: float | int | str) -> str:
    """A metric as it stands in a line: a word, an exact count, or a number to 3 significant digits.

    >>> dense_value(0.01478), dense_value(148.0), dense_value(7), dense_value("ocean")
    ('0 point 0 1 4 8', '1 4 8', '7', 'ocean')
    """
    if isinstance(value, str):
        return value
    if isinstance(value, int):
        return num(value)
    return num(_sig3(value), sig=3)


def table_value(value: float | int | str) -> str:
    """A parameter or a constant of the tables as it was put in, to at most 5 digits (``278.3``
    stays ``2 7 8 point 3``; a measured number would be cut to 3 by :func:`dense_value`)."""
    if isinstance(value, (str, int)):
        return dense_value(value)
    return num(float(value), sig=5)


# ---------------------------------------------------------------------------------------------
# the hand-offs: plain formulas on the numbers the lines print


def enrich(hydrogen: float, helium: float, generations: int) -> dict[str, float]:
    """Mass fractions of the cloud after ``generations`` of earlier stars (see the module docstring).

    >>> c = enrich(0.75, 0.25, 2)
    >>> round(c["hydrogen"], 6), round(c["helium"], 6), round(sum(c[el] for el in YIELDS), 6)
    (0.705675, 0.27955, 0.014775)
    """
    cloud = {"hydrogen": hydrogen, "helium": helium, "traces": 1.0 - hydrogen - helium, **dict.fromkeys(YIELDS, 0.0)}
    for _ in range(generations):
        h = cloud["hydrogen"]
        cloud["hydrogen"] = h - (HELIUM_YIELD + METAL_YIELD) * h
        cloud["helium"] += HELIUM_YIELD * h
        for el, share in YIELDS.items():
            cloud[el] += share * METAL_YIELD * h
    return cloud


def metallicity_of(hydrogen: float, generations: int) -> float:
    """Metal mass fraction after ``generations``, from the hydrogen the first hour left (not rounded)."""
    z, h = 0.0, hydrogen
    for _ in range(generations):
        z += METAL_YIELD * h
        h -= (HELIUM_YIELD + METAL_YIELD) * h
    return z


def metallicity_printed(hydrogen: float, generations: int) -> float:
    """:func:`metallicity_of` as the hand-off question asks it: from the printed hydrogen, in exact
    decimals, to 3 digits.

    >>> metallicity_printed(0.75, 2), metallicity_printed(0.663, 1), metallicity_printed(0.75, 0)
    (0.0148, 0.00663, 0.0)
    """
    z, h = Decimal(0), _dec(hydrogen)
    for _ in range(generations):
        z = _EXACT.add(z, _EXACT.multiply(_dec(METAL_YIELD), h))
        h = _EXACT.subtract(h, _EXACT.multiply(_EXACT.add(_dec(HELIUM_YIELD), _dec(METAL_YIELD)), h))
    return _half_up(z)


def cooling_of(metallicity: float) -> float:
    """``min(0.6, 0.1 + 20 * metallicity)`` to 2 decimals, in exact decimals, a 5 rounding up.

    >>> cooling_of(0.00725), cooling_of(0.00663), cooling_of(0.0), cooling_of(0.03)
    (0.25, 0.23, 0.1, 0.6)
    """
    gained = _EXACT.multiply(_dec(COOLING_PER_METAL), _dec(metallicity))
    c = min(_dec(COOLING_MAX), _EXACT.add(_dec(COOLING_PRIMORDIAL), gained))
    return float(c.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP, context=_EXACT))


def star_mass_of(largest_clump: float, cloud_mass: float = CLOUD_MASS) -> float:
    """``cloud_mass * largest_clump`` to 3 digits, in exact decimals, a 5 rounding up.

    >>> star_mass_of(0.59), star_mass_of(0.673), star_mass_of(0.453), star_mass_of(0.391)
    (0.885, 1.01, 0.68, 0.587)
    """
    return _half_up(_EXACT.multiply(_dec(cloud_mass), _dec(largest_clump)))


def fusion_of(star_mass: float) -> str:
    return "yes" if star_mass >= MIN_STAR else "no"


def luminosity_of(star_mass: float) -> float:
    """``star_mass ^ 3.5`` as products and a square root (the same bits everywhere); 0 without fusion."""
    if star_mass < MIN_STAR:
        return 0.0
    return star_mass * star_mass * star_mass * math.sqrt(star_mass)


def frost_line_of(star_mass: float) -> float:
    return planets.FROST_AU * math.sqrt(luminosity_of(star_mass))


def disc_mass_of(metallicity: float) -> float:
    return _half_up(_EXACT.multiply(_dec(DISC_SCALE), _dec(metallicity)))


def temperature_of(star_mass: float, orbit: float) -> int:
    """Surface temperature in whole kelvin of a planet at ``orbit`` au (a half rounds up)."""
    kelvin = BLACK_BODY * math.sqrt(math.sqrt(luminosity_of(star_mass))) / math.sqrt(orbit) + GREENHOUSE
    return math.floor(kelvin + 0.5)


def mix_of(temperature: int) -> str:
    return "ocean" if WATER_MIN_K <= temperature <= WATER_MAX_K else "rocky"


def monomers_of(organic: int) -> int:
    return MONOMERS_PER_ORGANIC * organic


@dataclass(frozen=True)
class Handoff:
    inputs: tuple[str, ...]
    rule: Callable[..., float | int | str]
    #: the constants the rule uses, shown in its record line
    constants: tuple[str, ...]


HANDOFFS: dict[str, Handoff] = {
    "metallicity": Handoff(("hydrogen", "generations"), metallicity_printed, ("helium_yield", "metal_yield")),
    "cooling": Handoff(("metallicity",), cooling_of, ("cooling_primordial", "cooling_per_metal", "cooling_max")),
    "star_mass": Handoff(("largest_clump",), star_mass_of, ("cloud_mass",)),
    "fusion": Handoff(("star_mass",), fusion_of, ("min_star",)),
    "luminosity": Handoff(("star_mass",), luminosity_of, ("luminosity_power", "min_star")),
    "frost_line": Handoff(("star_mass",), frost_line_of, ("frost_au", "luminosity_power")),
    "disc_mass": Handoff(("metallicity",), disc_mass_of, ("disc_scale",)),
    "temperature": Handoff(("star_mass", "orbit"), temperature_of, ("black_body_k", "greenhouse_k", "luminosity_power")),
    "mix": Handoff(("temperature",), mix_of, ("water_min_k", "water_max_k")),
    "monomers": Handoff(("organic",), monomers_of, ("monomers_per_organic",)),
}
#: hand-off input -> (low, high, whole number?) a prompt may name; the practice questions draw from PRACTICE
INPUT_RANGES: dict[str, tuple[float, float, bool]] = {
    "hydrogen": (0.0, 1.0, False), "generations": (0, MAX_GENERATIONS, True), "metallicity": (0.0, 1.0, False),
    "largest_clump": (0.0, 1.0, False), "star_mass": (0.0, 100.0, False), "orbit": (0.01, 1000.0, False),
    "temperature": (0, 100000, True), "organic": (0, 100000, True),
}
PRACTICE: dict[str, tuple[float, float]] = {
    "hydrogen": (0.5, 0.9), "generations": (0, MAX_GENERATIONS), "metallicity": (0.0, 0.03),
    "largest_clump": (0.02, 0.9), "star_mass": (0.03, 1.4), "orbit": (0.3, 5.0), "temperature": (150, 450),
    "organic": (0, 40),
}
#: the fusion question draws its star around the threshold, or nearly every answer would be yes
PRACTICE_OF: dict[str, dict[str, tuple[float, float]]] = {"fusion": {"star_mass": (0.02, 0.2)}}
#: practice hand-off questions :func:`generate` adds per world
PRACTICE_PER_SEED = 20


def organic_molecules(pool: chem.Pool) -> int:
    """Molecules of a level-4 pool that hold at least one carbon-hydrogen bond."""
    c, h = chem.ELEMENTS.index("c"), chem.ELEMENTS.index("h")
    a, b = pool.el[pool.bond_a], pool.el[pool.bond_b]
    ch = ((a == c) & (b == h)) | ((a == h) & (b == c))
    if not ch.any():
        return 0
    n = len(pool.el)
    graph = coo_matrix((np.ones(len(pool.bond_a)), (pool.bond_a, pool.bond_b)), shape=(n, n))
    _, labels = connected_components(graph, directed=False)
    return int(len(np.unique(labels[pool.bond_a[ch]])))


def _final_pool(level: Rollout) -> chem.Pool:
    """The pool a level-4 rollout ends with (a second pass: the rollout keeps numbers only)."""
    pool = None
    for _, _, pool in chem.cool(level.seed, level.params):
        pass
    return pool


def habitable_bodies(level: Rollout) -> list[dict]:
    """The rocky planets inside the habitable zone at the end of a level-3 run, innermost first."""
    lo, hi = (edge * math.sqrt(float(level.params["luminosity"])) for edge in planets.HABITABLE)
    found = [b for b in level.bodies[-1]
             if b["solid"] + b["gas"] >= planets.PLANET_MIN and b["type"] == "rocky" and lo <= b["a"] <= hi]
    return sorted(found, key=lambda b: b["a"])


# ---------------------------------------------------------------------------------------------
# the chain


def random_params(rng: random.Random) -> dict[str, float | int]:
    """``generations`` 0 to 3, ``spin`` 0.05 to 0.4 and ``t_gas`` 1 to 10 Myr (both to 2 decimals)."""
    return {"generations": rng.randint(0, MAX_GENERATIONS), "spin": round(rng.uniform(0.05, 0.4), 2),
            "t_gas": round(rng.uniform(1.0, 10.0), 2)}


def _is_seed(seed: object) -> bool:
    return isinstance(seed, int) and not isinstance(seed, bool) and seed >= 0


def _params(seed: int, given: dict) -> dict[str, float | int]:
    if not _is_seed(seed):
        raise ValueError(f"a world seed is a whole number from 0, not {seed!r}")
    unknown = set(given) - set(PARAM_KEYS) - set(DEFAULTS)
    if unknown:
        raise TypeError(f"unknown parameter(s) {sorted(unknown)}; known: {sorted((*PARAM_KEYS, *DEFAULTS))}")
    p = {**random_params(random.Random(seed)), **DEFAULTS, **given}
    for key in ("generations", "particles", "atoms"):
        if isinstance(p[key], bool) or int(p[key]) != p[key]:
            raise ValueError(f"{key} must be a whole number, not {p[key]!r}")
        p[key] = int(p[key])
    for key in ("spin", "t_gas", "cloud_mass"):
        p[key] = float(p[key])
    if not 0 <= p["generations"] <= MAX_GENERATIONS:
        raise ValueError(f"generations must be 0 to {MAX_GENERATIONS}, not {p['generations']}")
    if not (math.isfinite(p["spin"]) and p["spin"] >= 0 and math.isfinite(p["t_gas"]) and p["t_gas"] >= 0
            and math.isfinite(p["cloud_mass"]) and p["cloud_mass"] > 0 and p["atoms"] >= 1000):
        raise ValueError(f"bad parameters: {p}")
    return p


def star_era(largest_clump: float, metallicity: float, t_gas: float, cloud_mass: float = CLOUD_MASS) -> dict:
    """The ``star`` era from the numbers the eras before it print."""
    m = star_mass_of(largest_clump, cloud_mass)
    root = math.sqrt(luminosity_of(m))
    return {"era": "star", "fusion": fusion_of(m), "star_mass": m, "luminosity": _sig3(luminosity_of(m)),
            "frost_line": _sig3(frost_line_of(m)), "hz_inner": _sig3(planets.HABITABLE[0] * root),
            "hz_outer": _sig3(planets.HABITABLE[1] * root), "disc_mass": disc_mass_of(metallicity), "t_gas": t_gas}


def run(seed: int, **params: float | int) -> WorldRollout:
    """One world: the eras of ``seed``. Parameters not given are the seed's own (:func:`random_params`)
    or the default sizes; ``run(seed)`` is what :func:`check` replays."""
    p = _params(seed, params)
    levels: dict[str, Rollout] = {}

    first = nucleo.rollout(seed)
    levels["nucleo"] = first
    x, y = float(first.summary["hydrogen"]), float(first.summary["helium"])
    primordial, cloud = enrich(x, y, 0), enrich(x, y, p["generations"])
    z = _sig3(math.fsum(cloud[el] for el in YIELDS))
    steps: list[dict] = [
        {"era": "nucleo", **{k: _sig3(primordial[k]) for k in ERAS["nucleo"]}},
        {"era": "enrichment", "generations": p["generations"], "hydrogen": _sig3(cloud["hydrogen"]),
         "helium": _sig3(cloud["helium"]), "metallicity": z, **{el: _sig3(cloud[el]) for el in YIELDS}},
        {"era": "cloud", "spin": p["spin"], "cooling": cooling_of(z)},
    ]

    collapse = gravity.simulation().run(seed, spin=p["spin"], cooling=steps[-1]["cooling"], n=p["particles"])
    levels["gravity"] = collapse
    steps.append({"era": "disc", "flattening": collapse.summary["flattening"], "spiral": collapse.summary["spiral_ever"],
                  "clumps": collapse.summary["clumps"], "largest_clump": collapse.summary["largest_clump"],
                  "collapse_time": collapse.summary["collapse_time"]})
    star = star_era(steps[-1]["largest_clump"], z, p["t_gas"], p["cloud_mass"])
    steps.append(star)

    system = None
    if star["fusion"] == "yes" and star["disc_mass"] > 0:
        system = planets.simulation().run(seed, m_star=star["star_mass"], disc_mass=star["disc_mass"], t_gas=p["t_gas"])
        levels["planets"] = system
        steps.append({"era": "planets", **{k: system.summary[k] for k in ("planets", "rocky", "ice", "gas", "habitable")},
                      "largest_mass": _sig3(system.summary["largest_mass"])})
    else:
        steps.append({"era": "planets", "planets": 0, "rocky": 0, "ice": 0, "gas": 0, "habitable": 0, "largest_mass": 0.0})

    surfaces: list[dict] = []
    for k, body in enumerate(habitable_bodies(system) if system else [], start=1):
        orbit = _sig3(body["a"])
        temperature = temperature_of(star["star_mass"], orbit)
        level = chem.simulation().run(seed + k, mix=mix_of(temperature), t_start=CHEM_START_K, t_end=temperature,
                                      steps=CHEM_STEPS, atoms=p["atoms"])
        levels[f"chemistry_{k}"] = level
        surfaces.append({"era": f"chemistry_{k}", "orbit": orbit, "mass": _sig3(body["solid"] + body["gas"]),
                         "temperature": temperature, "mix": level.summary["mix"], "water": level.summary["h2o"],
                         "organic": organic_molecules(_final_pool(level))})
    steps += surfaces
    for k, surface in enumerate(surfaces, start=1):
        if surface["mix"] == "ocean" and surface["water"] > 0 and surface["organic"] > 0:
            level = life.run(seed + k, monomers=monomers_of(surface["organic"]))
            levels[f"life_{k}"] = level
            steps.append({"era": f"life_{k}", "monomers": level.params["monomers"],
                          **{key: level.summary[key] for key in ERAS["life"][1:]}})
    return WorldRollout(SIM, seed, p, steps, summarise(steps), levels, primordial, cloud)


def conditions(steps: list[dict]) -> list[str]:
    """The links of the chain that held, in the order of :data:`CONDITIONS`."""
    era = {s["era"]: s for s in steps}
    surfaces = [s for s in steps if s["era"].startswith("chemistry_")]
    vessels = [s for s in steps if s["era"].startswith("life_")]
    watery = [s for s in surfaces if s["mix"] == "ocean" and s["water"] > 0]
    held = {
        "metals": era["enrichment"]["metallicity"] > 0,
        "star": era["star"]["fusion"] == "yes",
        "planets": era["planets"]["planets"] > 0,
        "habitable": era["planets"]["habitable"] > 0,
        "water": bool(watery),
        "organic": any(s["organic"] > 0 for s in watery),
        "replicators": any(s["first_replicator_step"] >= 0 for s in vessels),
        "survivors": any(s["replicators"] > 0 for s in vessels),
    }
    return [c for c in CONDITIONS if held[c]]


def outcome_of(steps: list[dict]) -> str:
    """The first missing link, or how the planet with the most replicators ends (of two planets
    with equally many, the inner one)."""
    held = conditions(steps)
    for condition in CONDITIONS:
        if condition not in held:
            return MISSING[condition]
    vessels = [s for s in steps if s["era"].startswith("life_")]
    most = max(s["replicators"] for s in vessels)
    best = next(s for s in vessels if s["replicators"] == most)
    return "dominated" if best["outcome"] == "dominated" else "life"


def summarise(steps: list[dict]) -> dict[str, float | int | str]:
    era = {s["era"]: s for s in steps}
    surfaces = [s for s in steps if s["era"].startswith("chemistry_")]
    vessels = [s for s in steps if s["era"].startswith("life_")]
    return {"hydrogen": era["enrichment"]["hydrogen"], "helium": era["enrichment"]["helium"],
            "metallicity": era["enrichment"]["metallicity"], "clumps": era["disc"]["clumps"],
            "planets": era["planets"]["planets"], "habitable": era["planets"]["habitable"],
            "planets_with_water": sum(1 for s in surfaces if s["mix"] == "ocean" and s["water"] > 0),
            "planets_with_life": sum(1 for s in vessels if s["replicators"] > 0),
            "eras": len(steps), "outcome": outcome_of(steps)}


@lru_cache(maxsize=256)
def _replay(seed: int) -> WorldRollout:
    """The gate's own copy of a seed's world; never handed out."""
    return run(seed)


def rollout(seed: int) -> WorldRollout:
    """The world a seed names in the lines and in :func:`check` (cached; the caller gets a copy)."""
    if not _is_seed(seed):
        raise ValueError(f"a world seed is a whole number from 0, not {seed!r}")
    return copy.deepcopy(_replay(seed))


def canonical(r: Rollout) -> bool:
    """True when ``r`` is the world its seed names, the only one :func:`check` can replay."""
    return r.sim == SIM and _is_seed(r.seed) and r.params == {**random_params(random.Random(r.seed)), **DEFAULTS}


# ---------------------------------------------------------------------------------------------
# the gate: every level's own, and every hand-off


def _base(era: str) -> str:
    return era.split("_")[0] if era.split("_")[0] in NUMBERED_ERAS else era


def _timeline_holds(r: WorldRollout) -> str:
    names = [s["era"] for s in r.steps]
    surfaces = [n for n in names if n.startswith("chemistry_")]
    vessels = [n for n in names if n.startswith("life_")]
    if tuple(names[:len(FIXED_ERAS)]) != FIXED_ERAS or names[len(FIXED_ERAS):] != surfaces + vessels:
        return f"the eras are out of order: {' '.join(names)}"
    if surfaces != [f"chemistry_{k}" for k in range(1, len(surfaces) + 1)]:
        return "the chemistry eras are not numbered 1, 2, ..."
    numbers = [int(n.split("_")[1]) for n in vessels]
    if numbers != sorted(set(numbers)) or any(f"chemistry_{k}" not in surfaces for k in numbers):
        return "a life era without its chemistry era, or out of order"
    for s in r.steps:
        if set(s) != {"era", *ERA_KEYS[_base(s["era"])]}:
            return f"era {s['era']} does not hold its metrics"
    wanted = {"nucleo", "gravity", *surfaces, *vessels} | ({"planets"} if "planets" in r.levels else set())
    if set(r.levels) != wanted:
        return "the levels kept do not match the eras"
    return ""


def _levels_hold(r: WorldRollout) -> str:
    gates = {"nucleo": nucleo.conserved, "gravity": gravity.simulation().conserved,
             "planets": planets.simulation().conserved, "chem": chem.simulation().conserved,
             "life": life.simulation().conserved}
    for name, level in r.levels.items():
        verdict = gates[level.sim](level)
        if not verdict.ok:
            return f"level {name}: {verdict.reason}"
    return ""


def _mass_holds(r: WorldRollout) -> str:
    before, after, first = r.primordial, r.cloud, r.levels["nucleo"]
    g = int(r.params["generations"])
    if set(before) != set(SPECIES) or set(after) != set(SPECIES):
        return "the mass ledger does not list every species"
    for name, ledger in (("before", before), ("after", after)):
        if abs(math.fsum(ledger.values()) - 1.0) > 1e-12:
            return f"the mass fractions {name} the enrichment sum to {math.fsum(ledger.values())!r}, not 1"
        if min(ledger.values()) < -1e-12:
            return f"a negative mass fraction {name} the enrichment"
    if first.seed != r.seed or first.steps != nucleo.rollout(r.seed).steps:
        return "the nucleo level is not the seed's own run"
    if before["hydrogen"] != first.summary["hydrogen"] or before["helium"] != first.summary["helium"] \
            or any(before[el] for el in YIELDS):
        return "the cloud does not start as the nucleo run ends"
    burned = before["hydrogen"] - after["hydrogen"]
    z = math.fsum(after[el] for el in YIELDS)
    if abs(after["hydrogen"] - before["hydrogen"] * (1.0 - HELIUM_YIELD - METAL_YIELD) ** g) > 1e-12:
        return f"hydrogen after {g} generations does not follow from the yields"
    if abs(burned - (after["helium"] - before["helium"]) - z) > 1e-12 or after["traces"] != before["traces"]:
        return "the hydrogen burned is not the helium and the metals made"
    if abs(z - burned * METAL_YIELD / (HELIUM_YIELD + METAL_YIELD)) > 1e-12 \
            or any(abs(after[el] - share * z) > 1e-12 for el, share in YIELDS.items()):
        return "the metals are not shared out by the yields"
    era = r.era("nucleo")
    if any(era[k] != _sig3(before[k]) for k in ERAS["nucleo"]):
        return "era nucleo does not show the nucleo run's end"
    era = r.era("enrichment")
    if era["generations"] != g or era["metallicity"] != _sig3(z) \
            or any(era[k] != _sig3(after[k]) for k in ("hydrogen", "helium", *YIELDS)):
        return "era enrichment does not show the ledger"
    return ""


def _cloud_follows(r: WorldRollout) -> str:
    cloud, disc, level = r.era("cloud"), r.era("disc"), r.levels["gravity"]
    if cloud["spin"] != r.params["spin"] or cloud["cooling"] != cooling_of(r.era("enrichment")["metallicity"]):
        return "the cooling does not follow from the metallicity, or the spin is not the parameter"
    if level.seed != r.seed or level.params["spin"] != cloud["spin"] or level.params["cooling"] != cloud["cooling"] \
            or level.params["n"] != r.params["particles"]:
        return "the gravity level did not run with the cloud's spin and cooling"
    last = level.steps[-1]
    halved = [s["time"] for s in level.steps if s["radius"] < 0.5 * level.steps[0]["radius"]]
    want = {"flattening": last["flattening"], "spiral": "yes" if any(s["spiral"] == "yes" for s in level.steps) else "no",
            "clumps": last["clumps"], "largest_clump": last["largest_clump"],
            "collapse_time": halved[0] if halved else "never"}
    if any(disc[k] != v for k, v in want.items()):
        return "era disc does not show how the gravity level ended"
    return ""


def _star_follows(r: WorldRollout) -> str:
    want = star_era(r.era("disc")["largest_clump"], r.era("enrichment")["metallicity"], float(r.params["t_gas"]),
                    float(r.params["cloud_mass"]))
    if r.era("star") != want:
        return "era star does not follow from the largest clump and the metallicity"
    return ""


def _planets_follow(r: WorldRollout) -> str:
    star, era, level = r.era("star"), r.era("planets"), r.levels.get("planets")
    if (level is not None) != (star["fusion"] == "yes" and star["disc_mass"] > 0):
        return "the planets level must run exactly when there is a star and solids"
    if level is None:
        if any(era[k] != 0 for k in ERAS["planets"]) or r.numbered("chemistry"):
            return "planets or chemistry without a planets level"
        return ""
    if level.seed != r.seed or level.params["m_star"] != star["star_mass"] \
            or level.params["disc_mass"] != star["disc_mass"] or level.params["t_gas"] != star["t_gas"]:
        return "the planets level did not run with the star and the disc of era star"
    if not close(level.params["luminosity"], luminosity_of(star["star_mass"]), rel=1e-9) \
            or not close(level.params["r_frost"], frost_line_of(star["star_mass"]), rel=1e-9):
        return "luminosity or frost line differ from the planets level's"
    last = level.steps[-1]
    if era["habitable"] != level.summary["habitable"] or era["largest_mass"] != _sig3(last["largest_mass"]) \
            or any(era[k] != last[k] for k in ("planets", "rocky", "ice", "gas")):
        return "era planets does not show how the planets level ended"
    r_frost = level.params["r_frost"]
    merged = {e["kept"] for e in level.events}
    for b in level.bodies[-1]:
        if b["solid"] + b["gas"] < planets.PLANET_MIN or b["a"] > r_frost:
            continue
        if b["type"] == "ice" or (b["type"] == "gas" and b["id"] not in merged):
            return f"a {b['type']} planet inside the frost line at {b['a']!r} au"
    worlds = habitable_bodies(level)
    if len(worlds) != era["habitable"] or any(b["a"] >= r_frost for b in worlds):
        return "the habitable planets are not the rocky ones in the zone, inside the frost line"
    return ""


def _chemistry_follows(r: WorldRollout) -> str:
    surfaces, level = r.numbered("chemistry"), r.levels.get("planets")
    worlds = habitable_bodies(level) if level is not None else []
    if len(surfaces) != len(worlds):
        return "there must be one chemistry era per habitable planet"
    star_mass = r.era("star")["star_mass"]
    for k, (era, body) in enumerate(zip(surfaces, worlds), start=1):
        run4 = r.levels[era["era"]]
        if era["orbit"] != _sig3(body["a"]) or era["mass"] != _sig3(body["solid"] + body["gas"]):
            return f"{era['era']} is not the planet at {body['a']!r} au"
        if era["temperature"] != temperature_of(star_mass, era["orbit"]) or era["mix"] != mix_of(era["temperature"]):
            return f"{era['era']}: temperature or mix do not follow from the star and the orbit"
        want = {"mix": era["mix"], "t_start": CHEM_START_K, "t_end": era["temperature"], "steps": CHEM_STEPS,
                "atoms": r.params["atoms"]}
        if run4.seed != r.seed + k or run4.params != want:
            return f"{era['era']}: the chem level did not run with the planet's temperature and mix"
        if era["water"] != run4.summary["h2o"] or era["organic"] != organic_molecules(_final_pool(run4)) \
                or era["organic"] > run4.summary["molecules"]:
            return f"{era['era']}: water or organic are not the chem level's"
    return ""


def _life_follows(r: WorldRollout) -> str:
    for k, surface in enumerate(r.numbered("chemistry"), start=1):
        era, level = r.era(f"life_{k}"), r.levels.get(f"life_{k}")
        ready = surface["mix"] == "ocean" and surface["water"] > 0 and surface["organic"] > 0
        if (era is not None) != ready:
            return f"life_{k}: life must run exactly where there is liquid water and an organic molecule"
        if era is None:
            continue
        own = {**life.random_params(random.Random(r.seed + k)), "monomers": monomers_of(surface["organic"])}
        if level.seed != r.seed + k or level.params != own or era["monomers"] != own["monomers"]:
            return f"life_{k}: the life level did not start from the planet's organic molecules"
        if any(era[key] != level.summary[key] for key in ERAS["life"][1:]):
            return f"life_{k} does not show how the life level ended"
    return ""


def _summary_follows(r: WorldRollout) -> str:
    s = r.summary
    if s != summarise(r.steps):
        return "the summary does not follow from the eras"
    if not s["planets_with_life"] <= s["planets_with_water"] <= s["habitable"] <= r.era("planets")["rocky"] <= s["planets"]:
        return "the counts of the summary do not nest"
    if s["metallicity"] == 0 and s["planets"]:
        return "planets without metals"
    return ""


def conserved(r: Rollout) -> Verdict:
    """The chain's gate (see the module docstring for the list)."""
    if r.sim != SIM or not isinstance(r, WorldRollout) or not r.steps:
        return Verdict(False, None, "not a world rollout")
    try:
        for holds in (_timeline_holds, _levels_hold, _mass_holds, _cloud_follows, _star_follows, _planets_follow,
                      _chemistry_follows, _life_follows, _summary_follows):
            reason = holds(r)
            if reason:
                return Verdict(False, None, reason)
    except (KeyError, IndexError, TypeError, AttributeError, ValueError) as e:
        return Verdict(False, None, f"malformed world rollout: {type(e).__name__}: {e}")
    return Verdict(True, None, "every level's gate and every hand-off hold")


# ---------------------------------------------------------------------------------------------
# prompts and their judgement

_N = r"(0|[1-9](?: \d)*)"
_ERA_Q = re.compile(rf"^world seed {_N} era ([a-z]+)(?:_([1-9]\d*))? ([a-z0-9_]+)$")
_SEED_Q = re.compile(rf"^world seed {_N} (final|param|why) ([a-z0-9_]+)$")
_TIMELINE_Q = re.compile(rf"^world seed {_N} timeline$")
_TABLE_Q = re.compile(r"^world (constant|yield) ([a-z0-9_]+)$")
_HANDOFF_Q = re.compile(r"^world handoff ([a-z_]+) ([a-z0-9_ ]+)$")
_NUMBER_WORDS = frozenset({*"0123456789", "point", "minus", "e"})


def _number(answer: str) -> float | int | None:
    """The finite number an answer spells, else ``None`` (``1 e 9 9 9`` is infinity: not a number)."""
    got = parse_num(answer)
    try:
        return got if got is not None and math.isfinite(float(got)) else None
    except OverflowError:
        return None


def _handoff_inputs(name: str, text: str) -> tuple | None:
    """The input values of a hand-off prompt, each written the one way :func:`num` writes it."""
    words, values = text.split(" "), []
    i = 0
    for key in HANDOFFS[name].inputs:
        if i >= len(words) or words[i] != key:
            return None
        j = i + 1
        while j < len(words) and words[j] in _NUMBER_WORDS:
            j += 1
        value = _number(" ".join(words[i + 1:j]))
        lo, hi, whole = INPUT_RANGES[key]
        if value is None or num(value, sig=6) != " ".join(words[i + 1:j]) or not lo <= value <= hi \
                or (whole and not isinstance(value, int)):
            return None
        values.append(value if whole else float(value))
        i = j
    return tuple(values) if i == len(words) else None


def parse(prompt: str) -> tuple | None:
    """A world prompt as ``(kind, ...)``; ``None`` when it is not one. Kinds: ``("era", seed, era,
    key)``, ``("final" | "param" | "why", seed, key)``, ``("timeline", seed)``, ``("constant" |
    "yield", name)``, ``("handoff", name, inputs)``."""
    m = _ERA_Q.match(prompt)
    if m:
        base, number, key = m[2], m[3], m[4]
        if base not in ERA_KEYS or (number is not None) != (base in NUMBERED_ERAS) or key not in ERA_KEYS[base]:
            return None
        return "era", int(parse_num(m[1])), base + (f"_{number}" if number else ""), key
    m = _SEED_Q.match(prompt)
    if m:
        known = {"final": SUMMARY_KEYS, "param": PARAM_KEYS, "why": OUTCOMES}[m[2]]
        return (m[2], int(parse_num(m[1])), m[3]) if m[3] in known else None
    m = _TIMELINE_Q.match(prompt)
    if m:
        return "timeline", int(parse_num(m[1]))
    m = _TABLE_Q.match(prompt)
    if m:
        return (m[1], m[2]) if m[2] in (CONSTANTS if m[1] == "constant" else YIELDS) else None
    m = _HANDOFF_Q.match(prompt)
    if m and m[1] in HANDOFFS:
        inputs = _handoff_inputs(m[1], m[2])
        return ("handoff", m[1], inputs) if inputs is not None else None
    return None


def owns(prompt: str) -> bool:
    return parse(prompt) is not None


def _judge(key: str, value: float | int | str, answer: str, exact: bool = False) -> Verdict:
    """Words exactly; ``exact`` numbers and :data:`EXACT_KEYS` exactly; :data:`MEASURED_KEYS` as whole
    numbers within 1 or 5 percent (zero exactly); other numbers within 5 percent."""
    expected, answer = (table_value if exact else dense_value)(value), answer.strip()
    if isinstance(value, str):
        ok = answer == value
        return Verdict(ok, expected, "exact word" if ok else "wrong word")
    got = _number(answer)
    if got is None:
        return Verdict(False, expected, "not a number")
    if exact or key in EXACT_KEYS:
        ok = got == value and (isinstance(got, int) or not isinstance(value, int))
        return Verdict(ok, expected, "exact" if ok else "not the exact value")
    if key in MEASURED_KEYS:
        ok = isinstance(got, int) and (got == value if 0 in (got, value) else abs(got - value) <= 1 or close(got, value))
        return Verdict(ok, expected, "within 1 or 5 percent" if ok else "wrong count")
    ok = close(float(got), float(value), rel=0.05)
    return Verdict(ok, expected, "within 5 percent" if ok else "more than 5 percent off")


def _truth(q: tuple) -> float | int | str | None:
    kind = q[0]
    if kind == "handoff":
        return HANDOFFS[q[1]].rule(*q[2])
    if kind == "constant":
        return CONSTANTS[q[1]]["value"]
    if kind == "yield":
        return YIELDS[q[1]]
    r = _replay(q[1])
    if kind == "era":
        era = r.era(q[2])
        return None if era is None else era[q[3]]
    if kind == "final":
        return r.summary[q[2]]
    if kind == "param":
        return r.params[q[2]]
    if kind == "timeline":
        return " ".join(s["era"] for s in r.steps)
    if r.summary["outcome"] != q[2]:  # a why of an outcome the world did not have
        return None
    return " ".join(conditions(r.steps)) or "nothing"


def truth(prompt: str) -> float | int | str | None:
    """The chain's own value for a prompt; ``None`` when the prompt is not ours or cannot be
    answered (an era the seed does not have, a ``why`` of another outcome)."""
    q = parse(prompt)
    return None if q is None else _truth(q)


def check(prompt: str, answer: str) -> Verdict:
    """Replay the chain a prompt names (or the hand-off it spells out) and compare."""
    q = parse(prompt)
    if q is None:
        return Verdict(False, None, "not my question")
    value = _truth(q)
    if value is None:
        if q[0] == "era":
            return Verdict(False, None, f"world seed {q[1]} has no era {q[2]}")
        return Verdict(False, None, f"world seed {q[1]} did not end {q[2]}")
    key = {"era": q[-1], "final": q[-1], "handoff": q[1]}.get(q[0], "")
    return _judge(key, value, answer, exact=q[0] in ("param", "constant", "yield"))


# ---------------------------------------------------------------------------------------------
# lines


def params_line(r: Rollout) -> Line:
    """``world seed 3 params. generations 1. spin 0 point 2 6. t_gas 2 point 1 7.``"""
    fields = " ".join(f"{k} {table_value(r.params[k])}." for k in PARAM_KEYS)
    return Line(f"{SIM} seed {num(r.seed)} params. {fields}", topic=SIM, kind="record")


def param_lines(r: Rollout) -> list[Line]:
    return [Line(f"{SIM} seed {num(r.seed)} param {k}", table_value(r.params[k]), SIM, "fact") for k in PARAM_KEYS]


def timeline_line(r: Rollout) -> Line:
    """``world seed 2 timeline. eras nucleo enrichment cloud disc star planets. outcome no_metals.``"""
    names = " ".join(s["era"] for s in r.steps)
    return Line(f"{SIM} seed {num(r.seed)} timeline. eras {names}. outcome {r.summary['outcome']}.", topic=SIM,
                kind="record")


def era_line(r: Rollout, t: int) -> Line:
    """``world seed 3 era planets. planets 7. rocky 3. ice 4. gas 0. habitable 1. largest_mass 2 1 point 3.``"""
    step = r.steps[t]
    fields = " ".join(f"{k} {dense_value(step[k])}." for k in ERAS[_base(step["era"])])
    return Line(f"{SIM} seed {num(r.seed)} era {step['era']}. {fields}", topic=SIM, kind="record")


def era_queries(r: Rollout, t: int) -> list[Line]:
    step = r.steps[t]
    return [Line(f"{SIM} seed {num(r.seed)} era {step['era']} {k}", dense_value(step[k]), SIM, "fact")
            for k in ERA_KEYS[_base(step["era"])]]


def why_line(r: Rollout) -> Line:
    """``q world seed 2 why no_metals. a star.``: the links of the chain that held."""
    return Line(f"{SIM} seed {num(r.seed)} why {r.summary['outcome']}", " ".join(conditions(r.steps)) or "nothing",
                SIM, "calc")


def handoff_line(name: str, *inputs: float | int) -> Line:
    """``q world handoff cooling metallicity 0 point 0 1 4 8. a 0 point 4.``"""
    hand = HANDOFFS[name]
    asked = " ".join(f"{k} {num(v, sig=6)}" for k, v in zip(hand.inputs, inputs))
    return Line(f"{SIM} handoff {name} {asked}", dense_value(hand.rule(*inputs)), SIM, "calc")


def handoff_lines(r: Rollout) -> list[Line]:
    """The hand-offs of one world, asked with the numbers its eras print."""
    era = {s["era"]: s for s in r.steps}
    z, m = era["enrichment"]["metallicity"], era["star"]["star_mass"]
    out = [handoff_line("metallicity", era["nucleo"]["hydrogen"], era["enrichment"]["generations"]),
           handoff_line("cooling", z), handoff_line("star_mass", era["disc"]["largest_clump"]),
           handoff_line("fusion", m), handoff_line("luminosity", m), handoff_line("frost_line", m),
           handoff_line("disc_mass", z)]
    for s in r.steps:
        if s["era"].startswith("chemistry_"):
            out += [handoff_line("temperature", m, s["orbit"]), handoff_line("mix", s["temperature"])]
            if f"life_{s['era'].split('_')[1]}" in era:
                out.append(handoff_line("monomers", s["organic"]))
    return out


def practice_lines(rng: random.Random, n: int) -> list[Line]:
    """``n`` hand-off questions with inputs drawn from the ranges the worlds cover (:data:`PRACTICE`,
    :data:`PRACTICE_OF`); they need no simulation."""
    out = []
    for _ in range(n):
        name = rng.choice(tuple(HANDOFFS))
        inputs = []
        for key in HANDOFFS[name].inputs:
            lo, hi = PRACTICE_OF.get(name, PRACTICE)[key]
            inputs.append(rng.randint(int(lo), int(hi)) if INPUT_RANGES[key][2] else _sig3(rng.uniform(lo, hi)))
        out.append(handoff_line(name, *inputs))
    return out


def records() -> list[Line]:
    """The hand-offs as a table: one record per formula with its inputs and constants, and the yields."""
    out = []
    for name, hand in HANDOFFS.items():
        fields = " ".join(f"{c} {table_value(CONSTANTS[c]['value'])}." for c in hand.constants)
        out.append(Line(f"{SIM} handoff {name}. from {' '.join(hand.inputs)}. {fields}", topic=SIM, kind="record"))
    fields = " ".join(f"{el} {table_value(share)}." for el, share in YIELDS.items())
    out.append(Line(f"{SIM} yields. {fields}", topic=SIM, kind="record"))
    return out


def table_lines() -> list[Line]:
    """Every question the tables answer (fixed; no seed involved)."""
    return [Line(f"{SIM} constant {name}", table_value(row["value"]), SIM, "fact") for name, row in CONSTANTS.items()] \
        + [Line(f"{SIM} yield {el}", table_value(share), SIM, "fact") for el, share in YIELDS.items()]


def lines(r: Rollout) -> list[Line]:
    """Every line of one world, which must be :func:`rollout` of its seed: the parameter and
    timeline records, a record per era, then the questions: every metric of every era, the
    summary, the parameters, the timeline, the ``why`` and the hand-offs as they happened."""
    if not canonical(r):
        raise ValueError(f"lines only for the world a seed names (rollout({r.seed})): the gate replays that one")
    out = [params_line(r), timeline_line(r)] + [era_line(r, t) for t in range(len(r.steps))]
    for t in range(len(r.steps)):
        out += era_queries(r, t)
    out += summary_lines(r) + param_lines(r)
    out.append(Line(f"{SIM} seed {num(r.seed)} timeline", " ".join(s["era"] for s in r.steps), SIM, "fact"))
    out.append(why_line(r))
    return out + handoff_lines(r)


def generate(rng: random.Random, n: int) -> list[Line]:
    """``n`` lines: the table questions, then world after world on seeds drawn from ``rng``
    (each with :data:`PRACTICE_PER_SEED` practice hand-offs) until there are ``n``. A world costs
    about 2 s and gives 85 to 111 lines, practice included. To build training data from whole
    worlds, call ``lines(rollout(seed))`` per seed instead."""
    if n <= 0:
        return []
    out = table_lines()
    for seed in seeds(rng, min(9998, n // 80 + 1)):
        if len(out) >= n:
            break
        out += lines(_replay(seed)) + practice_lines(rng, PRACTICE_PER_SEED)
    return out[:n]


class World:
    """The chain as a :class:`haishool.cosmos.Simulation` and a :class:`haishool.truth.Gate`."""

    sim = SIM
    topic = SIM
    KEYS = KEYS

    def run(self, seed: int, **params: float | int) -> WorldRollout:
        return run(seed, **params)

    def rollout(self, seed: int) -> WorldRollout:
        return rollout(seed)

    def random_params(self, rng: random.Random) -> dict[str, float | int]:
        return random_params(rng)

    def conserved(self, r: Rollout) -> Verdict:
        return conserved(r)

    def check(self, prompt: str, answer: str) -> Verdict:
        return check(prompt, answer)

    def owns(self, prompt: str) -> bool:
        return owns(prompt)

    def lines(self, r: Rollout) -> list[Line]:
        return lines(r)

    def truth(self, prompt: str) -> float | int | str | None:
        return truth(prompt)

    def records(self) -> list[Line]:
        return records()

    def table_lines(self) -> list[Line]:
        return table_lines()

    def generate(self, rng: random.Random, n: int) -> list[Line]:
        return generate(rng, n)


_SIM = World()


def simulation() -> World:
    return _SIM
