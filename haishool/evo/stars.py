"""Level 7 of the toy universe: generations of stars turn a box of gas into heavier elements.

Nucleosynthesis in the first hour (level 1) leaves hydrogen and helium and nothing a planet or a
cell could be made of. Nearly every heavier atom was made in stars and given back to the gas by
their winds and explosions (beryllium, boron and part of the lithium are the exceptions: cosmic
rays make them by breaking heavier nuclei; this level has none of them). The level is a
one-zone, closed-box model of galactic chemical evolution, the simplest kind there is, here with
stars that live as long as their mass allows: a box of gas of one million solar masses that
nothing enters or leaves, in which stars form, live, and return part of themselves, enriched, to
the gas the next stars form from. Its ingredients are real and each is simplified; the numbers
it produces are those of the toy.

Published models the rules come from, and what is taken from each (authors and years from
memory, none looked up for this file):

* **Star formation** after Schmidt (1959): the rate is a power of the gas density. Schmidt
  argued for a power near two; the power one used here, ``d gas / dt = - efficiency * gas``, is
  the common simplification and is chosen because it can be integrated exactly within a step of
  ``DT`` = 0.25 Gyr: :func:`gas_to_stars` ``= gas * (1 - exp(-efficiency * dt))``.
* **Initial mass function** of Salpeter (1955): the number of stars per unit mass falls as
  ``M^-2.35``. Salpeter measured it from about 0.4 to 10 solar masses; it is stretched here over
  0.1 to 100, as many models of chemical evolution do. That is a simplification: the measured
  mass function is flatter below about half a solar mass (Kroupa 2001; Chabrier 2003), so this
  box locks more of its mass in red dwarfs than a real galaxy does. It is cut into mass bins:
  thirty (:data:`EDGES`, ten per decade), cut again at every mass whose lifetime is a whole
  number of steps (:func:`turnoff_mass`, :func:`bin_edges`; 83 bins in all), so that the stars of
  a bin all die in the same step. Numbers and masses per bin are the exact integrals
  (:func:`imf_number`, :func:`imf_mass`). It gives 7.42 stars above 8 solar masses per 1000
  solar masses formed (:func:`stars_above`), 6.03 of them between 8 and 25.
* **Luminosity and lifetime**, the textbook scalings ``L = M^3.5`` and ``lifetime = 10 Gyr *
  M / L = 10 Gyr * M^-2.5`` (fuel over burning rate). The relation between mass and luminosity
  was found in binary stars in the early 1920s and explained by theory (Eddington 1924); the
  power 3.5 is a rough average over the main sequence, normalised to the sun. The
  measured power is nearer 4 around one solar mass and flatter for red dwarfs and for the
  heaviest stars, so the two laws are right *at* one solar mass (one solar luminosity, 10 Gyr)
  and only roughly elsewhere. The simulation uses them for every mass all the same, which is a
  toy: a real star of 100 solar masses shines a few million years, not the 0.1 million of the
  power law, and in published stellar models stars of 2 to 5 solar masses live about 1.5 to 2
  times shorter than the law says (from memory), so this box returns their carbon and nitrogen
  and makes its white dwarfs somewhat late. What matters here survives: massive stars die
  within one step, a star of one solar mass after 10 Gyr, a red dwarf not within the run.
* **Fates by initial mass** (:func:`fate`): below 0.08 solar masses a ``brown_dwarf`` (the
  hydrogen-burning limit after Kumar 1963; it never shines and its mass stays locked; the mass
  function used here starts at 0.1, so the simulation forms none); from 0.08 to below 8 a
  ``white_dwarf``; from 8 to 25 a core-collapse supernova that leaves a ``neutron_star``; above
  25 a ``black_hole``. The 8 is the usual round value (the real limit is known to about one
  solar mass); the 25 is where black holes begin in Heger et al. (2003). Toy: in their picture
  the limits move with the star's metals and winds, and stars somewhat above 25 solar masses
  still explode, faintly; here the limits are fixed and a star above 25 makes no supernova.
* **Remnant masses** (:func:`remnant_mass`): white dwarfs by the linear initial-final mass
  relation after Kalirai et al. (2008), ``0.11 M + 0.39`` (measured for stars of about 1 to 7
  solar masses; never more than the star itself, which is this module's cap for the light stars
  the relation was not measured for); neutron stars 1.4 solar masses, the typical measured mass;
  a black hole keeps half of the star (toy value: the real share depends on winds and
  metallicity), the other half is blown off before the collapse.
* **Yields** (:data:`YIELDS`): for each fate, the share of the returned gas that is newly made
  helium, carbon, nitrogen, oxygen, neon, magnesium, silicon, iron and ``other`` (sulfur, argon,
  calcium, nickel and the rest of the metals, lumped). The new elements are made out of the
  star's hydrogen; everything else in the returned gas has the make-up the star was born with.
  The table for the supernovae has the proportions of a star of about 15 solar masses after
  Woosley and Weaver (1995), the one for white dwarfs the order of magnitude of the winds of
  giant stars after Karakas (2010). The numbers are rounded toy values, one table for all
  masses of a fate, and those of the supernovae are set low on purpose, see "What is real and
  what is toy". The supernova table's oxygen over iron, 2.5 times the sun's (astronomers'
  [O/Fe] = +0.4), is about the plateau measured in old stars poor in metals (roughly +0.4 to
  +0.5, from memory); the box starts there because the table does, it does not predict it.
* **Type Ia supernovae**. Taken from Matteucci and Greggio (1986): a small, fixed share of the
  stars that end as white dwarfs explodes long after the stars formed, and gives most of the
  iron. Theirs is a share of the binary stars, and the delay is the lifetime of the companion.
  The rule here is simpler and invented: a share ``ia_fraction`` of *all* white dwarfs explodes
  ``IA_DELAY`` = 1 Gyr after the white dwarf has formed. Each event destroys 1.4 solar masses of
  white-dwarf matter (:data:`IA_MASS`, the Chandrasekhar mass, taken out of the pool of
  remnants; a single white dwarf here has 0.49 to 1.27 solar masses, and the companion that
  would bring it to 1.4 is not modelled) and returns all of it as metals, half of it iron: 0.7
  solar masses (:func:`ia_iron`; the table :data:`IA_YIELD` is rounded from model W7 of Nomoto,
  Thielemann and Yokoi 1984, from memory, with ``other`` taking up what is left of the 1.4).
  Because this iron comes late, oxygen over iron starts high and falls (Tinsley 1979).
* **The closed box** (Searle and Sargent 1972; Tinsley 1980): if all returns were immediate, the
  metallicity of the gas would be ``Z = y * ln(1 / gas_fraction)`` with the yield ``y`` = new
  metals returned per mass locked up for good (:func:`closed_box`). The simulation has delays.
  While a box has gas to spare it therefore lags behind the formula; when the gas is nearly
  gone the late returns fall into little gas and the simulation can pass the formula (see the
  targets). The summary gives both (``metallicity`` and ``closed_box``).
* **The sun** for comparison (Asplund, Grevesse, Sauval and Scott 2009, from memory): metals
  0.0134 of the mass of its surface today (when it formed, 0.0142 of its mass, and helium 0.270:
  heavy atoms have since settled inward), oxygen over iron about 4.4 by mass. The level compares
  with the value of today, 0.0134 (:data:`Z_SUN`), which is also the limit of its ``mature``
  stage; the box at 9 Gyr, 0.0136, lies between the two. ``alpha_over_iron`` is oxygen (one of
  the alpha elements) over iron in units of that solar ratio (:func:`alpha_over_iron`), as a
  plain ratio; astronomers quote its logarithm, so 2.5 here is their [O/Fe] = +0.4.

One step (:data:`DT` = 0.25 Gyr), in this order:

1. stars form: ``gas_to_stars(gas, efficiency, DT)`` leaves the gas, with the gas's make-up;
2. every group of stars (one birth step, one mass bin) whose :func:`death_time` falls within the
   step dies: its remnants go to ``remnants``, the rest returns to the gas, enriched by
   :data:`YIELDS`. A group counts as born at the beginning of the step that formed it, so stars
   above 4.4 solar masses (lifetime under 0.25 Gyr) return their gas in the step that formed
   them, though not to the stars of that step;
3. the type Ia supernovae that are due (white dwarfs of four steps ago) go off.

Metrics per step (:data:`STATE_KEYS`; :data:`KEYS` has meaning and unit): ``time`` in Gyr; ``gas``,
``stars`` (living) and ``remnants`` as fractions of the box's mass; ``metallicity``, ``helium``,
``oxygen``, ``carbon``, ``iron``, ``nitrogen`` as mass fractions of the gas; ``supernovae_cc`` and
``supernovae_ia``, the events so far in the box of one million solar masses (expected numbers,
so not whole); ``alpha_over_iron``; ``stage`` (:func:`stage`): ``exhausted`` when less than 5
percent of the box is gas, else ``first_stars`` while the gas has less than a tenth of the sun's
metals, ``enriching`` up to the sun's, ``mature`` above. The stage names and their limits are
labels of this toy for the state of the gas, not the astronomers' populations of stars (a box
at time zero is ``first_stars`` before it has a star). ``metallicity`` is everything heavier
than helium. A step also has ``hydrogen``, ``neon``, ``magnesium``, ``silicon`` and ``other``
(:data:`EXTRA_KEYS`): the gate judges them, the lines leave them out to stay short.

Dense lines (:func:`lines`), measured numbers to 3 significant digits, parameters as given:

    stars seed 3 params. efficiency 0 point 5 9. t_end 6 point 7 5. hydrogen 0 point 7 6. helium 0 point 2 4. ia_fraction 0 point 0 0 7 4. metal_yield 0 point 0 0 8 8 3. return_fraction 0 point 2 9 1.
    stars seed 3 step 8. time 2. gas 0 point 3 9 8. stars 0 point 5 5 3. remnants 0 point 0 4 9 1. metallicity 0 point 0 0 5 6 2. helium 0 point 2 4 9. oxygen 0 point 0 0 2 7 8. ... supernovae_cc 4 5 2 0. supernovae_ia 7 1 point 4. alpha_over_iron 1 point 7 5. stage enriching.
    q stars seed 3 step 8 oxygen. a 0 point 0 0 2 7 8.
    q stars seed 3 step 8 next alpha_over_iron. a 1 point 6 1.   (what happens next)
    q stars seed 3 final metallicity. a 0 point 0 2 1 1.          (how it ends)

and lessons that carry their inputs (:data:`LESSONS`, topic ``predict_stars``; their functions
are the ones the simulation calls, so a lesson is true of every rollout):

    q stars predict lifetime mass 2. a 1 point 7 6 8.
    q stars predict turnoff age 1 0. a 1.
    q stars predict fate mass 1 2 point 5. a neutron_star.
    q stars predict closed_box metal_yield 0 point 0 1 gas_fraction 0 point 2. a 0 point 0 1 6 0 9.

The inputs of a lesson are drawn evenly from a range, and the ranges of the rules that answer
with a word are set so that no word crowds out the others (shares that follow from the ranges;
40000 lessons give them within 0.02): ``shines`` 0.01 to 0.15 solar masses, yes 0.50 and no
0.50; ``fate`` 0.02 to 42 solar masses, black_hole 0.40, neutron_star 0.40, white_dwarf 0.19,
brown_dwarf 0.001 (the limit 0.08 is what ``shines`` teaches); ``stage`` with gas up to 1 and
metallicity up to 0.027, mature 0.48, enriching 0.42, exhausted 0.05, first_stars 0.05 (an even
draw cannot make the two narrow stages common without leaving out most of what the box does).
Of the rules that answer with a number only ``remnant`` has a common answer: 1.4, the neutron
star, in 0.28 of its lessons (the masses 8 to 25 of 0.02 to 60); no other numeric answer
reaches 0.03.

The seed names a run and, through :func:`random_params`, picks its parameters; the simulation
itself has no chance in it (a box of a million solar masses holds enough stars for the expected
numbers), so two seeds with the same parameters give the same steps. ``rules`` may be passed
and must be 7: this level has no round-6 form. :func:`check` replays the seed's own parameters,
so only the lines of :func:`rollout` (see :func:`is_canonical`) are training data; a run with
other parameters under the same seed has the same prompts with other answers.

Gates (:meth:`Stars.conserved`): gas + living stars + remnants = the box, to 1e-9, at every step;
for every element the gas holds what it had at the start, less what went into stars, plus what
was returned (the ledger, per element, to 1e-9); the returned gas adds up to what the dead stars
did not keep; no star makes hydrogen; the mass fractions of the gas sum to 1; time strictly
increasing; no negative mass; supernova counts never fall; the stars formed in a step are
``gas_to_stars`` of the gas before; the core-collapse supernovae are the stars of 8 to 25 solar
masses formed so far; the type Ia supernovae are ``ia_fraction`` of the white dwarfs that had
formed 1 Gyr before (:func:`white_dwarfs_formed`: the stars between the :func:`turnoff_mass` of
their age and 8 solar masses) and match the mass they destroyed; stage and ``alpha_over_iron``
follow from the step's own numbers. :meth:`Stars.check` replays the seed and compares within 5
percent for measured numbers and exactly for words. :func:`check` and :func:`owns` judge the
lines of rollouts; lessons (``stars predict ...``) are judged by :data:`LESSONS`.

Plausibility targets (``tests/test_evo_stars.py`` holds them). Measured with the defaults
(efficiency 0.3 per Gyr, hydrogen 0.752, helium 0.248, ``ia_fraction`` 0.004) and over the
canonical seeds 1 to 2000. The first two are what the toy tables were tuned to, not findings:

    about solar metallicity at 8 to 10 Gyr   0.0120 at 8 Gyr, 0.0136 at 9, 0.0151 at 10 (the sun:
                                             0.0134 today, 0.0142 at its birth); the gas is then
                                             0.18, 0.15, 0.13 of the box
    the mix at 9 Gyr is the sun's            oxygen 0.0059, carbon 0.0022, iron 0.0013, neon 0.0012,
                                             nitrogen 0.00074, magnesium 0.00070, silicon 0.00069
                                             (sun: 0.0057, 0.0024, 0.0013, 0.0013, 0.00069, 0.00071,
                                             0.00067); helium 0.272 (the sun was born with 0.27)
    oxygen is the most abundant metal        at every step of 1976 of 2000 runs; in the other 24 the
                                             box is exhausted (gas below 0.014) and the iron of
                                             many type Ia supernovae overtakes it
    oxygen over iron falls with time         2.5 times solar (the ratio of the supernova table)
                                             until the first type Ia supernovae at 1.25 Gyr, 1.0 at
                                             9 Gyr, 0.70 at 13.5 Gyr; it rises in no step of any
                                             run, at any parameters tried
    the gas shrinks, the remnants grow       in every step of every run, at any parameters tried
                                             (efficiency 0.01 to 4, ``ia_fraction`` 0 to 0.1)
    hydrogen falls, the metals rise          in every step of the 2000 runs; helium rises too,
                                             except by under 1e-4 in boxes with less than 0.012
                                             gas left. Not a law of the model: an emptied box
                                             refills from the old stars it holds, which return
                                             the metal-poor gas they were born from. With
                                             efficiency 1 and no type Ia supernovae the
                                             metallicity peaks at 0.0202 at 7.25 Gyr and falls to
                                             0.0190; with ``ia_fraction`` 0.001, the corner of the
                                             canonical range, it still falls in 4 of the 54 steps
    an end time of zero                      no metals, no stars, no supernovae, the gas as put in
    the simulation lags the closed box       ``closed_box`` over ``metallicity`` with the defaults:
                                             1.44 at 1 Gyr, 1.25 at 4, 1.10 at 9, 1.02 at 13.5. In
                                             56 of the 2000 runs the simulation ends above the
                                             formula (ratio down to 0.946): all have under 0.064
                                             of the box left as gas, efficiency from 0.32 and
                                             ``ia_fraction`` from 0.0048
    one generation of stars                  returns 0.290 of its mass within 13.5 Gyr; yield 0.0079;
                                             6.03 core-collapse and 0.57 type Ia supernovae per
                                             1000 solar masses formed
    every seed                               the gates pass; worst error of the mass 5e-15
    how runs end                             enriching 844, exhausted 683, mature 447, first_stars 26;
                                             final metallicity 0.0007 to 0.042, median 0.016

What is real and what is toy. Real: the Salpeter slope; lifetimes that fall steeply with mass;
that stars end as white dwarfs, neutron stars or black holes according to their mass; the
white-dwarf mass relation; the 1.4 solar masses of a typical neutron star; that oxygen, neon and magnesium come promptly from massive stars, carbon and nitrogen
also from the slow deaths of lighter stars, and most iron late from type Ia supernovae; the
closed-box formula; the book-keeping of mass. Simplified: the linear Schmidt law; the Salpeter
slope outside the masses it was measured for; one lifetime law for all masses; fixed mass limits
of the fates; every type Ia supernova destroying 1.4 solar masses, the Chandrasekhar mass of
the classic picture (W7), though many may explode below it. Toy, and tuned: the yield tables and ``ia_fraction``. A closed box keeps every
atom its stars make, while a real galaxy loses enriched gas in winds and is diluted by gas
falling in; with yields as high as the published ones this box would end well above the sun's
metallicity (perhaps twice: an estimate from memory of the published tables, not a calculation
made here). The supernova
table is therefore set low (oxygen is 3.3 percent of the returned gas; a star of 15 solar
masses in Woosley and Weaver's tables returns about 5 percent, from memory), so that the box
holds the sun's share of metals when the sun formed, 9 Gyr after the Big Bang. For the same
reason the default ``ia_fraction`` 0.004 gives 0.57 type Ia supernovae per 1000 solar masses
formed, a third to a half of what supernova surveys find (one to two; Maoz and Mannucci 2012,
from memory), set so that oxygen over iron is the sun's at 9 Gyr. The agreement with the sun at
9 Gyr is thus put in, not predicted. Also toy: the rule for type Ia supernovae (a share of all
white dwarfs, one delay; the real delays spread from 0.1 to 10 Gyr); yields that do not depend
on the star's own metals (real nitrogen does) or, within a fate, on its mass; no stars above 100
or below 0.1 solar masses; black holes that swallow all their new oxygen and iron, and no
supernova above 25 solar masses; star formation from time zero (the first stars took one to a
few hundred million years); one well-mixed zone; steps of 0.25 Gyr, with the stars of a step
counted as born at its beginning; the stage names and their limits. A closed box is also known
to disagree with the stars near the sun: it makes too many long-lived stars that are poor in
metals (the G-dwarf problem, van den Bergh 1962; Schmidt 1963), which this level does not
count. ``run`` accepts parameters beyond those :func:`random_params` draws (efficiency up to 4
per Gyr, ``ia_fraction`` up to 0.1) for what-if runs; nothing was tuned there, and with
``ia_fraction`` 0.1 and efficiency 1 the last gas of the box is more than a quarter metals. A
run tells the toy's rules; its numbers are not measurements of the Milky Way.

Hand-off. *In* (:func:`handoff_in`): the level below is the first hour (``nucleo``); its summary
gives the primordial gas: ``hydrogen`` = its hydrogen plus deuterium, ``helium`` = its helium-4
plus helium-3 (the trace ratios are per hydrogen nucleus and are turned into mass fractions),
scaled to sum to one and rounded to four decimals. The first hour is nothing on a clock of Gyr:
this level's time zero is the Big Bang. *Out*: the summary gives the make-up of the gas at
``time`` = ``t_end`` Gyr, the age of the universe at which a later star and its planets form
from that gas: the mass fractions ``hydrogen``, ``helium``, ``carbon``, ``nitrogen``, ``oxygen``,
``neon``, ``magnesium``, ``silicon``, ``iron``, ``other`` and the sum of all metals,
``metallicity``; also ``gas_fraction`` and ``stars_formed`` (in units of the box; the second can
pass 1, since returned gas forms stars again), ``supernovae`` with ``supernovae_cc`` and
``supernovae_ia`` (events in the box of one million solar masses), ``alpha_over_iron``,
``closed_box``, ``stage`` and ``time`` (Gyr). :func:`cloud` gives the same make-up at any step
of a rollout.

Reproducibility: plain Python floats, no numpy, and no random numbers in the simulation
(:func:`random_params` and the lessons draw from ``random.Random``). Sums are added from left to
right in a fixed order by the module's own loop, not by the built-in ``sum``, whose last digit
changed with Python 3.12: steps and ledgers are the same bit for bit on Python 3.11 and 3.12.
The elementary functions (``exp``, ``log``, powers) may still differ in the last digit between
machines; no decision sits on such a digit (the tests nudge the parameters by 1e-12 and compare
lines; the mean mass of a bin never has a lifetime within 0.02 steps of a step's time, since
the bins are cut at exactly those masses, and a cut that misses a fixed edge by a rounding
error is dropped).
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

from haishool.cosmos import Rollout, close, query_lines, seeds, state_line, summary_lines
from haishool.evo import Input, LessonGate, Rule, sig
from haishool.truth import Line, Verdict, is_finite, num, parse_num

SIM = "stars"

DT = 0.25  # Gyr, one step
T_MAX = 13.5  # Gyr, the longest run (about the age of the universe)
MAX_STEPS = 54  # T_MAX / DT
T_SUN = 10.0  # Gyr, main-sequence lifetime of one solar mass
BOX_MASS = 1.0e6  # solar masses in the box; only the supernova counts depend on it
IMF_SLOPE = 2.35  # Salpeter (1955): dN/dM ~ M^-2.35
M_MIN, M_MAX = 0.1, 100.0  # solar masses, the range of the mass function
M_SHINE = 0.08  # solar masses; below it no hydrogen burning (brown dwarf)
M_SUPERNOVA = 8.0  # from here a star's core collapses
M_BLACK_HOLE = 25.0  # above it the collapse leaves a black hole
WD_SLOPE, WD_OFFSET = 0.11, 0.39  # white dwarf mass = 0.11 M + 0.39 (after Kalirai et al. 2008: 0.109, 0.394)
NS_MASS = 1.4  # solar masses, a neutron star
BH_SHARE = 0.5  # share of its mass a star above 25 solar masses keeps as a black hole (toy value)
IA_MASS = 1.4  # solar masses of white-dwarf matter destroyed by one type Ia supernova
IA_IRON = 0.7  # solar masses of iron from one type Ia supernova
IA_DELAY = 1.0  # Gyr from the white dwarf's birth to its explosion (toy: one delay for all)
IA_FRACTION = 0.004  # default share of white dwarfs that explode (tuned: oxygen over iron solar at 9 Gyr)
Z_SUN = 0.0134  # metals in the sun by mass (Asplund et al. 2009)
SOLAR_O_FE = 4.4  # oxygen over iron in the sun by mass (5.73e-3 / 1.29e-3, Asplund et al. 2009, from memory)
GAS_EXHAUSTED = 0.05  # below this gas fraction the box is "exhausted"
FIRST_STARS = 0.1  # below this share of Z_SUN the gas is that of the "first_stars"
T_EPS = 1e-9  # Gyr; a death within T_EPS after a step's time counts as at that time

ELEMENTS = ("hydrogen", "helium", "carbon", "nitrogen", "oxygen", "neon", "magnesium", "silicon", "iron", "other")
METALS = ELEMENTS[2:]
FATES = ("brown_dwarf", "white_dwarf", "neutron_star", "black_hole")
STAGES = ("first_stars", "enriching", "mature", "exhausted")

#: bin edges in solar masses: ten per decade (the preferred numbers 1, 1.25, 1.6, 2, 2.5, 3.15, 4,
#: 5, 6.3, 8), so that the fate limits 8 and 25 are edges
EDGES: tuple[float, ...] = (0.1, 0.125, 0.16, 0.2, 0.25, 0.315, 0.4, 0.5, 0.63, 0.8,
                            1.0, 1.25, 1.6, 2.0, 2.5, 3.15, 4.0, 5.0, 6.3, 8.0,
                            10.0, 12.5, 16.0, 20.0, 25.0, 31.5, 40.0, 50.0, 63.0, 80.0, 100.0)

#: fate -> element -> share of the returned gas that is newly made (toy tables, see the docstring)
YIELDS: dict[str, dict[str, float]] = {
    "brown_dwarf": {},
    # the winds of giant stars: helium, carbon and nitrogen dredged up from the burning shells
    "white_dwarf": {"helium": 0.025, "carbon": 0.0025, "nitrogen": 0.0012},
    # a core-collapse supernova: the proportions of a star of 15 solar masses, set low (see the docstring)
    "neutron_star": {"helium": 0.05, "carbon": 0.005, "nitrogen": 0.0006, "oxygen": 0.033, "neon": 0.007,
                     "magnesium": 0.004, "silicon": 0.003, "iron": 0.003, "other": 0.002},
    # what a star above 25 solar masses blows off before it collapses: mostly its hydrogen envelope
    "black_hole": {"helium": 0.05, "carbon": 0.002, "nitrogen": 0.001},
}
#: solar masses of each element from one type Ia supernova; they add up to IA_MASS. "other" is
#: sulfur, argon, calcium, nickel and the rest
IA_YIELD: dict[str, float] = {"carbon": 0.05, "oxygen": 0.14, "magnesium": 0.01, "silicon": 0.15, "iron": IA_IRON,
                              "other": 0.35}

PARAM_KEYS = ("efficiency", "t_end", "hydrogen", "helium", "ia_fraction", "metal_yield", "return_fraction")
#: the parameters that are put in; they are written as given, the derived ones to 3 digits
INPUT_KEYS = frozenset({"efficiency", "t_end", "hydrogen", "helium", "ia_fraction"})
STATE_KEYS = ("time", "gas", "stars", "remnants", "metallicity", "helium", "oxygen", "carbon", "iron", "nitrogen",
              "supernovae_cc", "supernovae_ia", "alpha_over_iron", "stage")
#: per step as well, judged by the gate, but not in the lines (they would make a line too long)
EXTRA_KEYS = ("hydrogen", "neon", "magnesium", "silicon", "other")
SUMMARY_KEYS = ("metallicity", "oxygen", "carbon", "iron", "nitrogen", "neon", "magnesium", "silicon", "other",
                "helium", "hydrogen", "gas_fraction", "stars_formed", "supernovae", "supernovae_cc", "supernovae_ia",
                "alpha_over_iron", "closed_box", "time", "stage")
WORD_KEYS = frozenset({"stage"})
_DIGITS = frozenset("0123456789")

KEYS: dict[str, str] = {
    "efficiency": "share of the gas turned into stars per Gyr (parameter)",
    "t_end": "end of the run, Gyr after the Big Bang, a multiple of 0.25 (parameter)",
    "ia_fraction": "share of the white dwarfs that explode as type Ia supernovae (parameter)",
    "metal_yield": "new metals returned by one generation of stars per mass it locks up for good "
                   "(stars that die within 13.5 Gyr, type Ia included)",
    "return_fraction": "share of a generation's mass that returns to the gas within 13.5 Gyr",
    "time": "time since the Big Bang, Gyr",
    "gas": "gas, as a fraction of the box's mass",
    "stars": "living stars, as a fraction of the box's mass",
    "remnants": "white dwarfs, neutron stars and black holes, as a fraction of the box's mass",
    "metallicity": "mass fraction of the gas that is heavier than helium (the seven named metals and other)",
    "hydrogen": "mass fraction of hydrogen in the gas (as a parameter: at the start)",
    "helium": "mass fraction of helium in the gas (as a parameter: at the start)",
    "carbon": "mass fraction of carbon in the gas",
    "nitrogen": "mass fraction of nitrogen in the gas",
    "oxygen": "mass fraction of oxygen in the gas",
    "neon": "mass fraction of neon in the gas",
    "magnesium": "mass fraction of magnesium in the gas",
    "silicon": "mass fraction of silicon in the gas",
    "iron": "mass fraction of iron in the gas",
    "other": "mass fraction of all other metals in the gas (sulfur, argon, calcium, nickel and the rest)",
    "supernovae_cc": "core-collapse supernovae so far in the box of one million solar masses (expected number)",
    "supernovae_ia": "type Ia supernovae so far in the box of one million solar masses (expected number)",
    "alpha_over_iron": "oxygen over iron in the gas, in units of the sun's ratio 4.4 (0 while there is no iron)",
    "stage": "first_stars, enriching, mature or exhausted",
    "gas_fraction": "gas at the end, as a fraction of the box's mass",
    "stars_formed": "mass turned into stars over the whole run, in units of the box's mass (returned gas counts again)",
    "supernovae": "all supernovae of the run in the box of one million solar masses, both kinds",
    "closed_box": "the metallicity the closed-box formula gives for the final gas fraction: metal_yield * ln(1 / gas_fraction)",
}


# ---------------------------------------------------------------------------------------------
# the rules: every one is a lesson (LESSONS) and is called by the simulation
# ---------------------------------------------------------------------------------------------

def luminosity(mass: float) -> float:
    """Solar luminosities of a main-sequence star of ``mass`` solar masses: ``mass^3.5``, a rough
    average normalised to the sun (exact only at one solar mass)."""
    return mass ** 3.5


def lifetime(mass: float) -> float:
    """Gyr on the main sequence: fuel over burning rate, ``10 * mass / luminosity = 10 mass^-2.5``."""
    return T_SUN * mass / luminosity(mass)


def death_time(born: float, mass: float) -> float:
    """Gyr at which a star of ``mass`` solar masses dies that formed at ``born`` Gyr."""
    return born + lifetime(mass)


def turnoff_mass(age: float) -> float:
    """Solar masses of the star whose lifetime is ``age`` Gyr, ``(10 / age)^0.4``: in a group of
    stars born together ``age`` Gyr ago every heavier star is dead."""
    return (T_SUN / age) ** (1.0 / 2.5)


def shines(mass: float) -> str:
    """``yes`` when a body of ``mass`` solar masses burns hydrogen (0.08 solar masses or more)."""
    return "yes" if mass >= M_SHINE else "no"


def fate(mass: float) -> str:
    """What a star of ``mass`` solar masses ends as: below 0.08 ``brown_dwarf``, below 8
    ``white_dwarf``, up to 25 ``neutron_star`` (after a core-collapse supernova), above ``black_hole``."""
    if shines(mass) == "no":
        return "brown_dwarf"
    if mass < M_SUPERNOVA:
        return "white_dwarf"
    return "neutron_star" if mass <= M_BLACK_HOLE else "black_hole"


def remnant_mass(mass: float) -> float:
    """Solar masses that stay locked when a star of ``mass`` solar masses is gone: all of a brown
    dwarf, ``0.11 mass + 0.39`` in a white dwarf (never more than the star), 1.4 in a neutron
    star, half the star in a black hole."""
    end = fate(mass)
    if end == "brown_dwarf":
        return mass
    if end == "white_dwarf":
        return min(mass, WD_SLOPE * mass + WD_OFFSET)
    return NS_MASS if end == "neutron_star" else BH_SHARE * mass


def returned_mass(mass: float) -> float:
    """Solar masses a star of ``mass`` solar masses gives back to the gas: ``mass - remnant_mass``."""
    return mass - remnant_mass(mass)


def gas_to_stars(gas: float, efficiency: float, dt: float) -> float:
    """Gas turned into stars within ``dt`` Gyr at ``efficiency`` per Gyr: the exact integral of
    ``d gas / dt = -efficiency * gas``, ``gas * (1 - exp(-efficiency * dt))``."""
    return gas * -math.expm1(-efficiency * dt)


def _sum(values) -> float:
    """Plain addition from left to right. The built-in ``sum`` adds floats with a correction term
    from Python 3.12 on, so its last digit depends on the Python version; this one does not."""
    total = 0.0
    for value in values:
        total += value
    return total


def _clip(mass: float) -> float:
    return min(max(mass, M_MIN), M_MAX)


def _mass_integral(lo: float, hi: float) -> float:
    return (lo ** (2 - IMF_SLOPE) - hi ** (2 - IMF_SLOPE)) / (IMF_SLOPE - 2)


#: the constant of the mass function, so that it holds one solar mass between M_MIN and M_MAX
IMF_NORM = 1.0 / _mass_integral(M_MIN, M_MAX)


def imf_number(lo: float, hi: float) -> float:
    """Stars between ``lo`` and ``hi`` solar masses per solar mass of stars formed."""
    lo, hi = _clip(lo), _clip(hi)
    return IMF_NORM * (lo ** (1 - IMF_SLOPE) - hi ** (1 - IMF_SLOPE)) / (IMF_SLOPE - 1)


def imf_mass(lo: float, hi: float) -> float:
    """Share of the mass of a generation that is in stars between ``lo`` and ``hi`` solar masses."""
    return IMF_NORM * _mass_integral(_clip(lo), _clip(hi))


def stars_above(formed: float, mass: float) -> float:
    """Number of stars heavier than ``mass`` among ``formed`` solar masses of new stars
    (``stars_above(1000, 8)`` is 7.4)."""
    return formed * imf_number(mass, M_MAX)


def ia_iron(events: float) -> float:
    """Solar masses of iron from ``events`` type Ia supernovae: 0.7 each."""
    return IA_IRON * events


def closed_box(metal_yield: float, gas_fraction: float) -> float:
    """Metallicity of a closed box with instant returns: ``metal_yield * ln(1 / gas_fraction)``."""
    return metal_yield * math.log(1.0 / gas_fraction)


def alpha_over_iron(oxygen: float, iron: float) -> float:
    """Oxygen over iron in units of the sun's ratio (4.4 by mass); 0 while there is no iron."""
    return 0.0 if iron <= 0 else oxygen / iron / SOLAR_O_FE


def stage(gas: float, metallicity: float) -> str:
    """``exhausted`` when under 5 percent of the box is gas; else ``first_stars`` below a tenth of
    the sun's metals (0.00134), ``enriching`` below the sun's (0.0134), ``mature`` from there on."""
    if gas < GAS_EXHAUSTED:
        return "exhausted"
    if metallicity < FIRST_STARS * Z_SUN:
        return "first_stars"
    return "enriching" if metallicity < Z_SUN else "mature"


RULES: dict[str, Rule] = {
    "luminosity": Rule((Input("mass", 0.2, 10, places=2),), luminosity,
                       "luminosity of a main sequence star in solar units. mass to the power 3 point 5. "
                       "mass in solar masses. a rough average normalised to the sun"),
    "lifetime": Rule((Input("mass", 0.1, 20, places=2),), lifetime,
                     "main sequence lifetime in gyr. 1 0 times mass over luminosity. "
                     "that is 1 0 times mass to the power minus 2 point 5. mass in solar masses. "
                     "a rough average normalised to the sun"),
    "death_time": Rule((Input("born", 0, 13.5, places=2), Input("mass", 0.5, 20, places=2)), death_time,
                       "time in gyr at which a star dies. born plus lifetime. lifetime 1 0 times mass to the power "
                       "minus 2 point 5"),
    "turnoff": Rule((Input("age", 0.05, 13.5, places=2),), turnoff_mass,
                    "mass in solar masses of the star whose lifetime is age gyr. 1 0 over age to the power "
                    "0 point 4. in a group born together every heavier star is dead"),
    # 0.01 to 0.15 puts the limit 0.08 in the middle: as many yes as no
    "shines": Rule((Input("mass", 0.01, 0.15, places=3),), shines,
                   "yes when a body burns hydrogen. mass at least 0 point 0 8 solar masses. below it a brown dwarf"),
    # up to 42 solar masses: as many black holes (25 to 42) as neutron stars (8 to 25)
    "fate": Rule((Input("mass", 0.02, 42, places=2),), fate,
                 "what a star ends as by its mass in solar masses. below 0 point 0 8 brown_dwarf. below 8 white_dwarf. "
                 "up to 2 5 neutron_star after a core collapse supernova. above 2 5 black_hole. the limits 8 and 2 5 "
                 "are round values of this toy"),
    "remnant": Rule((Input("mass", 0.02, 60, places=2),), remnant_mass,
                    "solar masses that stay locked. brown_dwarf all of it. white_dwarf 0 point 1 1 times mass plus "
                    "0 point 3 9 but never more than mass. neutron_star 1 point 4. black_hole half of mass. "
                    "the half is a toy value"),
    "returned": Rule((Input("mass", 0.02, 60, places=2),), returned_mass,
                     "solar masses a dead star gives back to the gas. mass minus remnant"),
    "gas_to_stars": Rule((Input("gas", 0, 1, places=3), Input("efficiency", 0.1, 1, places=2),
                          Input("dt", 0.05, 1, places=2)), gas_to_stars,
                         "gas turned into stars in one step. gas times one minus exp of minus efficiency times dt. "
                         "efficiency per gyr. dt in gyr"),
    "stars_above": Rule((Input("formed", 100, 100000, integer=True), Input("mass", 0.1, 100, places=1)), stars_above,
                        "number of stars heavier than mass among formed solar masses of new stars. salpeter mass "
                        "function with slope 2 point 3 5 from 0 point 1 to 1 0 0 solar masses. 7 point 4 above 8 "
                        "per 1 0 0 0 formed"),
    "ia_iron": Rule((Input("events", 0, 5000, integer=True),), ia_iron,
                    "solar masses of iron from type ia supernovae. 0 point 7 times events. a rounded model value"),
    "closed_box": Rule((Input("metal_yield", 0.001, 0.03, places=4), Input("gas_fraction", 0.01, 1, places=3)),
                       closed_box,
                       "metallicity of a closed box with instant returns. metal_yield times natural log of one over "
                       "gas_fraction"),
    "alpha_over_iron": Rule((Input("oxygen", 0, 0.02, places=5), Input("iron", 0, 0.005, places=5)), alpha_over_iron,
                            "oxygen over iron in units of the solar ratio 4 point 4. zero while there is no iron. "
                            "mass fractions of the gas"),
    # metallicity up to 0.027, twice the sun's: about as many mature as enriching
    "stage": Rule((Input("gas", 0, 1, places=3), Input("metallicity", 0, 0.027, places=5)), stage,
                  "stage of the box. exhausted when gas below 0 point 0 5. else first_stars when metallicity below "
                  "0 point 0 0 1 3 4. enriching below 0 point 0 1 3 4. else mature. the names and limits are labels "
                  "of this toy"),
}
LESSONS = LessonGate(SIM, RULES)


def lesson_gate() -> LessonGate:
    return LESSONS


# ---------------------------------------------------------------------------------------------
# the mass bins
# ---------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Bin:
    """One mass bin of the mass function; everything per solar mass of stars formed."""
    lo: float
    hi: float
    mass: float  # the mean mass of its stars, solar masses
    number: float  # stars
    share: float  # mass
    fate: str
    lifetime: float  # Gyr
    returned: float  # share of the bin's mass that returns to the gas when its stars die


def bin_edges() -> tuple[float, ...]:
    """:data:`EDGES`, cut again at every :func:`turnoff_mass` of a whole number of steps (4.37 solar
    masses for 0.25 Gyr down to 0.887 for 13.5 Gyr), so that all stars of a bin die in one and
    the same step after their birth."""
    edges = list(EDGES)
    for n in range(1, MAX_STEPS + 1):
        cut = turnoff_mass(n * DT)
        # the cut for 10 Gyr is the edge 1.0; a power function that missed it by a rounding error
        # must not leave a bin as wide as that error
        if all(abs(cut - e) > 1e-9 * e for e in edges):
            edges.append(cut)
    return tuple(sorted(edges))


def _bins() -> tuple[Bin, ...]:
    out = []
    edges = bin_edges()
    for lo, hi in zip(edges, edges[1:]):
        number, share = imf_number(lo, hi), imf_mass(lo, hi)
        mass = share / number
        out.append(Bin(lo, hi, mass, number, share, fate(mass), lifetime(mass), returned_mass(mass) / mass))
    return tuple(out)


BINS: tuple[Bin, ...] = _bins()


def generation(ia_fraction: float) -> dict[str, float]:
    """One generation of stars, per solar mass formed, counting what dies within :data:`T_MAX`:
    ``return_fraction``, new ``metals`` returned, ``metal_yield`` = metals / (1 - return_fraction),
    and the events ``supernovae_cc``, ``white_dwarfs``, ``supernovae_ia``."""
    returned = metals = cc = dwarfs = 0.0
    for b in BINS:
        if b.lifetime > T_MAX:
            continue
        gas = b.share * b.returned
        returned += gas
        metals += gas * _sum(v for k, v in YIELDS[b.fate].items() if k in METALS)
        cc += b.number if b.fate == "neutron_star" else 0.0
        dwarfs += b.number if b.fate == "white_dwarf" else 0.0
    ia = ia_fraction * dwarfs
    returned += ia * IA_MASS
    metals += ia * IA_MASS
    return {"return_fraction": returned, "metals": metals, "metal_yield": metals / (1.0 - returned),
            "supernovae_cc": cc, "white_dwarfs": dwarfs, "supernovae_ia": ia}


def white_dwarfs_formed(born: float, age: float) -> float:
    """White dwarfs that ``born`` solar masses of stars have left ``age`` Gyr after they formed:
    the stars from :func:`turnoff_mass` of that age up to 8 solar masses, counted with
    :func:`stars_above`. The gate recomputes the type Ia supernovae with it."""
    if age <= 0:
        return 0.0
    lightest = min(max(turnoff_mass(age), M_MIN), M_SUPERNOVA)
    return stars_above(born, lightest) - stars_above(born, M_SUPERNOVA)


# ---------------------------------------------------------------------------------------------
# lines
# ---------------------------------------------------------------------------------------------

def dense_value(value: float | int | str) -> str:
    """A metric as it stands in a line: a word, or a number to 3 significant digits.

    >>> dense_value(1052.53), dense_value(0.03234), dense_value(0.000412), dense_value("mature")
    ('1 0 5 0', '0 point 0 3 2 3', '4 point 1 2 e minus 4', 'mature')
    """
    if isinstance(value, str):
        return value
    if isinstance(value, int):
        return num(value)
    return num(sig(float(value)))


def param_value(key: str, value: float) -> str:
    """A parameter as it stands in the ``params`` record: an input as given, a derived one like
    any measured number.

    >>> param_value("t_end", 10.25), param_value("ia_fraction", 0.0065), param_value("metal_yield", 0.0073512)
    ('1 0 point 2 5', '0 point 0 0 6 5', '0 point 0 0 7 3 5')
    """
    return num(float(value), sig=5) if key in INPUT_KEYS else dense_value(value)


@dataclass
class StarsRollout(Rollout):
    """A :class:`Rollout` that also keeps the ledger of every step: per element the mass in the
    gas, the mass that went into stars and the mass that came back (fractions of the box), and
    the totals ``formed``, ``died`` and ``ia_mass``."""
    ledger: list[dict] = field(default_factory=list)

    def value(self, text: float | int | str) -> str:
        return dense_value(text)


def random_params(rng: random.Random, rules: int = 7) -> dict[str, float]:
    """Efficiency uniform 0.1-1 per Gyr, end time 1-13.5 Gyr in steps of 0.25, primordial helium
    uniform 0.23-0.27 (hydrogen the rest), ``ia_fraction`` uniform 0.001-0.008."""
    _rules(rules)
    helium = round(rng.uniform(0.23, 0.27), 3)
    return {"efficiency": round(rng.uniform(0.1, 1.0), 2),
            "t_end": DT * rng.randint(4, MAX_STEPS),
            "hydrogen": round(1.0 - helium, 3),
            "helium": helium,
            "ia_fraction": round(rng.uniform(0.001, 0.008), 4)}


def _rules(rules: int) -> None:
    if rules != 7:
        raise ValueError(f"stars is a round-7 level: rules must be 7, not {rules!r}")


def handoff_in(below: dict) -> dict[str, float]:
    """The summary of the first hour (``haishool.cosmos.nucleo``) as this level's primordial gas.

    ``below`` has the mass fractions ``hydrogen`` (hydrogen-1) and ``helium`` (helium-4) and the
    number ratios per hydrogen nucleus ``deuterium`` and ``helium3`` (missing ones count as 0).
    With nucleon numbers 2 and 3 their mass fractions are ``2 * deuterium * hydrogen`` and
    ``3 * helium3 * hydrogen``. Deuterium counts as hydrogen, helium-3 as helium; lithium (1e-9)
    is dropped. The two are scaled to sum to one: ``helium = round(he / (h + he), 4)`` and
    ``hydrogen = round(1 - helium, 4)``.

    >>> handoff_in({"hydrogen": 0.752, "helium": 0.248})
    {'hydrogen': 0.752, 'helium': 0.248}
    """
    h1, he4 = float(below["hydrogen"]), float(below["helium"])
    h = h1 + 2.0 * float(below.get("deuterium", 0.0)) * h1
    he = he4 + 3.0 * float(below.get("helium3", 0.0)) * h1
    if not (math.isfinite(h) and math.isfinite(he) and h > 0 and he >= 0):
        raise ValueError(f"the gas below must have hydrogen and no negative helium: {below!r}")
    helium = round(he / (h + he), 4)
    return {"hydrogen": round(1.0 - helium, 4), "helium": helium}


# ---------------------------------------------------------------------------------------------
# the simulation
# ---------------------------------------------------------------------------------------------

def _death_step(born_step: int, b: Bin) -> int:
    """The step in which the stars of bin ``b`` die that formed in step ``born_step`` (they count
    as born at its beginning): the first step whose time is not before their :func:`death_time`,
    and not before the step that formed them."""
    dies = death_time((born_step - 1) * DT, b.mass)
    return max(born_step, math.ceil((dies - T_EPS) / DT))


class Stars:
    """The simulation and its gate. Rollouts replayed by :meth:`check` are cached per seed."""

    sim = SIM
    topic = SIM
    KEYS = KEYS

    def __init__(self) -> None:
        self._cache: dict[int, StarsRollout] = {}

    def run(self, seed: int, efficiency: float = 0.3, t_end: float = T_MAX, hydrogen: float = 0.752,
            helium: float = 0.248, ia_fraction: float = IA_FRACTION, rules: int = 7) -> StarsRollout:
        """Deterministic for given parameters; the seed only names the run.

        Raises ``ValueError`` unless 0 < efficiency <= 4 (per Gyr), ``t_end`` is a multiple of
        0.25 from 0 to 13.5, hydrogen > 0 and helium >= 0 sum to one (to 1e-9), and
        0 <= ia_fraction <= 0.1."""
        _rules(rules)
        efficiency, t_end, hydrogen, helium, ia_fraction = (float(efficiency), float(t_end), float(hydrogen),
                                                            float(helium), float(ia_fraction))
        if not (math.isfinite(efficiency) and 0 < efficiency <= 4):
            raise ValueError(f"efficiency must be above 0 and at most 4 per Gyr: {efficiency!r}")
        n_steps = round(t_end / DT) if math.isfinite(t_end) else -1
        if not (0 <= n_steps <= MAX_STEPS and abs(n_steps * DT - t_end) <= 1e-9):
            raise ValueError(f"t_end must be a multiple of {DT} from 0 to {T_MAX}: {t_end!r}")
        if not (math.isfinite(hydrogen) and math.isfinite(helium) and hydrogen > 0 and helium >= 0
                and abs(hydrogen + helium - 1.0) <= 1e-9):
            raise ValueError(f"hydrogen and helium must sum to one: {hydrogen!r}, {helium!r}")
        if not (math.isfinite(ia_fraction) and 0 <= ia_fraction <= 0.1):
            raise ValueError(f"ia_fraction must be between 0 and 0.1: {ia_fraction!r}")
        gen = generation(ia_fraction)
        params = {"efficiency": efficiency, "t_end": t_end, "hydrogen": hydrogen, "helium": helium,
                  "ia_fraction": ia_fraction, "metal_yield": gen["metal_yield"],
                  "return_fraction": gen["return_fraction"]}
        r = StarsRollout(SIM, int(seed), params, [])

        total = hydrogen + helium
        gas = {e: 0.0 for e in ELEMENTS}
        gas["hydrogen"], gas["helium"] = hydrogen / total, helium / total
        astrated = {e: 0.0 for e in ELEMENTS}
        returned = {e: 0.0 for e in ELEMENTS}
        box = {"stars": 0.0, "remnants": 0.0, "formed": 0.0, "died": 0.0, "ia_mass": 0.0, "cc": 0.0, "ia": 0.0}
        deaths: dict[int, list[tuple[float, dict[str, float], Bin]]] = {}  # step -> (mass formed, make-up, bin)
        ia_due: dict[int, float] = {}  # step -> events in the box
        ia_steps = round(IA_DELAY / DT)

        def snapshot(k: int) -> None:
            mass = _sum(gas[e] for e in ELEMENTS)
            frac = {e: gas[e] / mass for e in ELEMENTS}
            metallicity = _sum(frac[e] for e in METALS)
            r.steps.append({
                "time": k * DT, "gas": mass, "stars": box["stars"], "remnants": box["remnants"],
                "metallicity": metallicity, "helium": frac["helium"], "oxygen": frac["oxygen"],
                "carbon": frac["carbon"], "iron": frac["iron"], "nitrogen": frac["nitrogen"],
                "supernovae_cc": box["cc"], "supernovae_ia": box["ia"],
                "alpha_over_iron": alpha_over_iron(frac["oxygen"], frac["iron"]),
                "stage": stage(mass, metallicity),
                "hydrogen": frac["hydrogen"], "neon": frac["neon"], "magnesium": frac["magnesium"],
                "silicon": frac["silicon"], "other": frac["other"]})
            r.ledger.append({"gas": dict(gas), "astrated": dict(astrated), "returned": dict(returned),
                             "formed": box["formed"], "died": box["died"], "ia_mass": box["ia_mass"]})

        snapshot(0)
        for k in range(1, n_steps + 1):
            # 1. stars form out of the gas as it is at the beginning of the step
            mass = _sum(gas[e] for e in ELEMENTS)
            formed = gas_to_stars(mass, efficiency, DT)
            makeup = {e: gas[e] / mass for e in ELEMENTS}
            for e in ELEMENTS:
                gas[e] -= formed * makeup[e]
                astrated[e] += formed * makeup[e]
            box["stars"] += formed
            box["formed"] += formed
            for b in BINS:
                deaths.setdefault(_death_step(k, b), []).append((formed, makeup, b))
            # 2. the stars whose time has come die
            for born_mass, born, b in deaths.pop(k, []):
                dying = born_mass * b.share
                back = dying * b.returned
                table = YIELDS[b.fate]
                new = _sum(table.values())
                scale = min(1.0, born["hydrogen"] / new) if new > 0 else 0.0  # no more than its hydrogen allows
                for e in ELEMENTS:
                    out = back * (born[e] - scale * new if e == "hydrogen" else born[e] + scale * table.get(e, 0.0))
                    gas[e] += out
                    returned[e] += out
                box["stars"] -= dying
                box["died"] += dying
                box["remnants"] += dying - back
                count = stars_above(born_mass * BOX_MASS, b.lo) - stars_above(born_mass * BOX_MASS, b.hi)
                if b.fate == "neutron_star":
                    box["cc"] += count
                elif b.fate == "white_dwarf" and ia_fraction > 0:
                    ia_due[k + ia_steps] = ia_due.get(k + ia_steps, 0.0) + ia_fraction * count
            # 3. the type Ia supernovae that are due
            events = ia_due.pop(k, 0.0)
            if events > 0:
                for e, solar_masses in IA_YIELD.items():
                    out = (ia_iron(events) if e == "iron" else solar_masses * events) / BOX_MASS
                    gas[e] += out
                    returned[e] += out
                box["remnants"] -= IA_MASS * events / BOX_MASS
                box["ia_mass"] += IA_MASS * events / BOX_MASS
                box["ia"] += events
            snapshot(k)

        last = r.steps[-1]
        r.summary = {
            **{k: last[k] for k in ("metallicity", "oxygen", "carbon", "iron", "nitrogen", "neon", "magnesium",
                                    "silicon", "other", "helium", "hydrogen")},
            "gas_fraction": last["gas"], "stars_formed": box["formed"],
            "supernovae": box["cc"] + box["ia"], "supernovae_cc": box["cc"], "supernovae_ia": box["ia"],
            "alpha_over_iron": last["alpha_over_iron"],
            "closed_box": closed_box(gen["metal_yield"], last["gas"]),
            "time": last["time"], "stage": last["stage"],
        }
        return r

    def rollout(self, seed: int) -> StarsRollout:
        """The canonical run of a seed: parameters from ``random_params(random.Random(seed))``."""
        if seed not in self._cache:
            if len(self._cache) >= 512:
                self._cache.clear()
            self._cache[seed] = self.run(seed, **random_params(random.Random(seed)))
        return self._cache[seed]

    def conserved(self, r: Rollout) -> Verdict:
        ledger = getattr(r, "ledger", None)
        if not ledger or len(ledger) != len(r.steps):
            return Verdict(False, None, "the rollout has no ledger for every step")
        first = ledger[0]["gas"]
        if abs(_sum(first.values()) - 1.0) > 1e-9:
            return Verdict(False, "1", "the box does not start as one unit of gas")
        prev = prev_led = None
        ia_steps = round(IA_DELAY / DT)
        for t, (s, led) in enumerate(zip(r.steps, ledger)):
            gas, astrated, returned = led["gas"], led["astrated"], led["returned"]
            if any(set(d) != set(ELEMENTS) for d in (gas, astrated, returned)):
                return Verdict(False, None, f"step {t}: the ledger does not list every element")
            total = s["gas"] + s["stars"] + s["remnants"]
            if abs(total - 1.0) > 1e-9:
                return Verdict(False, "1", f"step {t}: gas + stars + remnants = {total!r}, not the box")
            if min(s["gas"], s["stars"], s["remnants"]) < -1e-12 or min(gas.values()) < -1e-12:
                return Verdict(False, None, f"step {t}: a negative mass")
            if s["gas"] <= 0 or abs(_sum(gas.values()) - s["gas"]) > 1e-9:
                return Verdict(False, None, f"step {t}: the elements do not add up to the gas")
            for e in ELEMENTS:
                if abs(first[e] - astrated[e] + returned[e] - gas[e]) > 1e-9:
                    return Verdict(False, None, f"step {t}: {e} is not accounted for")
            if abs(_sum(astrated.values()) - led["formed"]) > 1e-9:
                return Verdict(False, None, f"step {t}: the stars formed do not match what left the gas")
            if abs(led["formed"] - led["died"] - s["stars"]) > 1e-9:
                return Verdict(False, None, f"step {t}: living stars are not formed minus died")
            if abs(led["died"] - _sum(returned.values()) - s["remnants"]) > 1e-9:
                return Verdict(False, None, f"step {t}: remnants are not the dead stars less the returned gas")
            if returned["hydrogen"] > astrated["hydrogen"] + 1e-12:
                return Verdict(False, None, f"step {t}: more hydrogen came back than went into stars")
            fractions = {e: gas[e] / s["gas"] for e in ELEMENTS}
            for e in ELEMENTS:
                if not close(s[e], fractions[e], rel=1e-9, abs_=1e-12):
                    return Verdict(False, None, f"step {t}: {e} is not its share of the gas")
            metals = _sum(s[e] for e in METALS)
            if not close(s["metallicity"], metals, rel=1e-9, abs_=1e-12):
                return Verdict(False, num(sig(metals)), f"step {t}: metallicity is not the sum of the metals")
            if abs(s["hydrogen"] + s["helium"] + s["metallicity"] - 1.0) > 1e-9:
                return Verdict(False, None, f"step {t}: the mass fractions of the gas do not sum to one")
            if s["stage"] != stage(s["gas"], s["metallicity"]):
                return Verdict(False, stage(s["gas"], s["metallicity"]), f"step {t}: wrong stage")
            if not close(s["alpha_over_iron"], alpha_over_iron(s["oxygen"], s["iron"]), rel=1e-9, abs_=1e-12):
                return Verdict(False, None, f"step {t}: alpha_over_iron does not follow from oxygen and iron")
            if abs(led["ia_mass"] * BOX_MASS - IA_MASS * s["supernovae_ia"]) > 1e-6:
                return Verdict(False, None, f"step {t}: the type Ia supernovae do not match the mass they destroyed")
            # stars formed in step j count as born at (j - 1) * DT; their white dwarfs explode IA_DELAY later
            due = float(r.params["ia_fraction"]) * _sum(
                white_dwarfs_formed((ledger[j]["formed"] - ledger[j - 1]["formed"]) * BOX_MASS,
                                    (t - ia_steps - j + 1) * DT)
                for j in range(1, t - ia_steps + 1))
            if not close(s["supernovae_ia"], due, rel=1e-9, abs_=1e-6):
                return Verdict(False, num(sig(due)),
                               f"step {t}: the type Ia supernovae are not ia_fraction of the white dwarfs formed "
                               f"{IA_DELAY} Gyr before")
            if prev is None:
                if s["time"] != 0 or s["stars"] != 0 or s["remnants"] != 0 or s["supernovae_cc"] != 0:
                    return Verdict(False, None, "step 0 is not a box of gas alone at time zero")
            else:
                if not s["time"] > prev["time"]:
                    return Verdict(False, None, f"step {t}: time does not increase")
                if s["supernovae_cc"] < prev["supernovae_cc"] or s["supernovae_ia"] < prev["supernovae_ia"]:
                    return Verdict(False, None, f"step {t}: a supernova count went down")
                if led["formed"] < prev_led["formed"] or led["died"] < prev_led["died"]:
                    return Verdict(False, None, f"step {t}: stars formed or died went down")
                if any(astrated[e] < prev_led["astrated"][e] or returned[e] < prev_led["returned"][e] - 1e-15
                       for e in ELEMENTS):
                    return Verdict(False, None, f"step {t}: the ledger went backwards")
                born = gas_to_stars(prev["gas"], float(r.params["efficiency"]), DT)
                if abs(led["formed"] - prev_led["formed"] - born) > 1e-9:
                    return Verdict(False, None, f"step {t}: the stars formed are not gas_to_stars of the gas before")
            solar_masses = led["formed"] * BOX_MASS
            massive = stars_above(solar_masses, M_SUPERNOVA) - stars_above(solar_masses, M_BLACK_HOLE)
            if not close(s["supernovae_cc"], massive, rel=1e-9, abs_=1e-9):
                return Verdict(False, num(sig(massive)),
                               f"step {t}: core-collapse supernovae are not the stars of 8 to 25 solar masses formed so far")
            prev, prev_led = s, led
        return Verdict(True, None, "mass, every element, the fractions, counts and time are consistent")

    def parse(self, prompt: str) -> tuple[int, str, int, str] | None:
        """``(seed, where, step, key)`` with ``where`` in ``step``, ``next``, ``final``, ``params``.
        A step must exist in the seed's canonical rollout (its length depends on ``t_end``)."""
        w = prompt.split(" ")
        if len(w) < 5 or w[0] != SIM or w[1] != "seed":
            return None
        i = _digits_end(w, 2)
        seed = parse_num(w[2:i])
        if not isinstance(seed, int) or i >= len(w) or i - 2 > 12 or num(seed).split() != w[2:i]:  # no leading zeros
            return None
        where, rest = w[i], w[i + 1:]
        if where in ("final", "params"):
            keys = SUMMARY_KEYS if where == "final" else PARAM_KEYS
            return (seed, where, -1, rest[0]) if len(rest) == 1 and rest[0] in keys else None
        if where != "step":
            return None
        j = _digits_end(rest, 0)
        digits, rest = rest[:j], rest[j:]
        step = parse_num(digits)
        if not isinstance(step, int) or not 0 <= step <= MAX_STEPS or num(step).split() != digits:
            return None
        if not rest or rest[-1] not in STATE_KEYS + EXTRA_KEYS or len(rest) > 2 or (len(rest) == 2 and rest[0] != "next"):
            return None
        where = "next" if len(rest) == 2 else "step"
        if step + (where == "next") >= len(self.rollout(seed).steps):
            return None
        return seed, where, step, rest[-1]

    def owns(self, prompt: str) -> bool:
        return self.parse(prompt) is not None

    def truth(self, prompt: str) -> float | int | str | None:
        """The simulation's own value for a prompt, or ``None`` when it is not this level's."""
        q = self.parse(prompt)
        if q is None:
            return None
        seed, where, step, key = q
        r = self.rollout(seed)
        if where == "final":
            return r.summary[key]
        if where == "params":
            return r.params[key]
        return r.steps[step + (where == "next")][key]

    def check(self, prompt: str, answer: str) -> Verdict:
        """Replay the seed and compare: words exactly, numbers within 5 percent."""
        value = self.truth(prompt)
        if value is None:
            return Verdict(False, None, "not my question")
        words = prompt.split()
        key = words[-1]
        expected = param_value(key, value) if words[-2] == "params" else dense_value(value)
        if isinstance(value, str):
            ok = answer.strip() == value
            return Verdict(ok, expected, "exact word" if ok else "wrong word")
        got = parse_num(answer)
        if not is_finite(got):  # "1 e 9 9 9" parses to infinity, which is close to nothing
            return Verdict(False, expected, "not a number")
        ok = close(float(got), float(value), rel=0.05, abs_=1e-12)
        return Verdict(ok, expected, "within 5 percent" if ok else "more than 5 percent off")

    def records(self) -> list[Line]:
        return []

    def generate(self, rng: random.Random, n: int) -> list[Line]:
        """All lines of ``n`` canonical rollouts with seeds drawn from ``rng``."""
        out: list[Line] = []
        for seed in seeds(rng, n):
            out += lines(self.rollout(seed))
        return out


def _digits_end(words: list[str], start: int) -> int:
    """Index after the run of single-digit tokens that begins at ``start``."""
    i = start
    while i < len(words) and words[i] in _DIGITS:
        i += 1
    return i


def params_line(r: Rollout) -> Line:
    """``stars seed 3 params. efficiency 0 point 3 1. t_end 1 0 point 2 5. hydrogen 0 point 7 5 1. ...``"""
    fields = " ".join(f"{k} {param_value(k, r.params[k])}." for k in PARAM_KEYS)
    return Line(f"{r.sim} seed {num(r.seed)} params. {fields}", topic=r.sim, kind="record")


def lines(r: Rollout, every: int = 1) -> list[Line]:
    """The parameter record, a state record and questions for every ``every``-th step (now and
    next), and the final questions. The ledger and the five extra keys per step are left out; the
    summary has every element at the end. Only the lines of a canonical rollout
    (:func:`is_canonical`, what :func:`rollout` and :meth:`Stars.generate` give) are true for
    :func:`check`, which replays the seed's own parameters; the lines of a what-if run show what
    that run did and must not be mixed into training data."""
    keys = list(STATE_KEYS)
    out = [params_line(r)]
    for t in range(0, len(r.steps), every):
        out.append(state_line(r, t, keys))
        out += query_lines(r, t, keys)
    return out + summary_lines(r)


def is_canonical(r: Rollout) -> bool:
    """True when ``r`` is :func:`rollout` of its seed: the run whose lines :func:`check` accepts.
    A run with other parameters under the same seed (a what-if run, a link of a chain) has the
    same prompts with other answers; its lines are not training data for this level."""
    if r.sim != SIM or type(r.seed) is not int or r.seed < 0:
        return False
    mine = _SIM.rollout(r.seed)
    return r.params == mine.params and r.steps == mine.steps and r.summary == mine.summary


def cloud(r: Rollout, time: float | None = None) -> dict[str, float]:
    """The make-up of the gas at ``time`` Gyr (default: the end): the ten mass fractions,
    ``metallicity`` and ``time``. This is what a star that forms then is made of. ``time`` must be
    a step of the rollout (a multiple of 0.25 Gyr up to its ``t_end``)."""
    k = len(r.steps) - 1 if time is None else round(float(time) / DT)
    if not 0 <= k < len(r.steps) or (time is not None and abs(k * DT - float(time)) > 1e-9):
        raise ValueError(f"the rollout has no step at {time!r} Gyr")
    step = r.steps[k]
    return {**{e: step[e] for e in ELEMENTS}, "metallicity": step["metallicity"], "time": step["time"]}


_SIM = Stars()


def simulation() -> Stars:
    return _SIM


def run(seed: int, **params) -> StarsRollout:
    return _SIM.run(seed, **params)


def rollout(seed: int) -> StarsRollout:
    return _SIM.rollout(seed)


def conserved(r: Rollout) -> Verdict:
    return _SIM.conserved(r)


def check(prompt: str, answer: str) -> Verdict:
    """Judge an answer to a question about a rollout (``stars seed ...``). Lessons
    (``stars predict ...``) are judged by :data:`LESSONS`."""
    return _SIM.check(prompt, answer)


def owns(prompt: str) -> bool:
    return _SIM.owns(prompt)
