"""Level 10 of the ladder: bodies get sensors and nerves when sensing pays.

A population of at most 200 agents lives on a ring (``dims`` 1) or a small grid (``dims`` 2) with
food and predators. Nothing moves on a screen: in every generation each agent's food intake, its
costs and its chance to escape are *computed* from its genome by the rules below, and only two
things are drawn from the seed: who is caught, and who leaves how many offspring with which
mutations. A run is 300 generations; every fifth is an output step, 60 in all.

**Genome** (seven genes): four sensors with a level 0 to 5 each (``smell`` for chemicals,
``eyes`` for light, ``touch``, ``ears`` for sound); a neuron gene (0 neurons, or 2, 4, 8 ... 1024);
a speed 1 to 5; a voice (0 or 1: the ability to make a sound, on which the next level builds its
signals). The 100 founders are blind, deaf, mute and without neurons, at speed 1.

**Sensing.** A sensor of level L has L * L receptors. Pooling n receptors improves the signal to
noise ratio by sqrt(n) (the standard pooling rule for independent noise), and the toy sets the
range in cells equal to that gain: range = reach * sqrt(L * L), with a reach of 1 cell for smell,
eyes and ears and of 0.1 for touch (touch works by contact). A range r covers 2 r + 1 cells on
the ring and (2 r + 1)^2 on the grid. Each sensor works only where its medium is:

    smell   finds food, slowly (following a trail takes time): a smelled cell counts 0.4
    eyes    find food, a seen cell counts ``light`` (0 dark .. 1 bright); notice a predator at
            0.5 * light * range (an eye looks one way)
    touch   finds food and feels a predator at its short range, in any light
    ears    hear a predator at their full range, and the voices of mates; food is silent

Two sensors watching the same cell find its food once: the best channel counts. What each sense
is good for is set by hand (toy rules), and so are the 0.4 and the 0.5.

**Nerves.** Using a sensor means linking what it reports to an action: one association per
further cell watched, summed over the sensors. One association is free (a sensor cell wired
straight to a muscle cell); every further one needs two neurons. An agent with n neurons whose
sensors would need k associations uses the share min(1, (1 + n // 2) / k) of what they could tell.

**The body** comes from level 9. With one kind of cell it has no organs: no sensors, no voice.
Neurons need at least 3 cell types, and at most a quarter of the cells can be neurons (rounded
down to a power of two, at most 1024).

**Food** has the mean density ``density`` (items per cell). With ``patchiness`` p it sits in
fewer places, each holding 1 / (1 - p) items. An agent meets food places at the rate
speed * density * (1 - p) * cells, where ``cells`` is 1 for an agent without sensors (its own
cell: random encounter) and else the cells it can make use of, and takes from a place what lies
within those cells, at most the whole clump (an agent that comes upon one item of a clump does
not know of the others unless its sensors show them):

    rate   = speed * density * (1 - p) * cells * min(1 / (1 - p), cells)
    intake = yield * rate / (1 + 0.5 * rate)                    (Holling's disc equation)

An item yields 100 energy units from a twentieth of today's oxygen on, and less below:
100 * (0.0625 + 0.9375 * min(1, oxygen / 0.05)); 0.0625 is 2 of 32 ATP per glucose.

**Costs** per generation, in the same units; ``cost`` multiplies the cost of tissue:

    sensors   cost * unit * (receptors + 3)    unit: smell 1, eyes 2, touch 0.5, ears 1; the 3 is
                                               the organ itself; nothing at level 0
    neurons   cost * 0.05 * neurons            (nerve tissue is expensive)
    movement  0.5 * speed^3 (the power against a drag that grows with the square of the speed),
              and of what is then left: 0.06 for calling (with a voice) and 0.4 * (1 - found)
              for the search for a mate

``found`` is the share of the mate search that sound saves: 0.5 * voice * (mean hearing of the
population) + 0.5 * (own hearing) * (share with a voice); hearing is ears / 5 times the share of
its sensor information the agent can use. So a voice costs, and is repaid only where there are
ears; and ears find callers only where there are voices.

**Predators.** An agent is attacked in a generation with probability
1 - exp(-predators * (2 s + 1)), where s is the predators' own sensor level (a hunter sweeps a
strip of width 2 s + 1). An attacked agent that notices the predator at distance D (its head
start) flees at its speed v from a predator of speed 6 to a refuge that is anywhere up to 4
cells away: it escapes with probability min(1, D v / ((6 - v) * 4)), and not at all when D is 0.
The predators answer one step (5 generations) late: looking at the prey of the step before, they
raise their sensor one level when their meal (2 s + 1) * (1 - mean escape) * (prey / 200) is
below 1.5 and lower it when one level less would still feed them. Prey that escape better are
hunted by sharper predators: an arms race.

**A generation.** (1) Every agent forages and pays its costs; one whose costs eat up its intake
starves, paying as far as the intake goes. (2) The others are attacked and caught or not (drawn).
(3) The survivors' surplus is turned into offspring at 0.5 units each, at most 200; each
offspring draws its parent with probability surplus / total surplus (fitness is net energy,
times survival through step 2); what is left over is lost with the adults (``e_stored``).
(4) Each gene of an offspring mutates with probability ``mutation``: a sensor level, the neuron
gene or the speed one step, down twice as often as up (a gene is easier to break than to
improve), the voice lost twice as often as gained. (5) The adults die.

**Stages**, the first that applies: ``starved`` (nobody left), ``arms_race`` (predators with
sensor level 3 or more), ``brained`` (32 neurons or more on average), ``seeing`` (half the agents
have eyes), ``smelling`` (half have some sensor, whichever it is), ``blind``. The summary's
``outcome`` is ``starved``, ``brained``, ``seeing``, ``sensing`` or ``blind`` by the same tests
without the predators. These are labels of the toy: ``brained`` says that the agents have 32
neurons on average by this level's count, and nothing about what a brain is.

Lines (:func:`lines`; metrics to 3 significant digits, parameters as given; seed 35 verbatim, its
stages being blind, seeing from step 2, brained from 17, arms_race from 23; the longest line of
seeds 1 to 40 has 86 tokens):

    senses seed 3 5 params. light 0 point 7 6. patchiness 0 point 2 3. density 0 point 4 9. predators 0 point 2 6.
        cost 1 point 3 5. mutation 0 point 0 2 4. dims 2. oxygen 0 point 2 0 2. body_size 6 5 9 1. cell_types 1 0.
        max_level 5. max_neurons 1 0 2 4. eye_pays 2.
    senses seed 3 5 step 4 0. population 2 0 0. smell 0 point 1. eyes 1 point 1 3. touch 0 point 0 9.
        ears 1 point 9 6. eyed 0 point 9 7. neurons 8 5 point 6. voice 0 point 5. speed 4. intake 1 7 4.
        survival 0 point 9 3 3. predator_sense 5. stage arms_race.
    q senses seed 3 5 step 4 0 eyed. a 0 point 9 7.
    q senses seed 3 5 step 4 0 next neurons. a 8 5 point 3.       (what happens next)
    q senses seed 3 5 final outcome. a brained.                    (how it ends)
    q senses predict escape head_start 3 speed 2 predator_speed 6. a 0 point 3 7 5.     (a lesson)

**Gate** (:meth:`Senses.conserved`), every one of the 300 generations: the energy is accounted
for, ``e_intake = e_sensors + e_neurons + e_movement + e_reproduction + e_stored + e_eaten`` (the
last is the surplus of the agents that were caught: it goes to the predators), to 1e-9 of the
intake; ``starved + caught + old = population``, and the next population is ``births`` (every
adult dies, so births minus deaths is births minus population); ``e_reproduction`` is 0.5 per
birth and no surplus for a further offspring is left unused below the cap; every probability and
share lies in 0..1; ``attack`` and ``survival`` follow from the predators and the mean escape; a
share and its mean level fit (``eyed <= eyes <= 5 * eyed``); no level, voice or neuron beyond
what the body allows; the predators hold their level within a step and change it by their rule;
the stage follows from the numbers. For the 60 steps the genomes are kept, and the metrics must
be what the rules give for them. The summary must follow from the last step.
:meth:`Senses.check` replays a seed (parameters from :func:`random_params` with
``random.Random(seed)``) and compares: words, counts and parameters exactly, other numbers
within 5 percent. Module level :func:`check` and :func:`owns` also answer the lessons.

**Lessons** (:data:`LESSONS`, 24 rules in :data:`RULES`): every rule above as a question that
carries its inputs (``senses predict <rule> <input> <value> ...``). The functions that answer
them are the functions the simulation calls, so a lesson is true of every rollout. The gate
judges bodies up to the largest level 9 can hand over (2^20 cells, 30 cell types). Inputs are
drawn evenly from their ranges, except where that gives one answer most of the time
(:class:`SensesLessons` says where, and how they are drawn instead). Over 2000 lessons the most
common answer of a rule is then at most about half of its lessons (``LESSONS.generate`` with
``random.Random`` 1 to 4, 500 each): ``max_level``, which has two answers, 5 in 0.51;
``break_even`` 2 in 0.44 (0.36 over 20000 draws, with 3 in 0.33 and 0 in 0.22); ``energy_yield``
100 in 0.40; ``stage`` brained in 0.31 (arms_race 0.20: the predators' level decides before
anything else is looked at, so it is drawn at 3 or more only once in four); every other rule in
0.30 or less.

Hand-off. :func:`handoff_in` turns the summary of level 9 (bodies) into parameters: body size
and cell types (which limit organs and neurons), the predators' share of the biomass (predator
density) and the oxygen (yield of food). The summary gives upward, to level 11 (signals):
``eared`` and ``voice`` (the shares with ears and with a voice), ``talkers`` (the share with
both), ``neurons`` (mean per agent), ``predator_pressure`` (the chance to be attacked in a
generation at the end), ``kin_gain`` and ``group`` (would an agent gain from staying near kin:
:func:`kin_gain`, ``yes`` from 0.01), ``population``, and ``stage``, ``eyes``, ``ears``,
``survival``, ``outcome``. :func:`handoff_out` picks from the summary what level 11 is to take.

Plausibility (``tests/test_evo_senses.py`` holds the targets). Measured in a world where every
sense can pay (patchiness 0.6, density 0.2, cost 1, mutation 0.02, today's oxygen, a body of
5000 cells with 6 cell types), means of the last step over seeds 1 to 8, ring / grid:

    no eyes in the dark (light 0, no predators)        eyes 0.04 / 0.08, at most 0.32; smell 3.9 / 2.7
    eyes with light (light 1, no predators)            eyes 2.9 / 2.4, every agent has one in every run
    neurons with the sensors in use (dark)             one sensor 24 / 75, two or more 43 / 185
    neurons with what they have to tell apart          bright, no predators: ring 20, grid 84
    neurons against the cost of tissue (bright, hunted)  cost 0.5: 36 / 221; cost 2: 19 / 69
    predators (density 0.2, dark) bring ears           0.16 / 0.12 without, 2.2 / 1.4 with
        speed                                          3.0 / 2.7 without, 4.0 / 3.9 with
        and a voice, where there are ears              0.07 / 0.07 without, 0.71 / 0.40 with

Clumped food does less than one might expect, and less than a first version of this text
claimed. In full light without predators the eyes end at the same level whether the food is
spread evenly or clumped (patchiness 0: 2.9 / 2.2; 0.8: 2.9 / 2.4), and the eye level that pays
by the rules is the same. What clumping changes is how poor the blind are (a founder takes in
18 units at patchiness 0, 3.9 at 0.8), so the first eye spreads sooner: half the agents have one
after 3.6 / 7.0 steps with food spread evenly, after 2.0 / 2.4 at patchiness 0.8.

Over the canonical seeds 1 to 300 (parameters from :func:`random_params`): outcomes sensing 143,
brained 62, seeing 61, blind 33, starved 1. 31 worlds have bodies of one kind of cell, which
stay blind; the counts that follow are over the other 269. The 47 dark worlds end with a mean
eye level of 0.03, at most 0.18. Of 121 bright worlds (light 0.5 or more) 81 end with eyes in
at least half the agents; of the 79 worlds with some light below 0.4, where a nose finds more
food than an eye of the same level at half the cost per receptor, 2 do. The mean eye level
rises with ``eye_pays`` (0.02, 0.38, 0.90, 1.19, 1.83 for 0 to 4). ``eye_pays`` asks about an
eye alone, not about an eye against a nose: the one world with 5 has a light of 0.32 and ends
with noses and ears (eyes 0.08). Ears 0.05 without predators (49 worlds) against 0.50 with them
(220), speed 2.7 against 3.1; a voice in 35 % of the agents where at least half have ears (59
worlds), in 5 % elsewhere; ``group`` is yes in 33 worlds; the predators end at level 3 or more
in 37 of the 116 worlds with a predator density of 0.15 or more. The worst relative error of
the energy ledger is 3e-16. A run takes 0.13 s (median; 0.15 s at most, on the machine this was
measured on).

Two things the numbers show that were not put in. A useless organ is not quite absent: mutation
keeps making level-1 eyes in the dark and selection removes them (a few per cent of the agents
carry one at any time). And evolution can be caught on a low hill (a local optimum, in the
picture of a fitness landscape after Wright 1932): on a dark grid a population either grows a
nose and the brain to use it, or stays with touch, which needs hardly any neurons and so pays
at once, but from which every first step towards a nose costs more than it brings (seeds 2 and
5 of the eight above).

What is taken from published models and what is invented.

* Holling's disc equation (Holling 1959), the type II functional response: intake rises with
  the encounter rate and levels off because every item takes time to handle. Taken as it is,
  with an invented handling time of 0.5.
* Encounter rate = speed * density * what the searcher covers: the kinetic picture of
  encounters (after Gerritsen and Strickler 1977, who worked it out for swimming plankton in
  three dimensions; only the proportionality is taken). The predators' strip of width 2 s + 1 is
  a searcher sweeping a strip. That the prey's rate goes with its detection *area* (r on a ring,
  r^2 on a grid) is this toy's picture of an agent that checks its whole neighbourhood at every
  move. That a clump is only taken as far as the agent's cells reach is invented; its effect is
  that the blind get less from clumped food, while an agent that watches as many cells as a
  clump holds items gets what it would get from food spread evenly.
* sqrt(n) gain from pooling n receptors with independent noise: standard signal averaging. That
  the range equals this gain, and that a level has level^2 receptors, is invented.
* What each sense is good for (a nose finds food slowly, an eye needs light and looks one way,
  touch is contact, ears hear predators and voices but no food) is a caricature set by hand; the
  0.4, the 0.5 and the reaches are invented.
* Fleeing: the prey runs for a refuge while the predator closes the head start at the difference
  of the speeds. Ydenberg and Dill 1986 treat the distance at which prey starts to flee as a
  decision that weighs the cost of fleeing against the risk; here there is no decision, the
  prey flees as soon as it notices, and only the race is taken. The uniform refuge distance and
  the two constants are invented.
* Poisson chance of at least one attack, 1 - exp(-rate): elementary.
* Nerve and sense tissue costs energy all the time. Aiello and Wheeler 1995 argued this for the
  human brain (the expensive tissue hypothesis: compared with other primates of their size,
  humans pay for a large brain with a small gut; that trade-off is not modelled), Niven and Laughlin 2008 review the energy cost of sensing as a
  pressure on the size of sense organs. Only the idea is taken: an organ costs in proportion to
  its size and spreads only where it brings in more. Cave fish without eyes are the usual
  example; whether they lost them to save energy is debated, and the toy does not settle it.
  Two neurons per association, one free association, the unit costs, the organ cost and the
  limits from cell types and body size are toy rules. The quarter of the cells is a round
  number; for scale, the worm C. elegans has 302 neurons among 959 body cells (from memory),
  nearly a third.
* Predators and prey sharpening each other's senses: the arms race idea (after Dawkins and Krebs
  1979). The predators' rule of thumb and its lag are invented.
* The chance that at least one of n notices a predator, 1 - (1 - p)^n: the many eyes idea of
  group vigilance (after Pulliam 1973), used for :func:`kin_gain` only; the population of this
  level is not grouped, and ``kin_gain`` and ``group`` are an index for the next level, not
  something the agents do.
* Aerobic against anaerobic yield (about 32 against 2 ATP per glucose, textbook values from
  memory): the ratio is taken; that the yield rises in proportion to the oxygen and is full at a
  twentieth of today's is a toy rule (the Pasteur point, below which yeast turn from respiration
  to fermentation, is often put near a hundredth of today's level; no real organism is meant).
* Selection is fitness-proportional sampling of parents, with replacement, in non-overlapping
  generations (like a Wright-Fisher population, but its size is set by the energy each
  generation and only capped at 200) with stepwise mutation; mutations that break being more common than mutations that build is a
  common observation, the ratio 2 to 1 is invented.
* The voice and the search for a mate are invented from end to end: many animals do call to be
  found, but the shares 0.06 and 0.4 and the rule for ``found`` come from no model.
* Not modelled: learning, behaviour in time, sex (a mate is searched and paid for, but an
  offspring has one parent), predators that hear voices, the smell of predators, colour, kin
  groups, the anatomy or the history of any real eye, ear or brain. The energy units are
  arbitrary. No number here is a measurement of a real animal, and nothing here says how or when
  eyes, ears, nerves or voices arose in the history of life: the level shows only that, under
  these rules, an organ spreads where it brings in more than it costs.

Reproducibility: pure Python floats; totals by ``math.fsum``, and the running sum by which
parents are drawn added in the order of the population; squares, cubes and powers are
multiplied out (a product and a square root are rounded the same way on every machine, ``pow``
is not); a run draws only ``random()`` from ``random.Random(7919 * seed + 10)`` (the Mersenne
Twister, whose ``random()`` gives the same numbers in every Python); a run sorts nothing; numpy
is not used. What is left: ``math.expm1`` (the chance of an attack) may differ in the last digit
between machines; a change ten thousand times larger moves no line of seeds 1 to 8 and no
answer of an ``attack`` lesson (see the tests). :func:`random_params` uses ``uniform``,
``choice``, ``randint`` and a power of ten, which have given the same numbers since Python 3.2
but are not promised to; the tests pin the parameters of seed 35.
"""

from __future__ import annotations

import math
import random
from bisect import bisect_right
from dataclasses import dataclass, field

from haishool.cosmos import Rollout, close, query_lines, seeds, state_line, summary_lines
from haishool.evo import Input, LessonGate, Rule, sig
from haishool.truth import Line, Verdict, is_finite, num, parse_num

SIM = "senses"

CAP = 200  # agents at most
START = 100  # agents at the start
STEPS = 60  # output steps
GENS_PER_STEP = 5  # generations from one output step to the next
SENSORS = ("smell", "eyes", "touch", "ears")
MAX_LEVEL = 5
MAX_NEURON_GENE = 10  # 2^10 = 1024 neurons
SPEEDS = (1, 5)
START_SPEED = 1
REFERENCE_SPEED = 2  # speed of the agent for which ``eye_pays`` is worked out
#: cells of range per unit of pooling gain
REACH = {"smell": 1.0, "eyes": 1.0, "touch": 0.1, "ears": 1.0}
#: energy per receptor and generation
UNIT_COST = {"smell": 1.0, "eyes": 2.0, "touch": 0.5, "ears": 1.0}
ORGAN = 3  # a sense organ costs as much as this many receptors before it has any (toy)
SMELL_SLOW = 0.4  # what a smelled cell counts against a seen one
EYE_FIELD = 0.5  # share of its range at which an eye notices a predator
CALL_SHARE = 0.06  # share of its spare energy an agent with a voice spends on calling
NEURON_COST = 0.05  # energy per neuron and generation
NEURONS_PER_ASSOCIATION = 2
FREE_ASSOCIATIONS = 1
SENSOR_CELL_TYPES = 2  # cell types a body needs before it can have sense organs or a voice
NEURON_CELL_TYPES = 3  # cell types a body needs before it can have neurons
NEURON_SHARE = 0.25  # share of a body's cells that can be neurons
MOVE_COST = 0.5  # energy per speed^3
MATE_SHARE = 0.4  # share of its spare energy an agent spends to find a mate without sound
HANDLING = 0.5  # handling time of one food item (Holling)
FOOD_ENERGY = 100.0  # energy of an item where there is oxygen enough
ANAEROBIC = 2.0 / 32.0  # share of that energy without oxygen (ATP per glucose, textbook values from memory)
O2_FULL = 0.05  # oxygen (today's air is 1) from which food yields its full energy (toy)
OFFSPRING_COST = 0.5
PREDATOR_SPEED = 6.0
REFUGE = 4.0  # cells: a refuge is anywhere up to this far
PREDATOR_NEED = 1.5
ARMS_LEVEL = 3
BRAIN = 32.0  # mean neurons from which a population counts as brained
GROUP_SIZE = 5  # kin within earshot, for kin_gain
GROUP_MIN = 0.01  # kin_gain from which staying near kin pays
LOSS_BIAS = 2.0 / 3.0  # share of mutations that go down
MAX_BODY = 2 ** 20  # cells of the largest body level 9 can hand over (the lessons judge up to here)
MAX_TYPES = 30  # and its most cell types

STAGES = ("blind", "smelling", "seeing", "brained", "arms_race", "starved")
OUTCOMES = ("blind", "sensing", "seeing", "brained", "starved")

INPUT_KEYS = ("light", "patchiness", "density", "predators", "cost", "mutation", "dims", "oxygen", "body_size",
              "cell_types")
PARAM_KEYS = INPUT_KEYS + ("max_level", "max_neurons", "eye_pays")
STATE_KEYS = ("population", "smell", "eyes", "touch", "ears", "eyed", "neurons", "voice", "speed", "intake",
              "survival", "predator_sense", "stage")
LEDGER_KEYS = ("e_intake", "e_sensors", "e_neurons", "e_movement", "e_reproduction", "e_stored", "e_eaten",
               "starved", "caught", "old", "births", "attack", "escape", "eared", "talkers", "sensing")
SUMMARY_KEYS = ("stage", "population", "eyes", "ears", "neurons", "voice", "eared", "talkers", "survival",
                "predator_pressure", "kin_gain", "group", "outcome")
COUNT_KEYS = frozenset({"population", "predator_sense", "dims", "body_size", "cell_types", "max_level",
                        "max_neurons", "eye_pays", "starved", "caught", "old", "births"})
WORD_KEYS = frozenset({"stage", "outcome", "group"})
_DIGITS = frozenset("0123456789")

KEYS: dict[str, str] = {
    "light": "how bright the world is, 0 dark to 1 bright (parameter)",
    "patchiness": "clumping of the food, 0 spread evenly to 0.8: a place holds 1 / (1 - patchiness) items (parameter)",
    "density": "food items per cell, on average (parameter)",
    "predators": "predators per cell of the strip they sweep, 0 to 0.3 (parameter)",
    "cost": "multiplier on the energy cost of sensors and neurons (parameter)",
    "mutation": "chance per gene and offspring of a mutation (parameter)",
    "dims": "1 for a ring, 2 for a grid (parameter)",
    "oxygen": "oxygen relative to today's air, 0 to 1; below 0.05 food yields less energy (parameter)",
    "body_size": "cells in a body, from level 9 (parameter)",
    "cell_types": "kinds of cell in a body, from level 9 (parameter)",
    "max_level": "highest sensor level a body can have: 5 with at least 2 cell types, else 0 (and no voice)",
    "max_neurons": "most neurons a body can have: none below 3 cell types, else a quarter of its cells, as a power of two up to 1024",
    "eye_pays": "the eye level that leaves most energy to an agent of speed 2 with that eye alone, by the rules, 0 to 5",
    "population": "agents alive in this generation (count)",
    "smell": "mean level of the chemical sense, 0 to 5",
    "eyes": "mean level of the eyes, 0 to 5",
    "touch": "mean level of touch, 0 to 5",
    "ears": "mean level of the ears, 0 to 5",
    "eyed": "share of the agents with an eye level above 0",
    "neurons": "mean number of neurons per agent",
    "voice": "share of the agents that can make a sound",
    "speed": "mean speed, cells per move, 1 to 5",
    "intake": "mean energy an agent takes in per generation, energy units",
    "survival": "mean chance of an agent not to be caught in this generation",
    "predator_sense": "sensor level of the predators, 0 to 5 (0 when there are none)",
    "stage": "blind, smelling, seeing, brained, arms_race or starved",
    "e_intake": "ledger: energy taken in by all agents",
    "e_sensors": "ledger: energy paid for sensors",
    "e_neurons": "ledger: energy paid for neurons",
    "e_movement": "ledger: energy paid for moving, calling and finding a mate",
    "e_reproduction": "ledger: energy turned into offspring, 0.5 units each",
    "e_stored": "ledger: surplus energy of the survivors that made no offspring",
    "e_eaten": "ledger: surplus energy of the agents that were caught",
    "starved": "ledger: agents whose costs ate up their intake (count)",
    "caught": "ledger: agents caught by predators (count)",
    "old": "ledger: agents that lived to reproduce and died at the end of the generation (count)",
    "births": "ledger: offspring, the next generation (count)",
    "attack": "ledger: chance of an agent to be attacked in this generation",
    "escape": "ledger: mean chance of an attacked agent to escape",
    "eared": "share of the agents with an ear level above 0",
    "talkers": "share of the agents with ears and a voice",
    "sensing": "ledger: share of the agents with any sensor",
    "predator_pressure": "chance of an agent to be attacked in the last generation",
    "kin_gain": "share of an agent's risk of death that warning calls of kin could take away, at the end",
    "group": "yes when kin_gain is 0.01 or more: staying near kin would pay, else no",
    "outcome": "blind, sensing, seeing, brained or starved",
}

Genome = tuple[int, int, int, int, int, int, int]  # smell, eyes, touch, ears, neuron gene, speed, voice


# ---------------------------------------------------------------------------------------------
# the rules: every function here is called by the simulation and answers a lesson


def receptors(level: int) -> int:
    """Receptors of a sensor of this level (toy rule): level squared."""
    return level * level


def pooling_gain(receptors_: float) -> float:
    """Signal-to-noise gain from pooling n receptors with independent noise: sqrt(n)."""
    return math.sqrt(receptors_)


def sensor_range(level: int, reach: float) -> float:
    """Range in cells (toy rule): the sensor's reach times the pooling gain of its level^2 receptors."""
    return reach * pooling_gain(receptors(level))


def detection_area(range_: float, dims: int) -> float:
    """Cells within ``range_``: 2 r + 1 on a ring, (2 r + 1)^2 on a grid (the agent's own cell included).
    Multiplied out, not ``**``: a product is rounded the same way on every machine, ``pow`` is not."""
    side = 2.0 * range_ + 1.0
    area = 1.0
    for _ in range(dims):
        area *= side
    return area


def neuron_count(gene: int) -> int:
    """Neurons for a neuron gene: 0, then 2, 4, 8 ... 1024."""
    return 0 if gene <= 0 else 2 ** gene


def usable_associations(neurons: int) -> int:
    """Sensor-action associations an agent can use: one free, one more per two neurons."""
    return FREE_ASSOCIATIONS + neurons // NEURONS_PER_ASSOCIATION


def neurons_needed(associations: int) -> int:
    """Fewest neurons (0 or a power of two) that can hold this many associations (toy rule)."""
    need = NEURONS_PER_ASSOCIATION * (associations - FREE_ASSOCIATIONS)
    if need <= 0:
        return 0
    n = 2
    while n < need:
        n *= 2
    return n


def sensor_use(neurons: int, associations: float) -> float:
    """Share of its sensor information an agent can use, 0 to 1."""
    if associations <= usable_associations(neurons):
        return 1.0
    return usable_associations(neurons) / associations


def max_neurons(body_size: int, cell_types: int) -> int:
    """Most neurons a body allows (toy rule): none below 3 cell types, else a quarter of the
    cells, rounded down to a power of two, at most 1024."""
    if cell_types < NEURON_CELL_TYPES:
        return 0
    limit = int(body_size * NEURON_SHARE)
    n = 0
    for gene in range(1, MAX_NEURON_GENE + 1):
        if neuron_count(gene) <= limit:
            n = neuron_count(gene)
    return n


def max_sensor_level(cell_types: int) -> int:
    """Highest sensor level a body allows (toy rule): 5 with at least 2 cell types (a sensor is
    made of cells of its own kind), else 0: a body of one kind of cell has no sense organs."""
    return MAX_LEVEL if cell_types >= SENSOR_CELL_TYPES else 0


def energy_yield(oxygen: float) -> float:
    """Energy of one food item: 6.25 without oxygen, rising in proportion to 100 at a twentieth
    of today's oxygen, 100 from there on (toy rule)."""
    return FOOD_ENERGY * (ANAEROBIC + (1.0 - ANAEROBIC) * min(1.0, oxygen / O2_FULL))


def encounter_rate(speed: float, density: float, patchiness: float, cells: float) -> float:
    """Food items met per unit time; ``cells`` 1 is an agent without sensors."""
    spread = 1.0 - patchiness
    return speed * density * spread * cells * min(1.0 / spread, cells)


def food_intake(rate: float, yield_: float) -> float:
    """Holling's disc equation with handling time 0.5: energy per generation."""
    return yield_ * rate / (1.0 + HANDLING * rate)


def sensor_cost(level: int, unit: float) -> float:
    """Energy per generation for a sensor: ``unit`` per receptor, and as much as 3 receptors for
    the organ itself; nothing at level 0."""
    return unit * (receptors(level) + ORGAN) if level > 0 else 0.0


def neuron_cost(neurons: int, cost: float) -> float:
    """Energy per generation for neurons: 0.05 each, times the cost multiplier."""
    return cost * NEURON_COST * neurons


def movement_cost(speed: float) -> float:
    """Energy per generation for moving: 0.5 * speed^3 (against a drag that grows with the square
    of the speed, the power goes with its cube). Multiplied out, see :func:`detection_area`."""
    return MOVE_COST * speed * speed * speed


def mate_found(voice: int, hearing: float, voice_share: float, mean_hearing: float) -> float:
    """Share of the mate search that sound saves: half by being heard, half by hearing."""
    return 0.5 * voice * mean_hearing + 0.5 * hearing * voice_share


def social_cost(voice: int, found: float, spare: float) -> float:
    """Energy spent on calling and on the search for a mate: of the spare energy, 0.06 for an
    agent with a voice and 0.4 times what of the search sound does not save."""
    return (CALL_SHARE * voice + MATE_SHARE * (1.0 - found)) * spare


def net_energy(intake: float, sensors: float, neurons: float, movement: float) -> float:
    """What is left: intake minus the costs, never below 0 (then the agent starves)."""
    return max(0.0, intake - sensors - neurons - movement)


def attack_probability(predators: float, predator_sense: int) -> float:
    """Chance of at least one attack in a generation: 1 - exp(-predators * (2 s + 1))."""
    return -math.expm1(-predators * (2 * predator_sense + 1))


def escape_probability(head_start: float, speed: float, predator_speed: float = PREDATOR_SPEED) -> float:
    """Chance to reach a refuge (up to 4 cells away) before a faster predator closes the head start."""
    if head_start <= 0:
        return 0.0
    if speed >= predator_speed:
        return 1.0
    return min(1.0, head_start * speed / ((predator_speed - speed) * REFUGE))


def survival_probability(attack: float, escape: float) -> float:
    """Chance not to be caught: 1 - attack * (1 - escape)."""
    return 1.0 - attack * (1.0 - escape)


def expected_offspring(fitness: float, total_fitness: float, births: int) -> float:
    """Offspring an agent expects: its share of the fitness times the births."""
    return births * fitness / total_fitness if total_fitness > 0 else 0.0


def births_from(surplus: float) -> int:
    """Offspring the survivors' surplus pays for at 0.5 each, at most 200."""
    return min(CAP, int(surplus // OFFSPRING_COST))


def predator_meal(predator_sense: int, escape: float, prey_share: float) -> float:
    """What the predators catch, against a need of 1.5: (2 s + 1) * (1 - escape) * prey share."""
    return (2 * predator_sense + 1) * (1.0 - escape) * prey_share


def predator_step(predator_sense: int, escape: float, prey_share: float) -> int:
    """The predators' next sensor level: up when hungry, down when one level less still feeds them."""
    if predator_meal(predator_sense, escape, prey_share) < PREDATOR_NEED:
        return min(MAX_LEVEL, predator_sense + 1)
    if predator_sense > 0 and predator_meal(predator_sense - 1, escape, prey_share) >= PREDATOR_NEED:
        return predator_sense - 1
    return predator_sense


def group_detection(p: float, n: int) -> float:
    """Chance that at least one of n notices: 1 - (1 - p)^n (many eyes). Multiplied out, see
    :func:`detection_area`."""
    none = 1.0
    for _ in range(n):
        none *= 1.0 - p
    return 1.0 - none


def kin_gain(attack: float, escape: float, hearing: float, callers: float) -> float:
    """Share of an agent's risk that warning calls of kin could remove (an index for the next
    level, not part of the dynamics): it is attacked, would not escape alone, hears, and at least
    one of 5 kin nearby both notices predators and has a voice (``callers``: that share)."""
    return attack * (1.0 - escape) * hearing * group_detection(callers, GROUP_SIZE)


def eye_net(level: int, speed: float, density: float, patchiness: float, light: float, cost: float, dims: int,
            oxygen: float = 1.0, neuron_limit: int = 2 ** MAX_NEURON_GENE) -> float:
    """Energy an agent keeps that has only an eye of this level and the neurons the eye needs
    (as far as its body allows): intake minus the cost of the eye and of those neurons."""
    extra = detection_area(sensor_range(level, REACH["eyes"]), dims) - 1.0
    neurons = min(neurons_needed(math.ceil(extra)), neuron_limit)
    cells = 1.0 + sensor_use(neurons, extra) * light * extra
    intake = food_intake(encounter_rate(speed, density, patchiness, cells), energy_yield(oxygen))
    return intake - sensor_cost(level, cost * UNIT_COST["eyes"]) - neuron_cost(neurons, cost)


def break_even_level(speed: float, density: float, patchiness: float, light: float, cost: float, dims: int,
                     oxygen: float = 1.0, neuron_limit: int = 2 ** MAX_NEURON_GENE) -> int:
    """The eye level 0 to 5 that leaves most energy to an agent with that eye alone (the lowest
    of equals; 0 when no eye pays). In about 1 of 70 worlds drawn like the lessons this is not the
    first level at which a better eye stops paying: an eye can pay again once the body affords
    the neurons to use it, so all six levels are compared."""
    best, best_net = 0, eye_net(0, speed, density, patchiness, light, cost, dims, oxygen, neuron_limit)
    for level in range(1, MAX_LEVEL + 1):
        net = eye_net(level, speed, density, patchiness, light, cost, dims, oxygen, neuron_limit)
        if net > best_net + 1e-9:
            best, best_net = level, net
    return best


def outcome_of(population: int, neurons: float, eyed: float, sensing: float) -> str:
    """``starved``, ``brained``, ``seeing``, ``sensing`` or ``blind``, the first that applies."""
    if population == 0:
        return "starved"
    if neurons >= BRAIN:
        return "brained"
    if eyed >= 0.5:
        return "seeing"
    return "sensing" if sensing >= 0.5 else "blind"


def stage_of(population: int, predator_sense: int, neurons: float, eyed: float, sensing: float) -> str:
    """The stage of a generation: as :func:`outcome_of` (``sensing`` is called ``smelling``),
    but ``arms_race`` once the predators' sensor level is 3 or more."""
    if population > 0 and predator_sense >= ARMS_LEVEL:
        return "arms_race"
    return outcome_of(population, neurons, eyed, sensing).replace("sensing", "smelling")


# ---------------------------------------------------------------------------------------------
# lessons

_LEVEL = Input("level", 0, MAX_LEVEL, True)
_DIMS = Input("dims", 1, 2, True)
_SPEED = Input("speed", SPEEDS[0], SPEEDS[1], True)
_SENSE = Input("predator_sense", 0, MAX_LEVEL, True)

RULES: dict[str, Rule] = {
    "detection_area": Rule(
        (Input("range", 0, 5, places=1), _DIMS), detection_area,
        "cells within range of a sensor. 2 times range plus 1 on a ring with dims 1. that number squared on a grid with dims 2"),
    "pooling_gain": Rule(
        (Input("receptors", 1, 400, True),), pooling_gain,
        "signal to noise gain from pooling receptors with independent noise. square root of receptors. the standard pooling rule"),
    "sensor_range": Rule(
        (_LEVEL, Input("reach", 0.1, 1, places=1)), sensor_range,
        "range in cells of a sensor. receptors are level squared. range is reach times square root of receptors. toy rule"),
    "encounter": Rule(
        (_SPEED, Input("density", 0.05, 0.5, places=2), Input("patchiness", 0, 0.8, places=2),
         Input("cells", 1, 121, places=1)), encounter_rate,
        "food items met per unit time. speed times density times 1 minus patchiness times cells times the smaller of cells and 1 over 1 minus patchiness. cells 1 is no sensor"),
    "intake": Rule(
        (Input("rate", 0, 30, places=2), Input("yield", 6.25, 100, places=2)), food_intake,
        "energy taken in per generation. disc equation. yield times rate over 1 plus 0 point 5 times rate"),
    "energy_yield": Rule(
        (Input("oxygen", 0, 1, places=3),), energy_yield,
        "energy of one food item. 6 point 2 5 plus 9 3 point 7 5 times oxygen over 0 point 0 5. at most 1 0 0. oxygen 1 is todays air. toy rule"),
    "sensor_cost": Rule(
        (_LEVEL, Input("unit", 0.25, 4, places=2)), sensor_cost,
        "energy per generation for a sensor. unit times receptors plus 3 for the organ. receptors are level squared. 0 at level 0"),
    "movement_cost": Rule(
        (Input("speed", SPEEDS[0], SPEEDS[1], places=1),), movement_cost,
        "energy per generation for moving. 0 point 5 times speed cubed"),
    "break_even": Rule(
        (_SPEED, Input("density", 0.05, 0.5, places=2), Input("patchiness", 0, 0.8, places=2),
         Input("light", 0, 1, places=2), Input("cost", 0.5, 2, places=2), _DIMS), break_even_level,
        "eye level 0 to 5 that leaves most energy to an agent with that eye alone. todays air. an eye costs 2 times cost per receptor and for the organ and needs neurons. the lowest of equals. 0 when no eye pays"),
    "escape": Rule(
        (Input("head_start", 0, 5, places=1), _SPEED, Input("predator_speed", 6, 8, True)), escape_probability,
        "chance to reach a refuge up to 4 cells away. head_start times speed over 4 times predator_speed minus speed. at most 1"),
    "attack": Rule(
        (Input("predators", 0, 0.3, places=2), _SENSE), attack_probability,
        "chance of at least one attack in a generation. 1 minus exp of minus predators times 2 predator_sense plus 1"),
    "survival": Rule(
        (Input("attack", 0, 1, places=2), Input("escape", 0, 1, places=2)), survival_probability,
        "chance not to be caught. 1 minus attack times 1 minus escape"),
    "energy_budget": Rule(
        (Input("intake", 0, 200, places=1), Input("sensors", 0, 40, places=1), Input("neurons", 0, 20, places=1),
         Input("movement", 0, 40, places=1)), net_energy,
        "energy an agent has left. intake minus sensors minus neurons minus movement. 0 when the costs are larger and the agent starves"),
    "offspring": Rule(
        (Input("fitness", 0, 100, places=1), Input("total_fitness", 1, 10000, places=0),
         Input("births", 0, CAP, True)), expected_offspring,
        "offspring an agent expects. births times fitness over total_fitness",
        lambda fitness, total_fitness, births: fitness <= total_fitness),
    "births": Rule(
        (Input("surplus", 0, 150, places=1),), births_from,
        "offspring the surplus energy of the survivors pays for. one per 0 point 5 units. rounded down. at most 2 0 0"),
    "neurons_needed": Rule(
        (Input("associations", 0, 500, True),), neurons_needed,
        "fewest neurons for that many sensor action associations. one association is free. each further one needs 2 neurons. rounded up to a power of two. toy rule"),
    "sensor_use": Rule(
        (Input("neurons", 0, 256, True), Input("associations", 1, 500, True)), sensor_use,
        "share of its sensor information an agent can use. 1 plus half its neurons rounded down. over associations. at most 1. toy rule"),
    "max_neurons": Rule(
        (Input("body_size", 1, MAX_BODY, True), Input("cell_types", 1, MAX_TYPES, True)), max_neurons,
        "most neurons a body allows. none below 3 cell_types. else a quarter of body_size rounded down to a power of two. at most 1 0 2 4. toy rule"),
    "max_level": Rule(
        (Input("cell_types", 1, MAX_TYPES, True),), max_sensor_level,
        "highest sensor level a body allows. 5 with 2 cell_types or more. else 0. toy rule"),
    "predator_step": Rule(
        (_SENSE, Input("escape", 0, 1, places=2), Input("prey_share", 0, 1, places=2)), predator_step,
        "next sensor level of the predators. meal is 2 predator_sense plus 1 times 1 minus escape times prey_share. one level up when meal is below 1 point 5. one down when one level less still gives 1 point 5"),
    "group_detection": Rule(
        (Input("p", 0, 1, places=2), Input("n", 1, 10, True)), group_detection,
        "chance that at least one of n notices a predator when each does with chance p. 1 minus 1 minus p to the power n. many eyes"),
    "mate_found": Rule(
        (Input("voice", 0, 1, True), Input("hearing", 0, 1, places=2), Input("voice_share", 0, 1, places=2),
         Input("mean_hearing", 0, 1, places=2)), mate_found,
        "share of the mate search that sound saves. half of voice times mean_hearing plus half of hearing times voice_share"),
    "kin_gain": Rule(
        (Input("attack", 0, 1, places=2), Input("escape", 0, 1, places=2), Input("hearing", 0, 1, places=2),
         Input("callers", 0, 1, places=2)), kin_gain,
        "share of the risk that warning calls of kin could remove. attack times 1 minus escape times hearing times the chance that one of 5 kin calls. that is 1 minus 1 minus callers to the power 5"),
    "stage": Rule(
        (Input("population", 0, CAP, True), _SENSE, Input("neurons", 0, 2 ** MAX_NEURON_GENE, places=1),
         Input("eyed", 0, 1, places=2), Input("sensing", 0, 1, places=2)), stage_of,
        "stage of a generation. starved with population 0. arms_race with predator_sense 3 or more. brained with 3 2 neurons or more. seeing with eyed 0 point 5 or more. smelling with sensing 0 point 5 or more. else blind",
        lambda population, predator_sense, neurons, eyed, sensing: eyed <= sensing),
}

class SensesLessons(LessonGate):
    """The lesson gate of this level. Only the drawing of inputs differs from :class:`LessonGate`,
    and only where numbers drawn evenly from the range would give one answer most of the time
    (measured: ``max_level`` 5 in 97 of 100 lessons, ``energy_yield`` 100 in 74, ``neurons_needed``
    1024 in 48, ``stage`` arms_race in 48) or would leave out the worlds of the rollouts. The ranges, and so what the gate
    judges, are those of :data:`RULES`; every judgement is the base class's.

    * ``max_level``: one cell type half the time (the answer 0), else 2 to 30 (the answer 5).
    * ``energy_yield``: oxygen below a twentieth of today's six times in ten (where the yield
      rises), else up to today's (where it is 100).
    * ``max_neurons``: the body size evenly over the doublings from 1 to 32767 cells (a doubling
      first, then half the time the power of two itself); every larger body allows 1024.
    * ``neurons_needed``: the associations evenly over the doublings, or none.
    * ``stage``: nobody left once in eight; the predators' level 3 to 5 once in four, else 0 to 2
      (drawn evenly, half the lessons would be ``arms_race``: 0.48 of 2000, measured); the neurons
      evenly over the doublings up to 1024, or none; ``sensing`` from ``eyed`` upward (an agent
      with an eye is an agent with a sensor).
    * ``break_even``: a dark world (light 0) with chance 0.15, as :func:`random_params` draws it.
    """

    def draw(self, rng: random.Random, name: str, spec: Input, values: list[float | int]) -> float | int:
        key = (name, spec.name)
        if key == ("max_level", "cell_types"):
            return SENSOR_CELL_TYPES - 1 if rng.random() < 0.5 else rng.randint(SENSOR_CELL_TYPES, int(spec.high))
        if key == ("energy_yield", "oxygen"):
            low, high = (0.0, O2_FULL) if rng.random() < 0.6 else (O2_FULL, spec.high)
            return round(rng.uniform(low, high), spec.places)
        if key == ("max_neurons", "body_size"):
            k = rng.randint(0, 14)
            return 1 << k if rng.random() < 0.5 else rng.randint(1 << k, (2 << k) - 1)
        if key == ("neurons_needed", "associations"):
            k = rng.randint(-1, int(spec.high).bit_length() - 1)
            return 0 if k < 0 else rng.randint(1 << k, min(int(spec.high), (2 << k) - 1))
        if key == ("stage", "population"):
            return 0 if rng.random() < 0.125 else rng.randint(1, int(spec.high))
        if key == ("stage", "predator_sense"):
            return rng.randint(ARMS_LEVEL, int(spec.high)) if rng.random() < 0.25 else rng.randint(0, ARMS_LEVEL - 1)
        if key == ("stage", "neurons"):
            k = rng.randint(-1, MAX_NEURON_GENE - 1)
            return 0.0 if k < 0 else round(rng.uniform(1 << k, 2 << k), spec.places)
        if key == ("stage", "sensing"):
            return round(rng.uniform(values[-1], spec.high), spec.places)
        if key == ("break_even", "light") and rng.random() < 0.15:
            return 0.0
        if spec.integer:
            return rng.randint(int(spec.low), int(spec.high))
        return round(rng.uniform(spec.low, spec.high), spec.places)

    def generate(self, rng: random.Random, n: int) -> list[Line]:
        """Exactly ``n`` lessons, rules drawn uniformly, inputs by :meth:`draw`."""
        names = sorted(self.rules)
        out: list[Line] = []
        tries = 0
        while len(out) < n:
            tries += 1
            if tries > 50 * n + 1000:
                raise RuntimeError(f"{self.sim}: the lesson rules reject almost every drawn input")
            name = rng.choice(names)
            values: list[float | int] = []
            for spec in self.rules[name].inputs:
                values.append(self.draw(rng, name, spec, values))
            ln = self.line(name, values)
            if ln is not None:
                out.append(ln)
        return out


LESSONS = SensesLessons(SIM, RULES)


def lesson_gate() -> LessonGate:
    return LESSONS


# ---------------------------------------------------------------------------------------------
# the simulation


@dataclass
class SensesRollout(Rollout):
    """A :class:`Rollout` that also keeps the genomes of every step and the record of every
    generation (``steps[t]`` is ``generations[5 * t]``)."""
    #: per step, per agent: (smell, eyes, touch, ears, neuron gene, speed, voice)
    genomes: list[list[Genome]] = field(default_factory=list)
    generations: list[dict] = field(default_factory=list)

    def value(self, text: float | int | str) -> str:
        return dense_value(text)


def dense_value(value: float | int | str) -> str:
    """A metric as it stands in a line: a word, an exact count, or a number to 3 significant digits.

    >>> dense_value(0.98512), dense_value(200), dense_value(97.33), dense_value("seeing")
    ('0 point 9 8 5', '2 0 0', '9 7 point 3', 'seeing')
    """
    if isinstance(value, str):
        return value
    if isinstance(value, int):
        return num(value)
    return num(sig(float(value)))


def param_value(key: str, value: float | int) -> str:
    """A parameter as it stands in the ``params`` record: as given.

    >>> param_value("light", 0.57), param_value("mutation", 0.026), param_value("body_size", 14228)
    ('0 point 5 7', '0 point 0 2 6', '1 4 2 2 8')
    """
    return num(value) if isinstance(value, int) else num(float(value), sig=6)


def random_params(rng: random.Random, rules: int = 7) -> dict[str, float | int]:
    """A world: dark with chance 0.15, without predators with chance 0.2, ring or grid, oxygen
    from a hundredth of today's to today's, a body of 10 to 30000 cells with 1 to 12 cell types."""
    light = 0.0 if rng.random() < 0.15 else round(rng.uniform(0.05, 1.0), 2)
    predators = 0.0 if rng.random() < 0.2 else round(rng.uniform(0.02, 0.3), 2)
    return {"light": light,
            "patchiness": round(rng.uniform(0.0, 0.8), 2),
            "density": round(rng.uniform(0.1, 0.5), 2),
            "predators": predators,
            "cost": round(rng.uniform(0.5, 2.0), 2),
            "mutation": round(rng.uniform(0.005, 0.03), 3),
            "dims": rng.choice((1, 2)),
            "oxygen": round(10 ** rng.uniform(-2.0, 0.0), 3),
            "body_size": int(round(10 ** rng.uniform(1.0, 4.5))),
            "cell_types": rng.randint(1, 12)}


def _first(below: dict, names: tuple[str, ...]):
    """The first of ``names`` that the summary below holds as a finite number."""
    for name in names:
        value = below.get(name)
        if isinstance(value, (int, float)) and not isinstance(value, bool) and is_finite(value):
            return value
    return None


def handoff_in(below: dict) -> dict[str, float | int]:
    """Parameters of this level from the summary of level 9 (bodies), or from what its
    ``handoff_out`` makes of it. Only what the bodies decide is returned; the chain draws the
    rest (light, food, cost, mutation, ring or grid).

    * ``body_size``: the cells of the largest common body (``body_size``, else ``max_size``), a
      whole number, at least 1.
    * ``cell_types``: its kinds of cell (``cell_types``), at least 1.
    * ``predators``: 3 times the predators' share of the biomass (``predator_share``), at most
      0.3, to 3 significant digits: a tenth of all biomass in predators is the densest hunting
      this level knows (toy scaling). 0 when level 9 says there are no predators (``predators``
      ``no``, or fewer than 3 ``trophic_levels``).
    * ``oxygen``: the oxygen of level 9 in present levels (``oxygen``), cut off at 1.

    A key the summary does not hold is left out, so the default or drawn value stays.

    >>> handoff_in({"max_size": 512, "cell_types": 7, "trophic_levels": 3, "oxygen": 0.0755, "predator_share": 0.0155})
    {'body_size': 512, 'cell_types': 7, 'predators': 0.0465, 'oxygen': 0.0755}
    >>> handoff_in({"body_size": 32, "cell_types": 4, "predators": "no", "predator_share": 0.00105, "oxygen": 2.0})
    {'body_size': 32, 'cell_types': 4, 'predators': 0.0, 'oxygen': 1.0}
    """
    out: dict[str, float | int] = {}
    size = _first(below, ("body_size", "max_size"))
    if size is not None:
        out["body_size"] = max(1, int(size))
    types = _first(below, ("cell_types",))
    if types is not None:
        out["cell_types"] = max(1, int(types))
    share = _first(below, ("predator_share",))
    levels = _first(below, ("trophic_levels",))
    if below.get("predators") == "no" or (levels is not None and levels < 3):
        out["predators"] = 0.0
    elif share is not None:
        out["predators"] = sig(min(0.3, 3.0 * max(0.0, share)))
    oxygen = _first(below, ("oxygen",))
    if oxygen is not None:
        out["oxygen"] = min(1.0, max(0.0, float(oxygen)))
    return out


@dataclass(frozen=True)
class _Traits:
    """What follows from a genome and the world alone (not from the others or the predators)."""
    intake: float
    sensors: float
    neurons_cost: float
    move: float
    escape: float
    hearing: float
    neurons: int
    notices: bool


def traits(g: Genome, p: dict) -> _Traits:
    """One agent by the rules: its intake, costs, chance to escape and hearing."""
    smell, eyes, touch, ears, gene, speed, _voice = g
    dims, light, cost = p["dims"], p["light"], p["cost"]
    levels = dict(zip(SENSORS, (smell, eyes, touch, ears)))
    ranges = {s: sensor_range(levels[s], REACH[s]) for s in SENSORS}
    extra = {s: detection_area(ranges[s], dims) - 1.0 for s in SENSORS}
    neurons = neuron_count(gene)
    use = sensor_use(neurons, math.fsum(extra.values()))
    cells = 1.0 + use * max(SMELL_SLOW * extra["smell"], light * extra["eyes"], extra["touch"])
    intake = food_intake(encounter_rate(speed, p["density"], p["patchiness"], cells), energy_yield(p["oxygen"]))
    head_start = use * max(EYE_FIELD * light * ranges["eyes"], ranges["ears"], ranges["touch"])
    sensors = math.fsum(sensor_cost(levels[s], cost * UNIT_COST[s]) for s in SENSORS)
    return _Traits(intake, sensors, neuron_cost(neurons, cost), movement_cost(speed),
                   escape_probability(head_start, speed, PREDATOR_SPEED), use * ears / MAX_LEVEL, neurons,
                   head_start > 0)


def _measure(pop: list[Genome], tr: list[_Traits]) -> dict[str, float | int]:
    """The metrics that are means and shares of a population (3 significant digits)."""
    n = len(pop)
    if n == 0:
        return {"population": 0, **{k: 0.0 for k in ("smell", "eyes", "touch", "ears", "eyed", "neurons", "voice",
                                                       "speed", "intake", "escape", "eared", "talkers", "sensing")}}

    def mean(values: list) -> float:
        return sig(math.fsum(values) / n)

    def share(flags) -> float:
        return sig(sum(1 for f in flags if f) / n)

    return {"population": n,
            "smell": mean([g[0] for g in pop]), "eyes": mean([g[1] for g in pop]),
            "touch": mean([g[2] for g in pop]), "ears": mean([g[3] for g in pop]),
            "eyed": share(g[1] > 0 for g in pop), "neurons": mean([x.neurons for x in tr]),
            "voice": share(g[6] for g in pop), "speed": mean([g[5] for g in pop]),
            "intake": mean([x.intake for x in tr]),
            "escape": math.fsum(x.escape for x in tr) / n,
            "eared": share(g[3] > 0 for g in pop), "talkers": share(g[3] > 0 and g[6] for g in pop),
            "sensing": share(any(g[:4]) for g in pop)}


def _mutate(rng: random.Random, g: Genome, mu: float, max_level: int, max_gene: int) -> Genome:
    """Each gene changes with probability ``mu``: one step, down twice as often as up, within
    its limits; the voice is lost twice as often as gained (and stays 0 in a body without organs)."""
    out = list(g)
    for i, (lo, hi) in enumerate(((0, max_level),) * 4 + ((0, max_gene), SPEEDS)):
        if rng.random() < mu:
            out[i] = min(hi, max(lo, out[i] + (-1 if rng.random() < LOSS_BIAS else 1)))
    if rng.random() < mu:
        out[6] = 0 if rng.random() < LOSS_BIAS or max_level == 0 else 1
    return tuple(out)  # type: ignore[return-value]


def _generation(rng: random.Random, pop: list[Genome], p: dict, sense: int, cache: dict[Genome, _Traits],
                max_level: int, max_gene: int) -> tuple[dict, list[Genome]]:
    """One generation: its record (metrics and ledger) and the offspring."""
    n = len(pop)
    for g in pop:
        if g not in cache:
            cache[g] = traits(g, p)
    tr = [cache[g] for g in pop]
    m = _measure(pop, tr)
    attack = attack_probability(p["predators"], sense) if p["predators"] > 0 and n else 0.0
    voice_share = sum(g[6] for g in pop) / n if n else 0.0
    mean_hearing = math.fsum(x.hearing for x in tr) / n if n else 0.0
    e_sensors, e_neurons, e_movement, surplus, survive = [], [], [], [], []
    starved = 0
    for g, x in zip(pop, tr):
        spare = net_energy(x.intake, x.sensors, x.neurons_cost, x.move)
        move = x.move + social_cost(g[6], mate_found(g[6], x.hearing, voice_share, mean_hearing), spare)
        left = net_energy(x.intake, x.sensors, x.neurons_cost, move)
        if left <= 0.0:  # starves: pays as far as the intake goes
            costs = x.sensors + x.neurons_cost + move
            paid = x.intake / costs if costs > 0 else 0.0
            starved += 1
            left = 0.0
        else:
            paid = 1.0
        e_sensors.append(paid * x.sensors)
        e_neurons.append(paid * x.neurons_cost)
        e_movement.append(paid * move)
        surplus.append(left)
        survive.append(survival_probability(attack, x.escape))
    # predation: one draw per agent that did not starve, in the order of the population
    alive = [s > 0.0 and not (attack > 0.0 and rng.random() >= q) for s, q in zip(surplus, survive)]
    caught = sum(1 for s, a in zip(surplus, alive) if s > 0.0 and not a)
    fitness = [s if a else 0.0 for s, a in zip(surplus, alive)]
    total = math.fsum(fitness)
    births = births_from(total)
    # reproduction: each offspring draws its parent by fitness share, then its mutations
    children: list[Genome] = []
    if births:
        cum, acc = [], 0.0
        for f in fitness:
            acc += expected_offspring(f, total, births)
            cum.append(acc)
        for _ in range(births):
            i = min(bisect_right(cum, rng.random() * births), n - 1)
            while fitness[i] <= 0.0:  # a draw on the edge of a parent without fitness
                i -= 1
            children.append(_mutate(rng, pop[i], p["mutation"], max_level, max_gene))
    e_repro = OFFSPRING_COST * births
    record = {
        "population": n, "smell": m["smell"], "eyes": m["eyes"], "touch": m["touch"], "ears": m["ears"],
        "eyed": m["eyed"], "neurons": m["neurons"], "voice": m["voice"], "speed": m["speed"], "intake": m["intake"],
        "survival": sig(math.fsum(survive) / n) if n else 0.0, "predator_sense": sense,
        "stage": stage_of(n, sense, m["neurons"], m["eyed"], m["sensing"]),
        "e_intake": math.fsum(x.intake for x in tr), "e_sensors": math.fsum(e_sensors),
        "e_neurons": math.fsum(e_neurons), "e_movement": math.fsum(e_movement), "e_reproduction": e_repro,
        "e_stored": max(0.0, total - e_repro), "e_eaten": math.fsum(s for s, a in zip(surplus, alive) if not a),
        "starved": starved, "caught": caught, "old": n - starved - caught, "births": births,
        "attack": attack, "escape": m["escape"], "eared": m["eared"], "talkers": m["talkers"],
        "sensing": m["sensing"],
    }
    return record, children


def _fail(reason: str, expected: str | None = None) -> Verdict:
    return Verdict(False, expected, reason)


class Senses:
    """The simulation and its gate. Rollouts replayed by :meth:`check` are cached per seed."""

    sim = SIM
    topic = SIM
    KEYS = KEYS

    def __init__(self) -> None:
        self._cache: dict[int, SensesRollout] = {}

    def run(self, seed: int, light: float = 0.7, patchiness: float = 0.5, density: float = 0.2,
            predators: float = 0.1, cost: float = 1.0, mutation: float = 0.02, dims: int = 2, oxygen: float = 1.0,
            body_size: int = 1000, cell_types: int = 6, rules: int = 7) -> SensesRollout:
        """Deterministic for a given seed and parameters. ``rules`` is accepted for the round-7
        convention (this level has no round-6 form). Raises ``ValueError`` for a parameter
        outside its meaning (a share above 1, a negative density, a world that is neither ring
        nor grid, a body without cells)."""
        if rules != 7:
            raise ValueError("senses is a round-7 level: rules must be 7")
        light, patchiness, density, predators = float(light), float(patchiness), float(density), float(predators)
        cost, mutation, oxygen = float(cost), float(mutation), float(oxygen)
        for name, value, lo, hi in (("light", light, 0, 1), ("patchiness", patchiness, 0, 0.95),
                                    ("density", density, 0, 10), ("predators", predators, 0, 10),
                                    ("cost", cost, 0, 100), ("mutation", mutation, 0, 1), ("oxygen", oxygen, 0, 1)):
            if not (math.isfinite(value) and lo <= value <= hi):
                raise ValueError(f"{name} must be between {lo} and {hi}: {value!r}")
        if isinstance(dims, bool) or dims not in (1, 2):
            raise ValueError(f"dims is 1 (ring) or 2 (grid): {dims!r}")
        for name, value in (("body_size", body_size), ("cell_types", cell_types)):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) \
                    or int(value) != value or value < 1:
                raise ValueError(f"{name} must be a positive whole number: {value!r}")
        dims, body_size, cell_types = int(dims), int(body_size), int(cell_types)
        p: dict[str, float | int] = {
            "light": light, "patchiness": patchiness, "density": density, "predators": predators, "cost": cost,
            "mutation": mutation, "dims": dims, "oxygen": oxygen, "body_size": body_size, "cell_types": cell_types,
            "max_level": max_sensor_level(cell_types), "max_neurons": max_neurons(body_size, cell_types)}
        max_level = p["max_level"]
        p["eye_pays"] = min(max_level, break_even_level(REFERENCE_SPEED, density, patchiness, light, cost, dims,
                                                        oxygen, p["max_neurons"]))
        max_gene = max(g for g in range(MAX_NEURON_GENE + 1) if neuron_count(g) <= p["max_neurons"])
        rng = random.Random(7919 * int(seed) + 10)
        r = SensesRollout(SIM, int(seed), p, [])
        pop: list[Genome] = [(0, 0, 0, 0, 0, START_SPEED, 0)] * START
        cache: dict[Genome, _Traits] = {}
        sense = 0
        for t in range(STEPS):
            if predators > 0 and t > 0:  # the predators answer the prey of one step ago
                sense = predator_step(sense, r.steps[-1]["escape"], r.steps[-1]["population"] / CAP)
            for k in range(GENS_PER_STEP):
                record, children = _generation(rng, pop, p, sense, cache, max_level, max_gene)
                r.generations.append(record)
                if k == 0:
                    r.steps.append(record)
                    r.genomes.append(pop)
                pop = children
        r.summary = _summary(r)
        return r

    def rollout(self, seed: int) -> SensesRollout:
        """The canonical run of a seed: parameters from ``random_params(random.Random(seed))``."""
        if seed not in self._cache:
            if len(self._cache) >= 256:
                self._cache.clear()
            self._cache[seed] = self.run(seed, **random_params(random.Random(seed)))
        return self._cache[seed]

    # -- the gate -----------------------------------------------------------------------------

    def conserved(self, r: Rollout) -> Verdict:
        """Energy, heads, probabilities, shares, limits, the predators' rule, stage and summary."""
        p = r.params
        generations = getattr(r, "generations", None)
        stride = GENS_PER_STEP
        if not generations or not r.steps or getattr(r, "genomes", None) is None:
            return _fail("no record of the generations or of the genomes: not a rollout of this level")
        if len(generations) != stride * len(r.steps) or any(r.steps[t] != generations[stride * t]
                                                            for t in range(len(r.steps))):
            return _fail("the steps are not every fifth generation")
        if p["max_neurons"] != max_neurons(p["body_size"], p["cell_types"]) \
                or p["max_level"] != max_sensor_level(p["cell_types"]):
            return _fail("max_level or max_neurons does not follow from the body")
        if generations[0]["population"] != START:
            return _fail("the first generation is not the 100 founders", num(START))
        for g, s in enumerate(generations):
            v = self._generation_ok(s, p)
            if v is not None:
                return _fail(f"generation {g}: {v}")
            if g + 1 < len(generations) and generations[g + 1]["population"] != s["births"]:
                return _fail(f"generation {g + 1}: population is not the births of the generation before",
                             num(s["births"]))
            if g % stride and s["predator_sense"] != generations[g - 1]["predator_sense"]:
                return _fail(f"generation {g}: the predators changed between two steps")
        for t, s in enumerate(r.steps):
            if p["predators"] > 0 and t > 0:
                before = r.steps[t - 1]
                want = predator_step(before["predator_sense"], before["escape"], before["population"] / CAP)
            else:
                want = 0
            if s["predator_sense"] != want:
                return _fail(f"step {t}: predator_sense is not what the predators' rule gives", num(want))
        v = self._genomes_ok(r)
        if v is not None:
            return _fail(v)
        if dict(r.summary) != _summary(r):
            return _fail("the summary does not follow from the last step")
        return Verdict(True, None, "energy, heads, probabilities, shares, limits, predators, stages and summary are consistent")

    @staticmethod
    def _generation_ok(s: dict, p: dict) -> str | None:
        n = s["population"]
        energy = [s[k] for k in ("e_sensors", "e_neurons", "e_movement", "e_reproduction", "e_stored", "e_eaten")]
        if min(energy) < 0 or s["e_intake"] < 0:
            return "negative energy"
        if abs(s["e_intake"] - math.fsum(energy)) > 1e-9 * max(1.0, s["e_intake"]):
            return (f"energy: intake {s['e_intake']!r} != what was paid, turned into offspring, stored and "
                    f"eaten {math.fsum(energy)!r}")
        if min(s["starved"], s["caught"], s["old"], s["births"], n) < 0 or s["starved"] + s["caught"] + s["old"] != n:
            return "starved + caught + old != population"
        if not 0 <= n <= CAP or s["births"] > CAP:
            return "more agents than the cap"
        if s["e_reproduction"] != OFFSPRING_COST * s["births"]:
            return "e_reproduction is not 0.5 per birth"
        if s["births"] < CAP and s["e_stored"] >= OFFSPRING_COST * (1 + 1e-9):
            return "surplus for a further offspring was left unused"
        if s["old"] == 0 and (s["births"] or s["e_reproduction"] + s["e_stored"] > 0):
            return "offspring or surplus without a survivor"
        for key in ("attack", "escape", "survival", "eyed", "voice", "eared", "talkers", "sensing"):
            if not 0.0 <= s[key] <= 1.0:
                return f"{key} is not between 0 and 1"
        want = attack_probability(p["predators"], s["predator_sense"]) if p["predators"] > 0 and n else 0.0
        if not close(s["attack"], want, rel=1e-12, abs_=1e-15):
            return "attack does not follow from the predators and their sensor level"
        if s["attack"] == 0 and s["caught"]:
            return "caught without an attack"
        if n and abs(s["survival"] - survival_probability(s["attack"], s["escape"])) > 0.006:
            return "survival != 1 - attack * (1 - escape)"
        if n and not close(s["intake"], s["e_intake"] / n, rel=0.006):
            return "intake is not the mean of e_intake"
        tol = 1.006
        for share, level in (("eyed", "eyes"), ("eared", "ears")):
            if not (s[share] <= s[level] * tol + 1e-12 and s[level] <= MAX_LEVEL * s[share] * tol + 1e-12):
                return f"{share} and {level} do not fit"
        if s["talkers"] > min(s["eared"], s["voice"]) * tol or s["sensing"] * tol < max(s["eyed"], s["eared"]):
            return "talkers or sensing do not fit the other shares"
        if any(not 0 <= s[k] <= p["max_level"] for k in SENSORS) or s["neurons"] > p["max_neurons"] * tol \
                or (p["max_level"] == 0 and s["voice"] > 0):
            return "sensors, a voice or neurons beyond what the body allows"
        if n and not SPEEDS[0] <= s["speed"] <= SPEEDS[1]:
            return "speed outside 1 to 5"
        if n == 0 and any(s[k] != 0 for k in STATE_KEYS[1:-2] + LEDGER_KEYS):
            return "numbers without a population"
        if type(s["predator_sense"]) is not int or not 0 <= s["predator_sense"] <= MAX_LEVEL:
            return "predator_sense is not a level 0 to 5"
        if s["stage"] != stage_of(n, s["predator_sense"], s["neurons"], s["eyed"], s["sensing"]):
            return "the stage does not follow from the numbers"
        return None

    @staticmethod
    def _genomes_ok(r: Rollout) -> str | None:
        """The metrics of every step are the means of its genomes, by the rules."""
        genomes = r.genomes
        if len(genomes) != len(r.steps):
            return "genomes and steps differ in number"
        p = r.params
        cache: dict[Genome, _Traits] = {}
        for t, (pop, s) in enumerate(zip(genomes, r.steps)):
            for g in pop:
                if len(g) != 7 or any(not 0 <= x <= p["max_level"] for x in g[:4]) \
                        or not SPEEDS[0] <= g[5] <= SPEEDS[1] or g[6] not in (0, min(1, p["max_level"])) \
                        or not 0 <= g[4] <= MAX_NEURON_GENE \
                        or neuron_count(g[4]) > p["max_neurons"]:
                    return f"step {t}: a genome outside its limits"
                if g not in cache:
                    cache[g] = traits(g, p)
            tr = [cache[g] for g in pop]
            for key, value in _measure(pop, tr).items():
                if s[key] != value:
                    return f"step {t}: {key} is not what the genomes give"
            if not close(s["e_intake"], math.fsum(x.intake for x in tr), rel=1e-12, abs_=1e-12):
                return f"step {t}: e_intake is not what the genomes take in"
        return None

    # -- questions ----------------------------------------------------------------------------

    def parse(self, prompt: str) -> tuple[int, str, int, str] | None:
        """``(seed, where, step, key)`` with ``where`` in ``step``, ``next``, ``final``, ``params``."""
        w = prompt.split(" ")
        if len(w) < 5 or w[0] != SIM or w[1] != "seed":
            return None
        i = _digits_end(w, 2)
        seed = parse_num(w[2:i])
        if not isinstance(seed, int) or i >= len(w) or num(seed).split() != w[2:i]:  # no leading zeros
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
        if not isinstance(step, int) or not 0 <= step < STEPS or num(step).split() != digits:
            return None
        if len(rest) == 2 and rest[0] == "next" and rest[1] in STATE_KEYS + LEDGER_KEYS and step + 1 < STEPS:
            return seed, "next", step, rest[1]
        if len(rest) == 1 and rest[0] in STATE_KEYS + LEDGER_KEYS:
            return seed, "step", step, rest[0]
        return None

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
        value = self.truth(prompt)
        if value is None:
            return Verdict(False, None, "not my question")
        words = prompt.split()
        key, is_param = words[-1], words[-2] == "params"
        expected = param_value(key, value) if is_param else dense_value(value)
        if isinstance(value, str):
            ok = answer.strip() == value
            return Verdict(ok, expected, "exact word" if ok else "wrong word")
        got = parse_num(answer)
        if not is_finite(got):  # "1 e 9 9 9" parses to infinity, which is close to nothing
            return Verdict(False, expected, "not a number")
        if isinstance(value, int) or is_param:
            ok = got == value
            return Verdict(ok, expected, "exact" if ok else "wrong count or parameter")
        ok = close(float(got), float(value), rel=0.05)
        return Verdict(ok, expected, "within 5 percent" if ok else "more than 5 percent off")

    def records(self) -> list[Line]:
        return []

    def generate(self, rng: random.Random, n: int) -> list[Line]:
        """All lines of ``n`` canonical rollouts with seeds drawn from ``rng``."""
        out: list[Line] = []
        for seed in seeds(rng, n):
            out += lines(self.rollout(seed))
        return out


def _summary(r: Rollout) -> dict[str, float | int | str]:
    """What the last step hands upward (see the module docstring)."""
    last = r.steps[-1]
    genomes = getattr(r, "genomes", None)
    pop = genomes[-1] if genomes else []
    gain = 0.0
    if pop:
        by = {g: traits(g, r.params) for g in set(pop)}
        callers = sum(1 for g in pop if g[6] and by[g].notices) / len(pop)
        hearing = math.fsum(by[g].hearing for g in pop) / len(pop)
        gain = sig(kin_gain(last["attack"], last["escape"], hearing, callers))
    return {"stage": last["stage"], "population": last["population"], "eyes": last["eyes"], "ears": last["ears"],
            "neurons": last["neurons"], "voice": last["voice"], "eared": last["eared"], "talkers": last["talkers"],
            "survival": last["survival"], "predator_pressure": sig(last["attack"]), "kin_gain": gain,
            "group": "yes" if gain >= GROUP_MIN else "no",
            "outcome": outcome_of(last["population"], last["neurons"], last["eyed"], last["sensing"])}


def handoff_out(summary: dict) -> dict[str, float | int | str]:
    """What the summary gives upward to level 11 (signals), under the names of the task: the
    shares with ears and with a voice (and with both), the neurons, whether agents would gain
    from staying near kin, and the predator pressure."""
    return {key: summary[key] for key in ("eared", "voice", "talkers", "neurons", "group", "kin_gain",
                                          "predator_pressure", "population", "outcome")}


def _digits_end(words: list[str], start: int) -> int:
    """Index after the run of single-digit tokens that begins at ``start``."""
    i = start
    while i < len(words) and words[i] in _DIGITS:
        i += 1
    return i


def params_line(r: Rollout) -> Line:
    """``senses seed 3 params. light 0 point 5 7. patchiness 0 point 5. ...``"""
    fields = " ".join(f"{k} {param_value(k, r.params[k])}." for k in PARAM_KEYS)
    return Line(f"{r.sim} seed {num(r.seed)} params. {fields}", topic=r.sim, kind="record")


def lines(r: Rollout, every: int = 1) -> list[Line]:
    """The parameter record, a state record and questions for every ``every``-th step (now and
    next), and the final questions. The ledger is left out: it is the gate's, not the model's."""
    keys = list(STATE_KEYS)
    out = [params_line(r)]
    for t in range(0, len(r.steps), every):
        out.append(state_line(r, t, keys))
        out += query_lines(r, t, keys)
    return out + summary_lines(r)


_SIM = Senses()


def simulation() -> Senses:
    return _SIM


def run(seed: int, **params) -> SensesRollout:
    """``simulation().run``."""
    return _SIM.run(seed, **params)


def conserved(r: Rollout) -> Verdict:
    """``simulation().conserved``."""
    return _SIM.conserved(r)


def owns(prompt: str) -> bool:
    """True for a question about a rollout of this level or one of its lessons."""
    return _SIM.owns(prompt) or LESSONS.owns(prompt)


def check(prompt: str, answer: str) -> Verdict:
    """Judge an answer about a rollout (replayed) or to a lesson (computed from its inputs)."""
    if LESSONS.owns(prompt):
        return LESSONS.check(prompt, answer)
    return _SIM.check(prompt, answer)
