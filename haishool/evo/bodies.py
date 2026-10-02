"""Level 9 of the toy universe: cells that stay together, and who eats whom.

Level 8 ends with cells, simple ones or complex ones that carry a partner. Here cells form
bodies, bodies eat each other, and the eaten get bigger. A world is one well-mixed vessel with
light for ``light`` biomass units of producers and a community of at most 60 **lineages**
(species). A lineage has a biomass and four heritable traits:

* ``size``: cells per body, a power of two from 1 to 1048576 (2^20);
* ``adhesion``: the share of daughter cells that stay attached, ``1 - 1 / size``
  (:func:`adhesion`; in this toy adhesion and size are one trait seen twice: a mutation of the
  adhesion doubles or halves the body);
* ``cell_types``: 1 to 30, never more than the size allows (:func:`max_cell_types`);
* ``guild``: ``producer`` (lives on light), ``grazer`` (eats producers) or ``predator`` (eats
  grazers, and where the energy pyramid supports a fourth level also predators with at most an
  eighth of its cells; such a predator counts as level 4 however long the chain of predators
  below it is).

A world starts with one lineage of single-celled producers at 5 % of the capacity and runs for
80 output steps (0 to 79) of 20 **ticks** each. Nobody knows in detail how bodies, cell types and
food webs arose; this level strings together simple published models, each reduced to a formula
a line can state, and says at the end which parts are invented. It is a toy that shows what
those rules do together, not a reconstruction of how multicellular life or animals came about.
A tick has no unit; think of a generation of single cells.

**One tick**, in this order, each phase acting on what the last one left
(:meth:`_World.tick`):

1. *Upkeep.* Every lineage loses ``upkeep * biomass``, with ``upkeep = 0.05 * pace * (1 + 0.03
   * (cell_types - 1))`` (:func:`upkeep_rate`). The ``pace`` is Kleiber's law per cell: a body of
   n cells turns over n^0.75 (:func:`kleiber_rate`), each of its cells n^-0.25
   (:func:`energy_per_cell`). So a large body needs less energy per cell, and it also grows and
   feeds that much slower: every rate of a body runs at its pace. 80 % of the upkeep is
   respired, 20 % is shed as dead matter.
2. *Feeding.* All but 0.5 biomass units of a guild are within reach of its consumers
   (:func:`reachable`; the hidden part is shared out in proportion). A consumer lineage with
   biomass C finds the food F = sum over its prey of ``vulnerability * biomass within reach``,
   with ``vulnerability = hunter cells / (hunter cells + prey cells)`` (:func:`vulnerability`):
   small prey are eaten more, large prey escape. Its pressure on fully vulnerable prey is
   ``attack * C / (1 + attack * handling * F)`` (:func:`predation_pressure`, the saturating
   response). A prey lineage under the pressures q_j of its consumers loses the share ``1 -
   exp(-sum of q_j * vulnerability_j)`` of its biomass within reach (:func:`fraction_eaten`),
   split among the consumers in proportion. A consumer keeps ``efficiency`` of what it eats;
   the rest goes to the decomposers. ``attack = 0.05 * predation * pace * supply``
   (:func:`attack_rate`), ``handling = 0.25 / (pace * supply)`` (:func:`handling_time`);
   predators hunt at ``min(1, oxygen / 0.02)`` of their attack rate
   (:func:`predator_activity`).
3. *Growth.* A producer lineage makes ``growth_rate * biomass * max(0, 1 - all producers /
   light)`` (:func:`production`: logistic growth on shared light), with ``growth_rate = 0.6 *
   pace * supply`` (:func:`growth_rate`).
4. *Deaths.* A consumer whose body is bigger than the oxygen allows (:func:`oxygen_max_size`)
   suffocates: its lineage loses 20 % per tick. A lineage below 0.01 biomass units is extinct;
   what is left of it goes to the decomposers.
5. *Mutation.* Every established lineage (0.5 biomass units or more) founds a mutant with
   chance ``mutation / 20``: the size doubles or halves, a cell type is gained or lost, or the
   guild moves up (producer to grazer, grazer to predator), each of the five as likely. The
   mutant takes 2 % of its parent's biomass; a mutant with the traits of a living lineage joins
   that lineage. A mutant that is not viable is not founded. When the vessel holds 60 lineages,
   a new one pushes out the rarest lineage if that is smaller than the founder (it is
   ``displaced`` and counts as an extinction), else it is not founded either. The last lineage
   of the producers is never pushed out (the rarest of the others is looked at instead): the
   cap of 60 is a limit of the computation, and it must not be what ends a world.
6. *Oxygen*, in units the toy calls the present level (oxygen 1 is where a consumer body may
   have 16384 cells; the unit is tied to nothing measured): set free with production (5e-5 per
   biomass unit made), used up by respiration and by the decomposers for the half of the dead
   matter that is not buried, and 0.1 % of it per tick by rocks and gases; never more than
   there is. What stays in the air is what burial keeps from being burnt again: the textbook
   reason why the Earth's air holds oxygen, here with invented numbers.

``supply`` (:func:`supply`) says how well the cells of a body are supplied, a free cell being 1.
The surface of a ball of n cells per volume falls as n^(-1/3) (:func:`surface_to_volume`), and
every cell type adds exchange surface: ``supply = min(1, cell_types * size^(-1/3))``.

**Viable** is a mutant

* with at least 1 cell type and at most ``1 + floor(1.5 * log2(size))``, never more than 30
  (:func:`max_cell_types`; a body that halves loses the cell types it can no longer hold);
* with two cell types, germ and soma, when it has more than 32 cells
  (:func:`needed_cell_types`: a larger body needs division of labour to keep reproducing);
* with at most 64 cells and 2 cell types when the world has no complex cells
  (``complex_cells no``; a toy rule, see below);
* that, as a consumer, is no bigger than the oxygen allows: ``16384 * oxygen^1.5`` cells,
  rounded down to a power of two (:func:`oxygen_max_size`). Producers are exempt: they make
  oxygen in their own cells;
* that, when it moves up a guild, has prey to hunt (``predation`` above 0) and a trophic level
  the energy pyramid supports (:func:`supported_levels`), and for a predator oxygen of at least
  0.02.

What drives the bodies. Among producers alone the single cell wins: with the pace on every
rate, a larger body is no cheaper to run, only slower, and worse supplied. With grazers the
smallest prey is eaten most, so producers that double escape; grazers that double catch them
again (:func:`vulnerability` rises with the hunter's size) and escape their own predators; and
a large body needs cell types to keep its cells supplied. The oxygen sets how far the consumers
can follow. Being eaten is the only reason for a larger body that this toy has. For real
organisms it is one proposed reason among several (in the experiment of Ratcliff et al. 2012
yeast became multicellular when selected for settling fast, with no predator), so "no
predation, no bodies" is a property of the toy and not a claim about the history of life.

**A step shows** (:data:`STATE_KEYS`): ``species`` (living lineages), the biomass of
``producers``, ``grazers`` and ``predators``, ``max_size`` and ``cell_types`` (the largest
body and the most cell types among the *common* lineages: established ones that hold at least a
tenth of their guild's biomass; the largest producer lineage always counts),
``consumer_size`` and ``consumer_types`` (the same among common grazers and predators, 0
without any), ``mean_size`` (all cells divided by all bodies), ``trophic_levels`` (the highest
level whose lineages together are established: 1 producers, 2 grazers, 3 predators, 4
predators that eat predators), ``oxygen``, ``extinctions`` (so far) and the ``stage``:

    collapse        the producers are below 1 % of the capacity (grazed down to their refuge)
    single_cells    the common bodies are single cells
    colonies        the largest common body has 2 to 32 cells
    multicellular   it has more than 32 cells (so it has germ and soma)
    differentiated  and some common body has 3 or more cell types
    food_web        and predators are established (three trophic levels)

The summary gives the last step's ``stage``, ``species``, ``max_size``, ``cell_types``,
``consumer_size``, ``consumer_types``, ``trophic_levels`` and ``oxygen``, the
``predator_share`` (predators in all biomass) and the ``outcome`` (:func:`outcome_of`): what
the bodies are at the end, whatever the producers' biomass. ``single_cells``, ``colonies`` and
``multicellular`` go by the largest common body, as the stages do. ``animals_like`` asks for
common *consumers* with more than 32 cells and three or more cell types in a world with
established predators: a differentiated body that eats, among hunters. That is all the word
means; a world whose only differentiated bodies are producers ends as ``multicellular``. A
fifth word, ``collapse``, is kept for a world that has lost its producers (the consumers that
are left starve). The rules do not quite exclude that: every producer lineage could fall below
the extinction threshold in the same tick (some fifty lineages of about equal size sharing the
refuge), or the last one left could be a body too poorly supplied to grow. It happened in none
of the canonical seeds 1 to 1000 and in none of 96 hand-set worlds with light from ten
thousand to a million; the word is there so that such a vessel is not handed upward as
``single_cells``.

**Lines** (:func:`lines`; numbers as digits, measured numbers to 3 significant digits,
parameters as given; these are seed 3 verbatim, the longest line of seeds 1 to 200 has 72
tokens, the longest of 8000 lessons 38; a rollout gives 1131 lines):

    bodies seed 3 params. light 5 1. efficiency 0 point 1 3. predation 1 point 2 9. mutation 1 point 9 5. start_oxygen 0 point 0 0 1 4. complex_cells yes.
    bodies seed 3 step 4 0. species 3 1. producers 3 0 point 1. grazers 1 0 point 2. predators 0 point 2 9 7. max_size 3 2. mean_size 2 1 point 1. cell_types 3. consumer_size 3 2. consumer_types 3. trophic_levels 2. oxygen 0 point 0 4 9 2. extinctions 3 7. stage colonies.
    q bodies seed 3 step 4 0 max_size. a 3 2.
    q bodies seed 3 step 7 9 stage. a food_web.
    q bodies seed 3 final outcome. a animals_like.
    q bodies predict vulnerability prey 4 0 9 6 hunter 1 3 1 0 7 2. a 0 point 9 6 9 7.

**Gate** (:meth:`Bodies.conserved`). Each step carries a ledger (:data:`LEDGER_KEYS`), summed
over its ticks, and the rollout keeps every lineage of every step and a log of foundings and
extinctions. Checked for every step:

* energy, guild by guild (Lindeman's budget of a trophic level): ``income = respired + growth
  + taken + lost + moved`` to 1e-9 of the capacity, where the income of the producers is their
  production and that of consumers what they ate, ``growth`` the change of the guild's biomass,
  ``taken`` what consumers ate of it (transferred up), ``lost`` what went to the decomposers
  (shed, not assimilated, suffocated, extinct, displaced) and ``moved`` the biomass of founders
  that changed guild. What is taken from one guild is the income of the next; over the whole
  community ``production = respiration + growth + lost``;
* no biomass or flow is negative; no consumer takes more than its prey has (``max_take``, the
  largest share of a prey lineage eaten in one tick, is at most 1); a consumer guild keeps at
  most ``efficiency`` of its food and passes on no more than it kept and founders brought;
* oxygen: its change equals what was produced minus what was consumed, produced is 5e-5 times
  the production, and it is never negative;
* the lineages: alive above the threshold, sizes powers of two, cell types within what the size
  allows and needs, no two lineages with the same traits, no body or cell type that needs
  complex cells in a world without them, no trophic level above what the pyramid supports; no
  consumer founded bigger than the oxygen allowed, no predator founded below oxygen 0.02, the
  last producer lineage never pushed out of a full vessel;
* every metric, the stage and the summary follow from the lineages (:func:`describe`), and
  ``extinctions`` matches the log.

:meth:`Bodies.check` replays a seed (parameters from :func:`random_params` with
``random.Random(seed)``) and compares: words, counts and parameters exactly, measured numbers
within 5 percent. Module level :func:`check` and :func:`owns` also answer the lessons.

**Lessons** (:data:`LESSONS`, :data:`RULES`): every rule above as a question that carries its
inputs: ``kleiber``, ``energy_per_cell``, ``surface_to_volume``, ``adhesion``,
``max_cell_types``, ``needed_cell_types``, ``supply``, ``upkeep``, ``growth_rate``,
``attack_rate``, ``handling_time``, ``vulnerability``, ``reachable``, ``predation_pressure``,
``fraction_eaten``, ``production``, ``logistic_step``, ``pyramid_flow``, ``supported_levels``,
``oxygen_max_size``, ``predator_activity``, and one whole tick of the predator-prey rule for
single-celled producers and grazers (``prey_step``, ``grazer_step``: the simulation's own
:meth:`_World.ecology` on two lineages). The functions that answer the lessons are the
functions the simulation calls; the one exception is ``logistic_step``, which the simulation
does not call by name: it is :func:`production` for a lineage that has the light to itself,
and the tests show that the growth of a lone lineage in a tick is that step, bit for bit.
The inputs are drawn where the rollouts are (:class:`BodyLessons` says how): body sizes evenly
over the doublings, producers as a share of their capacity, oxygen over the orders of
magnitude. Drawn evenly from their ranges, ``production`` answered 0 in 70 % of its lessons,
``supported_levels`` 4 in 82 %, ``needed_cell_types`` 2 in 76 % and ``predator_activity`` 1 in
62 %; now no answer of a rule has more than about half of its lessons.

**Hand-off.** *In* (:func:`handoff_in`): from the summary of level 8 (cells), ``complex_cells``
is ``yes`` when its outcome is ``complex_cells``, and its ``energy_per_cell`` sets the
``efficiency`` (invented for the chain: 0.05 at the 10 units of a level-8 cell without a
working partner, 0.2 at the 160 of a cell with one, by the logarithm in between). Light,
predation, mutation and the starting oxygen are not the cells' to give. *Out*: the summary, and
:func:`handoff_out` under the names level 10 (senses) asks for: ``body_size`` and
``cell_types`` (those of the common consumers, the bodies that could carry senses; of all
common bodies when there are no consumers), ``predators`` (yes when the third trophic level is
established), ``predator_share``, ``oxygen`` and ``species``.

**Plausibility targets**: what the rules above must produce, with the values measured over the
canonical seeds 1 to 1000 (``tests/test_evo_bodies.py`` holds each with a margin on seeds 1
to 200 and on worlds with one parameter forced). They are properties of the toy; none is a
finding about real organisms:

    every seed      the gate passes                              worst error 3.6e-12 biomass units
                    no trophic level above the pyramid's         1000 of 1000 (a rule, and the gate)
    no predation    bodies stay small (144 worlds)               single cells at the end in 143, never a
                                                                 common body above 2 cells
    predation from  size and cell types rise (484 worlds with    largest common body: median 128 cells,
    0.5, complex    complex cells)                               quartiles 32 and 256, most 1024; cell
    cells                                                        types: median 5, most 9
    no complex      no differentiated bodies (309 worlds)        never above 64 cells or 2 cell types;
    cells                                                        stage never differentiated or food_web
    efficiency      more of it, more trophic levels (worlds      three or four levels at the end: 34 % of
                    with predation from 0.5)                     the worlds below 0.08, 85 % from 0.16;
                                                                 four: none below 0.08, 35 % from 0.16
    light           richer worlds carry more levels and more     three or four levels: 16 % below 50, 93 %
                    oxygen, and they crash (worlds with          from 400; oxygen at the end: median 0.038
                    predation from 0.5)                          and 0.42; steps in collapse: in no world
                                                                 below light 129, 18 % of the steps from 400
    oxygen          no predators without it; it caps the         78 worlds stay below 0.02: no predator
                    consumers                                    lineage in any; of 806 worlds with common
                                                                 consumers 124 end with them at the cap
    outcomes        all four occur                               single_cells 228, colonies 300,
                                                                 multicellular 168, animals_like 304

With one parameter forced and the others drawn (seeds 1 to 80, the parameters of
:func:`random_params` with predation set to 1.2 and then the one parameter forced; "complex
cells" means complex_cells forced to yes as well): predation 0, 0.2, 0.5, 1, 2 gives a median
largest common body of 1, 8, 96, 128, 128 cells (complex cells); efficiency 0.05, 0.08, 0.12,
0.16, 0.2 gives three or more levels in 19, 27, 49, 64, 74 of the 80 worlds; light 20, 60, 200,
600, 1000 gives them in 0, 34, 67, 75, 78 of 80, steps in collapse 0, 0, 1, 21, 22 %, and a
median largest body of 8, 128, 256, 64, 64 cells: the bodies are largest in middling light and
smaller again in the rich worlds that crash (why was not isolated); mutation 0.2, 1, 3 gives 4,
128, 512 cells (complex cells).

A world of the canonical seeds 1 to 1000 takes about 0.09 seconds on average, at most about half
a second; the
busiest hand-set world of the tests (light a million, 20 mutants per lineage and step) 0.75
(timed on a machine busy with other work; the test asks for under 2 seconds).

**What is taken from published models and what is invented** (authors and years from memory,
none looked up for this file).

* Logistic growth (Verhulst 1838) for the producers: taken as it is, with all producer lineages
  sharing one capacity (competition for light with equal weights).
* The predator-prey model of Lotka (1925) and Volterra (1926), with logistic prey and a
  saturating consumer as in Rosenzweig and MacArthur (1963); the saturating response is
  Holling's disc equation (1959). Taken: the forms. Invented: the numbers 0.05 and 0.25, and
  the step: within a tick the pressure is held fixed and the prey's loss integrated,
  ``1 - exp(-pressure)``, the form of Nicholson and Bailey (1935) for discrete generations;
  it is what keeps a consumer from taking more than there is. The refuge of 0.5 units is the
  usual stabiliser of such models, here invented in its size and in being shared by a guild.
  That rich worlds cycle until the prey crash is the paradox of enrichment (Rosenzweig 1971);
  the stage ``collapse`` marks it.
* The trophic budget and the transfer efficiency of about 10 % after Lindeman (1942). Here the
  efficiency is the share of the eaten biomass that becomes consumer biomass, before the
  consumer's own upkeep; real efficiencies compare the production of two levels. That food
  chains are short where little energy arrives is the pyramid; :func:`supported_levels` is a toy
  threshold on it (the flow must pay the upkeep of 0.5 units of single cells), and how many
  levels a world really carries is left to the dynamics (after Oksanen et al. 1981: more
  productive worlds carry longer chains).
* Kleiber's law (1932), metabolic rate ~ mass^0.75, and with it the slower pace of all rates
  of large bodies (after Brown et al. 2004). Taken: the exponent. Toy: a body's mass is its
  number of cells, one pace applies to upkeep, growth, attack and handling alike, and the law
  is used from a single cell upward. It is a regularity measured across animals of very
  different size; that it also holds from one cell to a colony of a few is assumed here.
* Size-selective predation as a route to multicellularity after Boraas, Seale and Boxhorn
  (1998), whose single-celled alga under a flagellate predator turned into colonies within a
  hundred generations (they settled at eight cells). Taken: that small prey are eaten more.
  Invented: the form ``hunter / (hunter + prey)`` and that it holds for every size. Other
  routes have been shown in the laboratory (Ratcliff et al. 2012, see above); the toy has
  this one only.
* Germ and soma: among the volvocine algae the small colonies have one kind of cell, the large
  ones set aside body cells that no longer divide (after Kirk 1998). The threshold 32 and the
  rule that a body above it cannot reproduce without a second cell type are the toy's.
* More cells, more cell types (Bell and Mooers 1997; Bonner 1988): taken as a direction only.
  The cap ``1 + 1.5 * log2(cells)`` and the rule that each cell type adds exchange surface
  are invented; surface over volume of a ball is geometry.
* Oxygen: that the net source is the burial of organic matter is the textbook budget (after
  Berner and Canfield 1989); that a body living on diffusion can be as thick as the square
  root of the oxygen outside is the diffusion argument (after Runnegar 1982; Catling et al.
  2005); that carnivores are the first to go where oxygen is low is after Sperling et al.
  (2013). Invented: every constant (5e-5, one half buried, 0.1 % per tick, 16384 cells at the
  present level, the 0.02 for predators), the unit itself (the oxygen of a run is not a
  reconstruction of any era of the Earth), that producers are exempt, and that no body here
  has a circulation that would lift the limit.
* Lineages that found mutants with a small share of their biomass: the picture of adaptive
  dynamics (after Geritz et al. 1998), without its mathematics. The 2 %, the threshold 0.01,
  the five equally likely mutations (one of them turns a producer into a grazer or a grazer
  into a predator at a stroke), the cap of 60 lineages and the rule that a founder pushes out
  the rarest lineage of a full vessel, though never the last producers, are invented.
* ``complex_cells no`` caps bodies at 64 cells and 2 cell types. That is the toy rule that
  ties this level to level 8 (after the energy argument of Lane and Martin 2010 that level 8
  plays); real bacteria do build filaments with two or three cell types, and no law is known
  that stops them there.
* ``adhesion = 1 - 1 / size`` is invented: if a daughter cell stays with chance a, a chain
  grows to 1 / (1 - a) cells on average.

A world has no space, no seasons, no sex and no development, one resource, three guilds and
three traits. The biomass unit and the tick are arbitrary. In 80 steps the common bodies of
the canonical worlds reach at most 1024 cells and 10 cell types (the largest living lineage has
4096 cells), not the 2^20 and 30 the traits allow: mutations are the bottleneck, and large
bodies live slowly. Where grazers cannot live on their own, the founders that switch guild can
keep a small population of them going. The starting oxygen is still seen at the end of a dim
world: the rocks take up 0.1 % of it per tick, which leaves a fifth after the 1580 ticks of a
run, so of two worlds that differ only in the start (0.001 and 0.2) the second ends with 2.4
times the oxygen at light 30, 1.3 times at light 150 and 1.06 times at light 1000 (medians of
80 worlds each). ``max_size`` and ``cell_types`` are two maxima over the common bodies and
need not belong to the same lineage, and likewise ``consumer_size`` and ``consumer_types``.
The numbers of a run tell the toy's rules; they say nothing measured about the history of
life.

**Reproducibility.** ``random.Random(seed)`` gives two numbers per established lineage and
tick, in order of founding; nothing else is random. The simulation uses +, -, *, / and the
square root only (the exponential and the cube root are written out in those,
:func:`fraction_eaten`, :func:`_cbrt`), sums run in order of founding, ties are broken by the
order of founding, and there is no numpy: every IEEE 754 machine computes the same bits,
whatever its C library or numpy. :func:`random_params` and the lessons draw light and oxygen
over the orders of magnitude with the same means (:func:`_exp`, the logarithms written out as
constants) and round them. What still asks the C library: ``haishool.evo.sig`` takes a
``log10`` to find where the third digit is, and :func:`handoff_in` a ``log`` of the energy
below, rounded with a margin.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

from haishool.cosmos import Rollout, close, seeds, state_line, summary_lines
from haishool.evo import Input, LessonGate, Rule, sig
from haishool.truth import Line, Verdict, is_finite, num, parse_num

SIM = "bodies"

STEPS = 80          # output steps 0 .. 79
TICKS = 20          # ticks of the ecology between two output steps
MAX_K = 20          # a body has 2^k cells, k = 0 .. 20
MAX_SIZE = 1 << MAX_K  # 1048576 cells
MAX_TYPES = 30
MAX_LINEAGES = 60
GUILDS = ("producer", "grazer", "predator")
GUILD_KEYS = ("producers", "grazers", "predators")

R0 = 0.6            # growth of single-celled producers per tick, far below the capacity
M0 = 0.05           # upkeep of single cells per tick, as a share of their biomass
SHED = 0.2          # share of the upkeep that is shed as dead matter (the rest is respired)
A0 = 0.05           # attack rate of single-celled consumers at predation 1, per biomass and tick
H0 = 0.25           # handling time of single-celled consumers: ticks per biomass eaten per own biomass
TYPE_COST = 0.03    # each cell type beyond the first raises the upkeep by this share
GERM_SIZE = 32      # bodies of more cells need germ and soma (two cell types) to reproduce
PREY_RATIO = 8      # a predator eats other predators only when it has at least 8 times their cells
REFUGE = 0.5        # biomass of a guild that is hidden from its consumers
FOUND = 0.02        # share of the parent's biomass that founds a mutant lineage
EXTINCT = 0.01      # a lineage below this biomass is extinct
ESTABLISHED = EXTINCT / FOUND  # 0.5: from here on a lineage can found a mutant that is not extinct at once
COMMON = 0.1        # a lineage is common when it holds a tenth of its guild's biomass
START_SHARE = 0.05  # the first producers fill this share of the capacity
CRASH = 0.01        # producers below this share of the capacity: the stage is "collapse"
SUFFOCATE = 0.2     # share of a consumer's biomass lost per tick when its body is too big for the oxygen
N_PAL = 1 << 14     # cells of the largest consumer body at the present level of oxygen (toy)
O2_PREDATOR = 0.02  # oxygen from which predators hunt at full strength (toy)
O2_YIELD = 5e-5     # oxygen (present levels) set free per unit of biomass produced
BURIAL = 0.5        # share of the dead matter that is buried and not decomposed
O2_SINK = 0.001     # share of the oxygen that rocks and gases take up per tick
FLOW_MIN = M0 * ESTABLISHED  # 0.025: upkeep per tick of an established population of single cells
SIMPLE_MAX_K = 6    # without complex cells: at most 64 cells ...
SIMPLE_MAX_TYPES = 2  # ... and two cell types (toy)
DIFFERENTIATED = 3  # cell types from which a body counts as differentiated
MUTATIONS = ("size_up", "size_down", "type_gain", "type_loss", "guild")
GONE = ("extinct", "displaced")  # how a lineage ends: below the threshold, or pushed out of the full vessel
CELL_ENERGY, CELL_GAIN = 10.0, 16.0  # level 8 (cells.ENERGY, cells.ENDO_GAIN): energy of a cell without a
#                                      partner, and the factor by which a working partner multiplies it
#: natural logarithms, written out so that no C library is asked: of 20 and 1000 (light), of 0.001
#: and 0.2 (starting oxygen), of 1200 (the oxygen of a lesson spans 0.001 to 1.2), of 16 (CELL_GAIN)
LN_20, LN_1000, LN_0001, LN_02 = 2.995732273553991, 6.907755278982137, -6.907755278982137, -1.6094379124341003
LN_1200, LN_16 = 7.090076835776092, 2.772588722239781

STAGES = ("single_cells", "colonies", "multicellular", "differentiated", "food_web", "collapse")
OUTCOMES = ("single_cells", "colonies", "multicellular", "animals_like", "collapse")

PARAM_KEYS = ("light", "efficiency", "predation", "mutation", "start_oxygen", "complex_cells")
STATE_KEYS = ("species", "producers", "grazers", "predators", "max_size", "mean_size", "cell_types",
              "consumer_size", "consumer_types", "trophic_levels", "oxygen", "extinctions", "stage")
RAW_KEYS = ("raw_producers", "raw_grazers", "raw_predators", "raw_oxygen")
LEDGER_TERMS = ("income", "respired", "taken", "lost", "moved")
LEDGER_KEYS = RAW_KEYS + tuple(f"{term}_{g}" for term in LEDGER_TERMS for g in GUILD_KEYS) + (
    "o2_produced", "o2_consumed", "max_take")
SUMMARY_KEYS = ("stage", "species", "max_size", "cell_types", "consumer_size", "consumer_types",
                "trophic_levels", "oxygen", "predator_share", "outcome")
COUNT_KEYS = frozenset({"species", "max_size", "cell_types", "consumer_size", "consumer_types",
                        "trophic_levels", "extinctions"})
WORD_KEYS = frozenset({"stage", "outcome", "complex_cells"})
_DIGITS = frozenset("0123456789")

KEYS: dict[str, str] = {
    "light": "carrying capacity of the producers, biomass units (parameter)",
    "efficiency": "the toy's transfer efficiency, a gross growth efficiency: share of what a consumer eats that "
                  "becomes its biomass, before its upkeep (parameter; not Lindeman's ratio of the production of two levels)",
    "predation": "predation pressure: attack rate of consumers in units of 0.05 per biomass and tick (parameter)",
    "mutation": "mutants an established lineage founds per step on average, before viability (parameter)",
    "start_oxygen": "oxygen at step 0, in the toy's unit (parameter)",
    "complex_cells": "yes when level 8 gave complex cells; no caps bodies at 64 cells and 2 cell types (toy)",
    "species": "living lineages (count, at most 60)",
    "producers": "biomass of all producer lineages, biomass units",
    "grazers": "biomass of all grazer lineages, biomass units",
    "predators": "biomass of all predator lineages, biomass units",
    "max_size": "cells of the largest body among the common lineages (a power of two)",
    "mean_size": "mean cells per body over all bodies: all cells divided by all bodies",
    "cell_types": "most cell types of a body among the common lineages",
    "consumer_size": "cells of the largest body among the common grazers and predators (0 without any)",
    "consumer_types": "most cell types of a body among the common grazers and predators (0 without any)",
    "trophic_levels": "highest trophic level with an established population: 1 producers, 2 grazers, "
                      "3 predators, 4 predators of predators",
    "oxygen": "oxygen in the toy's unit: 1 is called the present level (a consumer body may then have 16384 cells)",
    "extinctions": "lineages that went extinct so far (count)",
    "stage": "single_cells, colonies, multicellular, differentiated, food_web or collapse",
    "predator_share": "final share of the predators in all biomass",
    "outcome": "single_cells, colonies or multicellular by the largest common body; animals_like when the "
               "common consumers are differentiated and predators established (collapse without producers)",
    "raw_producers": "ledger: biomass of the producers, unrounded",
    "raw_grazers": "ledger: biomass of the grazers, unrounded",
    "raw_predators": "ledger: biomass of the predators, unrounded",
    "raw_oxygen": "ledger: oxygen, unrounded",
    "o2_produced": "ledger: oxygen set free since the last step",
    "o2_consumed": "ledger: oxygen used up since the last step",
    "max_take": "ledger: largest share of a prey lineage eaten in one tick since the last step",
}
for _g in GUILD_KEYS:
    KEYS.update({
        f"income_{_g}": f"ledger: {'production' if _g == 'producers' else 'biomass eaten'} of the {_g} since the last step",
        f"respired_{_g}": f"ledger: biomass the {_g} respired since the last step",
        f"taken_{_g}": f"ledger: biomass of the {_g} eaten by consumers since the last step",
        f"lost_{_g}": f"ledger: biomass of the {_g} lost to the decomposers since the last step",
        f"moved_{_g}": f"ledger: biomass that left the {_g} with founders of another guild (negative: arrived)",
    })
del _g


# ---------------------------------------------------------------------------------------------
# the rules, one function each: the simulation calls them, and so do the lessons
# ---------------------------------------------------------------------------------------------

def kleiber_rate(mass: float) -> float:
    """Metabolic rate of a body of ``mass`` cells, in units of one free cell's rate: mass^0.75.

    Written as sqrt(mass * sqrt(mass)), which every IEEE machine rounds the same way.

    >>> kleiber_rate(1), kleiber_rate(16), kleiber_rate(10000)
    (1.0, 8.0, 1000.0)
    """
    return math.sqrt(mass * math.sqrt(mass))


def energy_per_cell(cells: float) -> float:
    """What one cell of a body of ``cells`` cells turns over, a free cell being 1: cells^-0.25.

    >>> energy_per_cell(1), energy_per_cell(16), energy_per_cell(65536)
    (1.0, 0.5, 0.0625)
    """
    return kleiber_rate(cells) / cells


def _cbrt(x: float) -> float:
    """Cube root of ``x >= 1`` by Newton's iteration from a power of two, rounded to 12 decimals:
    only +, * and /, so every IEEE machine gets the same bits (``math.cbrt`` may differ in the
    last digit from one C library to the next).

    >>> _cbrt(1), _cbrt(8), _cbrt(1000), _cbrt(2)
    (1.0, 2.0, 10.0, 1.259921049895)
    """
    y = float(1 << ((int(x).bit_length() + 2) // 3))
    for _ in range(12):
        y = (2.0 * y + x / (y * y)) / 3.0
    return round(y, 12)


def _exp(x: float) -> float:
    """e^x with +, * and / only (the series below 0.5, squared back up), so that the drawn
    parameters do not depend on the C library either. Good to a few units in the last place.

    >>> _exp(0.0), round(_exp(1.0), 12), round(_exp(-2.0), 12), round(_exp(6.907755278982137), 9)
    (1.0, 2.718281828459, 0.135335283237, 1000.0)
    """
    if x < 0.0:
        return 1.0 / _exp(-x)
    doublings = 0
    while x > 0.5:
        x *= 0.5
        doublings += 1
    y = 1.0
    for n in range(18, 0, -1):
        y = 1.0 + x / n * y
    for _ in range(doublings):
        y *= y
    return y


def surface_to_volume(cells: float) -> float:
    """Surface over volume of a ball of ``cells`` cells, each a ball of radius 1: 3 / cells^(1/3).

    >>> surface_to_volume(1), surface_to_volume(8), surface_to_volume(1000)
    (3.0, 1.5, 0.3)
    """
    return 3.0 / _cbrt(cells)


def adhesion(cells: int) -> float:
    """The share of daughter cells that stay attached in a body of ``cells`` cells: 1 - 1 / cells.

    >>> adhesion(1), adhesion(2), adhesion(64)
    (0.0, 0.5, 0.984375)
    """
    return 1.0 - 1.0 / cells


def max_cell_types(cells: int) -> int:
    """Toy rule: a body of ``cells`` cells has at most 1 + floor(1.5 * log2(cells)) cell types,
    never more than 30. Computed in whole numbers: floor(1.5 log2 n) = floor(log2(n^3) / 2).

    >>> [max_cell_types(n) for n in (1, 2, 3, 4, 8, 64, 1024, 1048576)]
    [1, 2, 3, 4, 5, 10, 16, 30]
    """
    return min(MAX_TYPES, 1 + ((int(cells) ** 3).bit_length() - 1) // 2)


def needed_cell_types(cells: int) -> int:
    """Toy rule: a body of more than 32 cells needs germ and soma, two cell types, to reproduce.

    >>> needed_cell_types(32), needed_cell_types(64)
    (1, 2)
    """
    return 1 if cells <= GERM_SIZE else 2


def supply(cells: int, cell_types: int) -> float:
    """How well a body's cells are supplied, 1 being a free cell: every cell type adds exchange
    surface, so supply = min(1, cell_types * cells^(-1/3)).

    >>> supply(1, 1), supply(8, 1), supply(8, 2), supply(1000, 5)
    (1.0, 0.5, 1.0, 0.5)
    """
    return min(1.0, cell_types * surface_to_volume(cells) / 3.0)


def upkeep_rate(cells: int, cell_types: int) -> float:
    """Share of its biomass a body burns or sheds per tick: Kleiber's pace, and 3 percent more
    for every cell type beyond the first.

    >>> upkeep_rate(1, 1), round(upkeep_rate(16, 3), 6)
    (0.05, 0.0265)
    """
    return M0 * energy_per_cell(cells) * (1.0 + TYPE_COST * (cell_types - 1))


def growth_rate(cells: int, cell_types: int) -> float:
    """Growth of a producer per tick far below the capacity: 0.6 * pace * supply, the pace being
    :func:`energy_per_cell`.

    >>> growth_rate(1, 1), round(growth_rate(16, 2), 9), growth_rate(4096, 16)
    (0.6, 0.238110158, 0.075)
    """
    return R0 * energy_per_cell(cells) * supply(cells, cell_types)


def attack_rate(predation: float, cells: int, cell_types: int) -> float:
    """Attack rate of a consumer per biomass and tick: 0.05 * predation * pace * supply.

    >>> attack_rate(1.0, 1, 1), round(attack_rate(2.0, 16, 2), 9)
    (0.05, 0.039685026)
    """
    return A0 * predation * energy_per_cell(cells) * supply(cells, cell_types)


def handling_time(cells: int, cell_types: int) -> float:
    """Ticks a consumer needs to eat its own biomass: 0.25 / (pace * supply). A fed consumer
    eats 1 / handling_time of its biomass per tick, however much food there is.

    >>> handling_time(1, 1), round(handling_time(16, 2), 9), handling_time(4096, 16)
    (0.25, 0.629960525, 2.0)
    """
    return H0 / (energy_per_cell(cells) * supply(cells, cell_types))


def vulnerability(prey_cells: int, hunter_cells: int) -> float:
    """Size-selective feeding: the share of encounters that end with the prey eaten,
    hunter / (hunter + prey). Small prey are eaten more.

    >>> vulnerability(1, 1), vulnerability(8, 1), vulnerability(1, 8)
    (0.5, 0.1111111111111111, 0.8888888888888888)
    """
    return hunter_cells / (hunter_cells + prey_cells)


def production(biomass: float, rate: float, total: float, capacity: float) -> float:
    """Logistic growth: what a producer lineage makes in one tick, ``total`` being the biomass of
    all producers that share the light: rate * biomass * max(0, 1 - total / capacity).

    >>> production(10.0, 0.5, 50.0, 100.0), production(10.0, 0.5, 120.0, 100.0)
    (2.5, 0.0)
    """
    return rate * biomass * max(0.0, 1.0 - total / capacity)


def logistic_step(biomass: float, rate: float, capacity: float) -> float:
    """One producer lineage alone, one tick of growth: biomass + production.

    >>> logistic_step(50.0, 0.5, 100.0)
    62.5
    """
    return biomass + production(biomass, rate, biomass, capacity)


def reachable(total: float) -> float:
    """The share of a guild's biomass its consumers can reach: all but 0.5 units hide.

    >>> reachable(0.25), reachable(1.0), reachable(50.0)
    (0.0, 0.5, 0.99)
    """
    return max(0.0, 1.0 - REFUGE / total) if total > 0 else 0.0


def predation_pressure(attack: float, hunters: float, handling: float, food: float) -> float:
    """What a consumer lineage takes per tick as a rate on fully vulnerable prey, with Holling's
    saturating response: attack * hunters / (1 + attack * handling * food).

    >>> predation_pressure(0.05, 10.0, 0.25, 80.0)
    0.25
    """
    return attack * hunters / (1.0 + attack * handling * food)


def fraction_eaten(pressure: float) -> float:
    """The share of the reachable prey eaten in one tick under a summed pressure: 1 - exp(-pressure).

    Computed without the C library, so that every IEEE machine gets the same bits: the series
    x - x^2/2 + x^3/6 - ... for x <= 0.5, and f(2x) = f(x) * (2 - f(x)) above.

    >>> fraction_eaten(0.0), round(fraction_eaten(0.25), 12), round(fraction_eaten(3.0), 12)
    (0.0, 0.221199216929, 0.950212931632)
    """
    if pressure <= 0.0:
        return 0.0
    x, doublings = pressure, 0
    while x > 0.5:
        x *= 0.5
        doublings += 1
    f = 1.0
    for n in range(18, 1, -1):
        f = 1.0 - x / n * f
    f *= x
    for _ in range(doublings):
        f *= 2.0 - f
    return f


def pyramid_flow(production_: float, efficiency: float, level: int) -> float:
    """The energy pyramid: of ``production_`` at level 1, production * efficiency^(level - 1)
    can reach trophic level ``level``.

    >>> pyramid_flow(1000.0, 0.5, 3), pyramid_flow(80.0, 0.1, 1)
    (250.0, 80.0)
    """
    flow = production_
    for _ in range(int(level) - 1):
        flow *= efficiency
    return flow


def max_production(light: float) -> float:
    """The most single-celled producers can make per tick: at half the capacity, 0.6 * light / 4.

    >>> max_production(100.0)
    15.0
    """
    return production(light / 2.0, R0, light / 2.0, light)


def supported_levels(light: float, efficiency: float) -> int:
    """Toy rule: the trophic levels (1 to 4) the energy pyramid supports. Level n counts when the
    flow that can reach it, 0.15 * light * efficiency^(n - 1), is at least 0.025 per tick: the
    upkeep of an established population (0.5 biomass units) of single cells.

    >>> supported_levels(20.0, 0.05), supported_levels(100.0, 0.05), supported_levels(100.0, 0.1)
    (2, 3, 3)
    >>> supported_levels(200.0, 0.1), supported_levels(1000.0, 0.05), supported_levels(25.0, 0.2)
    (4, 3, 4)
    """
    top = max_production(light)
    n = 0
    for level in (1, 2, 3, 4):
        if pyramid_flow(top, efficiency, level) < FLOW_MIN:
            break
        n = level
    return max(1, n)


def oxygen_max_size(oxygen: float) -> int:
    """Toy rule for the diffusion limit: the largest consumer body, in cells, that the oxygen
    allows: 16384 * oxygen^1.5, rounded down to a power of two, at least 1. (A ball that gets
    its oxygen by diffusion can be as thick as the square root of the oxygen outside allows, so
    its volume goes with oxygen^1.5; the 16384 cells at the present level are invented.)

    >>> [oxygen_max_size(x) for x in (0.0, 0.001, 0.01, 0.1, 0.25, 1.0)]
    [1, 1, 16, 512, 2048, 16384]
    """
    raw = N_PAL * oxygen * math.sqrt(oxygen) if oxygen > 0 else 0.0
    whole = int(min(raw, float(MAX_SIZE)))
    return 1 if whole < 1 else 1 << (whole.bit_length() - 1)


def predator_activity(oxygen: float) -> float:
    """Toy rule: predators hunt at full strength from 0.02 of the present oxygen level, below
    that in proportion: min(1, oxygen / 0.02).

    >>> predator_activity(0.005), predator_activity(0.3)
    (0.25, 1.0)
    """
    return min(1.0, oxygen / O2_PREDATOR)


# ---------------------------------------------------------------------------------------------
# the community
# ---------------------------------------------------------------------------------------------

class _World:
    """The mutable state: parallel lists, one entry per lineage, in order of founding."""

    def __init__(self, light: float, efficiency: float, predation: float, oxygen: float,
                 complex_cells: bool) -> None:
        self.light, self.efficiency, self.predation = light, efficiency, predation
        self.oxygen = oxygen
        self.max_k = MAX_K if complex_cells else SIMPLE_MAX_K
        self.max_types = MAX_TYPES if complex_cells else SIMPLE_MAX_TYPES
        self.levels = supported_levels(light, efficiency)
        self.ids: list[int] = []
        self.parent: list[int] = []
        self.guild: list[int] = []
        self.k: list[int] = []
        self.types: list[int] = []
        self.b: list[float] = []
        # what follows from the traits, kept per lineage
        self.size: list[int] = []
        self.m: list[float] = []      # upkeep rate
        self.r: list[float] = []      # growth rate (producers)
        self.a: list[float] = []      # attack rate (consumers), before the oxygen factor
        self.h: list[float] = []      # handling time (consumers)
        self.prey: list[list[tuple[int, float]]] = []
        self._stale = False           # lineages came or went since the links were worked out
        self.next_id = 0
        self.extinctions = 0
        self.events: list[dict] = []
        self.step = 0
        self.ledger = self._empty_ledger()
        self._respired = self._lost = self._made = 0.0

    # -- lineages -----------------------------------------------------------------------------

    def add(self, guild: int, k: int, types: int, biomass: float, parent: int = -1) -> int:
        size = 1 << k
        self.ids.append(self.next_id)
        self.parent.append(parent)
        self.guild.append(guild)
        self.k.append(k)
        self.types.append(types)
        self.b.append(biomass)
        self.size.append(size)
        self.m.append(upkeep_rate(size, types))
        self.r.append(growth_rate(size, types) if guild == 0 else 0.0)
        self.a.append(attack_rate(self.predation, size, types) if guild else 0.0)
        self.h.append(handling_time(size, types) if guild else 0.0)
        self.next_id += 1
        self._stale = True
        return self.next_id - 1

    def remove(self, i: int) -> None:
        for name in ("ids", "parent", "guild", "k", "types", "b", "size", "m", "r", "a", "h"):
            del getattr(self, name)[i]
        self._stale = True

    def _links(self) -> None:
        """Who eats whom: grazers eat producers; predators eat grazers and, where the pyramid
        supports a fourth level, predators with at most an eighth of their cells. Worked out
        anew, when lineages came or went, before the next feeding and before a step is told."""
        if not self._stale:
            return
        self._stale = False
        n = len(self.b)
        self.prey = [[] for _ in range(n)]
        for j in range(n):
            g = self.guild[j]
            if g == 0:
                continue
            for i in range(n):
                gi = self.guild[i]
                if (gi == g - 1) or (g == 2 and gi == 2 and self.levels >= 4
                                     and self.size[i] * PREY_RATIO <= self.size[j]):
                    self.prey[j].append((i, vulnerability(self.size[i], self.size[j])))

    def level(self, j: int) -> int:
        """Trophic level of a lineage: 1 producer, 2 grazer, 3 predator, 4 predator of predators
        (however long the chain of predators below it)."""
        self._links()
        g = self.guild[j]
        if g == 2 and any(self.guild[i] == 2 for i, _ in self.prey[j]):
            return 4
        return g + 1

    def last_producer(self) -> int:
        """The index of the only producer lineage, or -1 when there are several (or none)."""
        found = -1
        for i, g in enumerate(self.guild):
            if g == 0:
                if found >= 0:
                    return -1
                found = i
        return found

    # -- one tick of the ecology --------------------------------------------------------------

    @staticmethod
    def _empty_ledger() -> dict[str, float]:
        out = {f"{term}_{g}": 0.0 for term in LEDGER_TERMS for g in GUILD_KEYS}
        out.update(o2_produced=0.0, o2_consumed=0.0, max_take=0.0)
        return out

    def ecology(self) -> None:
        """Upkeep, feeding, growth, in this order, each acting on what the last one left."""
        self._links()
        b, guild, led = self.b, self.guild, self.ledger
        n = len(b)
        respired = lost = 0.0
        # 1. upkeep at Kleiber's pace; a share of it is shed as dead matter
        for i in range(n):
            u = self.m[i] * b[i]
            b[i] -= u
            shed = SHED * u
            key = GUILD_KEYS[guild[i]]
            led["respired_" + key] += u - shed
            led["lost_" + key] += shed
            respired += u - shed
            lost += shed
        # 2. feeding: Holling's saturating response, summed over the consumers of each prey
        totals = [0.0, 0.0, 0.0]
        for i in range(n):
            totals[guild[i]] += b[i]
        share = [reachable(t) for t in totals]
        within = [b[i] * share[guild[i]] for i in range(n)]
        pressure = [0.0] * n
        q = [0.0] * n
        activity = predator_activity(self.oxygen)
        for j in range(n):
            links = self.prey[j]
            if not links:
                continue
            food = 0.0
            for i, v in links:
                food += v * within[i]
            attack = self.a[j] * (activity if guild[j] == 2 else 1.0)
            q[j] = predation_pressure(attack, b[j], self.h[j], food)
            for i, v in links:
                pressure[i] += q[j] * v
        eaten = [fraction_eaten(pressure[i]) * within[i] if pressure[i] > 0.0 else 0.0 for i in range(n)]
        gain = [0.0] * n
        for j in range(n):
            if q[j] <= 0.0:
                continue
            got = 0.0
            for i, v in self.prey[j]:
                if eaten[i] > 0.0:
                    got += eaten[i] * (q[j] * v / pressure[i])
            gain[j] = got
        for i in range(n):
            if eaten[i] > 0.0:
                led["taken_" + GUILD_KEYS[guild[i]]] += eaten[i]
                if eaten[i] > led["max_take"] * b[i]:
                    led["max_take"] = eaten[i] / b[i]
                b[i] -= eaten[i]
        for j in range(n):
            if gain[j] > 0.0:
                kept = self.efficiency * gain[j]
                key = GUILD_KEYS[guild[j]]
                led["income_" + key] += gain[j]
                led["lost_" + key] += gain[j] - kept
                lost += gain[j] - kept
                b[j] += kept
        # 3. growth of the producers on the light they share
        total = 0.0
        for i in range(n):
            if guild[i] == 0:
                total += b[i]
        made = 0.0
        for i in range(n):
            if guild[i] == 0:
                grown = production(b[i], self.r[i], total, self.light)
                b[i] += grown
                made += grown
        led["income_producers"] += made
        self._respired, self._lost, self._made = respired, lost, made

    def deaths(self) -> None:
        """Consumers too big for the oxygen suffocate; lineages below the threshold go extinct."""
        b, led = self.b, self.ledger
        limit = oxygen_max_size(self.oxygen)
        for i in range(len(b)):
            if self.guild[i] and self.size[i] > limit:
                dead = SUFFOCATE * b[i]
                b[i] -= dead
                led["lost_" + GUILD_KEYS[self.guild[i]]] += dead
                self._lost += dead
        i = 0
        while i < len(b):
            if b[i] < EXTINCT:
                led["lost_" + GUILD_KEYS[self.guild[i]]] += b[i]
                self._lost += b[i]
                self.events.append({"step": self.step, "kind": "extinct", "id": self.ids[i]})
                self.extinctions += 1
                self.remove(i)
            else:
                i += 1

    def breathe(self) -> None:
        """Oxygen: set free by production; used up by respiration, by the decomposers for the
        dead matter that is not buried, and by rocks; never more than there is."""
        made = O2_YIELD * self._made
        wanted = O2_YIELD * (self._respired + (1.0 - BURIAL) * self._lost) + O2_SINK * self.oxygen
        used = min(wanted, self.oxygen + made)
        self.ledger["o2_produced"] += made
        self.ledger["o2_consumed"] += used
        self.oxygen = max(0.0, self.oxygen + made - used)

    # -- evolution ----------------------------------------------------------------------------

    def mutant(self, i: int, kind: str) -> tuple[int, int, int] | None:
        """The traits (guild, doublings, cell types) of a mutant of lineage ``i``; None when the
        mutant is not viable."""
        guild, k, types = self.guild[i], self.k[i], self.types[i]
        if kind == "size_up":
            k += 1
        elif kind == "size_down":
            k -= 1
        elif kind == "type_gain":
            types += 1
        elif kind == "type_loss":
            types -= 1
        else:
            guild += 1
        if not (0 <= k <= self.max_k and guild <= 2 and types >= 1):
            return None
        size = 1 << k
        if kind == "size_down":
            types = min(types, max_cell_types(size))  # a smaller body cannot keep all its cell types
        if types > min(self.max_types, max_cell_types(size)) or types < needed_cell_types(size):
            return None
        if guild and size > oxygen_max_size(self.oxygen):
            return None
        if kind == "guild":
            if guild + 1 > self.levels or self.predation <= 0.0:
                return None
            if guild == 2 and self.oxygen < O2_PREDATOR:
                return None
        return guild, k, types

    def mutate(self, rng: random.Random, chance: float) -> None:
        """Every established lineage founds a mutant with probability ``chance``: two random
        numbers per established lineage, in order of founding."""
        b = self.b
        for parent_id in [self.ids[i] for i in range(len(b)) if b[i] >= ESTABLISHED]:
            u, what = rng.random(), rng.random()
            if u >= chance or parent_id not in self.ids:  # pushed out earlier in this tick
                continue
            parent = self.ids.index(parent_id)
            kind = MUTATIONS[int(what * len(MUTATIONS))]
            traits = self.mutant(parent, kind)
            if traits is None:
                continue
            guild, k, types = traits
            founder = FOUND * b[parent]
            twin = next((i for i in range(len(b)) if (self.guild[i], self.k[i], self.types[i]) == traits), None)
            if twin is None and len(b) >= MAX_LINEAGES:
                # the vessel is full: the founder pushes out the rarest lineage, if that is smaller;
                # never the last lineage of the producers, on which everything else lives
                keep = self.last_producer()
                rarest = min((i for i in range(len(b)) if i != keep), key=lambda i: (b[i], self.ids[i]))
                if b[rarest] >= founder:
                    continue
                self.ledger["lost_" + GUILD_KEYS[self.guild[rarest]]] += b[rarest]
                self._lost += b[rarest]
                self.events.append({"step": self.step, "kind": "displaced", "id": self.ids[rarest]})
                self.extinctions += 1
                self.remove(rarest)
                parent = self.ids.index(parent_id)
            b[parent] -= founder
            if guild != self.guild[parent]:
                self.ledger["moved_" + GUILD_KEYS[self.guild[parent]]] += founder
                self.ledger["moved_" + GUILD_KEYS[guild]] -= founder
            if twin is not None:
                b[twin] += founder
                continue
            new = self.add(guild, k, types, founder, parent_id)
            self.events.append({"step": self.step, "kind": kind, "id": new, "parent": parent_id,
                                "guild": GUILDS[guild], "size": 1 << k, "cell_types": types,
                                "oxygen": self.oxygen})

    def tick(self, rng: random.Random, chance: float) -> None:
        self.ecology()
        self.deaths()
        self.mutate(rng, chance)
        self.breathe()

    # -- what a step shows ----------------------------------------------------------------------

    def snapshot(self) -> tuple[dict, list[dict]]:
        lineages = [{"id": self.ids[i], "parent": self.parent[i], "guild": GUILDS[self.guild[i]],
                     "size": self.size[i], "adhesion": adhesion(self.size[i]), "cell_types": self.types[i],
                     "level": self.level(i), "biomass": self.b[i]} for i in range(len(self.b))]
        state = describe(lineages, self.light, self.oxygen, self.extinctions)
        state.update(self.ledger)
        self.ledger = self._empty_ledger()
        return state, lineages


def describe(lineages: list[dict], light: float, oxygen: float, extinctions: int) -> dict[str, float | int | str]:
    """The metrics of a step from its lineages (the simulation and its gate both use this).

    ``max_size`` and ``cell_types`` are taken over the common lineages: established ones (0.5
    biomass units) that hold at least a tenth of their guild's biomass; the largest producer
    lineage always counts. A trophic level above the first counts when its lineages together are
    established."""
    totals = {g: 0.0 for g in GUILDS}
    top = mass = bodies = largest = 0.0
    for ln in lineages:
        totals[ln["guild"]] += ln["biomass"]
        mass += ln["biomass"]
        bodies += ln["biomass"] / ln["size"]
        if ln["level"] == 4:
            top += ln["biomass"]
        if ln["guild"] == "producer":
            largest = max(largest, ln["biomass"])
    common = [ln for ln in lineages
              if (ln["biomass"] >= ESTABLISHED and ln["biomass"] >= COMMON * totals[ln["guild"]])
              or (ln["guild"] == "producer" and ln["biomass"] == largest)]
    max_size = max((ln["size"] for ln in common), default=0)
    cell_types = max((ln["cell_types"] for ln in common), default=0)
    eaters = [ln for ln in common if ln["guild"] != "producer"]
    levels = (4 if top >= ESTABLISHED else 3 if totals["predator"] >= ESTABLISHED
              else 2 if totals["grazer"] >= ESTABLISHED else 1 if totals["producer"] > 0 else 0)
    return {
        "species": len(lineages),
        "producers": sig(totals["producer"]), "grazers": sig(totals["grazer"]),
        "predators": sig(totals["predator"]),
        "max_size": max_size, "mean_size": sig(mass / bodies) if bodies > 0 else 0.0,
        "cell_types": cell_types,
        "consumer_size": max((ln["size"] for ln in eaters), default=0),
        "consumer_types": max((ln["cell_types"] for ln in eaters), default=0),
        "trophic_levels": levels, "oxygen": sig(oxygen), "extinctions": extinctions,
        "stage": stage_of(totals["producer"], light, max_size, cell_types, levels),
        "raw_producers": totals["producer"], "raw_grazers": totals["grazer"],
        "raw_predators": totals["predator"], "raw_oxygen": oxygen,
    }


def stage_of(producers: float, light: float, max_size: int, cell_types: int, levels: int) -> str:
    """The stage follows from the numbers of a step: ``collapse`` while the producers are grazed
    down below a hundredth of the capacity; else by the common bodies: ``single_cells``,
    ``colonies`` (2 to 32 cells), ``multicellular`` (more cells: germ and soma), ``differentiated``
    (multicellular with at least three cell types), ``food_web`` (differentiated bodies in a world
    with established predators).

    >>> stage_of(50.0, 100.0, 16, 4, 3), stage_of(50.0, 100.0, 64, 2, 3), stage_of(50.0, 100.0, 64, 3, 2)
    ('colonies', 'multicellular', 'differentiated')
    >>> stage_of(50.0, 100.0, 64, 3, 3), stage_of(0.9, 100.0, 64, 3, 3), stage_of(50.0, 100.0, 1, 1, 1)
    ('food_web', 'collapse', 'single_cells')
    """
    if producers < CRASH * light:
        return "collapse"
    if max_size <= 1:
        return "single_cells"
    if max_size <= GERM_SIZE:
        return "colonies"
    if cell_types < DIFFERENTIATED:
        return "multicellular"
    return "food_web" if levels >= 3 else "differentiated"


def outcome_of(max_size: int, cell_types: int, levels: int, consumer_size: int | None = None,
               consumer_types: int | None = None) -> str:
    """What the bodies at the end are, whatever the producers' biomass. ``single_cells``,
    ``colonies`` and ``multicellular`` go by the largest common body, producer or consumer.
    ``animals_like`` asks more: common *consumers* of more than 32 cells and with three or more
    cell types, in a world with established predators; a world whose only differentiated bodies
    are producers stays ``multicellular``. Without the last two arguments the consumers are
    taken to be as large and as differentiated as the largest common body. A world without
    producers (``max_size`` 0) has the outcome ``collapse``; see the module docstring.

    >>> outcome_of(1, 1, 2, 1, 1), outcome_of(32, 5, 3, 32, 5), outcome_of(64, 3, 2, 64, 3), outcome_of(64, 3, 3, 64, 3)
    ('single_cells', 'colonies', 'multicellular', 'animals_like')
    >>> outcome_of(128, 5, 3, 16, 3), outcome_of(128, 5, 3, 64, 2), outcome_of(128, 5, 3), outcome_of(0, 0, 0)
    ('multicellular', 'multicellular', 'animals_like', 'collapse')
    """
    if consumer_size is None or consumer_types is None:
        consumer_size, consumer_types = max_size, cell_types
    if max_size < 1:
        return "collapse"
    if max_size == 1:
        return "single_cells"
    if max_size <= GERM_SIZE:
        return "colonies"
    animals = consumer_size > GERM_SIZE and consumer_types >= DIFFERENTIATED and levels >= 3
    return "animals_like" if animals else "multicellular"


def pair_step(producers: float, grazers: float, light: float, efficiency: float,
              predation: float) -> tuple[float, float]:
    """One tick of the ecology for single-celled producers and single-celled grazers alone:
    the simulation's own :meth:`_World.ecology` on a community of two lineages."""
    world = _World(light, efficiency, predation, 1.0, True)
    world.add(0, 0, 1, producers)
    world.add(1, 0, 1, grazers)
    world.ecology()
    return world.b[0], world.b[1]


def prey_step(producers: float, grazers: float, light: float, efficiency: float, predation: float) -> float:
    """The producers after one tick with grazers (see :func:`pair_step`)."""
    return pair_step(producers, grazers, light, efficiency, predation)[0]


def grazer_step(producers: float, grazers: float, light: float, efficiency: float, predation: float) -> float:
    """The grazers after one tick with producers (see :func:`pair_step`)."""
    return pair_step(producers, grazers, light, efficiency, predation)[1]


# ---------------------------------------------------------------------------------------------
# lessons: every rule is one of the functions above
# ---------------------------------------------------------------------------------------------

def _fits(cells: int, cell_types: int) -> bool:
    """A body the simulation could hold: as many cell types as its size needs and allows."""
    return needed_cell_types(cells) <= cell_types <= max_cell_types(cells)


_CELLS = Input("cells", 1, MAX_SIZE, True)
_TYPES = Input("cell_types", 1, MAX_TYPES, True)
_PAIR = (Input("producers", 1, 1000, places=1), Input("grazers", 0.1, 200, places=1),
         Input("light", 20, 1000, True), Input("efficiency", 0.05, 0.2, places=2),
         Input("predation", 0, 2, places=2))
_PAIR_TEXT = ("single cells. both lose 0 point 0 5 as upkeep. grazers reach all but 0 point 5 of the "
              "producers. pressure is half of attack times grazers over 1 plus attack times 0 point 1 2 5 "
              "times reach. attack is 0 point 0 5 times predation. eaten is reach times 1 minus exp of "
              "minus pressure. ")

RULES: dict[str, Rule] = {
    "kleiber": Rule(
        (Input("mass", 1, 100000, True),), kleiber_rate,
        "metabolic rate of a body of mass cells in units of one free cell. kleiber law. "
        "mass to the power 0 point 7 5"),
    "energy_per_cell": Rule(
        (_CELLS,), energy_per_cell,
        "what one cell of a body turns over. a free cell is 1. kleiber rate over cells. "
        "cells to the power minus 0 point 2 5"),
    "surface_to_volume": Rule(
        (_CELLS,), surface_to_volume,
        "surface over volume of a ball of cells. each cell a ball of radius 1. 3 over the cube root of cells"),
    "adhesion": Rule(
        (Input("cells", 1, 1024, True),), adhesion,
        "toy rule. share of daughter cells that stay attached in a body of cells. 1 minus 1 over cells"),
    "max_cell_types": Rule(
        (_CELLS,), max_cell_types,
        "toy rule. most cell types a body of cells can have. 1 plus 1 point 5 times log 2 of cells "
        "rounded down. never more than 3 0"),
    "supply": Rule(
        (_CELLS, _TYPES), supply,
        "toy rule. how well the cells of a body are supplied. a free cell is 1. cell_types over the "
        "cube root of cells. never more than 1", _fits),
    "upkeep": Rule(
        (_CELLS, _TYPES), upkeep_rate,
        "toy numbers. share of its biomass a body burns or sheds per tick. 0 point 0 5 times cells to "
        "the power minus 0 point 2 5. times 1 plus 0 point 0 3 for each cell type beyond the first", _fits),
    "growth_rate": Rule(
        (_CELLS, _TYPES), growth_rate,
        "toy numbers. growth of a producer per tick far below the capacity. 0 point 6 times cells to "
        "the power minus 0 point 2 5 times supply", _fits),
    "needed_cell_types": Rule(
        (_CELLS,), needed_cell_types,
        "toy rule. cell types a body of cells needs to reproduce. 1 up to 3 2 cells. above that 2. "
        "germ and soma"),
    "attack_rate": Rule(
        (Input("predation", 0.2, 2, places=2), _CELLS, _TYPES), attack_rate,
        "toy numbers. attack rate of a consumer per biomass and tick. 0 point 0 5 times predation "
        "times cells to the power minus 0 point 2 5 times supply", lambda predation, cells, cell_types: _fits(cells, cell_types)),
    "handling_time": Rule(
        (_CELLS, _TYPES), handling_time,
        "toy numbers. ticks a consumer needs to eat its own biomass. 0 point 2 5 over supply times "
        "cells to the power 0 point 2 5", _fits),
    "reachable": Rule(
        (Input("total", 0.1, 100, places=2),), reachable,
        "toy rule. share of the biomass of a guild its consumers can reach. all but 0 point 5 units "
        "hide. 1 minus 0 point 5 over total. never less than 0"),
    "oxygen_max_size": Rule(
        (Input("oxygen", 0.001, 1.2, places=4),), oxygen_max_size,
        "toy rule. cells of the largest consumer body the oxygen allows. 1 6 3 8 4 times oxygen "
        "to the power 1 point 5. rounded down to a power of 2. at least 1. oxygen as a share of the "
        "present level"),
    "predator_activity": Rule(
        (Input("oxygen", 0, 0.05, places=4),), predator_activity,
        "toy rule. share of their full strength at which predators hunt. oxygen over 0 point 0 2. "
        "never more than 1"),
    "production": Rule(
        (Input("biomass", 0.1, 1200, places=1), Input("rate", 0.01, 0.6, places=2),
         Input("total", 0.1, 1200, places=1), Input("capacity", 20, 1000, True)), production,
        "logistic growth. what a producer lineage makes in one tick. rate times biomass times 1 minus "
        "total over capacity. never less than 0. total is the biomass of all producers",
        lambda biomass, rate, total, capacity: total >= biomass),
    "logistic_step": Rule(
        (Input("biomass", 0.1, 1200, places=1), Input("rate", 0.01, 0.6, places=2),
         Input("capacity", 20, 1000, True)), logistic_step,
        "one producer lineage alone after one tick of growth. biomass plus rate times biomass times "
        "1 minus biomass over capacity. no growth above the capacity"),
    "vulnerability": Rule(
        (Input("prey", 1, MAX_SIZE, True), Input("hunter", 1, MAX_SIZE, True)), vulnerability,
        "toy rule. size selective feeding. share of encounters that end with the prey eaten. hunter "
        "cells over hunter cells plus prey cells"),
    "predation_pressure": Rule(
        (Input("attack", 0.001, 0.1, places=3), Input("hunters", 0.1, 500, places=1),
         Input("handling", 0.25, 8, places=2), Input("food", 0, 1000, places=1)), predation_pressure,
        "saturating response. rate at which a consumer lineage eats fully vulnerable prey per tick. "
        "attack times hunters over 1 plus attack times handling times food"),
    "fraction_eaten": Rule(
        (Input("pressure", 0, 5, places=3),), fraction_eaten,
        "share of the reachable prey eaten in one tick under a summed pressure. 1 minus exp of minus pressure"),
    "pyramid_flow": Rule(
        (Input("production", 1, 1000, places=1), Input("efficiency", 0.05, 0.2, places=2),
         Input("level", 1, 4, True)), pyramid_flow,
        "energy pyramid. what can reach a trophic level per tick. production times efficiency to the "
        "power level minus 1"),
    "supported_levels": Rule(
        (Input("light", 1, 1000, True), Input("efficiency", 0.05, 0.2, places=2)), supported_levels,
        "toy rule. trophic levels the energy pyramid supports. a level counts when 0 point 1 5 times "
        "light times efficiency to the power level minus 1 is at least 0 point 0 2 5. at most 4"),
    "prey_step": Rule(
        _PAIR, prey_step,
        "toy numbers. producers after one tick with grazers. " + _PAIR_TEXT
        + "the rest grows by 0 point 6 times rest times 1 minus rest over light. no growth above light"),
    "grazer_step": Rule(
        _PAIR, grazer_step,
        "toy numbers. grazers after one tick with producers. " + _PAIR_TEXT
        + "grazers gain efficiency times eaten"),
}

#: inputs that count cells: drawn evenly over the doublings, not over the numbers
SIZE_INPUTS = frozenset({"cells", "mass", "prey", "hunter"})
#: the largest body drawn for ``needed_cell_types``: the rule turns at 32 cells, so bodies up to
#: 1024 cells give its two answers about equally often (the gate judges every size)
NEEDED_TOP = 1 << 10
#: the oxygen drawn for ``predator_activity`` (the gate judges up to 0.05): a third at full strength
ACTIVITY_TOP = 0.03
#: a lesson shows producers above the capacity (no growth) now and then: up to this share of it
OVER = 1.15


class BodyLessons(LessonGate):
    """The lesson gate of this level. Every judgement is the base class's; only the drawing of
    the inputs differs (:meth:`draw`), because drawn evenly from their ranges the inputs show
    states the rollouts never visit, and several rules then give one answer most of the time:

    * a body size is drawn the way the simulation has them: a doubling first (2^0 to the top of
      the range, each as likely) and then, half the time, the power of two itself, else a whole
      number up to the next doubling. Drawn evenly from 1 to 1048576, hardly a lesson would
      show a body as small as those of the rollouts. For ``needed_cell_types`` the doublings
      stop at 1024 cells;
    * the number of cell types is drawn from what the drawn body needs to what it allows;
    * oxygen is drawn evenly over the orders of magnitude; for ``predator_activity`` evenly
      from 0 to 0.03; the biomass of ``reachable`` evenly over the orders of magnitude (a guild
      of less than 0.5 units is all hidden);
    * producers are drawn as a share of their capacity, up to 1.15 of it: ``production`` draws
      the capacity, then all producers, then the lineage as a share of them; ``logistic_step``
      the capacity, then the lineage; ``prey_step`` and ``grazer_step`` the light, then the
      producers. In a rollout the producers never exceed the light;
    * the light of ``supported_levels`` is drawn evenly over the orders of magnitude from 1 to
      1000, so that all four answers occur.

    Everything else is drawn evenly from its range. Over 8000 lessons (seeds 1 to 4, 2000 each)
    the most common answer of a rule then has these shares: ``needed_cell_types`` (two
    answers) 2 in 0.52, ``supply`` 1 in 0.38, ``predator_activity`` 1 in 0.34,
    ``supported_levels`` 4 in 0.32, ``reachable`` 0 in 0.28, ``oxygen_max_size`` 1 in 0.16,
    ``production`` 0 in 0.11, every other rule 0.1 or less."""

    def size(self, rng: random.Random, high: int) -> int:
        """A body size up to ``high`` cells, drawn over the doublings."""
        k = rng.randint(0, int(high).bit_length() - 1)
        if rng.random() < 0.5:
            return 1 << k
        return rng.randint(1 << k, min(int(high), (1 << (k + 1)) - 1))

    def spread(self, rng: random.Random, spec: Input) -> float | int:
        """One input drawn evenly from its range."""
        if spec.integer:
            return rng.randint(int(spec.low), int(spec.high))
        return round(rng.uniform(spec.low, spec.high), spec.places)

    def draw(self, rng: random.Random, name: str) -> list[float | int]:
        """The inputs of one lesson of the rule ``name``, in the rule's order."""
        specs = self.rules[name].inputs
        if name in ("production", "logistic_step"):
            capacity = rng.randint(20, 1000)
            biomass = capacity * rng.uniform(0.01, OVER)
            rate = round(rng.uniform(0.01, 0.6), 2)
            if name == "logistic_step":
                return [round(biomass, 1), rate, capacity]
            return [round(biomass * rng.uniform(0.02, 1.0), 1), rate, round(biomass, 1), capacity]
        if name in ("prey_step", "grazer_step"):
            light = rng.randint(20, 1000)
            producers = round(max(1.0, min(1000.0, light * rng.uniform(0.01, OVER))), 1)
            return [producers, round(rng.uniform(0.1, 200.0), 1), light, round(rng.uniform(0.05, 0.2), 2),
                    round(rng.uniform(0.0, 2.0), 2)]
        if name == "supported_levels":
            return [max(1, round(_exp(rng.uniform(0.0, LN_1000)))), round(rng.uniform(0.05, 0.2), 2)]
        if name == "predator_activity":
            return [round(rng.uniform(0.0, ACTIVITY_TOP), 4)]
        if name == "reachable":  # 0.1 to 100, evenly over the orders of magnitude
            return [round(0.1 * _exp(rng.uniform(0.0, LN_1000)), 2)]
        drawn: dict[str, float | int] = {}
        for spec in specs:
            if spec.name in SIZE_INPUTS:
                value = self.size(rng, NEEDED_TOP if name == "needed_cell_types" else int(spec.high))
            elif spec.name == "cell_types" and "cells" in drawn:
                cells = int(drawn["cells"])
                value = rng.randint(needed_cell_types(cells), max_cell_types(cells))
            elif spec.name == "oxygen":  # 0.001 to 1.2, evenly over the orders of magnitude
                value = round(spec.low * _exp(rng.uniform(0.0, LN_1200)), spec.places)
            else:
                value = self.spread(rng, spec)
            drawn[spec.name] = value
        return list(drawn.values())

    def generate(self, rng: random.Random, n: int) -> list[Line]:
        """Exactly ``n`` lessons, rules drawn uniformly."""
        names = sorted(self.rules)
        out: list[Line] = []
        tries = 0
        while len(out) < n:
            tries += 1
            if tries > 50 * n + 1000:
                raise RuntimeError(f"{self.sim}: the lesson rules reject almost every drawn input")
            name = rng.choice(names)
            ln = self.line(name, self.draw(rng, name))
            if ln is not None:
                out.append(ln)
        return out


LESSONS = BodyLessons(SIM, RULES)


def lesson_gate() -> LessonGate:
    return LESSONS


# ---------------------------------------------------------------------------------------------
# parameters, rollout, lines
# ---------------------------------------------------------------------------------------------

def dense_value(value: float | int | str) -> str:
    """A metric as it stands in a line: a word, an exact count, or a number to 3 significant digits.

    >>> dense_value(1052.53), dense_value(0.03234), dense_value(7), dense_value("colonies")
    ('1 0 5 0', '0 point 0 3 2 3', '7', 'colonies')
    """
    if isinstance(value, str):
        return value
    if isinstance(value, int):
        return num(value)
    return num(sig(float(value)))


def param_value(key: str, value: float | int | str) -> str:
    """A parameter as it stands in the ``params`` record: as given.

    >>> param_value("light", 250.0), param_value("efficiency", 0.07), param_value("complex_cells", "yes")
    ('2 5 0', '0 point 0 7', 'yes')
    """
    return value if isinstance(value, str) else num(float(value), sig=5)


@dataclass
class BodiesRollout(Rollout):
    """A :class:`Rollout` that also keeps every lineage at every step and the log of events."""
    #: per step, per living lineage: ``{"id", "parent", "guild", "size", "adhesion", "cell_types",
    #: "level", "biomass"}``, in order of founding
    lineages: list[list[dict]] = field(default_factory=list)
    #: ``{"step", "kind", "id"}``: kind is ``extinct``, ``displaced`` or the mutation that founded a new lineage
    #: (:data:`MUTATIONS`), then also ``"parent", "guild", "size", "cell_types"`` and the
    #: ``"oxygen"`` at that tick. A mutant that joined a living lineage is not logged.
    events: list[dict] = field(default_factory=list)

    def value(self, text: float | int | str) -> str:
        return dense_value(text)


def _rules(rules: int) -> None:
    if rules != 7:
        raise ValueError(f"bodies is a round-7 level: rules must be 7, not {rules!r}")


def random_params(rng: random.Random, rules: int = 7) -> dict[str, float | str]:
    """Light 20-1000 (log-uniform, whole), efficiency 0.05-0.2, predation 0 in 15 % of the worlds
    and else 0.2-2, mutation 0.2-3, starting oxygen 0.001-0.2 (log-uniform, 2 digits), complex
    cells in 70 % of the worlds. Six random numbers, always in this order.

    >>> random_params(random.Random(3))
    {'light': 51.0, 'efficiency': 0.13, 'predation': 1.29, 'mutation': 1.95, 'start_oxygen': 0.0014, 'complex_cells': 'yes'}
    """
    _rules(rules)
    light = float(round(_exp(rng.uniform(LN_20, LN_1000))))
    efficiency = round(rng.uniform(0.05, 0.2), 2)
    none, strength = rng.random() < 0.15, round(rng.uniform(0.2, 2.0), 2)
    mutation = round(rng.uniform(0.2, 3.0), 2)
    oxygen = float(f"{_exp(rng.uniform(LN_0001, LN_02)):.2g}")
    return {"light": light, "efficiency": efficiency, "predation": 0.0 if none else strength,
            "mutation": mutation, "start_oxygen": oxygen,
            "complex_cells": "yes" if rng.random() < 0.7 else "no"}


def handoff_in(below: dict) -> dict[str, float | str]:
    """Parameters of this level from the summary of level 8 (``haishool.evo.cells``).

    * ``complex_cells`` is ``yes`` when the outcome below is ``complex_cells`` (or, without an
      outcome, when at least half the cells have a partner: ``complex >= 0.5``), else ``no``.
    * ``efficiency``: cells with more energy turn more of their food into growth (invented for
      the chain, nothing measured): 0.05 at ``energy_per_cell`` 10 (level 8's cell without a
      working partner, ``cells.ENERGY``), 0.2 at 160 (every cell with a working partner,
      ``cells.ENDO_GAIN`` = 16 times as much), in between by the logarithm,
      ``0.05 + 0.15 * log(energy / 10) / log(16)``, clipped to 0.05 .. 0.2, two decimals
      (a value on the edge between two hundredths goes up: 0.13 at 40).

    A world whose level 8 ended in ``collapse`` has no cells to build bodies from: ValueError.
    Light, predation, mutation and the starting oxygen are not the cells' to give.

    >>> handoff_in({"outcome": "complex_cells", "complex": 0.93, "energy_per_cell": 152.0})
    {'complex_cells': 'yes', 'efficiency': 0.2}
    >>> handoff_in({"outcome": "cells", "complex": 0.0, "energy_per_cell": 10.0})
    {'complex_cells': 'no', 'efficiency': 0.05}
    >>> handoff_in({"outcome": "complex_cells", "complex": 0.6, "energy_per_cell": 81.9})
    {'complex_cells': 'yes', 'efficiency': 0.16}
    """
    outcome = below.get("outcome")
    if outcome == "collapse" or below.get("cells", 1) == 0:
        raise ValueError(f"no cells below, so no bodies: {below!r}")
    partner = outcome == "complex_cells" if outcome is not None else float(below.get("complex", 0.0)) >= 0.5
    energy = float(below.get("energy_per_cell", CELL_ENERGY))
    if not (math.isfinite(energy) and energy > 0):
        raise ValueError(f"energy_per_cell must be positive: {energy!r}")
    share = round(min(1.0, max(0.0, math.log(energy / CELL_ENERGY) / LN_16)), 9)
    # share and sum are rounded with a margin, so that an efficiency on the edge between two
    # hundredths (0.125 at four times the energy) is the same on every C library
    return {"complex_cells": "yes" if partner else "no", "efficiency": round(0.05 + 0.15 * share + 1e-9, 2)}


def handoff_out(summary: dict) -> dict[str, float | int | str]:
    """What the summary gives upward to level 10 (senses), under the names of the task: the body
    size and cell types of the common consumers (the bodies that could carry senses; those of all
    common bodies when there are no consumers), whether predators are established and their
    share of the biomass, the oxygen and the number of species."""
    eaters = summary["consumer_size"] > 0
    return {"body_size": summary["consumer_size"] if eaters else summary["max_size"],
            "cell_types": summary["consumer_types"] if eaters else summary["cell_types"],
            "predators": "yes" if summary["trophic_levels"] >= 3 else "no",
            "predator_share": summary["predator_share"], "oxygen": summary["oxygen"],
            "species": summary["species"]}


def summarise(steps: list[dict]) -> dict[str, float | int | str]:
    last = steps[-1]
    total = last["raw_producers"] + last["raw_grazers"] + last["raw_predators"]
    return {"stage": last["stage"], "species": last["species"], "max_size": last["max_size"],
            "cell_types": last["cell_types"], "consumer_size": last["consumer_size"],
            "consumer_types": last["consumer_types"], "trophic_levels": last["trophic_levels"],
            "oxygen": last["oxygen"],
            "predator_share": sig(last["raw_predators"] / total) if total > 0 else 0.0,
            "outcome": outcome_of(last["max_size"], last["cell_types"], last["trophic_levels"],
                                  last["consumer_size"], last["consumer_types"])}


class Bodies:
    """The simulation and its gate. Rollouts replayed by :meth:`check` are cached per seed."""

    sim = SIM
    topic = SIM
    KEYS = KEYS

    def __init__(self) -> None:
        self._cache: dict[int, BodiesRollout] = {}

    def run(self, seed: int, light: float = 200.0, efficiency: float = 0.1, predation: float = 1.0,
            mutation: float = 0.5, start_oxygen: float = 0.01, complex_cells: str | bool = "yes",
            rules: int = 7) -> BodiesRollout:
        """Deterministic for a given seed and parameters (``random.Random(seed)``).

        Raises ``ValueError`` for light below 1, an efficiency outside 0 .. 0.5, a negative
        predation or oxygen, a mutation rate outside 0 .. TICKS, or ``complex_cells`` that is
        neither ``yes`` nor ``no``."""
        _rules(rules)
        if isinstance(complex_cells, bool):
            complex_cells = "yes" if complex_cells else "no"
        light, efficiency, predation = float(light), float(efficiency), float(predation)
        mutation, start_oxygen = float(mutation), float(start_oxygen)
        if not all(math.isfinite(x) for x in (light, efficiency, predation, mutation, start_oxygen)):
            raise ValueError("the parameters must be finite numbers")
        if light < 1 or not 0 < efficiency <= 0.5 or predation < 0 or start_oxygen < 0:
            raise ValueError(f"light >= 1, 0 < efficiency <= 0.5, predation >= 0, start_oxygen >= 0: "
                             f"{light!r}, {efficiency!r}, {predation!r}, {start_oxygen!r}")
        if not 0 <= mutation <= TICKS or complex_cells not in ("yes", "no"):
            raise ValueError(f"0 <= mutation <= {TICKS}, complex_cells yes or no: {mutation!r}, {complex_cells!r}")
        rng = random.Random(int(seed))
        world = _World(light, efficiency, predation, start_oxygen, complex_cells == "yes")
        world.add(0, 0, 1, START_SHARE * light)
        params = {"light": light, "efficiency": efficiency, "predation": predation, "mutation": mutation,
                  "start_oxygen": start_oxygen, "complex_cells": complex_cells}
        r = BodiesRollout(SIM, int(seed), params, [])
        state, lineages = world.snapshot()
        r.steps.append(state)
        r.lineages.append(lineages)
        chance = mutation / TICKS
        for t in range(1, STEPS):
            world.step = t
            for _ in range(TICKS):
                world.tick(rng, chance)
            state, lineages = world.snapshot()
            r.steps.append(state)
            r.lineages.append(lineages)
        r.events = world.events
        r.summary = summarise(r.steps)
        return r

    def rollout(self, seed: int) -> BodiesRollout:
        """The canonical run of a seed: parameters from ``random_params(random.Random(seed))``."""
        if seed not in self._cache:
            if len(self._cache) >= 512:
                self._cache.clear()
            self._cache[seed] = self.run(seed, **random_params(random.Random(seed)))
        return self._cache[seed]

    # -- the gate -------------------------------------------------------------------------------

    def conserved(self, r: Rollout) -> Verdict:
        p = r.params
        light, efficiency = float(p["light"]), float(p["efficiency"])
        tol = 1e-9 * max(1.0, light)
        lineages = getattr(r, "lineages", None) or []
        events = getattr(r, "events", [])
        if len(lineages) != len(r.steps):
            return Verdict(False, None, "the lineages of the steps are missing")
        supported = supported_levels(light, efficiency)
        simple = p["complex_cells"] != "yes"
        passed = {g: 0.0 for g in GUILD_KEYS}
        kept = {g: 0.0 for g in GUILD_KEYS}
        prev = None
        for t, s in enumerate(r.steps):
            # energy: for every guild, income = respired + growth + taken + lost + moved
            growth_all = 0.0
            for g in GUILD_KEYS:
                growth = s["raw_" + g] - (prev["raw_" + g] if prev else s["raw_" + g])
                growth_all += growth
                spent = s["respired_" + g] + growth + s["taken_" + g] + s["lost_" + g] + s["moved_" + g]
                if abs(s["income_" + g] - spent) > tol:
                    return Verdict(False, num(spent, sig=6), f"step {t}: the energy of the {g} does not add up")
                if min(s["raw_" + g], s["income_" + g], s["respired_" + g], s["taken_" + g], s["lost_" + g]) < 0:
                    return Verdict(False, None, f"step {t}: a negative biomass or flow of the {g}")
            if abs(s["taken_producers"] - s["income_grazers"]) > tol:
                return Verdict(False, None, f"step {t}: the grazers did not eat what the producers lost")
            if abs(s["taken_grazers"] + s["taken_predators"] - s["income_predators"]) > tol:
                return Verdict(False, None, f"step {t}: the predators did not eat what their prey lost")
            if abs(math.fsum(s["moved_" + g] for g in GUILD_KEYS)) > tol:
                return Verdict(False, None, f"step {t}: founders changed guild and biomass with it")
            whole = (math.fsum(s["respired_" + g] + s["lost_" + g] for g in GUILD_KEYS) + growth_all)
            if abs(s["income_producers"] - whole) > 3 * tol:
                return Verdict(False, num(whole, sig=6),
                               f"step {t}: production != respiration + growth + lost to decomposers")
            for g in GUILD_KEYS[1:]:  # a consumer keeps at most the efficiency of what it eats
                if s["lost_" + g] < (1.0 - efficiency) * s["income_" + g] - tol:
                    return Verdict(False, None, f"step {t}: the {g} kept more than the efficiency allows")
            if not 0.0 <= s["max_take"] <= 1.0:
                return Verdict(False, None, f"step {t}: consumers took more than their prey had")
            # oxygen
            if abs(s["o2_produced"] - O2_YIELD * s["income_producers"]) > 1e-9:
                return Verdict(False, None, f"step {t}: oxygen set free does not match the production")
            before = prev["raw_oxygen"] if prev else s["raw_oxygen"]
            if abs(s["raw_oxygen"] - before - (s["o2_produced"] - s["o2_consumed"])) > 1e-9:
                return Verdict(False, None, f"step {t}: oxygen change != produced minus consumed")
            if s["raw_oxygen"] < 0 or s["o2_consumed"] < 0:
                return Verdict(False, None, f"step {t}: negative oxygen")
            # the lineages and what the step says about them
            alive = lineages[t]
            for ln in alive:
                size, types = ln["size"], ln["cell_types"]
                if not ln["biomass"] >= EXTINCT or ln["guild"] not in GUILDS:
                    return Verdict(False, None, f"step {t}: lineage {ln['id']} is below the threshold")
                if size < 1 or size > MAX_SIZE or size & (size - 1):
                    return Verdict(False, None, f"step {t}: lineage {ln['id']} has an impossible size")
                if not needed_cell_types(size) <= types <= max_cell_types(size):
                    return Verdict(False, None, f"step {t}: lineage {ln['id']} has cell types its size forbids")
                if simple and (size > 1 << SIMPLE_MAX_K or types > SIMPLE_MAX_TYPES):
                    return Verdict(False, None, f"step {t}: lineage {ln['id']} needs complex cells")
                if ln["level"] > supported or ln["level"] < GUILDS.index(ln["guild"]) + 1:
                    return Verdict(False, None, f"step {t}: lineage {ln['id']} sits above the pyramid")
            if len({(ln["guild"], ln["size"], ln["cell_types"]) for ln in alive}) != len(alive):
                return Verdict(False, None, f"step {t}: two lineages with the same traits")
            told = describe(alive, light, s["raw_oxygen"], s["extinctions"])
            for key, value in told.items():
                same = value == s[key] if key in STATE_KEYS else abs(value - s[key]) <= tol
                if not same:
                    return Verdict(False, dense_value(value), f"step {t}: {key} does not follow from the lineages")
            if s["trophic_levels"] > supported:
                return Verdict(False, num(supported), f"step {t}: more trophic levels than the pyramid supports")
            gone = sum(1 for e in events if e["kind"] in GONE and e["step"] <= t)
            if s["extinctions"] != gone or (prev and s["extinctions"] < prev["extinctions"]):
                return Verdict(False, num(gone), f"step {t}: extinctions do not match the log")
            # the energy pyramid: a guild passes on at most what it kept of its food and what founders brought
            for g in GUILD_KEYS[1:]:
                passed[g] += s["taken_" + g]
                kept[g] += efficiency * s["income_" + g] - s["moved_" + g]
                if passed[g] > kept[g] + tol:
                    return Verdict(False, None, f"step {t}: the {g} passed on more energy than reached them")
            prev = s
        producers = {0}  # the living producer lineages, as the log tells them
        for n, e in enumerate(events):
            if e["kind"] in GONE:
                if e["kind"] == "displaced" and producers == {e["id"]}:
                    return Verdict(False, None, f"event {n}: the last producer lineage was pushed out")
                producers.discard(e["id"])
                continue
            if e["kind"] not in MUTATIONS or not 1 <= e["step"] < len(r.steps):
                return Verdict(False, None, f"event {n}: unknown kind or step")
            if e["guild"] == "producer":
                producers.add(e["id"])
            if e["guild"] != "producer" and e["size"] > oxygen_max_size(e["oxygen"]):
                return Verdict(False, None, f"event {n}: a consumer was founded too big for the oxygen")
            if e["kind"] == "guild" and (e["guild"] == "producer" or float(p["predation"]) <= 0
                                         or (e["guild"] == "predator" and e["oxygen"] < O2_PREDATOR)):
                return Verdict(False, None, f"event {n}: a guild switch the rules forbid")
        if dict(r.summary) != summarise(r.steps):
            return Verdict(False, None, "the summary does not follow from the last step")
        return Verdict(True, None, "energy, biomass, feeding, oxygen and the lineages are consistent")

    def parse(self, prompt: str) -> tuple[int, str, int, str] | None:
        """``(seed, where, step, key)`` with ``where`` in ``step``, ``final``, ``params``."""
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
        return (seed, "step", step, rest[0]) if len(rest) == 1 and rest[0] in STATE_KEYS else None

    def owns(self, prompt: str) -> bool:
        return self.parse(prompt) is not None

    def truth(self, prompt: str) -> float | int | str | None:
        """The simulation's own value for a prompt, or ``None`` when it is not this topic's."""
        q = self.parse(prompt)
        if q is None:
            return None
        seed, where, step, key = q
        r = self.rollout(seed)
        if where == "final":
            return r.summary[key]
        if where == "params":
            return r.params[key]
        return r.steps[step][key]

    def check(self, prompt: str, answer: str) -> Verdict:
        value = self.truth(prompt)
        if value is None:
            return Verdict(False, None, "not my question")
        words = prompt.split(" ")
        key, given = words[-1], words[-2] == "params"
        expected = param_value(key, value) if given else dense_value(value)
        if isinstance(value, str):
            ok = answer.strip() == value
            return Verdict(ok, expected, "exact word" if ok else "wrong word")
        got = parse_num(answer)
        if not is_finite(got):  # "1 e 9 9 9" parses to infinity, which is close to nothing
            return Verdict(False, expected, "not a number")
        if key in COUNT_KEYS or given:
            ok = got == value
            return Verdict(ok, expected, "exact" if ok else "wrong count" if key in COUNT_KEYS else "not the parameter")
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


def _digits_end(words: list[str], start: int) -> int:
    """Index after the run of single-digit tokens that begins at ``start``."""
    i = start
    while i < len(words) and words[i] in _DIGITS:
        i += 1
    return i


def params_line(r: Rollout) -> Line:
    """``bodies seed 3 params. light 2 5 0. efficiency 0 point 1. ... complex_cells yes.``"""
    fields = " ".join(f"{k} {param_value(k, r.params[k])}." for k in PARAM_KEYS)
    return Line(f"{r.sim} seed {num(r.seed)} params. {fields}", topic=r.sim, kind="record")


def lines(r: Rollout, every: int = 1) -> list[Line]:
    """The parameter record, a state record and one question per metric for every ``every``-th
    step, and the final questions. The ledger is left out: it is the gate's, not the model's."""
    keys = list(STATE_KEYS)
    out = [params_line(r)]
    for t in range(0, len(r.steps), every):
        out.append(state_line(r, t, keys))
        out += [Line(f"{r.sim} seed {num(r.seed)} step {num(t)} {k}", r.value(r.steps[t][k]), r.sim, "fact")
                for k in keys]
    return out + summary_lines(r)


_SIM = Bodies()


def simulation() -> Bodies:
    return _SIM


def run(seed: int, **params) -> BodiesRollout:
    """One rollout; parameters as in :meth:`Bodies.run`."""
    return _SIM.run(seed, **params)


def rollout(seed: int) -> BodiesRollout:
    """The canonical rollout of a seed (parameters from :func:`random_params`)."""
    return _SIM.rollout(seed)


def conserved(r: Rollout) -> Verdict:
    return _SIM.conserved(r)


def check(prompt: str, answer: str) -> Verdict:
    """Judge an answer to a question about a rollout or to a lesson (``bodies predict ...``)."""
    if LESSONS.owns(prompt):
        return LESSONS.check(prompt, answer)
    return _SIM.check(prompt, answer)


def owns(prompt: str) -> bool:
    return _SIM.owns(prompt) or LESSONS.owns(prompt)
