"""Level 12 of round 7: groups that cooperate, teach and accumulate knowledge.

Not a history of mankind. Up to 40 groups live for 20,000 years on a land of 40 equal
territories: 80 output steps of 250 years, each ten generations of 25 years. A group is a small
record (people, share of cooperators, skill, stored food, territories), not a crowd of agents.
Nobody knows in this detail how cooperation, technology and states arose. What the level does is
run a handful of published, simple models side by side, each with numbers that can be recomputed,
and let them feed each other: the language (from level 11) sets how well a skill is taught, skill
unlocks technologies, technologies and cooperation set how many people the land feeds, and the
people are the learners who carry the skill.

**Cooperation** (every generation, in every group). The game is the linear public-goods game of
the experiments of Fehr and Gaechter (2000, 2002), here with five players: a cooperator pays in a
``stake``, the pot is multiplied and shared equally among all five. Cooperators also punish: a
defector pays the fine ``punishment`` to every cooperator in his game, and each fine costs the
punisher a third of it (the 1 : 3 of Fehr and Gaechter 2002). ``benefit`` b and ``cost`` c say
what one contribution gives the four others in all and costs its giver net, so
stake = c + b / 4 and multiplier = 5 b / (4 c + b). The players are kin: a partner of a
cooperator is a cooperator with probability r + (1 - r) x, of a defector with (1 - r) x (x the
share of cooperators, r the relatedness; the usual assortment form). The expected payoffs then
differ by

    cooperator - defector = r b - c + 4 punishment (1 - r) (x - (1 - x) / 3)

Without punishment this is Hamilton's rule (Hamilton 1964): cooperation grows when r b > c.
With punishment there are two stable states, as in the model of Boyd and Richerson (1992):
where most cooperate a defector is fined by many and cooperation holds even among strangers;
where few do, punishing costs more than it brings and cooperation cannot start (with a fine the
share must first pass 1 / 4 + 3 (c - r b) / (16 punishment (1 - r)); a fine also makes the start
harder where r b > c only just holds). The share moves by one step of the replicator equation
(Taylor and Jonker 1978) per generation, x + 0.2 x (1 - x) (difference).
Taken from these: the game and its payoffs, the 1 : 3, the assortment form, the replicator
step. The 1 : 3 is what the experimenters set for their subjects, not a constant of human
nature. Left out, and it is the hard part of this literature: here every cooperator punishes.
The cooperator who lets others do the punishing (the second-order free rider) does not exist in
the toy, nor does the punishing of cooperators that Herrmann, Thoeni and Gaechter (2008) found
in several societies; so the toy's punishment works better than punishment is known to.
Invented for the toy: r thins out as 25 / people in a group larger than a band of 25 (kin are a
smaller part of a larger group), so kin alone carry cooperation only in bands; 1 % of each kind
err each generation; a seeded sampling noise per step; and a group brings in the share
0.4 + 0.6 x of the full harvest.

**Cumulative culture.** Henrich (2004): every learner copies the most skilled member, mostly
worse (by ``alpha``) and with luck better (spread ``beta``); the mean skill of N learners changes
per generation by

    -alpha + beta (0.5772 + ln N)

so skill is lost below N* = exp(alpha / beta - 0.5772) learners and gained above; Henrich
proposed it for the tools the Tasmanians no longer made after their island was cut off. That is
a model, and a disputed one: others read the Tasmanian record differently, and comparisons of
population size and tool kits have come out both ways (for it Kline and Boyd 2010, against it
Vaesen and others 2016). Taken from it: this expectation (the simulation uses it, it draws no
single learners) and N*. Invented: the loss is alpha (1 + skill / 600) / fidelity, larger for a
complex skill and for poor teaching, so every number of learners has a skill at which it stops
gaining; the teaching
fidelity 0.2 + 0.6 * success * coverage (+ 0.2 with writing, at most 1), where ``success`` is the
communication success of the language and coverage = compositional + (1 - compositional) * words /
(words + 10): what the language says by rule it can say of anything new, the rest only as far as
single words reach; N = own people + ``connectedness`` * all other people (that density and
migration between groups, not the size of one group alone, decide whether skill accumulates is
what Powell, Shennan and Thomas 2009 argue from their simulations; the sum is a toy); a linked group
also closes 5 % of its gap to the most skilled linked group per generation. Without language the
fidelity is 0.2, the loss five times alpha, and N* is larger than the people the canonical lands
can hold. That a group without language keeps no skill at all is therefore a property of these
toy numbers and no finding. A group has the technologies whose skill threshold it has reached,
in one fixed order: fire 30, stone_tools 90, clothing 150, boats 220, pottery 320,
farming 450, metal 720, writing 900, mathematics 1260, printing 1620; writing (and what follows)
only in a group with three levels of hierarchy (a toy rule: an administration is what needs
it here). A group whose skill falls loses them again, last first. Order, thresholds and the
factors on the carrying capacity (1.2, 1.3, 1.2, 1.2, 1.2, 10, 2, 1, 1.2, 1.2) are toy values;
real regions took different orders, some states never wrote, and fire and stone tools are far
older than any language anyone can date.

**People and food.** Logistic growth (Verhulst 1838) toward a carrying capacity, the limit set
by food that Malthus (1798) wrote of, here a number: capacity = land of the group * factors of
its technologies * (0.4 + 0.6 x) * climate, people next = people + growth * people * (1 -
people / capacity) per generation. Food is counted in rations (one feeds one person for one
generation): produced = (1 + surplus) * min(people, capacity) with surplus 0.05 (0.1 with pottery, 0.25 with farming, 0.3 with metal); everybody eats
one ration as long as there is food, the hungry die; of what is left over a share keeps (0.2,
with pottery 0.5) and the rest is lost, so produced = eaten + stored + lost. Three quarters of
the people die of age per generation; births fill up to the logistic number. A dry period
(chance 0.1 per step) cuts every harvest to 0.6-0.9 for that step: the people then overshoot
what the land gives, and only a store saves them. All of this paragraph but the logistic law is
toy.

**Groups.** The run starts with 3 to 8 bands on one territory each; the groups stand in a ring,
a daughter next to her mother. A group of at least ten people at 70 % of its capacity sends half
of them to a free territory. With chance ``connectedness`` per step (1.5 times that with boats) a
group is linked: linked groups exchange 2 % of their people, pass on skill and fight each other
half as often. Two neighbours fight with chance 0.05 per step times how full the fuller of them
is; the winner is drawn with the ratio rule (Tullock 1980), strength = people * (0.4 + 0.6 x); a
fifth of the losers and a twentieth of the winners die. Foragers only take the store; a winner
who farms takes land and people. Linked farming neighbours may unite (chance 0.03 x x'). A group
of several territories whose cooperation falls below 0.3 breaks in two. A group below five people
dies out. After the year ``isolation`` (if not 0) nobody is linked and every group learns alone.
All of this is toy.

**Structure.** Levels of hierarchy by size: none below 150 people, then one more per factor of
six (150, 900, 5400, 32400, ...). The factor is the span of about six of scalar stress (Johnson
1982); the rule itself is a toy. The stage is that of the largest group: ``bands``, ``tribes``,
``villages``, ``chiefdoms``, ``states`` at 0, 1, 2, 3, 4 and more levels (the names of Service
1962 with villages put in between; the sizes are toy, the words are labels for sizes, and that
societies climb one such ladder is a scheme much criticised since), or ``collapse`` when there
are fewer than 60 % of the most people of the last 2000 years. Roles: 1 + floor(log2(1 + rations
in the store)), the store feeding those who do not gather food. Wealth: everybody owns 0.1; the store belongs to
the five fifths of a group in proportion to q ** levels (q = 1 .. 5), equally without hierarchy;
``gini`` is the Gini coefficient (Gini 1912) of these five values. Both are toy rules.

Output: 81 steps (year 0 to 20000), each a dict of the metrics in :data:`STATE_KEYS` plus a
ledger (:data:`LEDGER_KEYS`); every group of every step (``rollout.groups``) and the log of buds,
raids, conquests, mergers, splits and extinctions (``rollout.events``). Dense lines
(:func:`lines`, 2512 per rollout, the longest about 100 tokens of the 128 allowed): measured
numbers and the number of people to 3 significant digits, counts exactly, the parameters as given:

    society seed 3 params. land 1 9 5 0. relatedness 0 point 2 9. benefit 3 point 2. cost 1. ... hamilton no.
        fidelity 0 point 5 7 2. critical_population 6 9 point 4. doubling_years 7 7 point 7.
    society seed 3 step 4 0. year 1 0 0 0 0. population 5 2 0 0 0. groups 3 1. largest_group 3 8 9 0.
        cooperation 0 point 9 8 2. skill 5 6 3. technologies 6. farming yes. writing no. hierarchy 2. roles 1 0.
        gini 0 point 3 1 2. trade 1. conflicts 3 6. stage villages.            (one line each)
    q society seed 3 step 4 0 largest_group. a 3 8 9 0.
    q society seed 3 step 4 0 next skill. a 5 8 0.                (what happens next)
    q society seed 3 final technology_list. a fire stone_tools clothing boats pottery farming metal.
    q society seed 3 final first_writing. a never.                (how it ends)

and the lessons (:data:`LESSONS`, 22 rules; each is a function of this module, and the simulation
calls the same function):

    q society predict hamilton relatedness 0 point 5 9 benefit 1 point 5 cost 1. a no.
    q society predict skill_change alpha 1 0 point 6 5 beta 1 point 6 6 learners 9 6 7 6 0. a 9 point 3 6 5.
    q society predict hierarchy population 3 4 5 6 3. a 4.

The inputs of a lesson are drawn evenly from their ranges, except the inputs of the rules with
thresholds (:data:`LADDERS`: numbers of people, specialists, skill): there a bracket is drawn
first, so every level of hierarchy, every stage, every number of roles and every number of
technologies is asked about as often (drawn evenly, two lessons of three had the answer 3 or
``chiefdoms``, and a third of the ``technologies`` lessons the answer 7). Measured over 22000
lessons (``random.Random(11)``): ``hamilton`` is yes in 51 % (cost up to 4.5; 60 % with cost up
to 3); the most common answer has a share of 20 % for ``hierarchy`` (0), 31 % for ``stage``
(states, which spans two levels), 7 % for ``roles``, 19 % for ``technologies`` (7: skill enough
to write, but no administration), 18 % for ``payoff_defector`` (0: nobody contributes), 12 % for
``replicator`` (0 or 1: the share is kept within its bounds) and at most 1.5 % for every other
rule.

Gates (:meth:`Society.conserved`): every group's people = its people a step before + births -
deaths + moved in - moved out, exactly, and so the population; nobody who moved out is missing;
food produced = eaten + stored + lost and the stores add up (to 1e-6 of the food); the 40
territories add up; the payoffs of the last generation are recomputed from the game and the
cooperation is one replicator step on from them; every share is within 0 and 1; a group's
technologies follow from its skill and size, in their order; hierarchy, roles, gini and stage
follow from the largest group and the history; conflicts match the log. :meth:`Society.check`
replays the seed (parameters from :func:`random_params` with ``random.Random(seed)``) and
compares exactly for words and counts and within 5 % for measured numbers and the number of
people. A prompt has one spelling (no leading zeros), and an answer must be a finite number.

Plausibility targets (``tests/test_evo_society.py`` holds them), with the values measured over
the canonical seeds 1 to 300 and in runs with chosen parameters over seeds 1 to 40:

    every seed        the gates pass; the run starts as bands without technology   300 of 300 runs
    their outcome     bands / tribes / villages / chiefdoms / states / collapse    156 / 24 / 32 / 3 / 60 / 25
                      farming / writing reached at some time                       101 / 53 runs
                      year of the first farming                                    2500 .. 12750, median 5000
                      year of the first writing                                    4250 .. 20000, median 9250
    no language       success 0.02: no technology, bands throughout                40 of 40 runs, skill stays 0
    a full language   compositional 1, success 0.9, land >= 3000,                  farming 39, writing 10, states
                      connectedness >= 0.6                                         12 of 40 runs; with punishment
                                                                                   0.8: 39, 35, 37
                      the same language, land 1000, no links at all                no farming in 40 of 40, no
                                                                                   technology at the end in 35,
                                                                                   never any in 34
    cooperation       r b < c, no punishment: highest share after year 2000        0.073 (40 runs)
                      the same with punishment 1.0: share at the end               0.39 .. 0.98, median 0.89
                      r b = 3 > c in bands (land 1000), no punishment              0.975 in every run
    farming           most people within 2000 years over people before             7.8 .. 29.6 times, median 10.2
    isolation         bands with 5 technologies cut off in year 10000              none left at the end, 40 of 40
                                                                                   (6 or 7 where the links hold)
    inequality        median gini at 0, 1, 2, 3, 4, 5 levels of hierarchy          0, 0.13, 0.33, 0.40, 0.46, 0.49

    learners          default run with land 1000 / 2000 / 3000 / 5000 and no links:   0 / 79 / 197 / 275
                      median skill at the end (40 runs each)
                      the same with connectedness 0.3                              830 / 1440 / 1556 / 1609

A run takes about 0.1 seconds. In the toy, writing appears only where punishment holds large
groups together (a group needs 5400 people for three levels), which is why most runs with a full
language farm but only some write. These targets say what the toy's rules do when they are put
together; they were tuned to look plausible and are not evidence for anything outside the toy.

What is real and what is toy. Real, in the sense of taken from a published model: Hamilton's
rule; the public-goods game with punishment and its 1 : 3 cost; the replicator equation; Henrich's
formula and its critical population; logistic growth; the Gini coefficient; the ratio rule of a
contest. Toy, tuned to give plausible runs and not derived: every constant named ``toy value``
in the code; kin thinning with group size; the growth of the transmission loss with skill and
its fall with teaching fidelity; the list of technologies, its order, thresholds and factors; the
harvest factor of cooperation; rations, stores and their spoilage; fights, mergers and splits;
hierarchy from size alone; roles and the wealth of the fifths; dry periods; 40 territories on a
ring without a map. Not in the toy at all: individuals, families, sexes and ages; cooperators
who do not punish, reputation and reciprocity; religion, law and markets; disease; the plants
and animals that could be domesticated, which differed from region to region; any feedback of
inequality on cooperation. The years of a run tell the toy's rules; they are no
dates of history, and no number here is a measurement of a real society.

Hand-off. *In* (:func:`handoff_in`): the level below is ``signals``; from its summary this level
takes, as they are, the communication ``success`` (0 to 1), the number of ``words`` in use and
``compositional``, the share of what is said that the language expresses by rule (0 to 1). They
enter only through the teaching fidelity; the other parameters (land, relatedness, benefit, cost,
punishment, alpha, beta, connectedness, growth, isolation) are this level's own. *Out*: the summary
gives the
chain its last era: ``stage`` and ``outcome`` (bands, tribes, villages, chiefdoms, states or
collapse), ``population`` and ``groups`` (counts), ``cooperation`` (share), ``technologies``
(count) and ``technology_list`` (their names in order, or none), ``farming`` and ``writing``
(yes/no), ``hierarchy`` and ``roles`` (counts), ``gini`` (0 to 0.8), ``conflicts`` (count),
``first_farming`` and ``first_writing`` (year since the start of this level, or never). One step
is 250 years; a run is 20,000 years.

Reproducibility: plain Python floats and integers; random numbers only from ``random()`` of
``random.Random(seed)`` (the Mersenne Twister, fixed across Python versions), drawn in a fixed
order; sums that enter a metric are exactly rounded (``math.fsum``) or integer; the simulation
uses no numpy, no sorting and no sets. People, cooperation and food are computed with the four basic operations and
``sqrt`` alone, which every machine rounds alike. ``log`` (in the skill) and ``exp`` may differ in
the last digit between machines; that moves a line only if a skill sits within such a digit of a
technology's threshold or of a rounding step. The tests change alpha and beta by 1e-12, ten
thousand times such a difference, and they move every result of ``log``, ``exp`` and ``log1p``
by one and by a thousand last digits, up, down and at random: not one line of 20 seeds moves
(measured once over 60 seeds: 0 of 150720 lines). For all seeds this is likely, not proven. The
lessons are drawn with ``choice``, ``randint``, ``randrange`` and ``uniform`` of the same
generator and whole-number brackets.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

from haishool.cosmos import Rollout, close, seeds
from haishool.evo import Input, LessonGate, Rule, sig
from haishool.truth import Line, Verdict, is_finite, num, parse_num

SIM = "society"

# --- the clock ---------------------------------------------------------------------------------
STEPS = 80  # output steps after step 0
STEP_YEARS = 250
GENERATION_YEARS = 25
GENERATIONS = STEP_YEARS // GENERATION_YEARS  # generations per output step

# --- land and groups ---------------------------------------------------------------------------
TERRITORIES = 40  # the land is cut into 40 equal territories; a group holds at least one (so at most 40 groups)
MIN_GROUP = 5  # a group below five people dies out
BUD_SHARE = 0.7  # a group at 70 % of its capacity sends half its people to a free territory
LAND_MIN = 1000.0  # the least land: a territory then feeds 2 * MIN_GROUP people without any cooperation

# --- cooperation -------------------------------------------------------------------------------
GAME_SIZE = 5  # players of one public-goods game (toy value; Fehr and Gaechter played in fours)
PUNISH_COST = 1.0 / 3.0  # a punisher pays a third of the fine he imposes (Fehr and Gaechter 2002: 1 for 3)
REPLICATOR_RATE = 0.2  # step size of the replicator equation per generation (toy value)
TREMBLE = 0.01  # share that takes the other strategy by error each generation (toy value)
DRIFT = 1.0  # size of the seeded sampling noise of the share per step, in units of sqrt(x (1 - x) / n)
BAND = 25  # people; up to here everybody in a group is kin at the full relatedness (toy value)
HARVEST_FLOOR = 0.4  # share of the full harvest a group without any cooperation still brings in (toy value)

# --- cumulative culture ------------------------------------------------------------------------
EULER_GAMMA = 0.5772156649015329  # the epsilon of Henrich's formula: mean of a standard Gumbel draw
COMPLEXITY = 600.0  # skill at which a skill is twice as hard to pass on as at skill 0 (toy value)
F_IMITATION = 0.2  # teaching fidelity without language: imitation alone (toy value)
F_LANGUAGE = 0.8  # with a perfectly understood language that can say everything (toy value)
F_WRITING = 0.2  # added by writing (toy value)
WORDS_HALF = 10  # a holistic lexicon of this many words covers half of what there is to teach (toy value)
DIFFUSE = 0.05  # share of the gap to the most skilled linked group that is closed per generation (toy value)
LINK_BOATS = 1.5  # boats make a link this much more likely (toy value)
MIGRATION = 0.02  # share of a linked group that moves to the next linked group per step (toy value)
ADMIN_LEVELS = 3  # levels of hierarchy from which administration needs writing (toy value)

#: the technologies in their fixed order: name, skill it needs, factor on the carrying capacity
TECHS = (
    ("fire", 30.0, 1.2),
    ("stone_tools", 90.0, 1.3),
    ("clothing", 150.0, 1.2),
    ("boats", 220.0, 1.2),
    ("pottery", 320.0, 1.2),
    ("farming", 450.0, 10.0),
    ("metal", 720.0, 2.0),
    ("writing", 900.0, 1.0),
    ("mathematics", 1260.0, 1.2),
    ("printing", 1620.0, 1.2),
)
TECH_NAMES = tuple(name for name, _, _ in TECHS)
BOATS = TECH_NAMES.index("boats") + 1  # a group with at least this many technologies has boats
POTTERY = TECH_NAMES.index("pottery") + 1
FARMING = TECH_NAMES.index("farming") + 1
METAL = TECH_NAMES.index("metal") + 1
WRITING = TECH_NAMES.index("writing") + 1

# --- food and people ---------------------------------------------------------------------------
MORTALITY = 0.75  # share of the people alive at a generation's start who die within it (toy value)
STORE_KEEP = (0.2, 0.5)  # share of the food left over that keeps for a generation: without, with pottery
OWN = 0.1  # personal goods per head, in food units, that everybody owns whatever the rank (toy value)

# --- hierarchy and stages ----------------------------------------------------------------------
SPAN = 6  # one more level of hierarchy per factor of six in people (scalar stress, Johnson 1982, as a toy)
STAGES = ("bands", "tribes", "villages", "chiefdoms", "states", "collapse")
COLLAPSE_SHARE = 0.6  # collapse: fewer people than this share of the most in the last COLLAPSE_WINDOW steps
COLLAPSE_WINDOW = 8

# --- neighbours --------------------------------------------------------------------------------
WAR_RATE = 0.05  # chance per step that two fully crowded neighbours fight (toy value)
LOSER_DEATHS, WINNER_DEATHS = 0.2, 0.05  # share of each side killed in a fight (toy values)
MERGE_RATE = 0.03  # chance per step that two linked, fully cooperating farming neighbours unite (toy value)
SPLIT_BELOW = 0.3  # a group of several territories breaks in two when its cooperation is below this
CLIMATE_RATE = 0.1  # chance per step of a dry period
CLIMATE_RANGE = (0.6, 0.9)  # harvest factor of a dry period

_DIGITS = frozenset("0123456789")


# ---------------------------------------------------------------------------------------------
# the rules: every function here is a lesson, and the simulation calls the same function
# ---------------------------------------------------------------------------------------------

def hamilton(relatedness: float, benefit: float, cost: float) -> str:
    """Hamilton's rule: helping kin pays when relatedness times benefit exceeds the cost.

    >>> hamilton(0.5, 4, 1), hamilton(0.125, 4, 1)
    ('yes', 'no')
    """
    return "yes" if relatedness * benefit > cost else "no"


def stake_of(benefit: float, cost: float, group: int) -> float:
    """The stake of a public-goods game in which one contribution costs its giver ``cost`` net
    and gives the ``group - 1`` others ``benefit`` in all: ``cost + benefit / (group - 1)``."""
    return cost + benefit / (group - 1)


def multiplier_of(benefit: float, cost: float, group: int) -> float:
    """The multiplier of that game: ``benefit * group / ((group - 1) * cost + benefit)``.

    >>> stake_of(4, 1, 5), multiplier_of(4, 1, 5)
    (2.0, 2.5)
    """
    return benefit * group / ((group - 1) * cost + benefit)


def payoff_cooperator(group: int, cooperators: float, multiplier: float, stake: float, punishment: float) -> float:
    """What a cooperator gets from one public-goods game with ``cooperators`` contributors
    (himself included) among ``group`` players: his share of the multiplied pot, minus his stake,
    minus what punishing costs him (a third of the fine for each defector).

    >>> round(payoff_cooperator(5, 3, 2.5, 2, 0.6), 9)
    0.6
    """
    return (multiplier * stake * cooperators / group - stake
            - PUNISH_COST * punishment * (group - cooperators))


def payoff_defector(group: int, cooperators: float, multiplier: float, stake: float, punishment: float) -> float:
    """What a defector gets: his share of the pot, minus one fine from every cooperator.

    >>> payoff_defector(5, 3, 2.5, 2, 0.5)
    1.5
    """
    return multiplier * stake * cooperators / group - punishment * cooperators


def kin_relatedness(relatedness: float, population: int) -> float:
    """Relatedness between the players of a game in a group of ``population`` people: the full
    value up to a band of 25, then thinned as 25 / population (kin are a smaller part of a
    larger group).

    >>> kin_relatedness(0.5, 20), kin_relatedness(0.5, 100)
    (0.5, 0.125)
    """
    return relatedness * min(1.0, BAND / population)


def replicator_step(share: float, cooperator: float, defector: float, rate: float) -> float:
    """One step of the replicator equation for the share of cooperators:
    ``share + rate * share * (1 - share) * (cooperator - defector)``, kept within 0 and 1.

    >>> replicator_step(0.5, 2.0, 1.0, 0.2)
    0.55
    """
    return min(1.0, max(0.0, share + rate * share * (1.0 - share) * (cooperator - defector)))


def tremble(share: float) -> float:
    """Errors: 1 % of each kind take the other strategy, so no share is ever exactly 0 or 1."""
    return TREMBLE + (1.0 - 2.0 * TREMBLE) * share


def teaching_fidelity(success: float, words: int, compositional: float, writing: int) -> float:
    """How well a skill is passed on, between 0.2 (imitation alone) and 1.

    ``0.2 + 0.6 * success * coverage`` (+ 0.2 with writing): ``success`` is how often a signal
    is understood; ``coverage`` is how much of what there is to teach the language can say: the
    share ``compositional`` of it by rule, and of the rest ``words / (words + 10)`` by single words.

    >>> teaching_fidelity(1, 10, 1, 0), teaching_fidelity(1, 10, 0, 0), teaching_fidelity(0, 10, 1, 0)
    (0.8, 0.5, 0.2)
    """
    coverage = compositional + (1.0 - compositional) * words / (words + WORDS_HALF)
    return min(1.0, F_IMITATION + (F_LANGUAGE - F_IMITATION) * success * coverage + (F_WRITING if writing else 0.0))


def effective_alpha(alpha: float, skill: float, fidelity: float) -> float:
    """The loss per transmission at a given skill and teaching fidelity:
    ``alpha * (1 + skill / 600) / fidelity``.

    >>> effective_alpha(3, 600, 0.5)
    12.0
    """
    return alpha * (1.0 + skill / COMPLEXITY) / fidelity


def learners(own: int, others: int, connectedness: float) -> float:
    """Effective number of learners of a group among others: its own people plus
    ``connectedness`` times the people of all other groups."""
    return own + connectedness * others


def skill_change(alpha: float, beta: float, learners: float) -> float:
    """Henrich's (2004) change of the mean skill per generation:
    ``-alpha + beta * (0.5772 + ln learners)``.

    >>> round(skill_change(3, 1, 100), 4)
    2.1824
    """
    return -alpha + beta * (EULER_GAMMA + math.log(learners))


def critical_population(alpha: float, beta: float) -> float:
    """The number of learners above which skill accumulates: ``exp(alpha / beta - 0.5772)``.

    >>> round(critical_population(3, 1), 2)
    11.28
    """
    return math.exp(alpha / beta - EULER_GAMMA)


def technologies(skill: float, hierarchy: int) -> int:
    """How many technologies a group has: those whose skill threshold it has reached, in the
    fixed order; writing (and whatever comes after it) only with three levels of hierarchy.

    >>> technologies(330, 0), technologies(1000, 2), technologies(1000, 3)
    (5, 7, 8)
    """
    count = 0
    for _, threshold, _ in TECHS:
        if skill < threshold or (count == WRITING - 1 and hierarchy < ADMIN_LEVELS):
            break
        count += 1
    return count


def tech_multiplier(count: int) -> float:
    """The factor the first ``count`` technologies put on the carrying capacity."""
    out = 1.0
    for _, _, factor in TECHS[:count]:
        out *= factor
    return out


def capacity(land: float, technologies: int, cooperation: float) -> float:
    """Carrying capacity in people: ``land`` (people it feeds without technology) times the
    factors of the technologies, times ``0.4 + 0.6 * cooperation``.

    >>> round(capacity(100, 1, 1), 6), round(capacity(100, 0, 0), 6)
    (120.0, 40.0)
    """
    return land * tech_multiplier(technologies) * (HARVEST_FLOOR + (1.0 - HARVEST_FLOOR) * cooperation)


def surplus_rate(technologies: int) -> float:
    """Food a worker makes beyond what one person eats: 0.05, with pottery 0.1, with farming
    0.25, with metal 0.3."""
    return (0.05 + (0.05 if technologies >= POTTERY else 0.0) + (0.15 if technologies >= FARMING else 0.0)
            + (0.05 if technologies >= METAL else 0.0))


def food_produced(population: int, capacity: float, technologies: int) -> float:
    """Food of one generation, in rations (one feeds one person for one generation):
    ``(1 + surplus rate) * min(population, capacity)``.

    >>> round(food_produced(100, 80, 0), 6), round(food_produced(100, 200, 6), 6)
    (84.0, 125.0)
    """
    return (1.0 + surplus_rate(technologies)) * min(float(population), capacity)


def logistic_step(population: float, rate: float, capacity: float) -> float:
    """Logistic growth, one generation: ``population + rate * population * (1 - population / capacity)``.

    >>> logistic_step(100, 0.25, 200)
    112.5
    """
    return population + rate * population * (1.0 - population / capacity)


def doubling_years(rate: float) -> float:
    """Years until a population that grows exponentially at the continuous ``rate`` per year has
    doubled: ``ln 2 / rate`` (the simulation passes ``ln(1 + growth) / 25`` for its generations).

    >>> round(doubling_years(0.01), 2)
    69.31
    """
    return math.log(2.0) / rate


def hierarchy_levels(population: int) -> int:
    """Levels of hierarchy a group of this size has (the toy rule): none below 150 people, then
    one more per factor of six: 150, 900, 5400, 32400, ...

    >>> [hierarchy_levels(n) for n in (149, 150, 899, 900, 5400, 32400, 194400)]
    [0, 1, 1, 2, 3, 4, 5]
    """
    levels, size = 0, BAND * SPAN
    while population >= size:
        levels += 1
        size *= SPAN
    return levels


def stage_of(population: int) -> str:
    """The stage of a group by its size: bands below 150, tribes below 900, villages below 5400,
    chiefdoms below 32400, then states.

    >>> stage_of(30), stage_of(2000), stage_of(50000)
    ('bands', 'villages', 'states')
    """
    return STAGES[min(hierarchy_levels(population), 4)]


def role_count(specialists: int) -> int:
    """Different roles in a group whose store feeds ``specialists`` people who do not gather food:
    ``1 + floor(log2(1 + specialists))``.

    >>> [role_count(s) for s in (0, 1, 2, 3, 200)]
    [1, 2, 2, 3, 8]
    """
    return (int(specialists) + 1).bit_length()


def gini(poorest: float, poor: float, middle: float, rich: float, richest: float) -> float:
    """Gini coefficient of five equally large parts of a group, each with the given wealth per
    head: the mean difference between two parts, over twice the mean. 0 when all are equal;
    0.8 at most for five parts.

    >>> gini(1, 1, 1, 1, 1), gini(0, 0, 0, 0, 5), round(gini(1, 2, 3, 4, 5), 4)
    (0.0, 0.8, 0.2667)
    """
    w = (poorest, poor, middle, rich, richest)
    total = math.fsum(w)
    if total <= 0:
        return 0.0
    return math.fsum(abs(a - b) for a in w for b in w) / (2.0 * len(w) * total)


def wealth_fifths(population: int, store: float, levels: int) -> tuple[float, float, float, float, float]:
    """Wealth per head of the five fifths of a group, poorest first: everybody owns 0.1; the
    stored food is shared out in proportion to ``q ** levels`` for the fifths q = 1 .. 5 (equal
    shares without hierarchy)."""
    weights = [float(q ** levels) for q in range(1, 6)]
    total = math.fsum(weights)
    per_head = store / population if population > 0 else 0.0
    return tuple(OWN + per_head * 5.0 * w / total for w in weights)  # type: ignore[return-value]


def win_probability(size: int, cooperation: float, other_size: int, other_cooperation: float) -> float:
    """Chance that a group wins a fight: its strength over the sum of both strengths, strength =
    ``size * (0.4 + 0.6 * cooperation)`` (the same factor as in the harvest).

    >>> win_probability(100, 1, 100, 0)
    0.7142857142857143
    """
    a = size * (HARVEST_FLOOR + (1.0 - HARVEST_FLOOR) * cooperation)
    b = other_size * (HARVEST_FLOOR + (1.0 - HARVEST_FLOOR) * other_cooperation)
    return a / (a + b)


def _far_from_equal(relatedness: float, benefit: float, cost: float) -> bool:
    """A lesson is not asked where relatedness * benefit equals the cost to the last digit."""
    return abs(relatedness * benefit - cost) > 1e-9


def _words(values) -> str:
    """``fire 3 0. stone_tools 9 0. ...``: the table of the technologies in dense words."""
    return ". ".join(f"{name} {num(value)}" for name, value in zip(TECH_NAMES, values))


RULES: dict[str, Rule] = {
    "hamilton": Rule(
        (Input("relatedness", 0, 1, places=2), Input("benefit", 0.5, 10, places=1),
         Input("cost", 0.1, 4.5, places=1)),
        hamilton,
        "hamilton rule. yes when relatedness times benefit is more than cost. then helping kin pays",
        _far_from_equal),
    "stake": Rule(
        (Input("benefit", 0.5, 10, places=1), Input("cost", 0.1, 3, places=1), Input("group", 2, 10, integer=True)),
        stake_of,
        "stake of the public goods game in which one contribution costs its giver cost and gives the others "
        "benefit in all. cost plus benefit over group minus 1"),
    "multiplier": Rule(
        (Input("benefit", 0.5, 10, places=1), Input("cost", 0.1, 3, places=1), Input("group", 2, 10, integer=True)),
        multiplier_of,
        "multiplier of that public goods game. benefit times group over the sum of group minus 1 times cost "
        "and benefit"),
    "payoff_cooperator": Rule(
        (Input("group", 2, 10, integer=True), Input("cooperators", 1, 10, integer=True),
         Input("multiplier", 1, 5, places=1), Input("stake", 0.5, 5, places=1), Input("punishment", 0, 2, places=1)),
        payoff_cooperator,
        "payoff of a cooperator in a public goods game. multiplier times stake times cooperators over group. "
        "minus stake. minus a third of punishment for every defector. cooperators counts himself",
        lambda group, cooperators, *_: cooperators <= group),
    "payoff_defector": Rule(
        (Input("group", 2, 10, integer=True), Input("cooperators", 0, 9, integer=True),
         Input("multiplier", 1, 5, places=1), Input("stake", 0.5, 5, places=1), Input("punishment", 0, 2, places=1)),
        payoff_defector,
        "payoff of a defector in a public goods game. multiplier times stake times cooperators over group. "
        "minus punishment for every cooperator",
        lambda group, cooperators, *_: cooperators < group),
    "kin": Rule(
        (Input("relatedness", 0, 0.5, places=2), Input("population", 5, 5000, integer=True)), kin_relatedness,
        f"toy rule. relatedness between players in a group. relatedness up to {num(BAND)} people. above that "
        f"relatedness "
        f"times {num(BAND)} over population"),
    "replicator": Rule(
        (Input("share", 0, 1, places=2), Input("cooperator", -5, 10, places=2), Input("defector", -5, 10, places=2),
         Input("rate", 0.05, 0.5, places=2)),
        replicator_step,
        "one step of the replicator equation. share plus rate times share times 1 minus share times cooperator "
        "payoff minus defector payoff. kept between 0 and 1"),
    "fidelity": Rule(
        (Input("success", 0, 1, places=2), Input("words", 0, 60, integer=True),
         Input("compositional", 0, 1, places=2), Input("writing", 0, 1, integer=True)),
        teaching_fidelity,
        f"toy rule. teaching fidelity. {num(F_IMITATION)} plus {num(F_LANGUAGE - F_IMITATION)} times success times "
        f"coverage. plus {num(F_WRITING)} with writing. at most 1. coverage is compositional plus 1 minus "
        f"compositional "
        f"times words over words plus {num(WORDS_HALF)}"),
    "effective_alpha": Rule(
        (Input("alpha", 1, 5, places=2), Input("skill", 0, 2000, places=0), Input("fidelity", 0.2, 1, places=2)),
        effective_alpha,
        f"toy rule. loss per transmission. alpha times 1 plus skill over {num(COMPLEXITY)}. divided by fidelity"),
    "learners": Rule(
        (Input("own", 5, 2000, integer=True), Input("others", 0, 7000, integer=True),
         Input("connectedness", 0, 1, places=2)),
        learners,
        "toy rule. effective number of learners of a group among others. own plus connectedness times others"),
    "skill_change": Rule(
        (Input("alpha", 0.5, 12, places=2), Input("beta", 0.5, 2, places=2),
         Input("learners", 1, 100000, integer=True)),
        skill_change,
        "change of mean skill per generation after henrich. minus alpha plus beta times the sum of "
        "0 point 5 7 7 2 and the natural log of learners"),
    "critical_population": Rule(
        (Input("alpha", 0.5, 8, places=2), Input("beta", 0.5, 2, places=2)), critical_population,
        "number of learners above which skill accumulates. exp of alpha over beta minus 0 point 5 7 7 2",
        lambda alpha, beta: critical_population(alpha, beta) < 10000),
    "technologies": Rule(
        (Input("skill", 0, 2000, integer=True), Input("hierarchy", 0, 6, integer=True)), technologies,
        "toy rule. number of technologies in the fixed order. skill thresholds " + _words(int(t) for _, t, _ in TECHS)
        + f". writing and what follows only with hierarchy {num(ADMIN_LEVELS)} or more"),
    "capacity": Rule(
        (Input("land", 5, 300, places=0), Input("technologies", 0, 10, integer=True),
         Input("cooperation", 0, 1, places=2)),
        capacity,
        f"toy rule. carrying capacity in people. land times the factors of the technologies times "
        f"{num(HARVEST_FLOOR)} plus {num(1 - HARVEST_FLOOR)} times cooperation. factors "
        + _words(f for _, _, f in TECHS),
        lambda land, count, cooperation: land * tech_multiplier(count) < 10000),
    "food": Rule(
        (Input("population", 5, 7000, integer=True), Input("capacity", 5, 7000, places=0),
         Input("technologies", 0, 10, integer=True)),
        food_produced,
        "toy rule. rations produced in one generation. 1 plus surplus rate times the smaller of population and "
        "capacity. "
        f"surplus rate {num(surplus_rate(0))}. from pottery {num(surplus_rate(POTTERY))}. from farming "
        f"{num(surplus_rate(FARMING))}. from metal {num(surplus_rate(METAL))}"),
    "logistic": Rule(
        (Input("population", 1, 9999, integer=True), Input("rate", 0.05, 0.5, places=2),
         Input("capacity", 10, 9999, integer=True)),
        logistic_step,
        "logistic growth in one generation. population plus rate times population times 1 minus population "
        "over capacity",
        lambda population, rate, capacity: population <= 3 * capacity),
    "doubling_years": Rule(
        (Input("rate", 0.001, 0.05, places=5),), doubling_years,
        "years until a population has doubled at an exponential growth rate per year. natural log of 2 over rate"),
    "hierarchy": Rule(
        (Input("population", 1, 400000, integer=True),), hierarchy_levels,
        f"levels of hierarchy by the toy rule. 0 below {num(BAND * SPAN)} people. then one more for every factor "
        f"of {num(SPAN)}. 1 from {num(BAND * SPAN)}. 2 from {num(BAND * SPAN ** 2)}. 3 from {num(BAND * SPAN ** 3)}. "
        f"4 from {num(BAND * SPAN ** 4)}. 5 from {num(BAND * SPAN ** 5)}"),
    "stage": Rule(
        (Input("population", 1, 400000, integer=True),), stage_of,
        f"toy rule. stage of a group by its size. bands below {num(BAND * SPAN)}. tribes below "
        f"{num(BAND * SPAN ** 2)}. villages below {num(BAND * SPAN ** 3)}. chiefdoms below "
        f"{num(BAND * SPAN ** 4)}. then states"),
    "roles": Rule(
        (Input("specialists", 0, 100000, integer=True),), role_count,
        "toy rule. roles in a group whose store feeds specialists. 1 plus the whole part of the log base 2 of 1 plus "
        "specialists"),
    "gini": Rule(
        (Input("poorest", 0, 10, places=1), Input("poor", 0, 10, places=1), Input("middle", 0, 10, places=1),
         Input("rich", 0, 10, places=1), Input("richest", 0, 10, places=1)),
        gini,
        "gini coefficient of five equal parts of a group by wealth per head. sum of the differences of all "
        "pairs over 5 times the sum of the wealth. 0 when all are equal",
        lambda *w: sum(w) > 0),
    "win_probability": Rule(
        (Input("size", 5, 100000, integer=True), Input("cooperation", 0, 1, places=2),
         Input("other_size", 5, 100000, integer=True), Input("other_cooperation", 0, 1, places=2)),
        win_probability,
        f"chance to win a fight by the ratio rule. own strength over the sum of both strengths. toy strength is "
        f"size times "
        f"{num(HARVEST_FLOOR)} plus {num(1 - HARVEST_FLOOR)} times cooperation"),
}
#: Inputs whose answer has steps (people, specialists, skill), by rule and input: the edges of the
#: brackets a value is drawn from. Drawn evenly from 1 to 400000, two lessons of three would show a
#: group of the same stage and hardly one a band; so a bracket is drawn first (each as likely) and
#: then a whole number in it.
_PEOPLE = (1, BAND * SPAN, BAND * SPAN ** 2, BAND * SPAN ** 3, BAND * SPAN ** 4, BAND * SPAN ** 5, 400001)
LADDERS: dict[tuple[str, str], tuple[int, ...]] = {
    ("hierarchy", "population"): _PEOPLE,
    ("stage", "population"): _PEOPLE,
    ("roles", "specialists"): tuple((1 << k) - 1 for k in range(17)) + (100001,),  # one bracket per number of roles
    ("kin", "population"): (5, BAND + 1, 100, 1000, 5001),  # a quarter of the lessons within a band
    ("skill_change", "learners"): (1, 10, 100, 1000, 10000, 100001),  # evenly over the orders of magnitude
    #: one bracket per number of technologies: drawn evenly from 0 to 2000, a third of the answers were 7
    ("technologies", "skill"): (0,) + tuple(int(t) for _, t, _ in TECHS) + (2001,),
}
#: rules whose answer jumps at the edges: one draw in four is an edge or the number just below it
EDGE_RULES = frozenset({"hierarchy", "stage", "roles", "technologies"})
EDGE_SHARE = 0.25


class SocietyLessons(LessonGate):
    """The lesson gate of this level. Only the drawing of inputs differs from :class:`LessonGate`:
    an input that has an entry in :data:`LADDERS` is drawn bracket first, so every level of
    hierarchy, every stage, every number of roles and every number of technologies is asked about
    as often, and a quarter of these questions sit right at a threshold (150 or 149 people, skill
    450 or 449). Whole-number
    arithmetic only. Everything else, and every judgement, is the base class's."""

    def draw(self, rng: random.Random, name: str, spec: Input) -> float | int:
        edges = LADDERS.get((name, spec.name))
        if edges is not None:
            i = rng.randrange(len(edges) - 1)
            if name in EDGE_RULES and i > 0 and rng.random() < EDGE_SHARE:
                return edges[i] - rng.randrange(2)
            return rng.randint(edges[i], edges[i + 1] - 1)
        if spec.integer:
            return rng.randint(int(spec.low), int(spec.high))
        return round(rng.uniform(spec.low, spec.high), spec.places)

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
            ln = self.line(name, [self.draw(rng, name, spec) for spec in self.rules[name].inputs])
            if ln is not None:
                out.append(ln)
        return out


LESSONS = SocietyLessons(SIM, RULES)


def lesson_gate() -> LessonGate:
    return LESSONS


# ---------------------------------------------------------------------------------------------
# keys
# ---------------------------------------------------------------------------------------------

PARAM_KEYS = ("land", "relatedness", "benefit", "cost", "punishment", "alpha", "beta", "connectedness", "success",
              "words", "compositional", "isolation", "growth", "hamilton", "fidelity", "critical_population",
              "doubling_years")
#: the parameters that are put in; they are written as given, the derived ones to 3 digits
INPUT_KEYS = frozenset({"land", "relatedness", "benefit", "cost", "punishment", "alpha", "beta", "connectedness",
                        "success", "words", "compositional", "isolation", "growth"})
#: derived parameters the gate needs but the record does not show
HIDDEN_PARAM_KEYS = ("stake", "multiplier")
STATE_KEYS = ("year", "population", "groups", "largest_group", "cooperation", "skill", "technologies", "farming",
              "writing", "hierarchy", "roles", "gini", "trade", "conflicts", "stage")
LEDGER_KEYS = ("births", "deaths", "moved", "produced", "eaten", "stored", "lost", "store", "free_land", "climate")
SUMMARY_KEYS = ("stage", "population", "groups", "cooperation", "technologies", "technology_list", "farming",
                "writing", "hierarchy", "roles", "gini", "conflicts", "outcome", "first_farming", "first_writing")
#: whole numbers that are judged exactly
COUNT_KEYS = frozenset({"year", "groups", "technologies", "hierarchy", "roles", "conflicts", "words", "isolation",
                        "births", "deaths", "moved", "free_land", "first_farming", "first_writing"})
#: whole numbers that are written to three digits and judged within 5 %, like a measured number
ROUNDED_KEYS = frozenset({"population", "largest_group"})
WORD_KEYS = frozenset({"farming", "writing", "stage", "hamilton", "technology_list", "outcome"})
OUTCOMES = STAGES

KEYS: dict[str, str] = {
    "land": "people the whole land feeds without any technology and with full cooperation (parameter)",
    "relatedness": "relatedness between the members of a band, 0 to 1 (parameter)",
    "benefit": "what one act of cooperation gives the other players of a game in all, in units of cost 1 (parameter)",
    "cost": "what one act of cooperation costs the one who does it, net (parameter)",
    "punishment": "fine a defector pays to every cooperator he plays with; the punisher pays a third of it (parameter)",
    "alpha": "Henrich's alpha: skill lost per transmission at skill 0 and perfect teaching (parameter)",
    "beta": "Henrich's beta: spread of the learners' results, which lets the best exceed the model (parameter)",
    "connectedness": "chance per step that a group is linked to the others, and weight of their people as "
                     "learners, 0 to 1 (parameter)",
    "success": "communication success of the language from level 11, 0 to 1 (parameter)",
    "words": "size of the lexicon from level 11 (parameter)",
    "compositional": "share of what is said that the language from level 11 expresses by rule, 0 to 1 (parameter)",
    "isolation": "year after which no group is linked to another any more; 0: never (parameter)",
    "growth": "largest growth per generation of 25 years, as a share (parameter)",
    "hamilton": "yes when relatedness * benefit > cost: bands cooperate without punishment",
    "fidelity": "teaching fidelity with this language and without writing, 0.2 to 0.8",
    "critical_population": "learners above which skill accumulates from skill 0: "
                           "exp(alpha / (fidelity * beta) - 0.5772)",
    "doubling_years": "years in which a population far below its capacity doubles: ln 2 / (ln(1 + growth) / 25)",
    "stake": "stake of the public-goods game that has this benefit and cost for five players",
    "multiplier": "multiplier of that game",
    "year": "years since the start; one step is 250 years, ten generations",
    "population": "people in all groups",
    "groups": "groups (count)",
    "largest_group": "people in the largest group",
    "cooperation": "share of cooperators, mean over all people, 0 to 1",
    "skill": "skill level, mean over all people (toy units; the technologies need 30 to 1620)",
    "technologies": "technologies the most advanced group has (count, in the fixed order)",
    "farming": "yes when a group farms, else no",
    "writing": "yes when a group writes, else no",
    "hierarchy": "levels of hierarchy in the largest group (count)",
    "roles": "different roles in the largest group (count)",
    "gini": "Gini coefficient of wealth in the largest group, 0 to 0.8",
    "trade": "share of the groups that are linked to others in this step, 0 to 1",
    "conflicts": "fights between neighbouring groups so far (count)",
    "stage": "bands, tribes, villages, chiefdoms or states by the size of the largest group; collapse after a "
             "loss of people",
    "births": "ledger: people born in this step",
    "deaths": "ledger: people who died in this step, of age, hunger or in a fight",
    "moved": "ledger: people who changed their group in this step",
    "produced": "ledger: rations produced in this step (one ration feeds one person for a generation)",
    "eaten": "ledger: rations eaten in this step",
    "stored": "ledger: change of all stores in this step, rations",
    "lost": "ledger: rations spoiled or left behind in this step",
    "store": "ledger: rations in all stores at the end of the step",
    "free_land": "ledger: territories nobody holds (of 40)",
    "climate": "ledger: harvest factor of the step, 1 or 0.6 to 0.9 in a dry period",
    "technology_list": "the technologies of the most advanced group at the end, in order; none when there are none",
    "outcome": "the final stage, or collapse when the run ends with fewer than 60 % of its largest population",
    "first_farming": "year of the first step with farming, or never",
    "first_writing": "year of the first step with writing, or never",
}


def dense_value(key: str, value: float | int | str) -> str:
    """A metric as it stands in a line: a word, an exact count, or a number to 3 significant digits
    (the number of people too: 12345 people are written 1 2 3 0 0).

    >>> dense_value("stage", "bands"), dense_value("groups", 12), dense_value("population", 12345)
    ('bands', '1 2', '1 2 3 0 0')
    >>> dense_value("cooperation", 0.98765), dense_value("skill", 371.26), dense_value("first_writing", "never")
    ('0 point 9 8 8', '3 7 1', 'never')
    """
    if isinstance(value, str):
        return value
    if isinstance(value, int) and key not in ROUNDED_KEYS:
        return num(value)
    return num(sig(float(value)))


def param_value(key: str, value: float | int | str) -> str:
    """A parameter as it stands in the ``params`` record: an input as given (to at most 5 digits),
    a derived one like any measured number.

    >>> param_value("land", 2350.0), param_value("relatedness", 0.25), param_value("doubling_years", 77.656)
    ('2 3 5 0', '0 point 2 5', '7 7 point 7')
    """
    if isinstance(value, str):
        return value
    if key in INPUT_KEYS:
        return num(value) if isinstance(value, int) else num(float(value), sig=5)
    return dense_value(key, value)


@dataclass
class SocietyRollout(Rollout):
    """A :class:`Rollout` that also keeps every group at every step and the log of what happened
    between groups."""
    #: per step, per group (a group that ended in this step is listed once more with ``gone``)
    groups: list[list[dict]] = field(default_factory=list)
    #: ``{"step", "kind", "a", "b", "moved", "killed_a", "killed_b"}``; kinds: bud, raid, conquest,
    #: merge, split, extinct
    events: list[dict] = field(default_factory=list)

    def value(self, text: float | int | str) -> str:
        return text if isinstance(text, str) else dense_value("", text)


def random_params(rng: random.Random, rules: int = 7) -> dict[str, float | int]:
    """Land 1000-5000 people, relatedness 0.05-0.5, benefit 1.5-6 at cost 1, punishment 0 in three
    runs of ten and else 0.1-1.5, alpha 2.4-3.6, beta 0.9-1.1, connectedness 0.1-1, growth
    0.15-0.35. The language: in one run of ten hardly any (success below 0.2), else success 0.5-1;
    a lexicon of 2 to 40 words; ``compositional`` 0 in 35 runs of a hundred, 1 in 45, between in
    20. In three runs of twenty the links are cut for good at a year between 7500 and 15000
    (``isolation``)."""
    _rules(rules)
    land = 10.0 * int(100 + 400 * rng.random())
    relatedness = round(rng.uniform(0.05, 0.5), 2)
    benefit = round(rng.uniform(1.5, 6.0), 1)
    punishment = 0.0 if rng.random() < 0.3 else round(rng.uniform(0.1, 1.5), 2)
    alpha = round(rng.uniform(2.4, 3.6), 2)
    beta = round(rng.uniform(0.9, 1.1), 2)
    connectedness = round(rng.uniform(0.1, 1.0), 2)
    success = round(rng.uniform(0.0, 0.2), 2) if rng.random() < 0.1 else round(rng.uniform(0.5, 1.0), 2)
    words = 2 + int(39 * rng.random())
    kind = rng.random()
    compositional = 0.0 if kind < 0.35 else 1.0 if kind < 0.8 else round(rng.uniform(0.1, 0.99), 2)
    isolation = STEP_YEARS * (30 + int(31 * rng.random())) if rng.random() < 0.15 else 0
    growth = round(rng.uniform(0.15, 0.35), 2)
    return {"land": land, "relatedness": relatedness, "benefit": benefit, "cost": 1.0, "punishment": punishment,
            "alpha": alpha, "beta": beta, "connectedness": connectedness, "success": success, "words": words,
            "compositional": compositional, "isolation": isolation, "growth": growth}


def _rules(rules: int) -> None:
    if rules != 7:
        raise ValueError(f"society is a round-7 level: rules must be 7, not {rules!r}")


def handoff_in(below: dict) -> dict[str, float | int]:
    """The summary of the level below (``haishool.evo.signals``) as this level's language.

    Taken as they are: ``success`` (the communication success, 0 to 1), ``words`` (the distinct
    words in use, a whole number; 0 when missing) and ``compositional`` (the share of what is said
    that the language expresses by its rule, 0 to 1; 0 when missing; the words ``yes`` and ``no``
    count as 1 and 0). Nothing else of the level below is used.

    >>> handoff_in({"stage": "language", "success": 0.914, "words": 12, "compositional": 0.95})
    {'success': 0.914, 'words': 12, 'compositional': 0.95}
    >>> handoff_in({"stage": "calls", "success": 0.4, "words": 2})
    {'success': 0.4, 'words': 2, 'compositional': 0.0}
    """
    def share(key: str, value) -> float:
        if isinstance(value, (str, bool)) or not is_finite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"the level below must give {key} as a number between 0 and 1: {value!r}")
        return float(value)

    words = below.get("words", 0)
    if isinstance(words, (str, bool)) or not is_finite(words) or words < 0 or int(words) != words:
        raise ValueError(f"the number of words must be a whole number, not negative: {words!r}")
    rule = below.get("compositional", 0.0)
    rule = {"yes": 1.0, "no": 0.0}.get(rule, rule) if isinstance(rule, str) else rule
    return {"success": share("success", below.get("success")), "words": int(words),
            "compositional": share("compositional", rule)}


# ---------------------------------------------------------------------------------------------
# the simulation
# ---------------------------------------------------------------------------------------------

class _Group:
    __slots__ = ("id", "n", "territories", "x", "skill", "tech", "store", "linked", "busy",
                 "n_prev", "births", "deaths", "moved_in", "moved_out",
                 "n_before", "share_before", "kin", "payoff_c", "payoff_d")

    def __init__(self, gid: int, n: int, territories: int, x: float, skill: float, store: float) -> None:
        self.id, self.n, self.territories, self.x, self.skill, self.store = gid, n, territories, x, skill, store
        self.tech = technologies(skill, hierarchy_levels(n))
        self.linked = self.busy = False
        self.n_prev = 0
        self.births = self.deaths = self.moved_in = self.moved_out = 0
        self.n_before, self.share_before, self.kin, self.payoff_c, self.payoff_d = n, x, 0.0, 0.0, 0.0

    def record(self) -> dict:
        return {"id": self.id, "n": self.n, "territories": self.territories, "cooperation": self.x,
                "skill": self.skill, "technologies": self.tech, "store": self.store,
                "linked": "yes" if self.linked else "no", "gone": "yes" if self.n == 0 else "no",
                "n_prev": self.n_prev, "births": self.births, "deaths": self.deaths,
                "moved_in": self.moved_in, "moved_out": self.moved_out, "n_before": self.n_before,
                "share_before": self.share_before, "kin": self.kin, "payoff_c": self.payoff_c,
                "payoff_d": self.payoff_d}


def _round(x: float) -> int:
    return int(math.floor(x + 0.5))


class _World:
    """The mutable state of one run."""

    def __init__(self, rng: random.Random, p: dict) -> None:
        self.rng, self.p = rng, p
        self.territory = p["land"] / TERRITORIES
        self.groups: list[_Group] = []
        self.gone: list[_Group] = []
        self.events: list[dict] = []
        self.next_id = 0
        self.conflicts = 0
        self.climate = 1.0
        self.cut = False
        self.ledger = dict.fromkeys(("produced", "eaten", "stored", "lost"), 0.0)
        first = 3 + int(6 * rng.random())
        for _ in range(first):
            x = 0.2 + 0.6 * rng.random()
            k = capacity(self.territory, 0, x)
            n = max(2 * MIN_GROUP, int(k * (0.4 + 0.4 * rng.random())))
            g = self.new_group(n, 1, x, 0.0, 0.0)
            g.n_prev = n
            self.groups.append(g)

    def new_group(self, n: int, territories: int, x: float, skill: float, store: float) -> _Group:
        g = _Group(self.next_id, n, territories, x, skill, store)
        self.next_id += 1
        return g

    @property
    def free(self) -> int:
        return TERRITORIES - sum(g.territories for g in self.groups)

    def cap(self, g: _Group) -> float:
        return capacity(g.territories * self.territory, g.tech, g.x)

    def log(self, step: int, kind: str, a: _Group, b: _Group | None = None, moved: int = 0,
            killed_a: int = 0, killed_b: int = 0) -> None:
        self.events.append({"step": step, "kind": kind, "a": a.id, "b": -1 if b is None else b.id,
                            "moved": moved, "killed_a": killed_a, "killed_b": killed_b})

    # --- what happens between groups, once per step -------------------------------------------

    def begin(self, step: int) -> None:
        rng, p = self.rng, self.p
        for g in self.groups:
            g.n_prev, g.busy = g.n, False
            g.births = g.deaths = g.moved_in = g.moved_out = 0
        self.gone = []
        for key in self.ledger:
            self.ledger[key] = 0.0
        self.climate = 1.0
        if rng.random() < CLIMATE_RATE:
            self.climate = CLIMATE_RANGE[0] + (CLIMATE_RANGE[1] - CLIMATE_RANGE[0]) * rng.random()
        self.cut = p["isolation"] > 0 and step * STEP_YEARS > p["isolation"]
        for g in self.groups:
            chance = 0.0 if self.cut else min(1.0, p["connectedness"] * (LINK_BOATS if g.tech >= BOATS else 1.0))
            g.linked = rng.random() < chance
        linked = [g for g in self.groups if g.linked]
        if len(linked) < 2:
            for g in linked:
                g.linked = False
            linked = []
        self.migrate(linked)
        for g in self.groups:
            noise = DRIFT * (2.0 * rng.random() - 1.0) * math.sqrt(g.x * (1.0 - g.x) / g.n)
            g.x = min(1.0 - TREMBLE, max(TREMBLE, g.x + noise))
        self.bud(step)
        self.meet(step)
        self.split(step)
        for g in self.groups:
            g.tech = technologies(g.skill, hierarchy_levels(g.n))

    def migrate(self, linked: list[_Group]) -> None:
        """Every linked group sends 2 % of its people to the next linked group (in the order of
        the ring); they bring their share of cooperators along."""
        if not linked:
            return
        out = [int(MIGRATION * g.n) for g in linked]
        shares = [g.x for g in linked]
        for i, g in enumerate(linked):
            came, came_x = out[i - 1], shares[i - 1]
            stay = g.n - out[i]
            g.moved_out += out[i]
            g.moved_in += came
            g.n = stay + came
            g.x = (shares[i] * stay + came_x * came) / g.n

    def bud(self, step: int) -> None:
        """A crowded group sends half its people to a free territory."""
        for g in list(self.groups):
            if self.free <= 0:
                break
            if g.n >= 2 * MIN_GROUP and g.n >= BUD_SHARE * self.cap(g):
                moved = g.n // 2
                share = moved / g.n
                child = self.new_group(moved, 1, g.x, g.skill, g.store * share)
                child.moved_in, child.linked, child.busy = moved, g.linked, True
                g.store -= child.store
                g.n -= moved
                g.moved_out += moved
                self.groups.insert(self.groups.index(g) + 1, child)
                self.log(step, "bud", g, child, moved)

    def meet(self, step: int) -> None:
        """Neighbours in the ring may fight; linked farming neighbours may unite."""
        rng = self.rng
        ring = list(self.groups)
        if len(ring) < 2:
            return
        for i, a in enumerate(ring):
            b = ring[(i + 1) % len(ring)]
            if a is b or a.busy or b.busy or a.n == 0 or b.n == 0:
                continue
            crowd = max(min(1.0, a.n / self.cap(a)), min(1.0, b.n / self.cap(b)))
            both = a.linked and b.linked
            chance = WAR_RATE * crowd * (0.5 if both else 1.0)
            u = rng.random()
            if u < chance:
                a.busy = b.busy = True
                self.fight(step, a, b)
            elif both and a.tech >= FARMING and b.tech >= FARMING and u < chance + MERGE_RATE * a.x * b.x:
                a.busy = b.busy = True
                big, small = (a, b) if a.n >= b.n else (b, a)
                moved = small.n
                self.absorb(big, small)
                self.log(step, "merge", big, small, moved)

    def fight(self, step: int, a: _Group, b: _Group) -> None:
        self.conflicts += 1
        a_wins = self.rng.random() < win_probability(a.n, a.x, b.n, b.x)
        winner, loser = (a, b) if a_wins else (b, a)
        killed_w, killed_l = int(WINNER_DEATHS * winner.n), int(LOSER_DEATHS * loser.n)
        winner.n -= killed_w
        winner.deaths += killed_w
        loser.n -= killed_l
        loser.deaths += killed_l
        if winner.tech >= FARMING:  # farmers take the land and the people; foragers only the store
            moved = loser.n
            self.absorb(winner, loser)
            self.log(step, "conquest", winner, loser, moved, killed_w, killed_l)
        else:
            winner.store += loser.store
            loser.store = 0.0
            self.log(step, "raid", winner, loser, 0, killed_w, killed_l)

    def absorb(self, into: _Group, other: _Group) -> None:
        total = into.n + other.n
        into.x = (into.x * into.n + other.x * other.n) / total
        into.skill = max(into.skill, other.skill)
        into.moved_in += other.n
        other.moved_out += other.n
        into.n = total
        into.territories += other.territories
        into.store += other.store
        other.n, other.territories, other.store = 0, 0, 0.0
        self.groups.remove(other)
        self.gone.append(other)

    def split(self, step: int) -> None:
        """A group of several territories whose cooperation has failed breaks in two."""
        for g in list(self.groups):
            if g.territories < 2 or g.x >= SPLIT_BELOW:
                continue
            part = g.territories // 2
            moved = g.n * part // g.territories
            if moved < MIN_GROUP or g.n - moved < MIN_GROUP:
                continue
            child = self.new_group(moved, part, g.x, g.skill, g.store * part / g.territories)
            child.moved_in, child.linked, child.busy = moved, g.linked, True
            g.store -= child.store
            g.n -= moved
            g.territories -= part
            g.moved_out += moved
            self.groups.insert(self.groups.index(g) + 1, child)
            self.log(step, "split", g, child, moved)

    # --- one generation inside every group ------------------------------------------------------

    def generation(self, step: int) -> None:
        p = self.p
        people = 0 if self.cut else sum(g.n for g in self.groups)
        best = max((g.skill for g in self.groups if g.linked), default=0.0)
        for g in list(self.groups):
            n = g.n
            # cooperation: expected payoffs of the public-goods game among kin, one replicator step
            kin = kin_relatedness(p["relatedness"], n)
            with_c = 1.0 + (GAME_SIZE - 1) * (kin + (1.0 - kin) * g.x)
            with_d = (GAME_SIZE - 1) * (1.0 - kin) * g.x
            pay_c = payoff_cooperator(GAME_SIZE, with_c, p["multiplier"], p["stake"], p["punishment"])
            pay_d = payoff_defector(GAME_SIZE, with_d, p["multiplier"], p["stake"], p["punishment"])
            g.n_before, g.share_before, g.kin, g.payoff_c, g.payoff_d = n, g.x, kin, pay_c, pay_d
            g.x = tremble(replicator_step(g.x, pay_c, pay_d, REPLICATOR_RATE))
            # skill: Henrich's change with the loss scaled by complexity and teaching fidelity
            fid = teaching_fidelity(p["success"], p["words"], p["compositional"], 1 if g.tech >= WRITING else 0)
            pupils = learners(n, max(0, people - n), p["connectedness"])
            skill = max(0.0, g.skill + skill_change(effective_alpha(p["alpha"], g.skill, fid), p["beta"], pupils))
            if g.linked and best > skill:
                skill += DIFFUSE * (best - skill)
            g.skill = skill
            # food: produced = eaten + stored + lost
            k = capacity(g.territories * self.territory, g.tech, g.x) * self.climate
            produced = food_produced(n, k, g.tech)
            available = produced + g.store
            fed = min(n, int(available))
            rest = available - fed
            store = STORE_KEEP[1 if g.tech >= POTTERY else 0] * rest
            self.ledger["produced"] += produced
            self.ledger["eaten"] += fed
            self.ledger["stored"] += store - g.store
            self.ledger["lost"] += rest - store
            g.store = store
            # people: the hungry die; logistic growth of the fed toward the capacity
            target = max(0, _round(logistic_step(fed, p["growth"], k)))
            old = _round(MORTALITY * fed)
            if target >= fed - old:
                g.births += target - (fed - old)
                g.deaths += (n - fed) + old
            else:
                g.deaths += n - target
            g.n = target
            if g.n < MIN_GROUP:
                g.deaths += g.n
                g.n = 0
                self.ledger["stored"] -= g.store
                self.ledger["lost"] += g.store
                g.store, g.territories = 0.0, 0
                self.groups.remove(g)
                self.gone.append(g)
                self.log(step, "extinct", g)
            else:
                g.tech = technologies(g.skill, hierarchy_levels(g.n))

    # --- what a step shows --------------------------------------------------------------------

    def snapshot(self, step: int, history: list[int]) -> tuple[dict, list[dict]]:
        groups = self.groups
        people = sum(g.n for g in groups)
        top = max(groups, key=lambda g: g.n, default=None)  # the first of equally large groups
        largest = top.n if top else 0
        levels = hierarchy_levels(largest)
        tech = max((g.tech for g in groups), default=0)
        recent = history[-COLLAPSE_WINDOW:]
        collapsed = people == 0 or (bool(recent) and people < COLLAPSE_SHARE * max(recent))
        every = groups + self.gone
        state = {
            "year": step * STEP_YEARS,
            "population": people,
            "groups": len(groups),
            "largest_group": largest,
            "cooperation": math.fsum(g.x * g.n for g in groups) / people if people else 0.0,
            "skill": math.fsum(g.skill * g.n for g in groups) / people if people else 0.0,
            "technologies": tech,
            "farming": "yes" if tech >= FARMING else "no",
            "writing": "yes" if tech >= WRITING else "no",
            "hierarchy": levels,
            "roles": role_count(int(top.store)) if top else 0,
            "gini": gini(*wealth_fifths(top.n, top.store, levels)) if top else 0.0,
            "trade": sum(1 for g in groups if g.linked) / len(groups) if groups else 0.0,
            "conflicts": self.conflicts,
            "stage": "collapse" if collapsed else stage_of(largest),
            "births": sum(g.births for g in every),
            "deaths": sum(g.deaths for g in every),
            "moved": sum(g.moved_in for g in every),
            "produced": self.ledger["produced"], "eaten": self.ledger["eaten"],
            "stored": self.ledger["stored"], "lost": self.ledger["lost"],
            "store": math.fsum(g.store for g in groups),
            "free_land": self.free,
            "climate": self.climate,
        }
        return state, [g.record() for g in every]


class Society:
    """The simulation and its gate. Rollouts replayed by :meth:`check` are cached per seed."""

    sim = SIM
    topic = SIM
    KEYS = KEYS

    def __init__(self) -> None:
        self._cache: dict[int, SocietyRollout] = {}

    def run(self, seed: int, land: float = 3000.0, relatedness: float = 0.25, benefit: float = 4.0,
            cost: float = 1.0, punishment: float = 0.5, alpha: float = 3.0, beta: float = 1.0,
            connectedness: float = 0.6, success: float = 0.9, words: int = 20, compositional: float = 1.0,
            isolation: int = 0, growth: float = 0.25, rules: int = 7) -> SocietyRollout:
        """Deterministic for a given seed and parameters (``random.Random(seed)``, only its
        ``random()``). Raises ``ValueError`` for parameters outside their meaning."""
        _rules(rules)
        numbers = {"land": land, "relatedness": relatedness, "benefit": benefit, "cost": cost,
                   "punishment": punishment, "alpha": alpha, "beta": beta, "connectedness": connectedness,
                   "success": success, "compositional": compositional, "growth": growth}
        p: dict[str, float | int | str] = {}
        for key, value in numbers.items():
            if isinstance(value, (str, bool)):
                raise ValueError(f"{key} must be a number: {value!r}")
            value = float(value)
            if not math.isfinite(value):
                raise ValueError(f"{key} must be a finite number: {value!r}")
            p[key] = value
        if not (p["land"] >= LAND_MIN):
            raise ValueError(f"land must feed at least {LAND_MIN:g} people: {land!r}")
        for key in ("relatedness", "connectedness", "success", "compositional"):
            if not 0.0 <= p[key] <= 1.0:
                raise ValueError(f"{key} must be between 0 and 1: {p[key]!r}")
        for key in ("benefit", "cost", "alpha", "beta"):
            if not p[key] > 0:
                raise ValueError(f"{key} must be positive: {p[key]!r}")
        if p["punishment"] < 0 or not 0 < p["growth"] <= 1:
            raise ValueError(f"punishment must not be negative, growth must be in (0, 1]: {punishment!r}, {growth!r}")
        if int(words) != words or words < 0 or int(isolation) != isolation or isolation < 0:
            raise ValueError(f"words and isolation must be whole numbers, not negative: {words!r}, {isolation!r}")
        p["words"], p["isolation"] = int(words), int(isolation)
        fidelity = teaching_fidelity(p["success"], p["words"], p["compositional"], 0)
        p["hamilton"] = hamilton(p["relatedness"], p["benefit"], p["cost"])
        p["fidelity"] = fidelity
        p["critical_population"] = critical_population(effective_alpha(p["alpha"], 0.0, fidelity), p["beta"])
        p["doubling_years"] = doubling_years(math.log1p(p["growth"]) / GENERATION_YEARS)
        p["stake"] = stake_of(p["benefit"], p["cost"], GAME_SIZE)
        p["multiplier"] = multiplier_of(p["benefit"], p["cost"], GAME_SIZE)
        ordered = {k: p[k] for k in PARAM_KEYS + HIDDEN_PARAM_KEYS}

        world = _World(random.Random(seed), ordered)
        r = SocietyRollout(SIM, int(seed), ordered, [])
        history: list[int] = []
        state, groups = world.snapshot(0, history)
        r.steps.append(state)
        r.groups.append(groups)
        history.append(state["population"])
        for step in range(1, STEPS + 1):
            world.begin(step)
            for _ in range(GENERATIONS):
                world.generation(step)
            state, groups = world.snapshot(step, history)
            r.steps.append(state)
            r.groups.append(groups)
            history.append(state["population"])
        r.events = world.events
        last = r.steps[-1]
        peak = max(history)
        farming = next((s["year"] for s in r.steps if s["farming"] == "yes"), "never")
        writing = next((s["year"] for s in r.steps if s["writing"] == "yes"), "never")
        ended_low = last["stage"] == "collapse" or last["population"] < COLLAPSE_SHARE * peak
        r.summary = {
            **{k: last[k] for k in ("stage", "population", "groups", "cooperation", "technologies")},
            "technology_list": " ".join(TECH_NAMES[:last["technologies"]]) or "none",
            **{k: last[k] for k in ("farming", "writing", "hierarchy", "roles", "gini", "conflicts")},
            "outcome": "collapse" if ended_low else last["stage"],
            "first_farming": farming,
            "first_writing": writing,
        }
        return r

    def rollout(self, seed: int) -> SocietyRollout:
        """The canonical run of a seed: parameters from ``random_params(random.Random(seed))``."""
        if seed not in self._cache:
            if len(self._cache) >= 256:
                self._cache.clear()
            self._cache[seed] = self.run(seed, **random_params(random.Random(seed)))
        return self._cache[seed]

    # --- the gate -----------------------------------------------------------------------------

    def conserved(self, r: Rollout) -> Verdict:
        """People, food, land, payoffs, shares, the order of the technologies, hierarchy and stage."""
        p = r.params
        every = getattr(r, "groups", None) or []
        events = getattr(r, "events", [])
        if len(every) != len(r.steps) or not r.steps:
            return Verdict(False, None, "the groups of every step are missing")
        prev = None
        history: list[int] = []
        for t, s in enumerate(r.steps):
            rec = every[t]
            alive = [g for g in rec if g["gone"] == "no"]
            sizes = [g["n"] for g in alive]
            # counts
            if s["year"] != t * STEP_YEARS:
                return Verdict(False, num(t * STEP_YEARS), f"step {t}: year is not 250 times the step")
            if (s["population"], s["groups"], s["largest_group"]) != (sum(sizes), len(alive), max(sizes, default=0)):
                return Verdict(False, num(sum(sizes)),
                               f"step {t}: population, groups or largest_group do not match the groups")
            if any(n < MIN_GROUP for n in sizes) or any(g["n"] != 0 for g in rec if g["gone"] == "yes"):
                return Verdict(False, None,
                               f"step {t}: a living group below {MIN_GROUP} people or a dead one with people")
            # shares
            shares = [s["cooperation"], s["trade"], s["gini"]] + [g["cooperation"] for g in alive]
            if not all(is_finite(x) and 0.0 <= x <= 1.0 for x in shares):
                return Verdict(False, None, f"step {t}: a share outside 0 to 1")
            # people: births - deaths + moved in - moved out, per group and in all
            for g in rec:
                if min(g["births"], g["deaths"], g["moved_in"], g["moved_out"]) < 0:
                    return Verdict(False, None, f"step {t}: group {g['id']} has a negative count")
                if g["n"] != g["n_prev"] + g["births"] - g["deaths"] + g["moved_in"] - g["moved_out"]:
                    return Verdict(False, None, f"step {t}: the people of group {g['id']} do not add up")
            if s["births"] != sum(g["births"] for g in rec) or s["deaths"] != sum(g["deaths"] for g in rec):
                return Verdict(False, None, f"step {t}: births or deaths do not match the groups")
            if not s["moved"] == sum(g["moved_in"] for g in rec) == sum(g["moved_out"] for g in rec):
                return Verdict(False, None, f"step {t}: people who moved out did not all arrive")
            if prev is None:
                if s["births"] or s["deaths"] or s["moved"] or any(g["n_prev"] != g["n"] for g in rec):
                    return Verdict(False, None, "step 0: nothing has happened yet")
            else:
                before = {g["id"]: g["n"] for g in every[t - 1] if g["gone"] == "no"}
                if any(g["n_prev"] != before.get(g["id"], 0) for g in rec):
                    return Verdict(False, None, f"step {t}: a group does not start from where it stood")
                if set(before) - {g["id"] for g in rec}:
                    return Verdict(False, None, f"step {t}: a group vanished without a record")
                if s["population"] != prev["population"] + s["births"] - s["deaths"]:
                    return Verdict(False, num(prev["population"] + s["births"] - s["deaths"]),
                                   f"step {t}: population is not the last one plus births minus deaths")
            # food: produced = eaten + stored + lost
            scale = max(1.0, abs(s["produced"]))
            if abs(s["produced"] - (s["eaten"] + s["stored"] + s["lost"])) > 1e-6 * scale:
                return Verdict(False, None, f"step {t}: food produced is not eaten plus stored plus lost")
            if min(s["produced"], s["eaten"], s["store"]) < 0 or s["lost"] < -1e-6 * scale:
                return Verdict(False, None, f"step {t}: negative food")
            if abs(s["store"] - math.fsum(g["store"] for g in alive)) > 1e-6 * scale:
                return Verdict(False, None, f"step {t}: the stores of the groups do not add up to the store")
            if prev is not None and abs(s["store"] - prev["store"] - s["stored"]) > 1e-6 * max(scale, prev["store"]):
                return Verdict(False, None, f"step {t}: the store did not change by what was stored")
            # land
            if s["free_land"] + sum(g["territories"] for g in alive) != TERRITORIES or s["free_land"] < 0 \
                    or any(g["territories"] < 1 for g in alive):
                return Verdict(False, None, f"step {t}: the territories do not add up to {TERRITORIES}")
            if not (s["climate"] == 1.0 or CLIMATE_RANGE[0] <= s["climate"] <= CLIMATE_RANGE[1]):
                return Verdict(False, None, f"step {t}: climate outside its range")
            # payoffs of the last generation, recomputed, and the replicator step that followed
            if prev is not None:
                for g in alive:
                    kin = kin_relatedness(p["relatedness"], g["n_before"])
                    x = g["share_before"]
                    pay_c = payoff_cooperator(GAME_SIZE, 1.0 + (GAME_SIZE - 1) * (kin + (1.0 - kin) * x),
                                              p["multiplier"], p["stake"], p["punishment"])
                    pay_d = payoff_defector(GAME_SIZE, (GAME_SIZE - 1) * (1.0 - kin) * x,
                                            p["multiplier"], p["stake"], p["punishment"])
                    if not (abs(kin - g["kin"]) <= 1e-12 and abs(pay_c - g["payoff_c"]) <= 1e-9 * max(1.0, abs(pay_c))
                            and abs(pay_d - g["payoff_d"]) <= 1e-9 * max(1.0, abs(pay_d))):
                        return Verdict(False, None,
                                       f"step {t}: the payoffs of group {g['id']} are not those of the game")
                    if abs(tremble(replicator_step(x, pay_c, pay_d, REPLICATOR_RATE)) - g["cooperation"]) > 1e-12:
                        return Verdict(False, None,
                                       f"step {t}: the cooperation of group {g['id']} is not one replicator step on")
            # technologies in their order, hierarchy by size
            for g in alive:
                if g["technologies"] != technologies(g["skill"], hierarchy_levels(g["n"])) or g["skill"] < 0:
                    return Verdict(False, None,
                                   f"step {t}: the technologies of group {g['id']} do not follow from its skill")
            tech = max((g["technologies"] for g in alive), default=0)
            if s["technologies"] != tech or s["farming"] != ("yes" if tech >= FARMING else "no") \
                    or s["writing"] != ("yes" if tech >= WRITING else "no"):
                return Verdict(False, num(tech), f"step {t}: technologies, farming or writing do not match the groups")
            levels = hierarchy_levels(s["largest_group"])
            if s["hierarchy"] != levels:
                return Verdict(False, num(levels), f"step {t}: hierarchy does not fit the size of the largest group")
            top = max(alive, key=lambda g: g["n"], default=None)
            roles = role_count(int(top["store"])) if top else 0
            g_top = gini(*wealth_fifths(top["n"], top["store"], levels)) if top else 0.0
            if s["roles"] != roles or abs(s["gini"] - g_top) > 1e-12:
                return Verdict(False, None, f"step {t}: roles or gini do not follow from the largest group")
            recent = history[-COLLAPSE_WINDOW:]
            collapsed = s["population"] == 0 or (bool(recent) and s["population"] < COLLAPSE_SHARE * max(recent))
            stage = "collapse" if collapsed else stage_of(s["largest_group"])
            if s["stage"] != stage:
                return Verdict(False, stage, f"step {t}: the stage does not fit size and history")
            fights = sum(1 for e in events if e["kind"] in ("raid", "conquest") and e["step"] <= t)
            if s["conflicts"] != fights:
                return Verdict(False, num(fights), f"step {t}: conflicts do not match the log")
            history.append(s["population"])
            prev = s
        summary = r.summary
        if summary:
            names = [] if summary["technology_list"] == "none" else summary["technology_list"].split()
            if tuple(names) != TECH_NAMES[:len(names)] or len(names) != summary["technologies"]:
                return Verdict(False, None, "the technologies at the end are not in their order")
        return Verdict(True, None,
                       "people, food, land, payoffs, shares, technologies, hierarchy and stage are consistent")

    # --- questions ------------------------------------------------------------------------------

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
        if not isinstance(step, int) or not 0 <= step <= STEPS or num(step).split() != digits:
            return None
        if len(rest) == 2 and rest[0] == "next" and rest[1] in STATE_KEYS + LEDGER_KEYS and step < STEPS:
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
        """Replay the seed and compare: words and counts exactly, measured numbers (and the number
        of people) within 5 %."""
        value = self.truth(prompt)
        if value is None:
            return Verdict(False, None, "not my question")
        words = prompt.split()
        key = words[-1]
        expected = param_value(key, value) if words[-2] == "params" else dense_value(key, value)
        if isinstance(value, str):
            ok = answer == value
            return Verdict(ok, expected, "exact word" if ok else "wrong word")
        got = parse_num(answer)
        if not is_finite(got):  # also "never" where a year is the truth, and "1 e 9 9 9"
            return Verdict(False, expected, "not a number")
        if key in COUNT_KEYS:
            ok = got == value
            return Verdict(ok, expected, "exact count" if ok else "wrong count")
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
    """``society seed 3 params. land 3 9 6 0. relatedness 0 point 2 9. ... doubling_years 1 0 3.``"""
    fields = " ".join(f"{k} {param_value(k, r.params[k])}." for k in PARAM_KEYS)
    return Line(f"{r.sim} seed {num(r.seed)} params. {fields}", topic=r.sim, kind="record")


def state_line(r: Rollout, t: int) -> Line:
    """``society seed 3 step 1 2. year 3 0 0 0. population 7 1 2 0. ... stage tribes.``"""
    step = r.steps[t]
    fields = " ".join(f"{k} {dense_value(k, step[k])}." for k in STATE_KEYS)
    return Line(f"{r.sim} seed {num(r.seed)} step {num(t)}. {fields}", topic=r.sim, kind="record")


def query_lines(r: Rollout, t: int) -> list[Line]:
    """One question per metric at step ``t``, and what it is one step (250 years) later."""
    head = f"{r.sim} seed {num(r.seed)} step {num(t)}"
    step = r.steps[t]
    out = [Line(f"{head} {k}", dense_value(k, step[k]), r.sim, "fact") for k in STATE_KEYS]
    if t + 1 < len(r.steps):
        nxt = r.steps[t + 1]
        out += [Line(f"{head} next {k}", dense_value(k, nxt[k]), r.sim, "calc") for k in STATE_KEYS]
    return out


def summary_lines(r: Rollout) -> list[Line]:
    return [Line(f"{r.sim} seed {num(r.seed)} final {k}", dense_value(k, r.summary[k]), r.sim, "calc")
            for k in SUMMARY_KEYS]


def lines(r: Rollout, every: int = 1) -> list[Line]:
    """The parameter record, a state record and questions for every ``every``-th step (now and
    next), and the final questions. The ledger is left out: it is the gate's, not the model's."""
    out = [params_line(r)]
    for t in range(0, len(r.steps), every):
        out.append(state_line(r, t))
        out += query_lines(r, t)
    return out + summary_lines(r)


_SIM = Society()


def simulation() -> Society:
    return _SIM


def run(seed: int, **params) -> SocietyRollout:
    return _SIM.run(seed, **params)


def rollout(seed: int) -> SocietyRollout:
    return _SIM.rollout(seed)


def conserved(r: Rollout) -> Verdict:
    return _SIM.conserved(r)


def check(prompt: str, answer: str) -> Verdict:
    """Judge an answer to a question about a rollout (``society seed ...``). Lessons
    (``society predict ...``) are judged by :data:`LESSONS`."""
    return _SIM.check(prompt, answer)


def owns(prompt: str) -> bool:
    return _SIM.owns(prompt)
