"""Level 5 of the toy universe: molecules that copy themselves.

A pool of free monomers of four kinds (``a``, ``c``, ``g``, ``u``) and a vessel of polymers,
strings over those letters. Each step, in this order:

1. **Inflow**: ``inflow`` fresh monomers enter the pool, the same number of each kind.
2. **Decay**: a polymer of length ``L`` falls apart with probability ``min(1, decay / L)`` (long
   chains are sturdier; with ``decay`` 3.2 every string of 2 or 3 letters breaks and a 4-letter
   replicator dies four times in five, so short lineages can go extinct); its letters go back to
   the pool.
3. **Polymerisation**: a share ``poly_rate`` of the free monomers joins into random strings of
   2 to 16 letters (length 1 + geometric(0.3), mean 4.3), letters drawn from the pool without
   replacement, so a scarce kind is scarce in the strings too.
4. **Replication**: every polymer that carries the motif copies itself once, in random order,
   if the pool still holds the letters the copy needs (a copy with no free letters fails, so
   when monomers are scarce the replicators compete for them). Each letter of a copy is
   replaced by one of the other three letters with probability ``mu``, which can destroy the
   motif or, rarely, create one. The copy's letters leave the pool; the template is not used up.

The motif is ``ag`` followed by ``cu`` with at most ``gap`` letters in between (``gap`` 0 is the
strict ``agcu``, ``gap`` 3 the loose ``ag...cu``); a random 4-letter string has it with
probability 1/256, so the first replicator is a chance event of the soup. A motif needs all four
kinds of letter, which is why a lineage that eats up one kind starves itself.

What is real and what is toy. Real in kind, though not in numbers: matter is conserved (every
letter is accounted for, kind by kind); copies are made from a template with errors, so there
is heredity and mutation; lineages differ in survival and compete for a limited supply, so there
is selection; growth is exponential while letters are plentiful and stops when they run out;
which lineage founds the population is chance. Toy: the four letters only borrow the names of
the RNA bases, there is no chemistry behind them (no base pairing, a copy is identical instead
of complementary, no energy, no temperature, no space, no reaction rates, one synchronous step
for everything). That a string with ``agcu`` can copy itself is declared, not derived: no real
4-letter RNA does that, known ribozymes that copy RNA have well over a hundred letters and none
copies itself completely. Decay falling with length is invented to give selection something to
work on; real long chains have more bonds to break and hydrolyse sooner. The rates are chosen
so that something happens within 40 steps; they are not measured, and nobody knows the real
odds of a first replicator.

Gate. :meth:`Life.conserved` checks every step: ``free_monomers + bound == monomers + inflow *
step`` exactly (decayed letters are back in ``free_monomers``); no negative count; ``diversity
<= replicators <= polymers``; the most common of ``d`` sequences holds at least ``1 / d`` of the
replicators; ``mean_length == bound / polymers``; no step makes more copies than there are
templates; the stage follows from the numbers; the summary follows from the steps.
:func:`simulate` also checks the ledger of each kind of letter while it runs. :meth:`Life.check`
replays the seed and compares: words, the step, parameters and ``first_replicator_step``
exactly; a count of 0 exactly (0 against 1 is soup against life); other counts within 1 or
5 percent; other numbers within 5 percent.

The vessel holds at most :data:`MAX_POLYMERS` polymers (a runtime guard for hand-set
parameters; with :func:`random_params` the letter supply, at most 20000 + 400 * 40, runs out
first).

Lines (numbers as digits, metrics rounded to 3 significant digits; these are seed 7 verbatim,
the state record being one line of 55 tokens; the longest possible record of a seed below
10000 has 75, see the tests):

    life seed 7 params. monomers 1 0 3 0 5. inflow 2 4 0. poly_rate 0 point 0 0 2. decay 0 point 1. mu 0 point 0 2. gap 0.
    life seed 7 step 3 0. free_monomers 1 6 7 8 6. bound 7 1 9. polymers 1 5 8. replicators 2 8. longest 1 5.
        mean_length 4 point 5 5. diversity 2. dominant_share 0 point 9 6 4. generations 3 2. stage dominated.
    q life seed 7 step 2 6 replicators. a 2.
    q life seed 7 step 2 6 next replicators. a 4.                 (one step later)
    q life seed 7 step 3 0 stage. a dominated.
    q life seed 7 final first_replicator_step. a 2 3.
    q life seed 7 final outcome. a dominated.
    q life seed 7 param mu. a 0 point 0 2.

Stages (each follows from the record and the one before it): ``soup`` (no replicator yet),
``first_replicator`` (the step the first one appears), ``dominated`` (at least 10 replicators
and ``dominant_share`` at least 0.5), ``extinct`` (there were replicators, none is left; a new
one can still arise from the soup), ``competition`` (some copies failed for lack of letters, or
the replicators fell), ``growth`` (every copy found its letters and the replicators did not
fall). Outcomes: ``none``, ``coexist``, ``dominated``, ``extinct``.

Plausibility targets, with the values measured on this code (tests/test_cosmos_life.py holds
each of them with a margin):

* the ledger holds on every seed, in total and for each kind of letter (seeds 1 to 40, 50 to 79
  and 1000 to 1299: all);
* a first replicator is a chance event that most vessels see: 39 of 40 seeded runs, first at
  step 1 to 37, median 6;
* small ``mu`` (0.001) with ``inflow`` 200: one lineage dominates in most runs (25 of 30);
* large ``mu`` (0.05) with ``inflow`` 200: the copies drift apart, at least 20 distinct
  replicators or none at all (26 of the 29 runs that had one), few runs dominated (1 of 30);
* no inflow and no decay: the letters end up locked in polymers, one kind runs out and the
  replicators stop multiplying (the same number over the last five steps in 28 of the 28 runs
  that had one). 19 of them make no copy at all in those steps; in the others a few mutated
  copies that need no letter of the missing kind still find theirs, at most 2.2 percent of the
  replicators in the last step, and none of those carries the motif;
* ``decay`` 3.2 with the strict motif and ``poly_rate`` 0.002: extinct, dominated and none all
  occur (18, 11 and 15 of 60);
* selection for length: with ``decay`` 3.2 the replicators at the end have 12.5 letters on
  average, with ``decay`` 0.1 they have 7.7 (30 seeds each, ``gap`` 0);
* growth is doubling while the letters last: with ``decay`` 0.1 the replicators grow by a
  median factor of 1.95 per step in stage ``growth`` from 10 replicators on (every replicator
  copies once, a few copies lose the motif, a few polymers decay).

Reproducibility. All randomness comes from the raw 64-bit output of ``numpy.random.PCG64(seed)``
(the stream numpy guarantees across versions), turned into uniform numbers, lengths, draws
without replacement and orders by integer arithmetic, exact float steps and stable sorts in this
module. No ``Generator`` distribution, shuffle or permutation is used (numpy may change those
between versions) and nothing that rounds differently from one maths library to the next, such
as ``exp`` or ``log``. The tests pin the stream and three rollouts by
digest, so a machine that computes anything else fails loudly instead of judging wrongly.

Rules 7
-------

Everything above is round 6 and stays as it is: the version-5 models were trained on those
rollouts, and without ``rules`` every output is bit for bit what it was. A review found several
of the rules above wrong or arbitrary. ``run(seed, rules=7)`` runs a corrected vessel
(``random_params(rng, rules=7)``, ``Life().rollout(seed, rules=7)``, ``Life().generate(rng, n,
rules=7)``; :func:`lines`, :meth:`Life.conserved`, :meth:`Life.check` and :meth:`Life.owns` see
which rules a rollout or a prompt has). Such a rollout has ``"rules": 7`` in its ``params``,
:data:`STEPS7` = 80 steps, the parameters of :data:`PARAM_KEYS7` (``hydrolysis`` in place of
``decay``; the other five are the same numbers for the same seed), the metrics of :data:`KEYS7`
and the summary of :data:`SUMMARY_KEYS7`. Its lines say ``rules 7`` after the seed, so that no
prompt has two truths; a prompt without it means round 6. These are seed 7 verbatim (the state
record is one line of 85 tokens; no record of a seed below 10000 with seeded parameters can
have more than 117):

    life seed 7 rules 7 params. monomers 1 0 3 0 5. inflow 2 4 0. poly_rate 0 point 0 0 2. hydrolysis 0 point 0 1. mu 0 point 0 2. gap 0.
    life seed 7 rules 7 step 3 4. free_monomers 1 5 4 6 7. spent 9 7. bound 2 9 0 1. polymers 5 9 6. replicators 2 1 5.
        longest 1 2. mean_length 4 point 8 7. replicator_length 6 point 9 8. diversity 2 4. lineages 4.
        dominant_share 0 point 6 8 8. effective_lineages 1 point 9. copies 2 4 8. failed 0. origins 4. stage growth.
    q life seed 7 rules 7 step 3 4 replicators. a 2 1 5.
    q life seed 7 rules 7 step 3 4 next replicators. a 3 2 4.
    q life seed 7 rules 7 step 4 0 stage. a competition.
    q life seed 7 rules 7 final first_replicator_step. a 2 0.
    q life seed 1 8 3 rules 7 final first_replicator_step. a never.
    q life seed 7 rules 7 final outcome. a dominated.
    q life seed 7 rules 7 final max_length. a 1 6.
    q life seed 7 rules 7 param hydrolysis. a 0 point 0 1.
    q life7 predict intact_probability length 6 hydrolysis 0 point 0 5 7. a 0 point 7 4 5 7.    (a lesson)
    life7 predict copy_probability. probability that a template of length letters finishes a copy in one step. 4 over length and never more than 1.

One step under rules 7, in this order: inflow; hydrolysis; polymerisation; replication. What was
corrected, the real thing each correction follows, and what in it is still toy:

* **Letters are not recycled for nothing** (:data:`KEYS7` ``spent``). In round 6 the letters of
  a decayed chain went back to the pool ready to use, so a closed vessel copied for ever: a
  perpetual motion machine. Joining nucleotides in water is uphill (hydrolysis of the bond
  releases roughly 20 to 25 kJ per mol; about 22 was measured for a bond of DNA, after Dickson,
  Burns and Richardson 2000, from memory), so real polymerisation and copying need
  activated monomers, and a hydrolysed monomer is not activated again without a source of
  energy. Now only the start supply and the inflow are activated; a letter set free by
  hydrolysis goes to ``spent`` and is never used again. The ledger of every kind of letter is
  ``free + spent + bound == start + inflow // 4 * step``. Toy: nothing recharges a spent
  monomer, and there is no energy in the model beyond this one bit per letter.
* **Hydrolysis per bond** (:func:`intact_probability`). Round 6 let a chain fall apart with
  probability ``decay / l``, the reverse of chemistry. Each of the ``l - 1`` bonds of a chain
  breaks on its own, so a chain stays whole with probability ``(1 - hydrolysis) ^ (l - 1)`` and
  long chains break sooner. (Without a catalyst a bond of RNA in neutral water at room
  temperature breaks of the order of 1e-7 times per minute, after Li and Breaker 1999, from
  memory: a bond lasts years. Most of these breaks are not hydrolysis in the strict sense: the
  2' hydroxyl group next to the bond attacks it, and water comes in afterwards. The toy calls
  every break hydrolysis.) A broken chain is cut at one bond: a piece of 2 or more letters
  stays in the vessel (a piece that still holds the motif is a shorter replicator of the same
  lineage), a single letter is spent. Toy: the values of ``hydrolysis`` (0.002 to 0.1 per bond
  and step, :data:`HYDROLYSIS`; the model gives a step no physical length), one break per chain
  and step at most, every bond equally likely. :func:`hydrolysis_at` is the hand-off from a
  planet's temperature (warmer water breaks bonds faster, that much is real; the table is toy).
* **Copying takes time** (:func:`copy_probability`). In round 6 a template of 16 letters copied
  as fast as one of 4. A polymerase adds one letter after the other, so short templates
  out-copy long ones: Qbeta RNA selected for fast copying lost 83 percent of its length in 74
  transfers (after Mills, Peterson and Spiegelman 1967). A template of ``l`` letters now
  finishes a copy within a step with probability ``min(1, 4 / l)``. Toy: :data:`COPY_SPEED` 4,
  which defines the step (the time a 4-letter template needs for one copy, no physical unit),
  and a chance per step in place of a copy that grows over several steps.
* **A copy is the complement** (:func:`reverse_complement`). Template copying works by base
  pairing, ``a`` with ``u`` and ``g`` with ``c``, and gives the opposite strand read backwards;
  the original comes back after two rounds. The motif ``ag`` ... ``cu`` is its own reverse
  complement up to its gap letters, so both strands carry it; this is why the declared motif
  is not wholly arbitrary. A copy uses up the letters of the complement, and ``diversity``
  counts a strand and its opposite strand as one sequence (:func:`canonical`). Still declared,
  not derived: that such a site makes a string copy itself, and that the strands come apart.
* **A length limit of Eigen's form** (:func:`motif_fidelity`, :func:`growth_factor`,
  :func:`max_length`). Eigen's condition is that a genome of ``n`` letters copied with accuracy
  ``q`` per letter persists when ``q ^ n`` times its advantage over its mutants is above 1
  (after Eigen 1971 and Eigen and Schuster 1977); beyond this error threshold the mutants take
  over. In this vessel only the four motif letters are selected, and a copy that loses the
  motif does not copy at all. Mutants that do not copy cannot take over, so there is no error
  threshold in Eigen's sense here; what is left is an extinction threshold (after Bull, Sanjuan
  and Wilke 2007): replicators of a kind die out when each leaves less than one of its kind per
  step. Copy errors alone never get them there, because the template is not used up: a spoilt
  copy leaves the count where it was (and ``(1 - mu) ^ 4`` is at least 0.81 for ``mu`` up to
  0.05). The limit that bites comes from hydrolysis, which has Eigen's form (a survival per
  bond in place of an accuracy per letter, raised to the length): a replicator of ``l``
  letters multiplies while
  ``(1 - hydrolysis) ^ (l - 1) * (1 + min(1, 4 / l) * (1 - mu) ^ 4) > 1``. The
  summary key ``max_length`` is the longest such ``l``: 16, 16, 16, 13, 8, 5 for ``hydrolysis``
  0.002 to 0.1 at ``mu`` 0.001, and 12, 7, 5 for 0.02, 0.05, 0.1 at ``mu`` 0.05. Eigen's
  paradox says: without enzymes no accurate copying and so no long genome, without a long
  genome no enzymes. Here the short reach comes from bonds that break before a slow copy is
  done, not from copy errors, but the bind is the same, and it is the reason for the next
  level. Toy: the numbers, and that pieces of broken chains are left out of the formula.
* **Polymerisation needs two monomers to meet** (:func:`polymer_letters`). Round 6 joined a
  fixed share of the pool per step at any concentration. The rate of a reaction between two
  molecules goes with the square of the concentration, so the share that joins is now
  ``poly_rate * free / 10000``: a thin soup makes far fewer strings. Toy: the reference of
  10000 monomers (:data:`POOL_REFERENCE`, it fixes the volume of the vessel) and the lengths of
  the strings, which stay 1 + geometric(0.3), cut off at 16. For chains of 2 or more letters
  that is the most probable distribution of a step-growth polymer (after Flory 1936) at an
  extent of reaction of 0.7, a toy value.
* **Descent is recorded** (``lineages``, ``origins``, ``dominant_share``,
  ``effective_lineages``, ``failed``). Every replicator that arises in the soup founds a
  lineage; copies and pieces inherit it. Round 6 called a vessel ``dominated`` when one whole
  sequence held half of the replicators, which measured the mutation rate and the time since
  founding, not competition (seeds 100 to 139 with ``inflow`` 200: ``dominated`` in 33 of 40
  runs at ``mu`` 0.001 and 1 of 40 at 0.05; under rules 7 with ``hydrolysis`` 0.005 it is 26 of
  40 at both).
  ``dominant_share`` is now the share of the largest lineage and ``effective_lineages`` the
  inverse Simpson index ``1 / sum(share_i ^ 2)`` (after Simpson 1949).
* **Stages and outcomes follow from the record** (:func:`stage`, :func:`outcome`). Stages:
  ``soup``, ``first_replicator``, ``extinct`` (none now, some before), ``competition`` (a copy
  found no letters in this step), ``decline`` (no copy failed and the replicators fell),
  ``growth``. Whether a copy failed is now in the record (``failed``). Outcomes, first match:
  ``none`` (never a replicator), ``fizzled`` (none left and never 10), ``extinct`` (none left,
  there were 10 or more), ``declining`` (at most half of the peak left), ``dominated`` (fewer
  than 2 effective lineages), ``coexist``. Toy: the thresholds 10, one half and 2, and the run
  length of 80 steps, chosen so that a closed vessel has time to run out.
* **Never is a word.** ``first_replicator_step`` is ``never`` when no replicator arose (round 6
  said ``minus 1``, which is no step), as the levels below say it.
* **Names.** ``copies`` in place of ``generations`` (a copy event is not a generation, and the
  word means generations of stars elsewhere), ``hydrolysis`` in place of ``decay``. The
  rules-7 keys say replicators, never life: a common working definition of life is a
  self-sustaining chemical system capable of Darwinian evolution (after Joyce 1994), and these
  strings have heredity and mutation but no metabolism and no compartment. The prefix ``life``
  of the lines is the name of the level, no claim.
* **The gate is strict about small whole numbers** (:func:`agree7`). ``longest`` is exact (it
  is at most 16); ``diversity``, ``lineages``, ``origins`` and ``failed`` are exact below 20
  (round 6 took 2 for 1 and 17 for a longest of 16); larger counts stay within 1 or 5 percent,
  0 is exact, ``never`` is a word.
* **The odds of an origin, as computed.** "1 in 256" above holds for strings of exactly four
  letters. Over the lengths of the soup a new string carries the motif once in 158.7, 93.7,
  75.0 and 67.0 strings for ``gap`` 0 to 3 (the exact sum over the lengths 2 to 16 with the
  four letters equally common), and such a string has 8.40, 8.77, 9.05 and 9.26 letters on
  average. An origin is not a single event here: a median of 5 per run, more than one in 278
  of 300 runs. All known life descends from one common ancestor; whether life arose more than
  once is not known, nobody knows the real odds, and ``gap`` and ``poly_rate`` are knobs that
  make something happen within a run.

Unchanged under rules 7: letters are drawn from the pool without replacement; each letter of a
copy is replaced independently by one of the other three with probability ``mu`` (the
assumption of Eigen's model and of Jukes and Cantor 1969); the template is not used up, copies
are served in random order and one that does not fit is skipped; :data:`MAX_LEN` and the motif
with its ``gap``. :data:`MAX_POLYMERS` stops new strings and copies (a copy it stops counts as
``failed``); pieces of broken chains can lift the count above it. Only hand-set parameters get
there: with the seeded ones no vessel of seeds 1 to 300 held more than 11871 polymers.

Lessons (:data:`LESSONS`, topic ``predict_life7``, prompts ``life7 predict <rule> <input> <value>
...``): the ten functions of :data:`RULES`, each the function the rules-7 run or its gate
calls, so a lesson is true of the simulation. The word is ``life7`` (:data:`SIM7`), as the other
corrected levels have ``planets7`` and ``nucleo7``: ``life predict ...`` and the topic
``predict_life`` belong to the round-6 lessons of ``haishool.cosmos.predict``, which teach the
round-6 rules (their ``survival_probability`` is the ``decay / l`` that rules 7 replaces, and
their ``next_total`` counts free and bound letters, without the spent ones), so no prompt and
no topic has two truths. :func:`conserved7` is the gate of a rules-7 rollout and :func:`check7`
replays a rules-7 prompt.

Plausibility targets under rules 7, measured on seeds 1 to 300 with their seeded parameters
unless said otherwise (tests/test_cosmos_life_rules7.py holds each with a margin on 40 seeds):

* the ledger with the spent pool holds on every seed, in total and for each kind of letter;
* no perpetual motion: every copy takes at least 4 activated letters, so ``4 * copies`` never
  exceeds the letters that entered. A closed vessel (``inflow`` 0, ``monomers`` 14361,
  ``poly_rate`` 0.01, ``gap`` 1, seeds 100 to 139) with ``hydrolysis`` 0.01 has a median of
  1547.5, 817.5, 441.5 and 233.5 replicators at steps 20, 40, 60 and 80, with 0.05 it has 1129,
  59, 2 and 0, and 73 of the 80 runs make no copy in the last ten steps, the others at most 3
  (of 1338 or more in the run); the same vessel under round 6 (``decay`` 1.6) still makes a
  median of 1696.5 copies in its last ten steps;
* outcomes: declining 135, dominated 96, coexist 44, extinct 21, none 3, fizzled 1. All 21
  extinct runs are closed vessels; with inflow the replicators fall from their peak to what
  the inflow feeds (declining 105, dominated 86, coexist 41 of 232);
* selection for short replicators: founders have a median of 8 letters; the replicators at the
  end are shorter than the founder in 253 of the 275 runs that have any, with a median length
  of 5.68, 5.27, 4.63, 4.16, 4.01 and 4.0 letters for ``hydrolysis`` 0.002 to 0.1;
* :func:`growth_factor` predicts the growth: the replicators of a step without failed copies,
  from 10 replicators on, grow by a median of 1.024 times the mean ``growth_factor`` of the
  replicators present (quartiles 0.999 and 1.069, 3554 steps; pieces of broken replicators that
  keep the motif and new origins are the surplus);
* nothing longer than ``max_length`` lasts: no replicator at the end is longer in any of the 275
  runs that have one (with ``inflow`` 200 and ``hydrolysis`` 0.1, seeds 100 to 139, the share
  that is longer is 0 in the median and 0.026 at most);
* a run takes 0.1 s on average (0.12 s on seeds 1 to 300) and 0.25 to 0.38 s at most on the
  machines it was measured on, some of them busy with other work.
"""

from __future__ import annotations

import functools
import random
import re
from collections.abc import Callable, Iterator
from fractions import Fraction
from typing import NamedTuple

import numpy as np

from haishool.cosmos import Rollout, close, query_lines, seeds, state_line, summary_lines
from haishool.evo import Input, LessonGate, Rule
from haishool.truth import Line, Verdict, num, parse_num

SIM = "life"
LETTERS = "acgu"
#: padding value in the sequence array (never a letter)
PAD = 4
MAX_LEN = 16
STEPS = 40
MAX_POLYMERS = 20000
#: fewer replicators than this never count as ``dominated``
DOMINATED_MIN = 10
DOMINATED_SHARE = 0.5
#: step queries per seed in :meth:`Life.generate` (the records and final lines always go in)
QUERIES_PER_SEED = 120

KEYS: dict[str, str] = {
    "step": "step number, 0 is the soup before anything happened (in every prompt, so not a field of its own)",
    "free_monomers": "free monomers in the pool, all four kinds together",
    "bound": "letters bound in polymers (free_monomers + bound = monomers + inflow * step)",
    "polymers": "polymers in the vessel",
    "replicators": "polymers carrying the motif",
    "longest": "length of the longest polymer, letters",
    "mean_length": "mean polymer length, letters (3 significant digits)",
    "diversity": "distinct replicator sequences",
    "dominant_share": "share of the replicators that are the most common sequence, 0 to 1 (3 significant digits)",
    "generations": "copy events so far",
    "stage": "soup, first_replicator, growth, competition, dominated or extinct",
}
#: the metrics written into records and asked about (everything but the step number)
METRICS: list[str] = [k for k in KEYS if k != "step"]
#: asked as ``life seed 7 final replicators``
SUMMARY_KEYS: dict[str, str] = {
    "replicators": "replicators at the last step",
    "diversity": "distinct replicator sequences at the last step",
    "dominant_share": "dominant_share at the last step",
    "first_replicator_step": "step at which the first replicator appeared, minus 1 if none did",
    "peak_replicators": "most replicators at any step",
    "outcome": "none (no replicator ever), coexist, dominated or extinct",
}
PARAM_KEYS: dict[str, str] = {
    "monomers": "free monomers at step 0, 5000 to 20000, split evenly over a c g u",
    "inflow": "monomers added per step, a multiple of 4 split evenly, 0 or 40 to 400",
    "poly_rate": "share of the free monomers that joins into random strings per step",
    "decay": "a polymer of length l falls apart with probability min(1, decay / l) per step, 0.1 to 3.2",
    "mu": "probability that a letter of a copy is replaced by another letter",
    "gap": "motif strictness: ag then cu with at most gap letters between, 0 (strict) to 3",
}
STAGES = ("soup", "first_replicator", "growth", "competition", "dominated", "extinct")
OUTCOMES = ("none", "coexist", "dominated", "extinct")
#: answers that must match exactly: identities and table values, not outcomes of the dice
EXACT = {"step", "first_replicator_step", *PARAM_KEYS}

_POWERS = 5 ** np.arange(MAX_LEN, dtype=np.int64)
_N = r"(0|[1-9](?: \d)*)"
_PROMPT = re.compile(rf"^life seed {_N} (?:step {_N} (next )?([a-z_]+)|(final|param) ([a-z_]+))$")
_TO_UNIT = 2.0 ** -53


def _length_cdf() -> np.ndarray:
    """P(1 + geometric(0.3) <= 2), ..., P(<= 15), by repeated multiplication (no ``pow``)."""
    out, q = [], 1.0
    for _ in range(MAX_LEN - 2):
        q *= 0.7
        out.append(1.0 - q)
    return np.array(out, dtype=np.float64)


_LENGTH_CDF = _length_cdf()
Raw = Callable[[int], np.ndarray]


def random_params(rng: random.Random, rules: int = 6) -> dict[str, float | int]:
    """Seeded parameters, from short lists so the model sees each value many times.

    ``rules=7`` makes the same six draws in the same order, drops ``decay`` and draws
    ``hydrolysis`` last, so ``monomers``, ``inflow``, ``poly_rate``, ``mu`` and ``gap`` of a seed
    are the same under both rule sets."""
    rules = _rules(rules)
    p = {
        "monomers": rng.randint(5000, 20000),
        "inflow": 0 if rng.random() < 0.2 else 4 * rng.randint(10, 100),
        "poly_rate": rng.choice([0.002, 0.005, 0.01, 0.02]),
        "decay": rng.choice([0.1, 0.2, 0.4, 0.8, 1.6, 3.2]),
        "mu": rng.choice([0.001, 0.002, 0.005, 0.01, 0.02, 0.05]),
        "gap": rng.randint(0, 3),
    }
    if rules == 6:
        return p
    hydrolysis = rng.choice(list(HYDROLYSIS))
    return {"monomers": p["monomers"], "inflow": p["inflow"], "poly_rate": p["poly_rate"],
            "hydrolysis": hydrolysis, "mu": p["mu"], "gap": p["gap"]}


def _rules(rules: int) -> int:
    if isinstance(rules, bool) or rules not in (6, 7):
        raise ValueError(f"rules must be 6 (round 6, the default) or 7, not {rules!r}")
    return int(rules)


def _params(seed: int, given: dict[str, float | int]) -> dict[str, float | int]:
    unknown = set(given) - set(PARAM_KEYS)
    if unknown:
        raise ValueError(f"unknown parameter: {', '.join(sorted(unknown))}")
    p = {**random_params(random.Random(seed)), **given}
    for k in ("monomers", "inflow", "gap"):
        if isinstance(p[k], bool) or int(p[k]) != p[k]:
            raise ValueError(f"{k} must be a whole number")
        p[k] = int(p[k])
    if p["monomers"] < 0 or p["inflow"] < 0 or p["inflow"] % 4:
        raise ValueError("monomers and inflow must not be negative, inflow a multiple of 4 (split evenly over a c g u)")
    if not 0 <= p["gap"] <= MAX_LEN - 4:
        raise ValueError(f"gap must be 0 to {MAX_LEN - 4}")
    if not (0 <= p["poly_rate"] <= 1 and 0 <= p["mu"] <= 1 and p["decay"] >= 0):
        raise ValueError("poly_rate and mu must lie in 0 to 1, decay must not be negative")
    return p


def has_motif(seqs: np.ndarray, gap: int) -> np.ndarray:
    """Which rows of a (n, MAX_LEN) letter array hold ``ag`` then ``cu`` within ``gap`` letters."""
    a, c, g, u = (LETTERS.index(ch) for ch in "acgu")
    hit = np.zeros(len(seqs), dtype=bool)
    for k in range(gap + 1):
        for i in range(MAX_LEN - 3 - k):
            hit |= (seqs[:, i] == a) & (seqs[:, i + 1] == g) & (seqs[:, i + 2 + k] == c) & (seqs[:, i + 3 + k] == u)
    return hit


def letter_counts(seqs: np.ndarray) -> np.ndarray:
    """(n, 4) counts of a, c, g, u per row."""
    return np.stack([(seqs == k).sum(1, dtype=np.int64) for k in range(4)], axis=1)


def _sig3(x: float) -> float:
    return float(f"{x:.3g}")


def _uniform(raw: Raw, n: int) -> np.ndarray:
    """``n`` numbers in [0, 1): the top 53 bits of each raw word, an exact conversion."""
    return (raw(n) >> np.uint64(11)) * _TO_UNIT


def _order(keys: np.ndarray) -> np.ndarray:
    """The order that sorts ``keys``; equal keys keep their place, so nothing is left to the sort."""
    return np.argsort(keys, kind="stable")


def _draw(raw: Raw, free: np.ndarray, k: int) -> np.ndarray:
    """``k`` letters drawn from the pool one after the other without replacement.

    The pool is numbered kind by kind (all ``a``, then ``c``, ``g``, ``u``); numbers are drawn
    until ``k`` different ones came up, in the order they came up (or, when most of the pool is
    wanted, the whole pool is put in random order and its first ``k`` taken).
    """
    n = int(free.sum())
    if 2 * k > n:
        picked = _order(raw(n))[:k]
    else:
        picked = np.empty(0, dtype=np.int64)
        while len(picked) < k:
            cand = np.concatenate([picked, (raw(k - len(picked) + 8) % np.uint64(n)).astype(np.int64)])
            by_value = _order(cand)
            new = np.ones(len(cand), dtype=bool)
            new[1:] = cand[by_value][1:] != cand[by_value][:-1]
            picked = cand[np.sort(by_value[new])]  # first time each number came up, in draw order
        picked = picked[:k]
    return np.searchsorted(np.cumsum(free), picked, side="right").astype(np.uint8)


def simulate(seed: int, p: dict[str, float | int]) -> Iterator[tuple[np.ndarray, np.ndarray, int, int]]:
    """The vessel after step 0, 1, ..., STEPS: ``(free, seqs, generations, failed)`` with the free
    monomers per kind, the polymers as a (n, MAX_LEN) letter array padded with ``PAD``, the copy
    events so far and the copies that failed in this step. ``p`` must hold every parameter."""
    raw: Raw = np.random.PCG64(seed).random_raw
    monomers, per_kind, gap = int(p["monomers"]), int(p["inflow"]) // 4, int(p["gap"])
    free = np.array([monomers // 4 + (1 if k < monomers % 4 else 0) for k in range(4)], dtype=np.int64)
    start = free.copy()
    seqs = np.empty((0, MAX_LEN), dtype=np.uint8)
    generations = 0
    yield free.copy(), seqs, 0, 0
    for t in range(1, STEPS + 1):
        free += per_kind
        # decay: a chain of length l breaks with probability decay / l, its letters go back
        if len(seqs):
            lens = (seqs != PAD).sum(1, dtype=np.int64)
            dies = _uniform(raw, len(seqs)) < p["decay"] / lens
            free += letter_counts(seqs[dies]).sum(0)
            seqs = seqs[~dies]
        # polymerisation: random strings from the pool, letters drawn without replacement
        want = int(p["poly_rate"] * int(free.sum()))
        room = MAX_POLYMERS - len(seqs)
        if want >= 2 and room > 0:
            lengths = 2 + np.searchsorted(_LENGTH_CDF, _uniform(raw, min(room, want // 2)), side="right")
            n_new = int(np.searchsorted(np.cumsum(lengths), want, side="right"))
            if n_new:
                lengths = lengths[:n_new]
                letters = _draw(raw, free, int(lengths.sum()))
                rows = np.full((n_new, MAX_LEN), PAD, dtype=np.uint8)
                rows[np.arange(MAX_LEN)[None, :] < lengths[:, None]] = letters
                free -= np.bincount(letters, minlength=4)
                seqs = np.concatenate([seqs, rows])
        # replication: every replicator copies once, in random order, while the letters last
        failed = 0
        templates = np.flatnonzero(has_motif(seqs, gap))
        if len(templates):
            copies = seqs[templates[_order(raw(len(templates)))]]
            real = copies != PAD
            mut = np.zeros(copies.shape, dtype=bool)
            mut[real] = _uniform(raw, int(real.sum())) < p["mu"]
            copies[mut] = (copies[mut] + 1 + (raw(int(mut.sum())) % np.uint64(3)).astype(np.uint8)) % 4
            need = letter_counts(copies)
            ok = (need.cumsum(0) <= free).all(1)
            if not ok.all():  # from the first failure on, each copy is checked against what is left
                i = int(np.argmin(ok))
                ok[i:] = False
                left = (free - need[:i].sum(0)).tolist()
                # the pool only shrinks: a copy that does not fit now will not fit later either
                fits = i + np.flatnonzero((need[i:] <= np.array(left, dtype=np.int64)).all(1))
                for j, row in zip(fits.tolist(), need[fits].tolist()):
                    if row[0] <= left[0] and row[1] <= left[1] and row[2] <= left[2] and row[3] <= left[3]:
                        left = [left[0] - row[0], left[1] - row[1], left[2] - row[2], left[3] - row[3]]
                        ok[j] = True
            room = MAX_POLYMERS - len(seqs)
            if ok.sum() > room:
                ok[np.flatnonzero(ok)[room:]] = False
            free -= need[ok].sum(0)
            seqs = np.concatenate([seqs, copies[ok]])
            generations += int(ok.sum())
            failed = int(len(templates) - ok.sum())
        if free.min() < 0 or (free + letter_counts(seqs).sum(0) != start + per_kind * t).any():
            raise ArithmeticError(f"life seed {seed} step {t}: the letters of a kind do not add up")
        yield free.copy(), seqs, generations, failed


def run(seed: int, rules: int = 6, **params: float | int) -> Rollout:
    """The rollout of one seed; parameters not given come from ``random_params(Random(seed))``.

    ``rules=7`` runs the corrected vessel (section "Rules 7" of the module docstring): parameters
    :data:`PARAM_KEYS7`, :data:`STEPS7` steps, metrics :data:`KEYS7`, and ``"rules": 7`` in the
    rollout's ``params``. Without it nothing differs from round 6."""
    if _rules(rules) == 7:
        return _run7(seed, params)
    p = _params(seed, params)
    gap = int(p["gap"])
    first = -1
    peak = prev = 0
    steps: list[dict] = []
    for t, (free, seqs, generations, failed) in enumerate(simulate(seed, p)):
        n = len(seqs)
        rep = has_motif(seqs, gap)
        n_rep = int(rep.sum())
        lens = (seqs != PAD).sum(1, dtype=np.int64)
        bound = int(lens.sum())
        if n_rep:
            _, counts = np.unique(seqs[rep].astype(np.int64) @ _POWERS, return_counts=True)
            diversity, share = len(counts), _sig3(int(counts.max()) / n_rep)
        else:
            diversity, share = 0, 0.0
        stage = _stage(n_rep, share, first, prev, failed)
        if stage == "first_replicator":
            first = t
        peak, prev = max(peak, n_rep), n_rep
        steps.append({"step": t, "free_monomers": int(free.sum()), "bound": bound, "polymers": n,
                      "replicators": n_rep, "longest": int(lens.max()) if n else 0,
                      "mean_length": _sig3(bound / n) if n else 0.0, "diversity": diversity,
                      "dominant_share": share, "generations": generations, "stage": stage})
    return Rollout(SIM, seed, p, steps, _summary(steps))


def _stage(n_rep: int, share: float, first: int, prev: int, failed: int) -> str:
    """The stage of a step from its numbers: ``first`` is the step of the first replicator so
    far (-1 for none), ``prev`` the replicators one step earlier, ``failed`` the failed copies."""
    if n_rep == 0:
        return "extinct" if first >= 0 else "soup"
    if first < 0:
        return "first_replicator"
    if n_rep >= DOMINATED_MIN and share >= DOMINATED_SHARE:
        return "dominated"
    return "competition" if failed or n_rep < prev else "growth"


def _summary(steps: list[dict]) -> dict[str, float | int | str]:
    last = steps[-1]
    reps = [s["replicators"] for s in steps]
    first = next((t for t, n in enumerate(reps) if n), -1)
    outcome = ("none" if first < 0 else "extinct" if last["replicators"] == 0
               else "dominated" if last["stage"] == "dominated" else "coexist")
    return {"replicators": last["replicators"], "diversity": last["diversity"],
            "dominant_share": last["dominant_share"], "first_replicator_step": first,
            "peak_replicators": max(reps), "outcome": outcome}


@functools.lru_cache(maxsize=1024)
def _rollout(seed: int) -> Rollout:
    return run(seed)


def params_line(r: Rollout) -> Line:
    fields = " ".join(f"{k} {r.value(v)}." for k, v in r.params.items())
    return Line(f"{SIM} seed {num(r.seed)} params. {fields}", topic=SIM, kind="record")


def param_lines(r: Rollout) -> list[Line]:
    return [Line(f"{SIM} seed {num(r.seed)} param {k}", r.value(v), SIM, "fact") for k, v in r.params.items()]


def lines(r: Rollout, rules: int = 6) -> list[Line]:
    """Every line of one rollout: the parameters, a record per step, the final and parameter
    questions, then the step and next-step questions. A rollout made with ``rules=7`` gives the
    ``life seed 7 rules 7 ...`` lines whatever ``rules`` says here; asking a round-6 rollout
    for ``rules=7`` lines is an error (one rollout has one set of rules)."""
    if is_rules7(r):
        return lines7(r)
    if _rules(rules) == 7:
        raise ValueError("this rollout was made with the round-6 rules; run(seed, rules=7) makes the other")
    out = [params_line(r)] + [state_line(r, t, METRICS) for t in range(len(r.steps))] + summary_lines(r) + param_lines(r)
    for t in range(len(r.steps)):
        out += query_lines(r, t, METRICS)
    return out


class Life:
    sim = SIM
    KEYS = KEYS

    def run(self, seed: int, rules: int = 6, **params: float | int) -> Rollout:
        return run(seed, rules, **params)

    def rollout(self, seed: int, rules: int = 6) -> Rollout:
        """The rollout of ``seed`` with its seeded parameters, cached (do not change it)."""
        return _rollout7(seed) if _rules(rules) == 7 else _rollout(seed)

    def generate(self, rng: random.Random, n: int, rules: int = 6) -> list[Line]:
        """``n`` lines over fresh seeds: each seed's records and final lines plus
        :data:`QUERIES_PER_SEED` of its step questions."""
        if _rules(rules) == 7:
            return _generate7(rng, n)
        out: list[Line] = []
        for seed in seeds(rng, min(9000, n // 50 + 1)):
            r = self.rollout(seed)
            queries = [ln for t in range(len(r.steps)) for ln in query_lines(r, t, METRICS)]
            out += [params_line(r)] + [state_line(r, t, METRICS) for t in range(len(r.steps))]
            out += summary_lines(r) + param_lines(r) + rng.sample(queries, min(QUERIES_PER_SEED, len(queries)))
            if len(out) >= n:
                break
        return out[:n]

    def conserved(self, r: Rollout) -> Verdict:
        """The ledger and what must follow from it: free + bound letters equal the start plus
        the inflow at every step; no negative counts; replicators within polymers; the shares
        fit the counts; no more copies than templates; the stages and the summary follow from
        the numbers. A ``rules=7`` rollout is judged by :func:`conserved7`."""
        if is_rules7(r):
            return conserved7(r)
        total0, inflow = int(r.params["monomers"]), int(r.params["inflow"])
        first = -1
        for t, s in enumerate(r.steps):
            before = r.steps[t - 1] if t else None
            expect = total0 + inflow * t
            if s["step"] != t:
                return Verdict(False, num(t), f"step {t}: numbered {s['step']}")
            if s["free_monomers"] + s["bound"] != expect:
                return Verdict(False, num(expect), f"step {t}: free {s['free_monomers']} + bound {s['bound']} != {expect}")
            if any(s[k] < 0 for k in KEYS if k != "stage"):
                return Verdict(False, None, f"step {t}: negative count")
            if not s["diversity"] <= s["replicators"] <= s["polymers"] or (s["diversity"] == 0) != (s["replicators"] == 0):
                return Verdict(False, None, f"step {t}: diversity, replicators and polymers do not fit")
            if not 0 <= s["dominant_share"] <= 1:
                return Verdict(False, None, f"step {t}: dominant_share {s['dominant_share']}")
            if s["replicators"]:
                # the most common of d sequences has between n / d and n - d + 1 of the n replicators
                low, high = 1 / s["diversity"], (s["replicators"] - s["diversity"] + 1) / s["replicators"]
                if not low * 0.99 <= s["dominant_share"] <= high * 1.01:
                    return Verdict(False, None, f"step {t}: dominant_share {s['dominant_share']} does not fit the counts")
            elif s["dominant_share"]:
                return Verdict(False, None, f"step {t}: a share without replicators")
            if s["bound"] > s["polymers"] * MAX_LEN or s["bound"] < 2 * s["polymers"]:
                return Verdict(False, None, f"step {t}: bound letters do not fit the polymers")
            if s["polymers"]:
                if s["mean_length"] != _sig3(s["bound"] / s["polymers"]) or not 2 <= s["longest"] <= MAX_LEN \
                        or s["longest"] * s["polymers"] < s["bound"]:
                    return Verdict(False, None, f"step {t}: lengths do not fit the bound letters")
            elif s["longest"] or s["mean_length"]:
                return Verdict(False, None, f"step {t}: a length without polymers")
            copies = s["generations"] - (before["generations"] if before else 0)
            if copies < 0 or copies > s["replicators"]:
                return Verdict(False, None, f"step {t}: {copies} copies from {s['replicators']} replicators")
            if s["stage"] not in STAGES:
                return Verdict(False, None, f"step {t}: unknown stage {s['stage']}")
            prev = before["replicators"] if before else 0
            allowed = {_stage(s["replicators"], s["dominant_share"], first, prev, failed) for failed in (0, 1)}
            if s["stage"] not in allowed:
                return Verdict(False, " or ".join(sorted(allowed)), f"step {t}: stage {s['stage']} does not follow")
            if s["stage"] == "first_replicator":
                first = t
        if r.summary != _summary(r.steps):
            return Verdict(False, None, "the summary does not follow from the steps")
        return Verdict(True)

    def owns(self, prompt: str) -> bool:
        m = _PROMPT.match(prompt)
        if not m:
            return owns7(prompt)
        if m.group(4):
            return m.group(4) in KEYS
        return m.group(6) in (SUMMARY_KEYS if m.group(5) == "final" else PARAM_KEYS)

    def check(self, prompt: str, answer: str) -> Verdict:
        m = _PROMPT.match(prompt)
        if not m:
            return check7(prompt, answer)
        if not self.owns(prompt):
            return Verdict(False, None, "not my question")
        r = self.rollout(parse_num(m.group(1)))
        if m.group(5) == "final":
            key, expected = m.group(6), r.summary[m.group(6)]
        elif m.group(5) == "param":
            key, expected = m.group(6), r.params[m.group(6)]
        else:
            t = parse_num(m.group(2)) + (1 if m.group(3) else 0)
            if not 0 <= t < len(r.steps):
                return Verdict(False, None, "no such step")
            key, expected = m.group(4), r.steps[t][m.group(4)]
        want, ok = r.value(expected), agree(expected, answer, exact=key in EXACT)
        return Verdict(ok, want, "" if ok else f"expected {want}")


def agree(expected: float | int | str, answer: str, exact: bool = False) -> bool:
    """Words exactly; ``exact`` numbers exactly; a count of 0 exactly, other counts within 1 or
    5 percent and written as whole numbers; other numbers within 5 percent."""
    if isinstance(expected, str):
        return answer == expected
    got = parse_num(answer)
    if got is None:
        return False
    if exact:
        return got == expected
    if isinstance(expected, int):
        if not isinstance(got, int) or expected == 0 or got == 0:
            return got == expected and isinstance(got, int)
        return abs(got - expected) <= 1 or close(got, expected)
    return close(got, expected)


# ---------------------------------------------------------------------------------------------
# rules 7: the corrected vessel (see "Rules 7" in the module docstring). Nothing above calls
# into this part unless a rollout is made with ``rules=7``.
# ---------------------------------------------------------------------------------------------

#: steps of a rules-7 run: long enough for a closed vessel to use up its activated letters
STEPS7 = 80
#: the values ``hydrolysis`` is drawn from: probability per bond and step
HYDROLYSIS = (0.002, 0.005, 0.01, 0.02, 0.05, 0.1)
#: temperatures in kelvin from which :func:`hydrolysis_at` gives the next value of the list
HYDROLYSIS_EDGES_K = (260, 275, 290, 305, 320)
#: letters a template has copied per step, so a copy of ``l`` letters is finished within a step
#: with probability ``min(1, COPY_SPEED / l)`` (toy value; it defines the length of a step)
COPY_SPEED = 4
#: free monomers at which ``poly_rate`` is the share that polymerises per step (toy value: it
#: fixes the volume of the vessel)
POOL_REFERENCE = 10000
#: a population that never reached this many replicators and is gone has ``fizzled``; one that
#: reached it and is gone is ``extinct`` (toy threshold)
FIZZLE_PEAK = 10
#: fewer effective lineages than this at the end is ``dominated`` (toy threshold)
COEXIST_LINEAGES = 2

KEYS7: dict[str, str] = {
    "step": "step number, 0 is the soup before anything happened (in every prompt, so not a field of its own); "
            "one step is the time a 4-letter template needs for one copy, no physical unit",
    "free_monomers": "activated monomers in the pool, all four kinds together; only these can be built into a string",
    "spent": "monomers set free by hydrolysis so far; they are not activated and are never used again",
    "bound": "letters bound in polymers (free_monomers + spent + bound = monomers + inflow * step)",
    "polymers": "polymers in the vessel",
    "replicators": "polymers carrying the motif",
    "longest": "length of the longest polymer, letters",
    "mean_length": "mean polymer length, letters (3 significant digits)",
    "replicator_length": "mean length of the replicators, letters (3 significant digits), 0 without replicators",
    "diversity": "distinct replicator sequences, a strand and its reverse complement counted as one",
    "lineages": "lines of descent that have a replicator in the vessel; a line starts with a replicator that arose in the soup",
    "dominant_share": "share of the replicators that belong to the largest lineage, 0 to 1 (3 significant digits)",
    "effective_lineages": "inverse Simpson index of the lineages, 1 over the sum of their squared shares "
                          "(3 significant digits), 0 without replicators",
    "copies": "copies made so far",
    "failed": "copies that were attempted in this step and found no letters",
    "origins": "replicators that arose in the soup so far, each the start of a lineage",
    "stage": "soup, first_replicator, growth, competition, decline or extinct",
}
#: the metrics written into rules-7 records and asked about (everything but the step number)
METRICS7: list[str] = [k for k in KEYS7 if k != "step"]
#: asked as ``life seed 7 rules 7 final replicators``
SUMMARY_KEYS7: dict[str, str] = {
    "replicators": "replicators at the last step",
    "diversity": "distinct replicator sequences at the last step",
    "lineages": "lineages that still have a replicator at the last step",
    "dominant_share": "dominant_share at the last step",
    "effective_lineages": "effective_lineages at the last step",
    "replicator_length": "replicator_length at the last step",
    "origins": "replicators that arose in the soup during the run",
    "first_replicator_step": "step at which the first replicator appeared, or never",
    "peak_replicators": "most replicators at any step",
    "max_length": "longest replicator that can multiply at this hydrolysis and mu while letters are plentiful, "
                  "4 to 16 letters, 0 if none can",
    "outcome": "none (no replicator ever), fizzled, extinct, declining, dominated or coexist",
}
PARAM_KEYS7: dict[str, str] = {
    "monomers": "activated monomers at step 0, 5000 to 20000, split evenly over a c g u",
    "inflow": "activated monomers added per step from outside the vessel, a multiple of 4 split evenly, 0 or 40 to 400",
    "poly_rate": "share of the free monomers that joins into random strings per step when 10000 are free; "
                 "the share goes with the number that are free (two monomers must meet)",
    "hydrolysis": "probability per bond and step that a bond of a polymer breaks, 0.002 to 0.1",
    "mu": "probability that a letter of a copy is replaced by another letter",
    "gap": "motif strictness: ag then cu with at most gap letters between, 0 (strict) to 3",
}
STAGES7 = ("soup", "first_replicator", "growth", "competition", "decline", "extinct")
OUTCOMES7 = ("none", "fizzled", "extinct", "declining", "dominated", "coexist")
#: answers that must match exactly under rules 7: identities and table values
EXACT7 = {"step", "first_replicator_step", "max_length", "longest", *PARAM_KEYS7}
#: small whole numbers that are exact below :data:`SMALL7` (2 against 1 is another answer, not a near miss)
SMALL_EXACT7 = {"diversity", "lineages", "origins", "failed"}
SMALL7 = 20
#: the word of the rules-7 lessons (``life7 predict intact_probability ...``, topic
#: ``predict_life7``). A rules-7 *story* keeps the word ``life`` and says ``rules 7`` after its
#: seed. ``life predict ...`` (topic ``predict_life``) are the round-6 lessons of
#: ``haishool.cosmos.predict``; with a word of their own the two never share a prompt or a topic.
SIM7 = "life7"

_PROMPT7 = re.compile(rf"^life seed {_N} rules 7 (?:step {_N} (next )?([a-z_]+)|(final|param) ([a-z_]+))$")
_COLS = np.arange(MAX_LEN, dtype=np.int64)[None, :]


def is_rules7(r: Rollout) -> bool:
    """True for a rollout made with ``run(seed, rules=7)``."""
    return r.params.get("rules") == 7


# the rules: each is a lesson (LESSONS) and is called by the rules-7 simulation or its gate

def intact_probability(length: int, hydrolysis: float) -> float:
    """Probability that none of the ``length - 1`` bonds of a chain breaks in one step,
    ``(1 - hydrolysis) ** (length - 1)``, by repeated multiplication (no ``pow``).

    >>> intact_probability(2, 0.1), intact_probability(4, 0.5), intact_probability(16, 0.0)
    (0.9, 0.125, 1.0)
    """
    q = 1.0
    for _ in range(int(length) - 1):
        q *= 1.0 - hydrolysis
    return q


def copy_probability(length: int) -> float:
    """Probability that a template of ``length`` letters finishes a copy within one step: the
    copy grows by :data:`COPY_SPEED` letters per step, so ``min(1, 4 / length)``.

    >>> copy_probability(4), copy_probability(8), copy_probability(16), copy_probability(2)
    (1.0, 0.5, 0.25, 1.0)
    """
    return min(1.0, COPY_SPEED / length)


def motif_fidelity(mu: float) -> float:
    """Probability that a copy keeps the four motif letters, ``(1 - mu) ** 4`` by four
    multiplications.

    >>> motif_fidelity(0.0), motif_fidelity(0.5), motif_fidelity(1.0)
    (1.0, 0.0625, 0.0)
    """
    q = 1.0
    for _ in range(4):
        q *= 1.0 - mu
    return q


def growth_factor(length: int, hydrolysis: float, mu: float) -> float:
    """Expected replicators of ``length`` letters one step later per replicator now, while
    letters are plentiful: it stays whole, and with :func:`copy_probability` adds a copy that
    keeps the motif. Pieces of broken chains and new strings from the soup are not counted.

    >>> growth_factor(4, 0.0, 0.0), growth_factor(8, 0.0, 0.0), growth_factor(4, 0.5, 0.0)
    (2.0, 1.5, 0.25)
    """
    return intact_probability(length, hydrolysis) * (1.0 + copy_probability(length) * motif_fidelity(mu))


def max_length(hydrolysis: float, mu: float) -> int:
    """The longest replicator, 4 to :data:`MAX_LEN` letters, whose :func:`growth_factor` is above
    1, or 0 when even 4 letters cannot multiply. Longer ones break faster than they copy.

    >>> max_length(0.002, 0.001), max_length(0.02, 0.001), max_length(0.1, 0.001), max_length(0.1, 0.05), max_length(0.3, 0.05)
    (16, 13, 5, 5, 0)
    """
    best = 0
    for length in range(4, MAX_LEN + 1):
        if growth_factor(length, hydrolysis, mu) > 1.0:
            best = length
    return best


def polymer_letters(free: int, poly_rate: float) -> int:
    """Letters that join into new strings in one step. Two monomers must meet, so the rate goes
    with the square of the concentration: the share ``poly_rate`` holds when
    :data:`POOL_REFERENCE` monomers are free and changes in proportion to their number (the
    vessel has one fixed volume); never more than there are.

    ``poly_rate * free * free / 10000`` rounded down, computed without rounding error from
    ``poly_rate`` as it is written in decimals: in floating point 0.043 * 20000 * 2 comes out a
    hair below 1720 and would be cut to 1719, which is not what the rule says.

    >>> polymer_letters(10000, 0.01), polymer_letters(5000, 0.01), polymer_letters(20000, 0.01), polymer_letters(100, 1.0), polymer_letters(50000, 0.5)
    (100, 25, 400, 1, 50000)
    >>> polymer_letters(20000, 0.043), polymer_letters(41000, 0.01), polymer_letters(9999, 0.002), polymer_letters(0, 0.02)
    (1720, 1681, 19, 0)
    """
    free = int(free)
    return min(free, int(Fraction(str(float(poly_rate))) * free * free // POOL_REFERENCE))


def inflow_per_kind(inflow: int) -> int:
    """Monomers of each of the four kinds that enter per step.

    >>> inflow_per_kind(240)
    60
    """
    return int(inflow) // 4


def next_total(total: int, inflow: int) -> int:
    """All letters of the vessel one step later (free, spent and bound together): only the
    inflow adds any.

    >>> next_total(10305, 240)
    10545
    """
    return int(total) + int(inflow)


def stage(replicators: int, previous: int, failed: int, seen_before: int) -> str:
    """The stage of a step from its record: ``replicators`` now, ``previous`` one step earlier,
    ``failed`` copies in this step, ``seen_before`` 1 if an earlier step had a replicator.

    >>> stage(0, 0, 0, 0), stage(1, 0, 0, 0), stage(0, 3, 0, 1), stage(9, 5, 2, 1), stage(4, 5, 0, 1), stage(5, 5, 0, 1)
    ('soup', 'first_replicator', 'extinct', 'competition', 'decline', 'growth')
    """
    if replicators == 0:
        return "extinct" if seen_before else "soup"
    if not seen_before:
        return "first_replicator"
    if failed > 0:
        return "competition"
    return "decline" if replicators < previous else "growth"


def outcome(peak: int, final: int, effective_lineages: float) -> str:
    """How a run ended, first match: ``none`` (never a replicator), ``fizzled`` (none left, never
    :data:`FIZZLE_PEAK`), ``extinct`` (none left, there were that many), ``declining`` (at most
    half of the peak left), ``dominated`` (fewer than 2 effective lineages), ``coexist``.

    >>> outcome(0, 0, 0.0), outcome(3, 0, 0.0), outcome(50, 0, 0.0), outcome(50, 25, 3.0), outcome(50, 40, 1.2), outcome(50, 40, 2.0)
    ('none', 'fizzled', 'extinct', 'declining', 'dominated', 'coexist')
    """
    if peak == 0:
        return "none"
    if final == 0:
        return "fizzled" if peak < FIZZLE_PEAK else "extinct"
    if 2 * final <= peak:
        return "declining"
    return "dominated" if effective_lineages < COEXIST_LINEAGES else "coexist"


def hydrolysis_at(temperature: float) -> float:
    """The ``hydrolysis`` value for a vessel at ``temperature`` kelvin, for the level that hands a
    planet over (not used inside this level): the next value of :data:`HYDROLYSIS` every 15 K
    from 260 K on, roughly a doubling per step. Warmer water breaks bonds faster; the numbers
    are toy (rates of real reactions commonly double or triple every 10 K, so the table rises
    more gently than real chemistry would, and pure water is ice below 273 K).

    >>> [hydrolysis_at(t) for t in (246, 260, 274.9, 275, 288, 300, 319, 320, 400)]
    [0.002, 0.005, 0.005, 0.01, 0.01, 0.02, 0.05, 0.1, 0.1]
    """
    return HYDROLYSIS[sum(1 for edge in HYDROLYSIS_EDGES_K if temperature >= edge)]


def reverse_complement(seqs: np.ndarray) -> np.ndarray:
    """Each row read backwards with ``a`` and ``u``, ``c`` and ``g`` exchanged: the strand that
    pairs with it. With :data:`LETTERS` ``acgu`` the partner of letter ``x`` is ``3 - x``."""
    lens = (seqs != PAD).sum(1, dtype=np.int64)
    idx = lens[:, None] - 1 - _COLS
    out = np.full(seqs.shape, PAD, dtype=np.uint8)
    real = idx >= 0
    out[real] = 3 - np.take_along_axis(seqs, np.maximum(idx, 0), axis=1)[real]
    return out


def canonical(seqs: np.ndarray) -> np.ndarray:
    """One whole number per row that a strand shares with its reverse complement and with no
    other sequence: the smaller of the two base-5 codes."""
    return np.minimum(seqs.astype(np.int64) @ _POWERS, reverse_complement(seqs).astype(np.int64) @ _POWERS)


class Vessel(NamedTuple):
    """The rules-7 vessel after a step."""
    #: activated monomers per kind
    free: np.ndarray
    #: hydrolysed monomers per kind, never used again
    spent: np.ndarray
    #: the polymers, a (n, MAX_LEN) letter array padded with ``PAD``
    seqs: np.ndarray
    #: per polymer the lineage it descends from, -1 when it does not carry the motif
    lineage: np.ndarray
    #: copies made so far
    copies: int
    #: copies attempted in this step that found no letters
    failed: int
    #: replicators that arose in the soup so far (the next lineage number)
    origins: int


def _params7(seed: int, given: dict[str, float | int]) -> dict[str, float | int]:
    unknown = set(given) - set(PARAM_KEYS7)
    if unknown:
        raise ValueError(f"unknown parameter under rules 7: {', '.join(sorted(unknown))}")
    p = {**random_params(random.Random(seed), rules=7), **given}
    for k in ("monomers", "inflow", "gap"):
        if isinstance(p[k], bool) or int(p[k]) != p[k]:
            raise ValueError(f"{k} must be a whole number")
        p[k] = int(p[k])
    if p["monomers"] < 0 or p["inflow"] < 0 or p["inflow"] % 4:
        raise ValueError("monomers and inflow must not be negative, inflow a multiple of 4 (split evenly over a c g u)")
    if not 0 <= p["gap"] <= MAX_LEN - 4:
        raise ValueError(f"gap must be 0 to {MAX_LEN - 4}")
    if not (0 <= p["poly_rate"] <= 1 and 0 <= p["mu"] <= 1 and 0 <= p["hydrolysis"] <= 1):
        raise ValueError("poly_rate, mu and hydrolysis must lie in 0 to 1")
    return {k: p[k] for k in PARAM_KEYS7}


def simulate7(seed: int, p: dict[str, float | int]) -> Iterator[Vessel]:
    """The rules-7 vessel after step 0, 1, ..., STEPS7. ``p`` must hold every parameter of
    :data:`PARAM_KEYS7`. Each step: inflow, hydrolysis, polymerisation, replication."""
    raw: Raw = np.random.PCG64(seed).random_raw
    monomers, per_kind, gap = int(p["monomers"]), inflow_per_kind(int(p["inflow"])), int(p["gap"])
    intact = np.array([1.0, 1.0] + [intact_probability(n, p["hydrolysis"]) for n in range(2, MAX_LEN + 1)])
    pcopy = np.array([0.0, 0.0] + [copy_probability(n) for n in range(2, MAX_LEN + 1)])
    free = np.array([monomers // 4 + (1 if k < monomers % 4 else 0) for k in range(4)], dtype=np.int64)
    start = free.copy()
    spent = np.zeros(4, dtype=np.int64)
    seqs = np.empty((0, MAX_LEN), dtype=np.uint8)
    lineage = np.empty(0, dtype=np.int64)
    copies = origins = 0
    yield Vessel(free.copy(), spent.copy(), seqs, lineage, 0, 0, 0)
    for t in range(1, STEPS7 + 1):
        free += per_kind
        # hydrolysis: a chain stays whole with probability (1 - h)^(l - 1); a broken one is cut
        # at one bond; single letters are spent, longer pieces stay
        if len(seqs):
            lens = (seqs != PAD).sum(1, dtype=np.int64)
            broken = np.flatnonzero(_uniform(raw, len(seqs)) >= intact[lens])
            if len(broken):
                rows, n_let = seqs[broken], lens[broken]
                cut = 1 + (raw(len(broken)) % (n_let - 1).astype(np.uint64)).astype(np.int64)
                left = np.where(_COLS < cut[:, None], rows, PAD).astype(np.uint8)
                src = _COLS + cut[:, None]
                right = np.where(src < n_let[:, None],
                                 np.take_along_axis(rows, np.minimum(src, MAX_LEN - 1), axis=1), PAD).astype(np.uint8)
                keep_left, keep_right = cut >= 2, n_let - cut >= 2
                lone = np.concatenate([rows[~keep_left, 0], rows[np.flatnonzero(~keep_right), n_let[~keep_right] - 1]])
                spent += np.bincount(lone, minlength=4)
                pieces = np.concatenate([left[keep_left], right[keep_right]])
                ids = np.concatenate([lineage[broken][keep_left], lineage[broken][keep_right]])
                ids[~has_motif(pieces, gap)] = -1  # a piece without the motif is no replicator
                whole = np.ones(len(seqs), dtype=bool)
                whole[broken] = False
                seqs = np.concatenate([seqs[whole], pieces])
                lineage = np.concatenate([lineage[whole], ids])
        # polymerisation: random strings from the activated pool, letters drawn without replacement
        want = polymer_letters(int(free.sum()), p["poly_rate"])
        room = MAX_POLYMERS - len(seqs)
        if want >= 2 and room > 0:
            lengths = 2 + np.searchsorted(_LENGTH_CDF, _uniform(raw, min(room, want // 2)), side="right")
            n_new = int(np.searchsorted(np.cumsum(lengths), want, side="right"))
            if n_new:
                lengths = lengths[:n_new]
                letters = _draw(raw, free, int(lengths.sum()))
                rows = np.full((n_new, MAX_LEN), PAD, dtype=np.uint8)
                rows[_COLS < lengths[:, None]] = letters
                free -= np.bincount(letters, minlength=4)
                ids = np.full(n_new, -1, dtype=np.int64)
                born = np.flatnonzero(has_motif(rows, gap))  # each one founds a lineage
                ids[born] = origins + np.arange(len(born), dtype=np.int64)
                origins += len(born)
                seqs = np.concatenate([seqs, rows])
                lineage = np.concatenate([lineage, ids])
        # replication: a template of l letters finishes a copy with probability min(1, 4 / l); the
        # copy is the reverse complement, with errors; copies are served in random order
        failed = 0
        templates = np.flatnonzero(lineage >= 0)
        if len(templates):
            lens = (seqs[templates] != PAD).sum(1, dtype=np.int64)
            templates = templates[_uniform(raw, len(templates)) < pcopy[lens]]
        if len(templates):
            templates = templates[_order(raw(len(templates)))]
            new = reverse_complement(seqs[templates])
            real = new != PAD
            mut = np.zeros(new.shape, dtype=bool)
            mut[real] = _uniform(raw, int(real.sum())) < p["mu"]
            new[mut] = (new[mut] + 1 + (raw(int(mut.sum())) % np.uint64(3)).astype(np.uint8)) % 4
            need = letter_counts(new)
            ok = (need.cumsum(0) <= free).all(1)
            if not ok.all():  # from the first failure on, each copy is checked against what is left
                i = int(np.argmin(ok))
                ok[i:] = False
                left_over = (free - need[:i].sum(0)).tolist()
                # the pool only shrinks: a copy that does not fit now will not fit later either
                fits = i + np.flatnonzero((need[i:] <= np.array(left_over, dtype=np.int64)).all(1))
                for j, row in zip(fits.tolist(), need[fits].tolist()):
                    if row[0] <= left_over[0] and row[1] <= left_over[1] and row[2] <= left_over[2] and row[3] <= left_over[3]:
                        left_over = [left_over[0] - row[0], left_over[1] - row[1], left_over[2] - row[2], left_over[3] - row[3]]
                        ok[j] = True
            room = max(0, MAX_POLYMERS - len(seqs))
            if ok.sum() > room:
                ok[np.flatnonzero(ok)[room:]] = False
            ids = lineage[templates].copy()
            ids[~has_motif(new, gap)] = -1  # a copy that lost the motif is no replicator
            free -= need[ok].sum(0)
            seqs = np.concatenate([seqs, new[ok]])
            lineage = np.concatenate([lineage, ids[ok]])
            copies += int(ok.sum())
            failed = int(len(templates) - ok.sum())
        if free.min() < 0 or (free + spent + letter_counts(seqs).sum(0) != start + per_kind * t).any():
            raise ArithmeticError(f"life seed {seed} rules 7 step {t}: the letters of a kind do not add up")
        yield Vessel(free.copy(), spent.copy(), seqs, lineage, copies, failed, origins)


def _run7(seed: int, given: dict[str, float | int]) -> Rollout:
    p = _params7(seed, given)
    seen = False
    prev = 0
    steps: list[dict] = []
    for t, v in enumerate(simulate7(seed, p)):
        n = len(v.seqs)
        rep = v.lineage >= 0
        n_rep = int(rep.sum())
        lens = (v.seqs != PAD).sum(1, dtype=np.int64)
        bound = int(lens.sum())
        if n_rep:
            diversity = len(np.unique(canonical(v.seqs[rep])))
            sizes = np.unique(v.lineage[rep], return_counts=True)[1].tolist()
            lineages = len(sizes)
            share = _sig3(max(sizes) / n_rep)
            effective = _sig3(n_rep * n_rep / sum(k * k for k in sizes))
            rep_length = _sig3(int(lens[rep].sum()) / n_rep)
        else:
            diversity, lineages, share, effective, rep_length = 0, 0, 0.0, 0.0, 0.0
        steps.append({"step": t, "free_monomers": int(v.free.sum()), "spent": int(v.spent.sum()), "bound": bound,
                      "polymers": n, "replicators": n_rep, "longest": int(lens.max()) if n else 0,
                      "mean_length": _sig3(bound / n) if n else 0.0, "replicator_length": rep_length,
                      "diversity": diversity, "lineages": lineages, "dominant_share": share,
                      "effective_lineages": effective, "copies": v.copies, "failed": v.failed,
                      "origins": v.origins, "stage": stage(n_rep, prev, v.failed, int(seen))})
        seen, prev = seen or n_rep > 0, n_rep
    return Rollout(SIM, seed, {**p, "rules": 7}, steps, _summary7(steps, p))


def _summary7(steps: list[dict], p: dict[str, float | int]) -> dict[str, float | int | str]:
    last = steps[-1]
    reps = [s["replicators"] for s in steps]
    first = next((t for t, n in enumerate(reps) if n), -1)
    return {"replicators": last["replicators"], "diversity": last["diversity"], "lineages": last["lineages"],
            "dominant_share": last["dominant_share"], "effective_lineages": last["effective_lineages"],
            "replicator_length": last["replicator_length"], "origins": last["origins"],
            "first_replicator_step": first if first >= 0 else "never", "peak_replicators": max(reps),
            "max_length": max_length(p["hydrolysis"], p["mu"]),
            "outcome": outcome(max(reps), last["replicators"], last["effective_lineages"])}


@functools.lru_cache(maxsize=1024)
def _rollout7(seed: int) -> Rollout:
    return run(seed, rules=7)


def _head7(r: Rollout) -> str:
    return f"{SIM} seed {num(r.seed)} rules 7"


def params_line7(r: Rollout) -> Line:
    """``life seed 7 rules 7 params. monomers 1 0 3 0 5. ... hydrolysis 0 point 0 1. ...``"""
    fields = " ".join(f"{k} {r.value(r.params[k])}." for k in PARAM_KEYS7)
    return Line(f"{_head7(r)} params. {fields}", topic=SIM, kind="record")


def param_lines7(r: Rollout) -> list[Line]:
    return [Line(f"{_head7(r)} param {k}", r.value(r.params[k]), SIM, "fact") for k in PARAM_KEYS7]


def state_line7(r: Rollout, t: int) -> Line:
    """``life seed 7 rules 7 step 3 0. free_monomers 1 2 3 4. spent 5 6. ... stage growth.``"""
    fields = " ".join(f"{k} {r.value(r.steps[t][k])}." for k in METRICS7)
    return Line(f"{_head7(r)} step {num(t)}. {fields}", topic=SIM, kind="record")


def query_lines7(r: Rollout, t: int) -> list[Line]:
    """One question per metric at step ``t`` and, if there is a next step, what it is then."""
    head = f"{_head7(r)} step {num(t)}"
    out = [Line(f"{head} {k}", r.value(r.steps[t][k]), SIM, "fact") for k in METRICS7]
    if t + 1 < len(r.steps):
        out += [Line(f"{head} next {k}", r.value(r.steps[t + 1][k]), SIM, "calc") for k in METRICS7]
    return out


def summary_lines7(r: Rollout) -> list[Line]:
    return [Line(f"{_head7(r)} final {k}", r.value(v), SIM, "calc") for k, v in r.summary.items()]


def lines7(r: Rollout) -> list[Line]:
    """Every line of one rules-7 rollout, in the order of :func:`lines`."""
    if not is_rules7(r):
        raise ValueError("not a rules-7 rollout")
    out = [params_line7(r)] + [state_line7(r, t) for t in range(len(r.steps))] + summary_lines7(r) + param_lines7(r)
    for t in range(len(r.steps)):
        out += query_lines7(r, t)
    return out


def _generate7(rng: random.Random, n: int) -> list[Line]:
    out: list[Line] = []
    for seed in seeds(rng, min(9000, n // 50 + 1)):
        r = _rollout7(seed)
        queries = [ln for t in range(len(r.steps)) for ln in query_lines7(r, t)]
        out += [params_line7(r)] + [state_line7(r, t) for t in range(len(r.steps))]
        out += summary_lines7(r) + param_lines7(r) + rng.sample(queries, min(QUERIES_PER_SEED, len(queries)))
        if len(out) >= n:
            break
    return out[:n]


def conserved7(r: Rollout) -> Verdict:
    """The gate of a rules-7 rollout. Every step: ``free_monomers + spent + bound`` is the start
    supply plus the inflow so far (:func:`next_total`); ``spent`` never falls; the copies made
    so far hold no more letters than ever entered (4 per copy at least: no letter is used
    twice); no negative count; ``diversity`` and ``lineages`` within ``replicators`` within
    ``polymers``; no more lineages than origins; the shares fit the counts; the lengths fit the
    bound letters; copies and failed copies together are no more than the replicators; the stage
    is :func:`stage` of the record; the summary follows from the steps and parameters."""
    p = r.params
    total, inflow = int(p["monomers"]), int(p["inflow"])
    seen = False
    for t, s in enumerate(r.steps):
        before = r.steps[t - 1] if t else None
        if t:
            total = next_total(total, inflow)
        if s["step"] != t:
            return Verdict(False, num(t), f"step {t}: numbered {s['step']}")
        if s["free_monomers"] + s["spent"] + s["bound"] != total:
            return Verdict(False, num(total), f"step {t}: free {s['free_monomers']} + spent {s['spent']} + bound "
                                              f"{s['bound']} != {total}")
        if any(s[k] < 0 for k in KEYS7 if k != "stage"):
            return Verdict(False, None, f"step {t}: negative count")
        if before and s["spent"] < before["spent"]:
            return Verdict(False, None, f"step {t}: spent monomers came back")
        if 4 * s["copies"] > total:
            return Verdict(False, None, f"step {t}: more copies than the letters that ever entered allow")
        n_rep = s["replicators"]
        if not s["diversity"] <= n_rep <= s["polymers"] or not s["lineages"] <= n_rep:
            return Verdict(False, None, f"step {t}: diversity, lineages, replicators and polymers do not fit")
        if any((s[k] == 0) != (n_rep == 0) for k in ("diversity", "lineages", "dominant_share", "effective_lineages",
                                                    "replicator_length")):
            return Verdict(False, None, f"step {t}: replicators and what is measured on them do not fit")
        if s["origins"] < (before["origins"] if before else 0) or s["lineages"] > s["origins"]:
            return Verdict(False, None, f"step {t}: lineages and origins do not fit")
        if n_rep:
            # the largest of k lineages has between n / k and n - k + 1 of the n replicators
            low, high = 1 / s["lineages"], (n_rep - s["lineages"] + 1) / n_rep
            if not low * 0.99 <= s["dominant_share"] <= high * 1.01:
                return Verdict(False, None, f"step {t}: dominant_share {s['dominant_share']} does not fit the counts")
            # the sum of squared shares lies between the largest share squared and the largest share
            if not 0.98 / s["dominant_share"] <= s["effective_lineages"] <= 1.03 / (s["dominant_share"] * s["dominant_share"]) \
                    or not 0.99 <= s["effective_lineages"] <= s["lineages"] * 1.01:
                return Verdict(False, None, f"step {t}: effective_lineages {s['effective_lineages']} does not fit")
            if not 4 * 0.99 <= s["replicator_length"] <= s["longest"] * 1.01:
                return Verdict(False, None, f"step {t}: replicator_length {s['replicator_length']} does not fit")
        if s["bound"] > s["polymers"] * MAX_LEN or s["bound"] < 2 * s["polymers"]:
            return Verdict(False, None, f"step {t}: bound letters do not fit the polymers")
        if s["polymers"]:
            if s["mean_length"] != _sig3(s["bound"] / s["polymers"]) or not 2 <= s["longest"] <= MAX_LEN \
                    or s["longest"] * s["polymers"] < s["bound"]:
                return Verdict(False, None, f"step {t}: lengths do not fit the bound letters")
        elif s["longest"] or s["mean_length"]:
            return Verdict(False, None, f"step {t}: a length without polymers")
        made = s["copies"] - (before["copies"] if before else 0)
        if made < 0 or made + s["failed"] > n_rep:
            return Verdict(False, None, f"step {t}: {made} copies and {s['failed']} failed from {n_rep} replicators")
        if (s["origins"] > 0) != (seen or n_rep > 0):
            return Verdict(False, None, f"step {t}: the first replicator must come from the soup")
        want = stage(n_rep, before["replicators"] if before else 0, s["failed"], int(seen))
        if s["stage"] != want:
            return Verdict(False, want, f"step {t}: stage {s['stage']} does not follow")
        seen = seen or n_rep > 0
    if r.summary != _summary7(r.steps, p):
        return Verdict(False, None, "the summary does not follow from the steps")
    return Verdict(True)


def owns7(prompt: str) -> bool:
    """True for a question about a rules-7 rollout: ``life seed 7 rules 7 step 3 replicators``."""
    m = _PROMPT7.match(prompt)
    if not m:
        return False
    if m.group(4):
        return m.group(4) in KEYS7
    return m.group(6) in (SUMMARY_KEYS7 if m.group(5) == "final" else PARAM_KEYS7)


def check7(prompt: str, answer: str) -> Verdict:
    """Replay the seed under rules 7 and compare with :func:`agree7`."""
    m = _PROMPT7.match(prompt)
    if not m or not owns7(prompt):
        return Verdict(False, None, "not my question")
    r = _rollout7(parse_num(m.group(1)))
    if m.group(5) == "final":
        key, expected = m.group(6), r.summary[m.group(6)]
    elif m.group(5) == "param":
        key, expected = m.group(6), r.params[m.group(6)]
    else:
        t = parse_num(m.group(2)) + (1 if m.group(3) else 0)
        if not 0 <= t < len(r.steps):
            return Verdict(False, None, "no such step")
        key, expected = m.group(4), r.steps[t][m.group(4)]
    want, ok = r.value(expected), agree7(key, expected, answer)
    return Verdict(ok, want, "" if ok else f"expected {want}")


def agree7(key: str, expected: float | int | str, answer: str) -> bool:
    """The tolerance of the rules-7 gate: words exactly (``never`` is a word); the keys of
    :data:`EXACT7` exactly; ``diversity``, ``lineages``, ``origins`` and ``failed`` exactly below
    :data:`SMALL7`; everything else as :func:`agree` (0 exactly, counts within 1 or 5 percent and
    whole, other numbers within 5 percent)."""
    if isinstance(expected, str) or key in EXACT7:
        return agree(expected, answer, exact=True)
    if key in SMALL_EXACT7 and isinstance(expected, int) and expected < SMALL7:
        got = parse_num(answer)
        return isinstance(got, int) and got == expected
    return agree(expected, answer)


#: the lessons of this level: every rule is a function the rules-7 simulation or its gate calls
RULES: dict[str, Rule] = {
    "intact_probability": Rule((Input("length", 2, MAX_LEN, integer=True), Input("hydrolysis", 0, 0.2)),
                               intact_probability,
                               "probability that a chain of length letters stays whole for one step. "
                               "one minus hydrolysis multiplied length minus one times. one factor per bond"),
    "copy_probability": Rule((Input("length", 2, MAX_LEN, integer=True),), copy_probability,
                             "probability that a template of length letters finishes a copy in one step. "
                             "4 over length and never more than 1"),
    "motif_fidelity": Rule((Input("mu", 0, 0.2),), motif_fidelity,
                           "probability that a copy keeps all 4 motif letters. one minus mu multiplied 4 times"),
    "growth_factor": Rule((Input("length", 4, MAX_LEN, integer=True), Input("hydrolysis", 0, 0.2), Input("mu", 0, 0.2)),
                          growth_factor,
                          "expected replicators one step later per replicator while letters are plentiful. "
                          "intact_probability times one plus copy_probability times motif_fidelity"),
    "max_length": Rule((Input("hydrolysis", 0, 0.2), Input("mu", 0, 0.2)), max_length,
                       "longest replicator of 4 to 1 6 letters with growth_factor above 1. 0 if none. "
                       "longer chains break faster than they copy"),
    "polymer_letters": Rule((Input("free", 0, 60000, integer=True), Input("poly_rate", 0, 0.05)), polymer_letters,
                            "letters that join into new strings in one step. poly_rate times free times free over "
                            "1 0 0 0 0 rounded down. never more than free. two monomers must meet"),
    "inflow_per_kind": Rule((Input("inflow", 0, 400, integer=True),), inflow_per_kind,
                            "monomers of each of the 4 kinds that enter per step. inflow over 4",
                            valid=lambda inflow: inflow % 4 == 0),
    "next_total": Rule((Input("total", 0, 1000000, integer=True), Input("inflow", 0, 10000, integer=True)), next_total,
                       "all letters one step later. free and spent and bound together. total plus inflow. "
                       "inflow is a multiple of 4",
                       valid=lambda total, inflow: inflow % 4 == 0),
    "stage": Rule((Input("replicators", 0, 20000, integer=True), Input("previous", 0, 20000, integer=True),
                   Input("failed", 0, 20000, integer=True), Input("seen_before", 0, 1, integer=True)), stage,
                  "stage of a step. no replicators is soup or after seen_before 1 extinct. the first is first_replicator. "
                  "then competition if failed is above 0. decline if replicators is below previous. else growth",
                  valid=lambda replicators, previous, failed, seen_before:
                      failed <= replicators and (seen_before == 1 or previous == 0)),
    "outcome": Rule((Input("peak", 0, 20000, integer=True), Input("final", 0, 20000, integer=True),
                     Input("effective_lineages", 0, 20000, places=2)), outcome,
                    "how a run ended. first match. none if peak is 0. final 0 is fizzled if peak is below 1 0 else extinct. "
                    "declining if 2 times final is at most peak. dominated if effective_lineages is below 2. else coexist",
                    valid=lambda peak, final, effective_lineages:
                        final <= peak and (effective_lineages == 0 if final == 0 else 1 <= effective_lineages <= final)),
}


class LifeLessons(LessonGate):
    """The lesson gate with a draw that reaches every branch. Prompts, truth and judgement are
    those of :class:`haishool.evo.LessonGate`; only :meth:`generate` differs: uniform draws of
    the two word rules would hardly ever show ``soup``, ``fizzled`` or ``none``, and inflows
    would mostly be refused, so their inputs are drawn case by case."""

    def draw(self, name: str, rng: random.Random) -> list[float | int]:
        if name == "stage":
            seen = int(rng.random() < 0.8)
            replicators = 0 if rng.random() < 0.2 else _magnitude(rng, 20000)
            previous = _magnitude(rng, 20000) if seen and rng.random() < 0.9 else 0
            failed = _magnitude(rng, replicators) if replicators and rng.random() < 0.4 else 0
            return [replicators, previous, failed, seen]
        if name == "outcome":
            peak = 0 if rng.random() < 0.1 else _magnitude(rng, 20000)
            final = 0 if peak == 0 or rng.random() < 0.3 else rng.randint(1, peak)
            return [peak, final, round(rng.uniform(1, min(final, 3)), 2) if final else 0]
        if name in ("inflow_per_kind", "next_total"):
            spec = self.rules[name].inputs
            return [rng.randint(int(s.low), int(s.high)) for s in spec[:-1]] + [4 * rng.randint(0, int(spec[-1].high) // 4)]
        return [rng.randint(int(s.low), int(s.high)) if s.integer else round(rng.uniform(s.low, s.high), s.places)
                for s in self.rules[name].inputs]

    def generate(self, rng: random.Random, n: int) -> list[Line]:
        """Exactly ``n`` lessons, rules drawn uniformly."""
        names = sorted(self.rules)
        out: list[Line] = []
        while len(out) < n:
            name = rng.choice(names)
            ln = self.line(name, self.draw(name, rng))
            if ln is not None:
                out.append(ln)
        return out


def _magnitude(rng: random.Random, high: int) -> int:
    """A whole number from 1 to ``high``, small ones as likely as large ones: the number of
    digits is drawn first."""
    top = rng.choice([k for k in (9, 99, 999, 9999, 99999) if k < high] + [high])
    return rng.randint(1, max(1, top))


LESSONS = LifeLessons(SIM7, RULES)


def lesson_gate() -> LessonGate:
    """The gate of the lessons ``life7 predict <rule> <input> <value> ...`` (topic ``predict_life7``)."""
    return LESSONS


def conserved(r: Rollout) -> Verdict:
    """:meth:`Life.conserved` for a rollout of either rule set."""
    return Life().conserved(r)


def owns(prompt: str) -> bool:
    """True for a story question of either rule set or a lesson of :data:`LESSONS`."""
    return Life().owns(prompt) or LESSONS.owns(prompt)


def check(prompt: str, answer: str) -> Verdict:
    """Judge a story question of either rule set (replayed) or a lesson (recomputed)."""
    if LESSONS.owns(prompt):
        return LESSONS.check(prompt, answer)
    return Life().check(prompt, answer)


def simulation() -> Life:
    return Life()
