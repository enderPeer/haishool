"""Level 11 of the ladder: a population that can hear and call settles on signals, then on a language.

At most 60 agents play three games on one clock of 60 output steps: a signalling game that gives
a few calls (steps 1 to 20), a naming game that gives a lexicon (21 to 40), and iterated learning
that turns whole phrases into a grammar (41 to 60). At the end the language can be read: words,
an order, utterances. Nobody knows how language began; each stage is a published toy model,
simplified, and the last section says what is taken from it and what is invented. That the three
games follow each other, calls first and grammar last, 20 steps each, is the timetable of this
toy and no claim about the order in which anything arose.

**The world of a run**, all from the seed. ``meanings`` things that matter, taken from a fixed
vocabulary of 12 (``predator`` is always one of them), in a drawn rank order. The meaning of rank
r is talked about with the Zipf share (1 / r) / sum(1 / k) (assumed: Zipf's law is about the
frequencies of words, here it is put on the meanings); predators come up
1 + 4 * ``predator_pressure`` times as often as their rank says (toy rule), and the shares are
scaled to add up to 1. A phoneme inventory of ``consonants`` (out of k t m n s l p b) and
``vowels`` (out of a i u e o): a syllable is one consonant and one vowel, a word is 1 to 3
syllables, written as separate tokens (``ka``, ``mi tu``). An agent holds ``neurons // 8`` words
(toy rule), its ``capacity``; only the ``capacity`` most talked-about meanings can ever have a
signal (they are *nameable*). The ``population`` lives in one group, or with ``split`` 1 in two
groups that never meet (the larger half is the first group). In a group the share ``ears_voice``
(rounded half up) has ears and a voice: the *talkers*. A group with fewer than two talkers, or
with a single call to choose from (one syllable, or one nameable meaning: log2(1) = 0 bits),
stays silent.

**Stage 1, calls**: the signalling game. Every talker keeps one table of weights, meaning x
call, all 1 at the start; the calls are single syllables, as many as there are nameable meanings
(at most all the syllables). An output step is 200 rounds. In a round the talkers pair off at
random; in each pair one of them sees what is the case (a nameable meaning, drawn by the shares
of the talk) and calls, the call drawn in proportion to its weights for that meaning; the other
acts on the call, the meaning drawn in proportion to its weights for that call. When the act
fits, both add 2 to the weight of that meaning and call, and 1 + 2 * pressure (rounded, a half
upwards) times as much for ``predator`` (a warning understood is worth more: toy rule); when it
does not, the caller takes 1 from the weight it used and the hearer 1 from the weight it used,
never below 1. A call is *settled* for a meaning when more than half of the talkers send it for
that meaning more often than not and more than half take it for that meaning more often than
not. Two calls that both mean the same thing are understood and yet not settled: neither is
what most talkers send (this happens, see the open points). A brain that holds fewer than 5
words coins no words (toy rule): its group plays this game to step 60.

**Stage 2, words**: the naming game. A settled call becomes the first word of the talkers who
favour it (toy rule: whether words descend from calls is not known, and many doubt it). An
output step is 60 games per talker: a random speaker, a random hearer, a topic drawn by the
shares of the talk; the hearer sees what is meant.

* The speaker has no word for the topic: it takes the hearer's (its shortest; not when the
  speaker uses that form for something else), or, when the hearer has none either, the two
  coin one: 3 random syllables. A word is never coined while half as many words as there are
  talkers are in circulation for the topic (toy bound).
* The speaker says its shortest word (of several a random one). A hearer that does not know the
  form keeps it for the topic. A hearer that has it for the topic understands: both drop every
  word for the topic that is not shorter. A hearer that has the form for another meaning
  misunderstands, and the speaker gives the form up for this topic: nobody holds one form for
  two meanings.
* Least effort (toy rule for Zipf's law of abbreviation): a word that was understood loses its
  last syllable with probability 0.015 / talkers, when the shorter form is free for both (and
  is in circulation already or the bound above leaves room for it). Words that are used more
  wear down more. Frequent meanings end with shorter words for a second reason that has nothing
  to do with wear: they are the ones whose one-syllable call settled in stage 1, while every
  word coined in stage 2 starts with 3 syllables. The plausibility table gives both parts.

**Stage 3, grammar**: iterated learning. The lexicon is now the word of the majority for every
meaning. The *things* are the ``things`` most talked-about meanings that have a word; the new
meanings are the pairs thing x property (near, far, big, small; ``properties`` of them). One
pair (two from 16 pairs on) is never taught. Stage 3 needs two things and room in the brain for
the words of a rule (``capacity`` >= words that are no things + things + properties); the room
left over is the number of whole phrases an agent can remember. Otherwise the group stays in
stage 2. Step 41 is generation 0: everybody holds the same stock of holistic phrases, one
unanalysable random word (1 to 3 syllables) per taught pair, as far as the room goes. Every
later step is one generation:

* a teacher is drawn among the adults; each learner hears ``bottleneck`` utterances of it, for
  pairs drawn evenly among the taught ones (with ``bottleneck`` 0 every taught pair once: no
  bottleneck; a run takes 0 to 1000, and with at most 22 taught pairs anything above a few
  hundred is no bottleneck either);
* the teacher says the whole it remembers; for a pair it does not remember, the word of the
  thing with its word for the property, in its order (thing first or property first); a
  property word it lacks it invents (1 to 3 random syllables);
* a learner looks for the order and the property words that explain most of what it heard (an
  utterance is explained when it is the word of its thing plus the commonest remainder of its
  property; of two equally good orders its own coin decides), remembers as wholes what its rule
  does not give, and replaces the adults.

A whole that a learner does not hear is lost, a rule is not: with a tight bottleneck the
language turns compositional, without one it stays holistic. The settled language of a group is
what more than half of its talkers hold: an order (a talker has one once it has a property
word), a word for each property, an utterance for each pair; a pair without a majority, and a
pair that was never taught, is said by the rule if there is one.

**Metrics** of a step, all for the first group (:data:`STATE_KEYS`, units in :data:`KEYS`):
``success``, the expected accuracy when a random agent tells a random other agent a meaning,
every meaning equally likely (single meanings in stages 1 and 2, the pairs in stage 3). It is
computed from the tables, inventories and learners, not sampled: a hearer that does not know a
signal guesses among all meanings, one that knows it for several is right in its share of them,
and a couple in which one cannot hear or call guesses. ``alarm`` is the same for ``predator``
alone: a hit rate, how often a warning is taken for one. It does not count false alarms (a call
about something else taken for a warning), and those are many under predator pressure.
``words`` (distinct words in use), ``synonyms`` (words beyond one per meaning),
``homonyms`` (forms in use for two meanings), ``word_length`` (mean syllables), ``meanings``
(with a settled utterance, pairs included), ``compositional`` (share of the taught pairs that
the settled language says by its rule), ``order``, ``dialects`` (share of the settled meanings
for which the second group says something else) and ``stage``: ``silent`` (nothing settled),
``calls``, ``words`` (stage 2, not every nameable meaning has its word yet), ``lexicon``,
``grammar`` (some pairs follow a rule), ``language`` (at least 0.9 of them do). The summary adds
``topographic`` (correlation between the distance of two taught pairs in meaning and the edit
distance of their utterances in syllables) and the ``outcome``: silent, calls, lexicon, language.

Lines (:func:`lines`; metrics to 3 significant digits, parameters as given):

    signals seed 8 7 params. meanings 7. consonants 8. vowels 2. population 2 5. ... coverage 0 point 7 5 5.
    signals seed 8 7 inventory. consonants m p t n s k l b. vowels i e.
    signals seed 8 7 meanings. named predator water shelter rival storm fire mate. things predator water
        shelter rival. properties far small big near. unseen predator small shelter far.
    signals seed 8 7 step 1 2. success 0 point 4 3 9. alarm 0 point 9 9 2. words 4. ... stage calls.
    q signals seed 8 7 step 1 2 success. a 0 point 4 3 9.
    q signals seed 8 7 step 2 0 next stage. a words.
    q signals seed 8 7 final outcome. a language.

and the language of the first group, at steps 20 and 40 (``... seed 8 7 step 2 0 lexicon.``) and
at the end, without a step:

    signals seed 8 7 lexicon. water ne. shelter ti. predator le. rival se. storm be se be. fire ti ni. ...
    signals seed 8 7 grammar. order thing_first. compositional 1.
    signals seed 9 1 phrases. sky_danger far me.                  (wholes the rule does not give)
    q signals seed 8 7 word near. a ti ne.
    q signals seed 8 7 say predator near. a le ti ne.
    q signals seed 8 7 meaning le ti ne. a predator near.
    q signals seed 8 7 say predator small. a le le li.           (a pair nobody was ever taught)
    q signals seed 8 7 order. a thing_first.
    q signals predict coverage meanings 1 1 bottleneck 1 2. a 0 point 6 8 1 4.     (a lesson)

A meaning without a settled word, an utterance without a meaning and a pair the language cannot
say are answered ``none``.

**Gate** (:meth:`Signals.conserved`), from the agents kept in the rollout, at every step: no
weight is negative; every word is 1 to 3 syllables of the inventory, so there are never more
words than the inventory can form; no talker uses one form for two meanings; never more words
for a meaning in circulation than half the talkers; no learner remembers more wholes than it
has room for, or a pair that is never taught; every metric is computed again from the agents
(the success exactly); the lexicon is the word of the majority, counted again talker by talker;
the rule is the word of the thing and the word of the property in the order; every utterance
leads back to its meaning; the compositional share is the share of the taught pairs that follow
the rule; step 0 is a population that guesses; the summary is the last step.
:meth:`Signals.check` replays a seed (parameters from :func:`random_params` with
``random.Random(seed)``): words, counts and utterances exactly, other numbers within 5 percent.
``say`` accepts every utterance the language has for the pair (what the majority says, and what
the rule gives), ``meaning`` every meaning of the utterance. Module level :func:`check` and
:func:`owns` also answer the lessons.

**Lessons** (:data:`LESSONS`): ``expected_success``, ``signal_bits``, ``syllables``,
``possible_words``, ``holistic_words``, ``compositional_words``, ``coverage``,
``zipf_frequency``, ``naming_max_words``, ``lexicon_capacity``. The functions that answer them
are the functions the simulation calls: the guess of a hearer is ``expected_success`` with
nothing shared (and weight tables with k settled calls give its value, which the tests hold),
the shares of the talk are ``zipf_frequency`` before predators are counted more often, the
inventory is ``syllables`` and bounds the words by ``possible_words``, the brain holds
``lexicon_capacity`` words, a naming game never has more than ``naming_max_words`` words for a
meaning, stage 3 has ``holistic_words`` pairs and needs room for ``compositional_words``, a
learner hears the share ``coverage`` of the taught pairs (the parameter ``coverage`` of a run is
this function of its taught pairs and its bottleneck), and a group with ``signal_bits`` 0 is
silent. The ``coverage`` lesson asks for 1 utterance or more: ``bottleneck 0`` in a run means no
bottleneck, not no utterance, and a lesson must not say otherwise.

Hand-off. :func:`handoff_in` turns the summary of level 10 (senses) into parameters: the share
with ears and a voice, the neurons, the predator pressure, and from group living the size of
the band and whether there are two. The summary gives upward (to level 12, society):
``success`` (how well the group communicates, 0 to 1), ``words`` (the size of its lexicon),
``meanings`` (what it can say), ``compositional`` and ``order`` (whether, and how, it has a
grammar), ``dialects``, ``stage`` and ``outcome``. What level 12 makes of them is its own rule.

Plausibility (what a run must look like; ``tests/test_evo_signals.py`` holds it). Measured over
the canonical seeds 1 to 400: every gate passes; outcomes lexicon 178, language 122, calls 62,
silent 38; a run takes 0.2 s on average and under 1 s at most (0.74 s on the slowest seed, on the
machine of the review; the time depends on the machine and its load), and gives 1419 to 1543
lines of at most 89 tokens. The table below was measured on seeds 1 to 30; seeds 201 to 230 give
the same picture (success 0.40 after the calls and 0.99 after the naming game, compositional
1.00 at bottleneck 12, 0.86 at 24, 0.41 at 36, 0.00 at 96, dialects 0.99 at the end). With
chosen parameters (8 meanings, 40 agents, 12 syllables, 4 things x 3 properties, 30 seeds each):

    without ears or voice    silent, success at the guess 1/8, no word, at every step
    success rises            0.125 at step 0, 0.38 after the calls, 0.99 after the naming game
                             (0.93 at least), 0.98 at the end (0.94 at least; bottleneck 12)
    the alarm call first     pressure 1: the warning is understood 0.98 of the time at step 10,
                             the other meanings 0.26 at step 20; pressure 0: 0.23 at step 10.
                             A warning call is settled at step 20 in 7, 30, 29, 27, 24 of 30
                             runs at pressure 0, 0.25, 0.5, 0.75, 1 (see the open points)
    false alarms             hearers take a call about another nameable thing for a warning
                             0.41 of the time at step 10 and 0.35 at step 20 under pressure 1
                             (0.11 and 0.10 at pressure 0; a guess among 8 is 0.125): predators
                             are much of the talk there, and a call that is not settled is
                             read as the likeliest thing. Calls only, 4 nameable, step 60:
                             pressure 1 leaves 0.25 false alarms and 2.9 of 4 calls settled
    punishing failures       calls only (32 neurons, pressure 0, 4 nameable meanings): 3.8 of 4
                             calls settled at step 60, success 0.54 of 0.5625 possible; with
                             the punishment switched off 2.5 calls, success 0.39
    the naming game          10 meanings, 30 agents: 56 words in circulation at the peak (46 to
                             67), 0.9 synonyms left at step 40 (3 at most), a word for 9.97 of 10
    Zipf's abbreviation      same runs, step 40: the 3 most frequent meanings 1.24 syllables,
                             the 3 rarest 2.61. With the wear switched off 1.91 and 3.00: most
                             of that difference is the calls of stage 1, kept as words. What
                             wear alone does shows in the words coined in stage 2: the more
                             frequent half of them 1.84 syllables, the rarer half 2.58 (the
                             frequent half shorter in 28 of 30 runs; 3 and 3 without wear)
    Kirby's bottleneck       11 taught pairs. bottleneck 6, 8, 12: compositional 1.0 (0.91 at
                             least), a language in all runs; 24: 0.87; 36: 0.46; 48: 0.14; 96:
                             0.00 (0.09 at most); 192 or no bottleneck: 0 in every run
    too tight a bottleneck   the rule is there but the property words do not get through:
                             success 0.52, 0.67, 0.85, 0.93, 0.98 at bottleneck 3, 4, 6, 8, 12
    what the rule buys       a compositional language says the pair nobody was taught; a
                             holistic one cannot (success 0.924 = (11 + 1/12) / 12 at best)
    dialects                 two groups differ in 0.95 of the meanings at step 40, 0.98 at the
                             end (they never meet and coin at random: drift by construction)
    topographic similarity   0.74 in the languages of the canonical seeds, -0.01 in the holistic ones

What is taken from published models and what is invented.

* The signalling game with states, signals and acts and a common interest is Lewis's (1969).
  Learning it by reinforcement, a choice in proportion to accumulated payoffs, is Roth and
  Erev's rule (1995). With two equally likely states, two signals and two acts that rule finds a
  signalling system (shown by Argiento, Pemantle, Skyrms and Volkov 2009, from memory); with
  more states, or states that are not equally likely, it can stay in partial pooling for good
  (two meanings on one signal, two signals for one meaning), which is why failures are punished
  here (both the pooling and reinforcement with punishment: after Barrett 2006, from memory;
  the table above has this toy's own comparison). Invented: one table per agent for sending and
  for understanding (the game has a sender's and a receiver's strategy; an agent that does both
  by the same associations is a simplification), the payoffs 2 and 1, the floor of 1, the
  pairing in rounds, the Zipf shares of the states and everything about predators.
* Alarm calls that mean a kind of predator exist (vervet monkeys: Seyfarth, Cheney and Marler
  1980). Nothing of them is modelled; here a warning only pays more and comes up more often.
  That calls come before words, and become words, is the toy's timetable (see the top).
* The naming game is Steels's (1995), in the minimal form of Baronchelli and others (2006):
  invent, adopt on failure, drop the competitors on success. Invented for the toy: several
  meanings at once, a speaker without a word takes the hearer's, that after a success a word
  shorter than the one used is kept (in the minimal game every other word is dropped), the bound
  of half the agents on the words for a meaning, one form for one meaning (the idea of mutual
  exclusivity in word learning, after Markman and Wachtel 1988), and the wearing down of words.
* Zipf's law of abbreviation (Zipf 1935, 1949): frequent words are short. That use wears a word
  down by a syllable with a fixed chance is this toy's stand-in for least effort; real
  shortening has many causes. In this toy frequent meanings are short for two reasons, wear
  and the one-syllable calls they keep from stage 1; only the first is meant as a model of
  the law, the second follows from the timetable.
* Iterated learning is Kirby's (2001; in the laboratory Kirby, Cornish and Smith 2008): a
  language that must pass through a bottleneck becomes compositional, because rules pass and
  unheard wholes do not. Taken: the bottleneck, learners that generalise, invention that uses
  what the speaker knows. Invented: one teacher for a whole generation, the learner's simple
  search for remainders, the stock of phrases at the start, that the words of the things come
  from stage 2, the room in the brain. That a bottleneck can also be too tight for a stable
  language is known from these models as well; here it shows as property words that do not
  reach every learner. Topographic similarity as a measure of compositionality is after
  Brighton and Kirby (2006): a correlation between how far two meanings are apart and how far
  their signals are; here the meanings differ in 1 or 2 parts and the signals by their edit
  distance in syllables, over the taught pairs only.
* log2 n bits in one of n equally likely signals: Shannon (1948).
* Not modelled: sounds, hearing errors, meaning beyond a list of words, syntax beyond two
  words, how a brain stores a word (8 neurons a word is arbitrary), children, lying, any
  reason to talk other than a common interest, and natural selection: nothing here is
  inherited by genes, the agents learn (stages 1 and 2) and are replaced by learners (stage 3).
  The numbers of a run tell the toy's rules; none of them is a measured fact about real
  language, real animals or the history of either.

Open points, said plainly. Under high predator pressure several calls come to mean predator
and none is the majority's (synonyms: the partial pooling named above, which the larger reward
makes more likely and which nothing in the game punishes), so the lexicon of step 20 lists no
warning call in up to a fifth of the runs though the warning is understood (``alarm`` near 1).
``alarm`` near 1 is less than it sounds: under pressure the hearers take a third and more of
the other calls for a warning as well (the table has the numbers), so "the alarm call first"
is true of what is sent and settled, while what is heard is at first mostly a readiness to
flee. No metric of a step shows the false alarms; the tests compute them from the weights.
A bottleneck below about 8 utterances loses property words on the way (success 0.85 at 6), and
the step after the stock shows a dip in ``success`` (0.92 to 0.47 at bottleneck 12) while the
learners that missed a pair have nothing to say. ``success`` counts single meanings up to step
40 and pairs from step 41 on, so it is not one curve across that border. The lesson rules
``expected_success`` and ``zipf_frequency`` reject about half of the drawn inputs, so they give
about half as many lessons as the other rules. Some syllables are also ordinary words of other
lines (``no``, ``so``, ``to``, ``be``, ``me``).

Reproducibility. Stages 2 and 3 use ``random.Random``; stage 1 uses the raw PCG64 bit stream of
the seed, turned into fractions here and not by ``Generator`` (the tests pin it). The weights of
stage 1 are whole numbers held in float64, so every sum that decides a draw is exact; the
success is summed talker by talker and with ``math.fsum``; no numpy sum, sort or random
routine touches a number that is not whole, except one stable sort of distinct fractions; no
set is ever walked, so the hash seed does not matter. ``coverage``, ``bits`` and ``topographic``
go through pow, log2 and sqrt and are printed to 3 digits. The streams of a seed are
``random.Random(8 * seed)`` for the world, ``8 * seed + 1`` and ``+ 2`` for the two groups and
PCG64 of ``8 * seed + 5`` and ``+ 6`` for their calls; the canonical parameters come from
``random.Random(seed)``, a stream that another seed uses for its world or a group (harmless:
the draws serve different things, and no claim rests on seeds being independent). The tests
pin a digest of whole rollouts, so a numpy that changed anything would be noticed.
"""

from __future__ import annotations

import bisect
import math
import random
from dataclasses import dataclass, field

import numpy as np

from haishool.cosmos import Rollout, close, query_lines, seeds, state_line, summary_lines
from haishool.evo import Input, LessonGate, Rule, sig
from haishool.truth import Line, Verdict, is_finite, num, parse_num

SIM = "signals"

CAP = 60  # agents at most
STEPS = 60  # output steps after step 0
PHASE = 20  # output steps of each of the three stages
SNAPSHOTS = (20, 40, 60)  # steps at which the language is written down (the last one is "final")
MAX_SYLLABLES = 3  # syllables in a word at most
#: the fixed vocabulary of meanings; a run uses 2 to 12 of them, ``predator`` always
VOCABULARY = ("predator", "food", "water", "mate", "sky_danger", "shelter", "young", "rival", "fire", "path",
              "storm", "night")
PROPERTIES = ("near", "far", "big", "small")
CONSONANTS = ("k", "t", "m", "n", "s", "l", "p", "b")
VOWELS = ("a", "i", "u", "e", "o")
ORDERS = ("thing_first", "property_first")
STAGES = ("silent", "calls", "words", "lexicon", "grammar", "language")
OUTCOMES = ("silent", "calls", "lexicon", "language")

NEURONS_PER_WORD = 8  # toy rule: neurons that one word (a sound pattern tied to a meaning) takes
LEXICON_MIN = 5  # toy rule: a brain that holds fewer words than this keeps to calls and coins no words
FAMILY = 6  # toy rule of the hand-off: agents that do not live in groups talk within a family of this size
PRESSURE_GAIN = 4.0  # toy rule: predators come up (1 + 4 * pressure) times as often as their Zipf rank says
STAKE = 2  # toy rule: a warning understood pays 1 + round(2 * pressure) times what another signal pays
REWARD = 2  # stage 1: added to the weights a talker used when the act fits the state
PUNISH = 1  # stage 1: taken from them when it does not; a weight never falls below 1, where it starts
ROUNDS = 200  # stage 1: rounds per output step; in a round every talker plays once
GAMES = 60  # stage 2: naming games per talker and output step
WEAR = 0.015  # stage 2: a word understood loses its last syllable with probability WEAR / talkers
LANGUAGE_SHARE = 0.9  # compositional share from which a grammar counts as a language
MAX_SEED = 10 ** 15  # the largest seed a run takes, and so the largest a question can name
MAX_BOTTLENECK = 1000  # utterances per learner at most: far beyond the 22 pairs taught at most, and fast
_TWO_M53 = 1.0 / 9007199254740992.0
_DIGITS = frozenset("0123456789")
_SYLLABLES = frozenset(c + v for c in CONSONANTS for v in VOWELS)

INPUT_KEYS = ("meanings", "consonants", "vowels", "population", "predator_pressure", "ears_voice", "neurons",
              "bottleneck", "split", "things", "properties")
PARAM_KEYS = INPUT_KEYS + ("syllables", "possible_words", "capacity", "talkers", "bits", "pairs", "coverage")
STATE_KEYS = ("success", "alarm", "words", "synonyms", "homonyms", "word_length", "meanings", "compositional",
              "order", "dialects", "stage")
SUMMARY_KEYS = ("stage", "success", "words", "meanings", "compositional", "order", "word_length", "dialects",
                "topographic", "outcome")
COUNT_KEYS = frozenset({"words", "synonyms", "homonyms", "meanings", "consonants", "vowels", "population",
                        "neurons", "bottleneck", "split", "things", "properties", "syllables", "possible_words",
                        "capacity", "talkers", "pairs"})
WORD_KEYS = frozenset({"order", "stage", "outcome"})

KEYS: dict[str, str] = {
    "meanings": "step: meanings with a settled utterance (single meanings, and pairs from stage 3 on; count). "
                "parameter: single meanings that matter in this world, 2 to 12",
    "consonants": "consonants in the phoneme inventory, 1 to 8 (parameter)",
    "vowels": "vowels in the phoneme inventory, 1 to 5 (parameter)",
    "population": "agents, at most 60 (parameter)",
    "predator_pressure": "chance to be attacked in a generation, 0 to 1, from the senses level (parameter)",
    "ears_voice": "share of the agents with ears and a voice, 0 to 1 (parameter)",
    "neurons": "neurons per agent; a word takes 8 (parameter, toy rule)",
    "bottleneck": "utterances a learner hears in stage 3, 1 to 1000; 0 is no bottleneck, the whole language "
                  "(parameter)",
    "split": "1 when the population lives in two groups that never meet, else 0 (parameter)",
    "things": "things that are talked about with a property in stage 3, 2 to 6 (parameter)",
    "properties": "properties in stage 3, 2 to 4 (parameter)",
    "syllables": "syllables the inventory gives: consonants times vowels",
    "possible_words": "words of 1 to 3 syllables the inventory gives",
    "capacity": "words an agent can hold: neurons // 8 (toy rule)",
    "talkers": "agents with ears and a voice (count)",
    "bits": "information in one word when the nameable meanings are equally likely: log2 of their number, bits",
    "pairs": "meanings of stage 3: things that have a word times properties, what a holistic language needs "
             "words for; 0 when stage 3 was not reached",
    "coverage": "expected share of the taught pairs a learner hears through the bottleneck; 1 without a "
                "bottleneck, 0 when stage 3 was not reached",
    "success": "expected communication accuracy in the first group, 0 to 1: a random speaker, a random other "
               "hearer, every meaning equally likely; computed from the agents' states, not sampled",
    "alarm": "the same accuracy for the meaning predator alone, 0 to 1: warnings taken for a warning (false "
             "alarms are not counted)",
    "words": "distinct words in use in the first group (count)",
    "synonyms": "words in use beyond one per meaning, summed over the meanings (count)",
    "homonyms": "forms in use for more than one meaning (count)",
    "word_length": "mean number of syllables of the words in use",
    "compositional": "share of the taught pairs that the settled language expresses by its rule, 0 to 1",
    "order": "thing_first, property_first, or none while no rule is settled",
    "dialects": "share of the meanings for which the two groups have settled on different utterances, 0 to 1",
    "stage": "silent, calls, words, lexicon, grammar or language",
    "topographic": "final correlation between how far two taught pairs are in meaning and in syllables, -1 to 1",
    "outcome": "silent, calls, lexicon or language",
}

Form = tuple  # a word or an utterance: a tuple of syllables


# ---------------------------------------------------------------------------------------------
# the rules, one function each: the simulation calls them, and the lessons ask for them


def expected_success(signals: int, states: int, shared: int) -> float:
    """Accuracy of a signalling system with ``states`` equally likely states and ``signals``
    signals when ``shared`` states have a signal of their own that everybody understands. In any
    other state the sender picks a signal at random; a hearer that hears a signal without a
    settled meaning guesses among all the states.

    >>> expected_success(4, 4, 4), expected_success(4, 4, 0), expected_success(4, 4, 3)
    (1.0, 0.25, 0.765625)
    """
    return shared / states + (states - shared) / states * (signals - shared) / signals / states


def signal_bits(signals: int) -> float:
    """Information in one of ``signals`` equally likely signals, in bits.

    >>> signal_bits(8), signal_bits(1)
    (3.0, 0.0)
    """
    return math.log2(signals)


def syllables(consonants: int, vowels: int) -> int:
    """Syllables of one consonant and one vowel."""
    return consonants * vowels


def possible_words(syllables: int, length: int) -> int:  # noqa: A002 - the lesson's input name
    """Words of 1 to ``length`` syllables.

    >>> possible_words(4, 3), possible_words(1, 3)
    (84, 3)
    """
    return sum(syllables ** i for i in range(1, length + 1))


def holistic_words(things: int, properties: int) -> int:
    """Words a holistic language needs: one for every pair of a thing and a property."""
    return things * properties


def compositional_words(things: int, properties: int) -> int:
    """Words a compositional language needs: one per thing and one per property."""
    return things + properties


def coverage(meanings: int, bottleneck: int) -> float:
    """Expected share of ``meanings`` equally likely meanings among ``bottleneck`` utterances.

    >>> coverage(2, 1), coverage(2, 2), coverage(12, 0)
    (0.5, 0.75, 0.0)
    """
    return 1.0 - (1.0 - 1.0 / meanings) ** bottleneck


def zipf_frequency(rank: int, meanings: int) -> float:
    """Share of the talk about the meaning of this rank: 1 / rank over the sum of 1 / r.

    >>> zipf_frequency(1, 2), zipf_frequency(1, 1)
    (0.6666666666666666, 1.0)
    """
    return (1.0 / rank) / math.fsum(1.0 / r for r in range(1, meanings + 1))


def naming_max_words(agents: int) -> int:
    """Most words for one meaning in circulation among ``agents`` (toy bound of this level)."""
    return agents // 2


def lexicon_capacity(neurons: int) -> int:
    """Words an agent can hold (toy rule)."""
    return neurons // NEURONS_PER_WORD


def stage_of(mode: str, settled: int, nameable: int, compositional: float) -> str:
    """The stage of a step, from the game that is played (``mode``: silent, calls, naming, grammar),
    the single meanings with a settled signal, how many could have one, and the compositional share."""
    if mode == "silent" or (mode == "calls" and settled == 0):
        return "silent"
    if mode == "calls":
        return "calls"
    if compositional >= LANGUAGE_SHARE:
        return "language"
    if compositional > 0:
        return "grammar"
    return "lexicon" if settled >= nameable else "words"


def outcome_of(stage: str) -> str:
    return {"silent": "silent", "calls": "calls", "language": "language"}.get(stage, "lexicon")


RULES: dict[str, Rule] = {
    "expected_success": Rule(
        (Input("signals", 1, 12, True), Input("states", 1, 12, True), Input("shared", 0, 12, True)),
        expected_success,
        "accuracy of a signalling system with equally likely states. shared states have their own signal that "
        "all understand. in another state the sender takes any signal and a hearer of an unsettled signal "
        "guesses among all states. shared over states plus states minus shared over states times signals "
        "minus shared over signals over states",
        lambda signals, states, shared: shared <= min(signals, states)),
    "signal_bits": Rule(
        (Input("signals", 1, 4096, True),), signal_bits,
        "information in one of that many equally likely signals in bits. log2 of signals"),
    "syllables": Rule(
        (Input("consonants", 1, 20, True), Input("vowels", 1, 10, True)), syllables,
        "syllables of one consonant and one vowel. consonants times vowels"),
    "possible_words": Rule(
        (Input("syllables", 1, 40, True), Input("length", 1, 4, True)), possible_words,
        "words of 1 up to length syllables. syllables plus syllables squared and so on up to the power length"),
    "holistic_words": Rule(
        (Input("things", 1, 20, True), Input("properties", 1, 20, True)), holistic_words,
        "words a holistic language needs. one for every pair. things times properties"),
    "compositional_words": Rule(
        (Input("things", 1, 20, True), Input("properties", 1, 20, True)), compositional_words,
        "words a compositional language needs. one per thing and one per property. things plus properties"),
    # from 1 utterance on: in the parameters of a run ``bottleneck 0`` means no bottleneck (coverage 1)
    "coverage": Rule(
        (Input("meanings", 2, 60, True), Input("bottleneck", 1, 200, True)), coverage,
        "expected share of equally likely meanings that a learner hears among bottleneck utterances. "
        "1 minus 1 minus 1 over meanings to the power bottleneck. bottleneck is 1 or more"),
    "zipf_frequency": Rule(
        (Input("rank", 1, 40, True), Input("meanings", 1, 40, True)), zipf_frequency,
        "zipf share of the talk about the meaning of that rank before predators are counted more often. "
        "1 over rank divided by the sum of 1 over r for r from 1 to meanings",
        lambda rank, meanings: rank <= meanings),
    "naming_max_words": Rule(
        (Input("agents", 2, 1000, True),), naming_max_words,
        "most words for one meaning in circulation among that many agents. half the agents rounded down. "
        "toy bound. a word is only coined when neither speaker nor hearer has one"),
    "lexicon_capacity": Rule(
        (Input("neurons", 0, 4096, True),), lexicon_capacity,
        "words an agent can hold. neurons divided by 8 rounded down. toy rule"),
}

LESSONS = LessonGate(SIM, RULES)


def lesson_gate() -> LessonGate:
    return LESSONS


# ---------------------------------------------------------------------------------------------
# values in lines


def dense_value(value: float | int | str) -> str:
    """A metric as it stands in a line: a word, an exact count, or a number to 3 significant digits.

    >>> dense_value(0.98512), dense_value(16), dense_value(1.5), dense_value("language")
    ('0 point 9 8 5', '1 6', '1 point 5', 'language')
    """
    if isinstance(value, str):
        return value
    if isinstance(value, int):
        return num(value)
    return num(sig(float(value)))


def param_value(key: str, value: float | int) -> str:
    """A parameter as it stands in the ``params`` record: an input as given, a derived number to
    3 significant digits.

    >>> param_value("predator_pressure", 0.57), param_value("neurons", 256), param_value("coverage", 0.64412)
    ('0 point 5 7', '2 5 6', '0 point 6 4 4')
    """
    if isinstance(value, int):
        return num(value)
    return num(float(value), sig=6) if key in INPUT_KEYS else dense_value(value)


def text_of(form: Form | None) -> str:
    """An utterance as dense words: its syllables, or ``none``."""
    return " ".join(form) if form else "none"


@dataclass
class SignalsRollout(Rollout):
    """A :class:`Rollout` that also keeps the world of the run, the agents of every step and
    the language of the first group at the three snapshot steps."""
    #: phonemes, syllables, the meanings in rank order, their shares of the talk, what can be named, ...
    world: dict = field(default_factory=dict)
    #: per step, per group: the state the metrics are computed from (:func:`view`)
    states: list[tuple[dict, ...]] = field(default_factory=list)
    #: step -> the first group's language: lexicon, properties, say, rule, order, compositional
    languages: dict[int, dict] = field(default_factory=dict)

    def value(self, text: float | int | str) -> str:
        return dense_value(text)


def random_params(rng: random.Random, rules: int = 7) -> dict[str, float | int]:
    """A world: 4 to 12 meanings, 2 to 8 consonants and 2 to 5 vowels, 8 to 60 agents; everybody
    has ears and a voice with chance 0.6, nobody with chance 0.1, else any share; 32 to 1024
    neurons (a power of two); no bottleneck with chance 0.15, else 6 to 192 utterances (evenly
    on a logarithmic scale); two groups with chance 0.3; 4 to 6 things and 3 or 4 properties."""
    share = rng.random()
    return {"meanings": rng.randint(4, 12),
            "consonants": rng.randint(2, 8),
            "vowels": rng.randint(2, 5),
            "population": rng.randint(8, CAP),
            "predator_pressure": round(rng.uniform(0.0, 1.0), 2),
            "ears_voice": 1.0 if share < 0.6 else 0.0 if share < 0.7 else round(rng.uniform(0.0, 1.0), 2),
            "neurons": 2 ** rng.randint(5, 10),
            "bottleneck": 0 if rng.random() < 0.15 else int(round(6 * 2 ** rng.uniform(0.0, 5.0))),
            "split": 1 if rng.random() < 0.3 else 0,
            "things": rng.randint(4, 6),
            "properties": rng.randint(3, 4)}


def _number(below: dict, names: tuple[str, ...]) -> float | None:
    for name in names:
        value = below.get(name)
        if isinstance(value, (int, float)) and not isinstance(value, bool) and is_finite(value):
            return float(value)
    return None


def handoff_in(below: dict) -> dict[str, float | int]:
    """Parameters of this level from the summary of level 10 (senses). Only what the senses decide
    is returned; the chain draws the rest (meanings, phonemes, bottleneck, things, properties).

    * ``ears_voice``: the share with ears and a voice (``talkers``; else the smaller of ``eared``
      and ``voice``, which is the most that share can be), cut to 0..1.
    * ``neurons``: the mean neurons per agent (``neurons``), rounded.
    * ``predator_pressure``: the chance to be attacked in a generation (``predator_pressure``), cut
      to 0..1.
    * ``population`` and ``split`` from group living (``group`` yes or no, ``population``): agents
      that live in groups talk in a band of at most 60, and form two bands (``split`` 1, which in
      this toy never meet) when there are 120 or more of them; agents that do not live in groups talk
      within a family of at most 6 (toy rules).

    A key the summary does not hold is left out, so the default or drawn value stays.

    >>> handoff_in({"talkers": 0.9, "neurons": 181.3, "predator_pressure": 0.45, "group": "yes", "population": 200})
    {'ears_voice': 0.9, 'neurons': 181, 'predator_pressure': 0.45, 'population': 60, 'split': 1}
    """
    out: dict[str, float | int] = {}
    share = _number(below, ("talkers",))
    if share is None:
        eared, voice = _number(below, ("eared",)), _number(below, ("voice",))
        share = min(eared, voice) if eared is not None and voice is not None else None
    if share is not None:
        out["ears_voice"] = sig(min(1.0, max(0.0, share)))
    neurons = _number(below, ("neurons",))
    if neurons is not None:
        out["neurons"] = max(0, int(round(neurons)))
    pressure = _number(below, ("predator_pressure",))
    if pressure is not None:
        out["predator_pressure"] = sig(min(1.0, max(0.0, pressure)))
    population = _number(below, ("population",))
    group = below.get("group")
    if population is not None:
        living = group != "no"
        out["population"] = int(min(CAP if living else FAMILY, max(0.0, population)))
        out["split"] = 1 if living and population >= 2 * CAP else 0
    elif group == "no":
        out["population"] = FAMILY
        out["split"] = 0
    return out


# ---------------------------------------------------------------------------------------------
# the world of a run


def _world(seed: int, p: dict) -> dict:
    """Phonemes, syllables, meanings and how often each is talked about: all from the seed."""
    rng = random.Random(8 * seed)
    consonants = tuple(rng.sample(CONSONANTS, p["consonants"]))
    vowels = tuple(rng.sample(VOWELS, p["vowels"]))
    forms = [c + v for c in consonants for v in vowels]
    rng.shuffle(forms)
    n = p["meanings"]
    others = rng.sample([m for m in VOCABULARY if m != "predator"], n - 1)
    place = rng.randrange(n)
    meanings = tuple(others[:place] + ["predator"] + others[place:])
    properties = tuple(rng.sample(PROPERTIES, p["properties"]))
    raw = [zipf_frequency(rank, n) * (1.0 + PRESSURE_GAIN * p["predator_pressure"] if m == "predator" else 1.0)
           for rank, m in enumerate(meanings, start=1)]
    total = math.fsum(raw)
    weights = tuple(x / total for x in raw)
    priority = sorted(range(n), key=lambda i: (-weights[i], i))
    named = tuple(priority[:min(n, p["capacity"])])
    cum, acc = [], 0.0
    share = math.fsum(weights[i] for i in named)
    for i in named:
        acc += weights[i] / share
        cum.append(acc)
    if cum:
        cum[-1] = 1.0
    return {"consonants": consonants, "vowels": vowels, "syllables": tuple(forms), "meanings": meanings,
            "properties": properties, "weights": weights, "named": named, "cum": tuple(cum),
            "calls": min(len(named), len(forms)), "predator": meanings.index("predator"),
            "payoff": 1 + int(STAKE * p["predator_pressure"] + 0.5), "capacity": p["capacity"],
            "bottleneck": p["bottleneck"], "things": p["things"], "guess": expected_success(1, n, 0)}


def _can_signal(world: dict) -> bool:
    """One of ``calls`` equally likely calls carries log2(calls) bits: nothing with a single call
    (one syllable, or one meaning that can be named), so such a population stays silent."""
    return world["calls"] >= 1 and signal_bits(world["calls"]) > 0


def _compose(order: str, word: Form, morph: Form) -> Form:
    return word + morph if order == "thing_first" else morph + word


def _strip(utterance: Form, word: Form, order: str) -> Form:
    """What is left of an utterance when the thing's word is taken off its front (thing_first)
    or its end (property_first); ``()`` when the word is not there or nothing is left."""
    k = len(word)
    if len(utterance) <= k:
        return ()
    if order == "thing_first":
        return utterance[k:] if utterance[:k] == word else ()
    return utterance[:-k] if utterance[-k:] == word else ()


def induce(seen: dict[int, Form], lexicon: dict[int, Form], pairs: list[tuple[int, int]],
           coin: str) -> tuple[str, dict[int, Form]]:
    """What a learner makes of the utterances it has heard: the order and the property words that
    explain most of them. An utterance is explained when it is the word of its thing with the
    same remainder as the other utterances of its property, the most frequent remainder (of
    equally frequent ones the first heard). Of two orders that explain equally many the
    learner's own ``coin`` wins."""
    best: tuple[str, dict[int, Form]] = (coin, {})
    best_score = 0
    for order in (coin,) + tuple(o for o in ORDERS if o != coin):
        rests: dict[int, dict[Form, int]] = {}
        for x, utterance in seen.items():
            t, prop = pairs[x]
            rest = _strip(utterance, lexicon[t], order)
            if rest:
                counts = rests.setdefault(prop, {})
                counts[rest] = counts.get(rest, 0) + 1
        morph = {prop: max(counts, key=counts.get) for prop, counts in rests.items()}
        score = sum(rests[prop][rest] for prop, rest in morph.items())
        if score > best_score:
            best, best_score = (order, morph), score
    return best


def _shortest(words: tuple | list) -> list[Form]:
    least = min(len(w) for w in words)
    return [w for w in words if len(w) == least]


class _Group:
    """One band of talkers that only meet each other, through the three stages."""

    def __init__(self, world: dict, talkers: int, seed: int, index: int) -> None:
        self.w = world
        self.T = talkers
        self.rng = random.Random(8 * seed + 1 + index)
        self.bits = np.random.PCG64(8 * seed + 5 + index)
        k = len(world["named"])
        # a call must carry information: two talkers, and at least two calls to choose from
        self.mode = "calls" if talkers >= 2 and _can_signal(world) else "silent"
        if self.mode == "calls":
            # one table of weights per talker, meaning x call, used both to send and to understand
            self.A = np.ones((talkers, len(world["meanings"]), world["calls"]))
            self.cum = np.array(world["cum"])
            self.named = np.array(world["named"])
            self.pay = np.full(k, float(REWARD))
            if world["predator"] in world["named"]:
                self.pay[world["named"].index(world["predator"])] = float(REWARD * world["payoff"])
        self.seen = 0.0

    # -- stage 1: the signalling game ---------------------------------------------------------

    def _uniform(self, size: int) -> np.ndarray:
        """Fractions straight from the PCG64 bit stream (the top 53 bits), which numpy keeps fixed."""
        return (self.bits.random_raw(size) >> np.uint64(11)).astype(np.float64) * _TWO_M53

    def _signal(self) -> None:
        """ROUNDS rounds. In a round the talkers pair off at random; in each pair one sees what is
        the case and calls, with a call drawn by its weights for that meaning; the other acts on
        the call, with a meaning drawn by its weights for that call. When the act fits, both add
        the payoff to the weight of that meaning and call; when it does not, each takes PUNISH
        from the weight it used, never below 1."""
        A, T = self.A, self.T
        half = T // 2
        _, n_meanings, n_calls = A.shape
        flat = A.reshape(-1)  # a view: A is one contiguous block
        # the fractions of all rounds in one draw: the same stream, in the same order, as one
        # draw per use (pairing T, states, calls and acts half each, round after round)
        per = T + 3 * half
        u = self._uniform(ROUNDS * per).reshape(ROUNDS, per)
        for k in range(ROUNDS):
            row = u[k]
            perm = np.argsort(row[:T], kind="stable")
            send, recv = perm[:half], perm[half:2 * half]
            state = np.searchsorted(self.cum, row[T:T + half], side="right")
            meant = self.named[state]
            c = np.cumsum(A[send, meant], axis=1)
            call = (c > (row[T + half:T + 2 * half] * c[:, -1])[:, None]).argmax(axis=1)
            c = np.cumsum(A[recv, :, call], axis=1)
            act = (c > (row[T + 2 * half:] * c[:, -1])[:, None]).argmax(axis=1)
            ok = act == meant
            no = ~ok
            # positions in the flat tables; every talker plays once a round, so none repeats
            sent = (send * n_meanings + meant) * n_calls + call
            taken = (recv * n_meanings + act) * n_calls + call  # where the act fits: the meaning meant
            pay = self.pay[state[ok]]
            flat[sent[ok]] += pay
            flat[taken[ok]] += pay
            for at in (sent[no], taken[no]):
                flat[at] = np.maximum(1.0, flat[at] - PUNISH)

    # -- stage 2: the naming game -------------------------------------------------------------

    def _start_naming(self) -> None:
        """The settled calls become the first words: a talker keeps the call of the majority where
        it is its own favourite; everything else has to be named anew."""
        w = self.w
        named, calls = w["named"], w["syllables"]
        settled = view(self.state(), w, full=False)["lexicon"]
        self.inv: list[dict[int, list[Form]]] = [{} for _ in range(self.T)]
        self.own: list[dict[Form, int]] = [{} for _ in range(self.T)]
        self.circ: dict[int, dict[Form, int]] = {m: {} for m in named}
        favourite = _favourites(self.A, 2)
        for m in named:
            form = settled.get(w["meanings"][m])
            if form is not None:  # a call is settled for one meaning only
                call = calls.index(form[0])
                for i in range(self.T):
                    if favourite[i][m] == call:
                        self._add(i, m, form)
        self.mode = "naming"
        self.bound = naming_max_words(self.T)
        self.wear = WEAR / self.T

    def _add(self, i: int, m: int, form: Form) -> None:
        self.inv[i].setdefault(m, []).append(form)
        self.own[i][form] = m
        held = self.circ[m]
        held[form] = held.get(form, 0) + 1

    def _drop(self, i: int, m: int, form: Form) -> None:
        self.inv[i][m].remove(form)
        del self.own[i][form]
        held = self.circ[m]
        held[form] -= 1
        if not held[form]:
            del held[form]

    def _keep(self, i: int, m: int, form: Form) -> None:
        """After a success: the word that worked stays, and so does anything shorter."""
        for other in [x for x in self.inv[i][m] if x != form and len(x) >= len(form)]:
            self._drop(i, m, other)

    def _coin(self, *owners: dict) -> Form | None:
        syl = self.w["syllables"]
        for _ in range(20):
            form = tuple(self.rng.choice(syl) for _ in range(MAX_SYLLABLES))
            if not any(form in own for own in owners):
                return form
        return None

    def _name(self) -> None:
        """GAMES games per talker: a speaker names a topic for a hearer who sees what is meant."""
        rng, T, inv, own = self.rng, self.T, self.inv, self.own
        named, cum = self.w["named"], self.w["cum"]
        for _ in range(GAMES * T):
            i = rng.randrange(T)
            j = rng.randrange(T - 1)
            if j >= i:
                j += 1
            m = named[bisect.bisect_right(cum, rng.random())]
            mine = inv[i].get(m)
            if not mine:
                theirs = inv[j].get(m)
                if theirs:  # the speaker has no word and takes the hearer's
                    form = _shortest(theirs)[0]
                    if form not in own[i]:
                        self._add(i, m, form)
                elif len(self.circ[m]) < self.bound:  # neither has one: a new word is coined
                    form = self._coin(own[i], own[j])
                    if form is not None:
                        self._add(i, m, form)
                        self._add(j, m, form)
                continue
            short = _shortest(mine)
            form = short[rng.randrange(len(short))] if len(short) > 1 else short[0]
            known = own[j].get(form)
            if known is None:  # a new word for the hearer, who sees what is meant and keeps it
                self._add(j, m, form)
            elif known == m:  # understood
                if len(form) > 1 and rng.random() < self.wear:
                    worn = form[:-1]  # least effort: the word loses its last syllable
                    if own[i].get(worn, m) == m and own[j].get(worn, m) == m \
                            and (worn in self.circ[m] or len(self.circ[m]) < self.bound):
                        for a in (i, j):
                            if worn not in own[a]:
                                self._add(a, m, worn)
                        form = worn
                self._keep(i, m, form)
                self._keep(j, m, form)
            else:  # the hearer uses this form for something else: the speaker gives it up
                self._drop(i, m, form)

    # -- stage 3: iterated learning -----------------------------------------------------------

    def _random_word(self) -> Form:
        syl = self.w["syllables"]
        return tuple(self.rng.choice(syl) for _ in range(self.rng.randint(1, MAX_SYLLABLES)))

    def _start_grammar(self) -> bool:
        """Generation 0: everybody holds the settled lexicon and a shared stock of holistic
        phrases, one unanalysable word per taught pair, as far as the brain has room."""
        w = self.w
        settled = view(self.state(), w, full=False)["lexicon"]
        lexicon = {m: settled[w["meanings"][m]] for m in w["named"] if w["meanings"][m] in settled}
        things = [m for m in w["named"] if m in lexicon][:w["things"]]
        n_prop = len(w["properties"])
        # the words of a compositional language must fit: the lexicon holds the things already
        room = w["capacity"] - (len(lexicon) - len(things)) - compositional_words(len(things), n_prop)
        if len(things) < 2 or room < 0:
            return False
        pairs = [(t, prop) for t in things for prop in range(n_prop)]
        assert len(pairs) == holistic_words(len(things), n_prop)
        held = sorted(self.rng.sample(range(len(pairs)), 1 if len(pairs) < 16 else 2))
        trained = [x for x in range(len(pairs)) if x not in held]
        used = set(lexicon.values())
        stock: dict[int, Form] = {}
        for x in trained:
            if len(stock) >= room:
                break
            word = lexicon[pairs[x][0]]
            for _ in range(40):
                form = self._random_word()
                if form not in used and not any(_strip(form, word, o) for o in ORDERS):
                    stock[x] = form
                    used.add(form)
                    break
        self.lexicon, self.things, self.pairs, self.held, self.trained = lexicon, things, pairs, held, trained
        self.room = room
        self.agents = [(dict(stock), {}, ORDERS[self.rng.randrange(2)]) for _ in range(self.T)]
        self.mode = "grammar"
        self.seen = 1.0
        return True

    def _utter(self, agent: tuple, x: int) -> Form | None:
        """What a teacher says for a pair: the whole it remembers, else the thing's word with
        its word for the property, which it invents when it has none."""
        memory, morph, order = agent
        if x in memory:
            return memory[x]
        t, prop = self.pairs[x]
        if prop not in morph:
            taken = set(self.lexicon.values()) | set(morph.values())
            for _ in range(40):
                form = self._random_word()
                if form not in taken:
                    morph[prop] = form
                    break
            else:
                return None
        return _compose(order, self.lexicon[t], morph[prop])

    def _teach(self) -> None:
        """One generation: a teacher drawn from the adults says ``bottleneck`` pairs to every
        learner; the learners generalise, remember what their rule does not give, and replace
        the adults."""
        rng, trained, b = self.rng, self.trained, self.w["bottleneck"]
        teacher = self.agents[rng.randrange(self.T)]
        learners, heard = [], 0
        for _ in range(self.T):
            coin = ORDERS[rng.randrange(2)]
            seen: dict[int, Form] = {}
            for x in (trained if b == 0 else [trained[rng.randrange(len(trained))] for _ in range(b)]):
                if x not in seen:
                    utterance = self._utter(teacher, x)
                    if utterance is not None:
                        seen[x] = utterance
            heard += len(seen)
            order, morph = induce(seen, self.lexicon, self.pairs, coin)
            memory: dict[int, Form] = {}
            for x, utterance in seen.items():
                t, prop = self.pairs[x]
                if prop in morph and _compose(order, self.lexicon[t], morph[prop]) == utterance:
                    continue  # the rule gives it
                if len(memory) < self.room:
                    memory[x] = utterance
            learners.append((memory, morph, order))
        self.agents = learners
        self.seen = heard / (self.T * len(trained))

    # -- the clock ----------------------------------------------------------------------------

    def advance(self, t: int) -> None:
        """From output step ``t - 1`` to ``t``."""
        if self.mode == "calls" and t == PHASE + 1 and self.w["capacity"] >= LEXICON_MIN:
            self._start_naming()
        if self.mode == "naming" and t == 2 * PHASE + 1 and self._start_grammar():
            return  # step 41 shows generation 0
        if self.mode == "calls":
            self._signal()
        elif self.mode == "naming":
            self._name()
        elif self.mode == "grammar":
            self._teach()

    def state(self) -> dict:
        """What :func:`view` and the gate need, copied."""
        if self.mode == "calls":
            return {"mode": "calls", "A": self.A.copy()}
        if self.mode == "naming":
            return {"mode": "naming", "inventories": tuple({m: tuple(ws) for m, ws in inv.items() if ws}
                                                           for inv in self.inv)}
        if self.mode == "grammar":
            return {"mode": "grammar", "lexicon": dict(self.lexicon), "things": tuple(self.things),
                    "held": tuple(self.held), "room": self.room, "seen": self.seen,
                    "agents": tuple((dict(memory), dict(morph), order) for memory, morph, order in self.agents)}
        return {"mode": "silent"}


# ---------------------------------------------------------------------------------------------
# what a state says: the exact success and the settled language


def _favourites(A: np.ndarray, axis: int) -> list[list[int]]:
    """What a talker does more often than not. Along the calls (``axis`` 2): per meaning the call
    that holds more than half of the weight of the meaning. Along the meanings (``axis`` 1): per
    call the meaning that holds more than half of the weight of the call. -1 where nothing does."""
    most = 2.0 * A > A.sum(axis=axis, keepdims=True)  # whole numbers: exact
    return np.where(most.any(axis=axis), most.argmax(axis=axis), -1).tolist()


def _majority(uses: dict, talkers: int) -> dict:
    """key -> the form that more than half of the talkers use for it."""
    return {key: form for key, forms in uses.items() for form, c in forms.items() if 2 * c > talkers}


def _counts(circulation: dict) -> tuple[list[Form], int, int]:
    """Distinct forms in use, the synonyms (forms beyond one per meaning) and the homonyms (forms
    in use for more than one meaning)."""
    used_for: dict[Form, int] = {}
    synonyms = 0
    for forms in circulation.values():
        synonyms += max(0, len(forms) - 1)
        for form in forms:
            used_for[form] = used_for.get(form, 0) + 1
    return sorted(used_for), synonyms, sum(1 for c in used_for.values() if c > 1)


def _empty_view(world: dict) -> dict:
    return {"mode": "silent", "success": world["guess"], "alarm": world["guess"], "forms": [], "synonyms": 0,
            "homonyms": 0, "lexicon": {}, "properties": {}, "say": {}, "rule": {}, "order": "none",
            "compositional": 0.0, "settled": 0, "things": (), "held": (), "trained": 0}


def view(state: dict, world: dict, full: bool = True) -> dict:
    """Everything the metrics need, computed from one group's state alone: the exact expected
    success among its talkers (``full``), the words in use, the settled lexicon and grammar."""
    mode = state["mode"]
    if mode == "calls":
        return _view_calls(state, world, full)
    if mode == "naming":
        return _view_naming(state, world, full)
    if mode == "grammar":
        return _view_grammar(state, world, full)
    return _empty_view(world)


def _single_success(per_named: list[float], world: dict) -> tuple[float, float]:
    """Accuracy over all meanings (the unnameable ones are guessed), and for ``predator``."""
    n, named, guess = len(world["meanings"]), world["named"], world["guess"]
    success = math.fsum(per_named + [guess] * (n - len(named))) / n
    alarm = per_named[named.index(world["predator"])] if world["predator"] in named else guess
    return success, alarm


def _view_calls(state: dict, world: dict, full: bool) -> dict:
    A = state["A"]
    T, _, n_calls = A.shape
    names, named, calls = world["meanings"], world["named"], world["syllables"]
    out = _empty_view(world)
    out["mode"] = "calls"
    uses: dict[int, dict[Form, int]] = {m: {} for m in named}
    for row in _favourites(A, 2):
        for m in named:
            if row[m] >= 0:
                form = (calls[row[m]],)
                uses[m][form] = uses[m].get(form, 0) + 1
    reads: dict[Form, dict[int, int]] = {(call,): {} for call in calls[:n_calls]}
    for row in _favourites(A, 1):
        for call, m in zip(calls, row):
            if m >= 0:
                reads[(call,)][m] = reads[(call,)].get(m, 0) + 1
    read = _majority(reads, T)
    out["forms"], out["synonyms"], out["homonyms"] = _counts(uses)
    # a call is settled for a meaning when most talkers send it for that meaning more often than
    # not, and most talkers take it for that meaning more often than not
    out["lexicon"] = {names[m]: form for m, form in _majority(uses, T).items() if read.get(form) == m}
    out["settled"] = len(out["lexicon"])
    if full:
        send = A / A.sum(axis=2, keepdims=True)  # the weights are whole numbers: the sums are exact
        recv = A / A.sum(axis=1, keepdims=True)
        a, b, own = np.zeros(A.shape[1:]), np.zeros(A.shape[1:]), np.zeros(A.shape[1:])
        for i in range(T):  # one talker after the other: the same sum on every machine
            a += send[i]
            b += recv[i]
            own += send[i] * recv[i]
        rows = (a * b - own).tolist()
        out["success"], out["alarm"] = _single_success([math.fsum(rows[m]) / (T * (T - 1)) for m in named], world)
    return out


def _view_naming(state: dict, world: dict, full: bool) -> dict:
    inventories = state["inventories"]
    T = len(inventories)
    names, named, guess = world["meanings"], world["named"], world["guess"]
    out = _empty_view(world)
    out["mode"] = "naming"
    uses: dict[int, dict[Form, int]] = {m: {} for m in named}
    circulation: dict[int, dict[Form, int]] = {m: {} for m in named}
    holders: dict[Form, int] = {}
    for inv in inventories:
        for m, words in inv.items():
            favourite = _shortest(words)[0]
            uses[m][favourite] = uses[m].get(favourite, 0) + 1
            for form in words:
                circulation[m][form] = circulation[m].get(form, 0) + 1
                holders[form] = holders.get(form, 0) + 1
    out["forms"], out["synonyms"], out["homonyms"] = _counts(circulation)
    out["lexicon"] = {names[m]: form for m, form in _majority(uses, T).items()}
    out["settled"] = len(out["lexicon"])
    if full:
        per_named = []
        for m in named:
            terms = []
            for inv in inventories:
                words = inv.get(m)
                if not words:  # nothing to say: the hearer guesses
                    terms.append((T - 1) * guess)
                    continue
                short = _shortest(words)
                # a hearer that holds the form for this meaning understands; one that does not
                # know the form guesses; one that holds it for another meaning misunderstands
                terms.append(math.fsum((circulation[m][form] - 1) + (T - holders[form]) * guess
                                       for form in short) / len(short))
            per_named.append(math.fsum(terms) / (T * (T - 1)))
        out["success"], out["alarm"] = _single_success(per_named, world)
    return out


def _view_grammar(state: dict, world: dict, full: bool) -> dict:
    names, props = world["meanings"], world["properties"]
    lexicon, agents, things = state["lexicon"], state["agents"], state["things"]
    T = len(agents)
    pairs = [(t, prop) for t in things for prop in range(len(props))]
    m = len(pairs)
    held = set(state["held"])
    out = _empty_view(world)
    out.update(mode="grammar", things=tuple(names[t] for t in things), trained=m - len(held),
               held=tuple((names[pairs[x][0]], props[pairs[x][1]]) for x in state["held"]))
    out["lexicon"] = {names[i]: form for i, form in lexicon.items()}
    out["settled"] = len(lexicon)
    # what every talker says for every pair (None: it would have to invent)
    said: list[list[Form | None]] = []
    uses: dict[int, dict[Form, int]] = {x: {} for x in range(m)}
    morphs: dict[int, dict[Form, int]] = {prop: {} for prop in range(len(props))}
    orders: dict[str, dict[str, int]] = {"order": {}}
    words: dict[Form, int] = dict.fromkeys(lexicon.values(), 1)
    for memory, morph, order in agents:
        row = []
        for x, (t, prop) in enumerate(pairs):
            utterance = memory.get(x)
            if utterance is None and prop in morph:
                utterance = _compose(order, lexicon[t], morph[prop])
            row.append(utterance)
            if utterance is not None:
                uses[x][utterance] = uses[x].get(utterance, 0) + 1
        said.append(row)
        for prop, form in morph.items():
            morphs[prop][form] = morphs[prop].get(form, 0) + 1
            words[form] = 1
        for form in memory.values():
            words[form] = 1
        if morph:
            orders["order"][order] = orders["order"].get(order, 0) + 1
    out["forms"] = sorted(words)
    out["synonyms"] = sum(max(0, len(forms) - 1) for forms in uses.values())
    used_for: dict[Form, set] = {}
    for i, form in lexicon.items():
        used_for.setdefault(form, set()).add(("single", i))
    for x, forms in uses.items():
        for form in forms:
            used_for.setdefault(form, set()).add(("pair", x))
    out["homonyms"] = sum(1 for found in used_for.values() if len(found) > 1)
    # the settled language: what more than half of the talkers say
    order = _majority(orders, T).get("order", "none")
    settled_morph = _majority(morphs, T)
    settled = _majority(uses, T)
    out["order"] = order
    out["properties"] = {props[prop]: form for prop, form in settled_morph.items()} if order != "none" else {}
    regular = 0
    for x, (t, prop) in enumerate(pairs):
        key = (names[t], props[prop])
        rule = _compose(order, lexicon[t], settled_morph[prop]) if order != "none" and prop in settled_morph else None
        if rule is not None:
            out["rule"][key] = rule
        # a pair that is never taught can only be said by the rule
        utterance = rule if x in held else settled.get(x, rule)
        if utterance is not None:
            out["say"][key] = utterance
        if x not in held and rule is not None and utterance == rule:
            regular += 1
    out["compositional"] = regular / (m - len(held))
    out["alarm"] = 1.0 if names[world["predator"]] in out["lexicon"] else world["guess"]
    if full:
        guess = expected_success(1, m, 0)
        tables: list[dict[Form, list[int]]] = []
        for memory, morph, agent_order in agents:
            table: dict[Form, list[int]] = {}
            for x, utterance in memory.items():
                table.setdefault(utterance, []).append(x)
            for x, (t, prop) in enumerate(pairs):
                if prop in morph:
                    found = table.setdefault(_compose(agent_order, lexicon[t], morph[prop]), [])
                    if x not in found:
                        found.append(x)
            tables.append(table)
        understood: dict[tuple[Form, int], float] = {}
        known: dict[Form, int] = {}
        for table in tables:
            for utterance, found in table.items():
                known[utterance] = known.get(utterance, 0) + 1
                for x in found:
                    understood[(utterance, x)] = understood.get((utterance, x), 0.0) + 1.0 / len(found)
        terms = []
        for row, table in zip(said, tables):
            for x, utterance in enumerate(row):
                if utterance is None:  # nothing to say: the hearer guesses
                    terms.append((T - 1) * guess)
                else:  # all hearers but the speaker: who reads it right, and who can only guess
                    terms.append(understood[(utterance, x)] - 1.0 / len(table[utterance])
                                 + (T - known[utterance]) * guess)
        out["success"] = math.fsum(terms) / (T * (T - 1) * m)
    return out


def _language(v: dict) -> dict:
    """The part of a view that is the language: what the records and the questions tell."""
    return {key: v[key] for key in ("lexicon", "properties", "say", "rule", "order", "compositional", "things",
                                    "held")}


def _dialects(a: dict, b: dict) -> float:
    """Share of the meanings with a settled utterance in either group that differ between them."""
    differ = total = 0
    for part in ("lexicon", "properties", "say"):
        for key in list(a[part]) + [k for k in b[part] if k not in a[part]]:
            total += 1
            differ += a[part].get(key) != b[part].get(key)
    return differ / total if total else 0.0


def meanings_of(language: dict, form: Form) -> list[str]:
    """Every meaning the language gives this utterance: single meanings, properties, pairs."""
    out = [name for name, word in language["lexicon"].items() if word == form]
    out += [name for name, word in language["properties"].items() if word == form]
    seen = []
    for part in ("say", "rule"):
        for key, utterance in language[part].items():
            if utterance == form and key not in seen:
                seen.append(key)
    return out + [f"{thing} {prop}" for thing, prop in seen]


def utterances_for(language: dict, thing: str, prop: str) -> list[Form]:
    """Every utterance the language has for a pair: what the group says, and what its rule gives."""
    out = []
    for part in ("say", "rule"):
        utterance = language[part].get((thing, prop))
        if utterance is not None and utterance not in out:
            out.append(utterance)
    return out


def _distance(a: Form, b: Form) -> int:
    """Edit distance between two utterances, counted in syllables."""
    row = list(range(len(b) + 1))
    for i, x in enumerate(a, start=1):
        last, row[0] = row[0], i
        for j, y in enumerate(b, start=1):
            last, row[j] = row[j], min(row[j] + 1, row[j - 1] + 1, last + (x != y))
    return row[-1]


def topographic(language: dict) -> float:
    """Correlation, over all couples of taught pairs with a settled utterance, between the number
    of parts in which the meanings differ (1 or 2) and the edit distance of the utterances in
    syllables; 0 when either does not vary."""
    said = [(key, utterance) for key, utterance in language["say"].items() if key not in language["held"]]
    xs, ys = [], []
    for i, (ka, ua) in enumerate(said):
        for kb, ub in said[i + 1:]:
            xs.append(float((ka[0] != kb[0]) + (ka[1] != kb[1])))
            ys.append(float(_distance(ua, ub)))
    if len(xs) < 2:
        return 0.0
    mx, my = math.fsum(xs) / len(xs), math.fsum(ys) / len(ys)
    vx, vy = math.fsum((x - mx) ** 2 for x in xs), math.fsum((y - my) ** 2 for y in ys)
    if vx <= 0 or vy <= 0:
        return 0.0
    return math.fsum((x - mx) * (y - my) for x, y in zip(xs, ys)) / math.sqrt(vx * vy)


def _fail(reason: str, expected: str | None = None) -> Verdict:
    return Verdict(False, expected, reason)


def _whole(value, lo: int, hi: int) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value) \
        and int(value) == value and lo <= value <= hi


def _step(views: list[dict], world: dict, sizes: tuple[int, int]) -> dict:
    """The metrics of one output step from the views of the groups (the first group's, and the
    distance to the second)."""
    v = views[0]
    agents, talkers = sizes
    # only two agents with ears and a voice can signal; any other couple guesses
    can = talkers * (talkers - 1) / (agents * (agents - 1)) if agents >= 2 and talkers >= 2 else 0.0
    guess = world["guess"] if v["mode"] != "grammar" else expected_success(1, len(v["things"]) * len(
        world["properties"]), 0)
    forms = v["forms"]
    return {
        "success": sig(can * v["success"] + (1.0 - can) * guess),
        "alarm": sig(can * v["alarm"] + (1.0 - can) * world["guess"]),
        "words": len(forms),
        "synonyms": v["synonyms"],
        "homonyms": v["homonyms"],
        "word_length": sig(sum(len(f) for f in forms) / len(forms)) if forms else 0.0,
        "meanings": v["settled"] + len(v["say"]),
        "compositional": sig(v["compositional"]),
        "order": v["order"],
        "dialects": sig(_dialects(v, views[1])) if len(views) > 1 else 0.0,
        "stage": stage_of(v["mode"], v["settled"], len(world["named"]), v["compositional"]),
    }


class Signals:
    """The simulation and its gate. Rollouts replayed by :meth:`check` are cached per seed."""

    sim = SIM
    topic = SIM
    KEYS = KEYS

    def __init__(self) -> None:
        self._cache: dict[int, SignalsRollout] = {}

    def run(self, seed: int, meanings: int = 8, consonants: int = 4, vowels: int = 3, population: int = 40,
            predator_pressure: float = 0.5, ears_voice: float = 1.0, neurons: int = 512, bottleneck: int = 12,
            split: int = 0, things: int = 4, properties: int = 3, rules: int = 7) -> SignalsRollout:
        """Deterministic for a given seed and parameters. ``rules`` is accepted for the round-7
        convention (this level has no round-6 form). Raises ``ValueError`` for a parameter outside
        its meaning."""
        if rules != 7:
            raise ValueError("signals is a round-7 level: rules must be 7")
        if not _whole(seed, 0, MAX_SEED):
            raise ValueError(f"seed must be a whole number from 0 to {MAX_SEED}: {seed!r}")
        for name, value, lo, hi in (("meanings", meanings, 2, len(VOCABULARY)),
                                    ("consonants", consonants, 1, len(CONSONANTS)),
                                    ("vowels", vowels, 1, len(VOWELS)), ("population", population, 0, CAP),
                                    ("neurons", neurons, 0, 10 ** 9), ("bottleneck", bottleneck, 0, MAX_BOTTLENECK),
                                    ("split", split, 0, 1), ("things", things, 2, 6),
                                    ("properties", properties, 2, len(PROPERTIES))):
            if not _whole(value, lo, hi):
                raise ValueError(f"{name} must be a whole number from {lo} to {hi}: {value!r}")
        predator_pressure, ears_voice = float(predator_pressure), float(ears_voice)
        for name, value in (("predator_pressure", predator_pressure), ("ears_voice", ears_voice)):
            if not (math.isfinite(value) and 0.0 <= value <= 1.0):
                raise ValueError(f"{name} must be between 0 and 1: {value!r}")
        seed = int(seed)
        p: dict[str, float | int] = {
            "meanings": int(meanings), "consonants": int(consonants), "vowels": int(vowels),
            "population": int(population), "predator_pressure": predator_pressure, "ears_voice": ears_voice,
            "neurons": int(neurons), "bottleneck": int(bottleneck), "split": int(split), "things": int(things),
            "properties": int(properties)}
        p["syllables"] = syllables(p["consonants"], p["vowels"])
        p["possible_words"] = possible_words(p["syllables"], MAX_SYLLABLES)
        p["capacity"] = lexicon_capacity(p["neurons"])
        sizes = _sizes(p["population"], p["split"], ears_voice)
        p["talkers"] = sum(t for _, t in sizes)
        nameable = min(p["meanings"], p["capacity"])
        p["bits"] = signal_bits(nameable) if nameable else 0.0
        p["pairs"], p["coverage"] = 0, 0.0  # known when stage 3 begins
        world = _world(seed, p)
        assert len(world["syllables"]) == p["syllables"]
        groups = [_Group(world, talkers, seed, g) for g, (_, talkers) in enumerate(sizes)]
        r = SignalsRollout(SIM, seed, p, [], world=world)
        for t in range(STEPS + 1):
            if t:
                for group in groups:
                    group.advance(t)
            states = tuple(group.state() for group in groups)
            views = [view(s, world, full=g == 0) for g, s in enumerate(states)]
            r.states.append(states)
            r.steps.append(_step(views, world, sizes[0]))
            if t in SNAPSHOTS:
                r.languages[t] = _language(views[0])
        last = r.steps[-1]
        final = r.languages[STEPS]
        world["talked"], world["unseen"] = final["things"], final["held"]  # for the ``meanings`` record
        p["pairs"], p["coverage"] = _taught(final, p)
        r.summary = {**{k: last[k] for k in ("stage", "success", "words", "meanings", "compositional", "order",
                                              "word_length", "dialects")},
                     "topographic": sig(topographic(final)), "outcome": outcome_of(last["stage"])}
        return r

    def rollout(self, seed: int) -> SignalsRollout:
        """The canonical run of a seed: parameters from ``random_params(random.Random(seed))``."""
        if seed not in self._cache:
            if len(self._cache) >= 256:
                self._cache.clear()
            self._cache[seed] = self.run(seed, **random_params(random.Random(seed)))
        return self._cache[seed]

    # -- the gate -----------------------------------------------------------------------------

    def conserved(self, r: Rollout) -> Verdict:
        """Weights, bounds, the exact success, the majority lexicon, the grammar and the summary:
        everything is counted again from the agents kept in the rollout."""
        p = r.params
        world, states = getattr(r, "world", None), getattr(r, "states", None)
        languages = getattr(r, "languages", None)
        if not world or not states or languages is None:
            return _fail("the rollout does not hold its agents")
        if len(r.steps) != STEPS + 1 or len(states) != len(r.steps):
            return _fail(f"a run has {STEPS + 1} steps with the agents of each")
        if p["syllables"] != syllables(p["consonants"], p["vowels"]) or len(world["syllables"]) != p["syllables"] \
                or len(set(world["syllables"])) != p["syllables"]:
            return _fail("the syllables are not consonants times vowels", num(syllables(p["consonants"], p["vowels"])))
        if p["possible_words"] != possible_words(p["syllables"], MAX_SYLLABLES):
            return _fail("possible_words does not follow from the inventory")
        if p["capacity"] != lexicon_capacity(p["neurons"]) or len(world["named"]) != min(p["meanings"], p["capacity"]):
            return _fail("the capacity does not follow from the neurons")
        sizes = _sizes(p["population"], p["split"], p["ears_voice"])
        if p["talkers"] != sum(t for _, t in sizes):
            return _fail("the talkers do not follow from the population and the share with ears and a voice")
        if p["bits"] != (signal_bits(len(world["named"])) if world["named"] else 0.0):
            return _fail("bits is not log2 of the meanings that can be named")
        if abs(math.fsum(world["weights"]) - 1.0) > 1e-12 or min(world["weights"]) <= 0:
            return _fail("the shares of the talk do not add up to 1")
        for t, (step, groups) in enumerate(zip(r.steps, states)):
            if len(groups) != len(sizes):
                return _fail(f"step {t}: a group is missing")
            try:
                for g, state in enumerate(groups):
                    reason = _state_fault(state, world, p, sizes[g][1])
                    if reason:
                        return _fail(f"step {t} group {g + 1}: {reason}")
                views = [view(state, world, full=g == 0) for g, state in enumerate(groups)]
                again = _step(views, world, sizes[0])
            except (KeyError, IndexError, TypeError, ValueError, AttributeError, ZeroDivisionError) as error:
                return _fail(f"step {t}: the agents cannot be read ({type(error).__name__})")
            for key in STATE_KEYS:
                if step.get(key) != again[key]:
                    return _fail(f"step {t}: {key} is not what the agents give", dense_value(again[key]))
            if not (0.0 <= step["success"] <= 1.0 and 0.0 <= step["alarm"] <= 1.0
                    and 0.0 <= step["compositional"] <= 1.0 and 0.0 <= step["dialects"] <= 1.0):
                return _fail(f"step {t}: a share is outside 0..1")
            if step["words"] > p["possible_words"]:
                return _fail(f"step {t}: more words than the inventory can form", num(p["possible_words"]))
            if any(len(form) > MAX_SYLLABLES for form in views[0]["forms"]):
                return _fail(f"step {t}: a word of more than {MAX_SYLLABLES} syllables")
            language = _language(views[0])
            reason = _language_fault(language, groups[0], world)
            if reason:
                return _fail(f"step {t}: {reason}")
            if t in SNAPSHOTS and languages.get(t) != language:
                return _fail(f"step {t}: the language written down is not the language of the agents")
        if set(languages) != set(SNAPSHOTS):
            return _fail("the language is written down at steps 20, 40 and 60")
        if (p["pairs"], p["coverage"]) != _taught(languages[STEPS], p):
            return _fail("pairs and coverage do not follow from the things, the properties and the bottleneck")
        first = r.steps[0]
        if first["success"] != sig(world["guess"]) or first["words"] or first["meanings"] or first["stage"] != "silent":
            return _fail("step 0 is not a silent population that guesses", dense_value(sig(world["guess"])))
        if sizes[0][1] < 2 and any(s["stage"] != "silent" or s["success"] != sig(world["guess"]) or s["words"]
                                   for s in r.steps):
            return _fail("signals without two agents that can hear and call")
        last = r.steps[-1]
        expected = {**{k: last[k] for k in SUMMARY_KEYS if k in last},
                    "topographic": sig(topographic(languages[STEPS])), "outcome": outcome_of(last["stage"])}
        if dict(r.summary) != expected:
            return _fail("the summary is not the last step")
        return Verdict(True, None, "weights, bounds, success, lexicon, grammar and summary are consistent")

    # -- questions ----------------------------------------------------------------------------

    def parse(self, prompt: str) -> tuple | None:
        """``(kind, seed, ...)``: ``step``/``next`` with a step and a key, ``final``/``params``
        with a key, ``word``/``meaning`` with the snapshot step and a word or an utterance,
        ``say`` with a thing and a property, ``order``."""
        w = prompt.split()
        if len(w) < 4 or w[0] != SIM or w[1] != "seed" or " ".join(w) != prompt:
            return None
        i = _digits_end(w, 2)
        seed = parse_num(w[2:i])
        if not isinstance(seed, int) or i >= len(w) or num(seed).split() != w[2:i]:  # no leading zeros
            return None
        if seed > MAX_SEED:  # no run takes such a seed, so there is nothing to replay
            return None
        rest, snapshot = w[i:], STEPS
        if rest[0] == "step":
            j = _digits_end(rest, 1)
            step = parse_num(rest[1:j])
            if not isinstance(step, int) or step > STEPS or num(step).split() != rest[1:j]:
                return None
            tail = rest[j:]
            if len(tail) == 1 and tail[0] in STATE_KEYS:
                return "step", seed, step, tail[0]
            if len(tail) == 2 and tail[0] == "next" and tail[1] in STATE_KEYS and step < STEPS:
                return "next", seed, step, tail[1]
            if step not in SNAPSHOTS[:-1] or not tail or tail[0] not in ("word", "meaning"):
                return None
            rest, snapshot = tail, step
        head, args = rest[0], rest[1:]
        if head == "word" and len(args) == 1 and args[0] in VOCABULARY + PROPERTIES:
            return "word", seed, snapshot, args[0]
        if head == "meaning" and 1 <= len(args) <= 2 * MAX_SYLLABLES and all(a in _SYLLABLES for a in args):
            return "meaning", seed, snapshot, tuple(args)
        if w[i] == "step":
            return None
        if head in ("final", "params") and len(args) == 1 and args[0] in (SUMMARY_KEYS if head == "final"
                                                                         else PARAM_KEYS):
            return head, seed, args[0]
        if head == "say" and len(args) == 2 and args[0] in VOCABULARY and args[1] in PROPERTIES:
            return "say", seed, args[0], args[1]
        if head == "order" and not args:
            return "order", seed
        return None

    def owns(self, prompt: str) -> bool:
        return self.parse(prompt) is not None

    def answers(self, prompt: str) -> list[str] | None:
        """Every answer the settled language accepts for a ``word``, ``meaning``, ``say`` or
        ``order`` question, the one the lines give first; ``None`` for any other prompt."""
        q = self.parse(prompt)
        if q is None or q[0] not in ("word", "meaning", "say", "order"):
            return None
        r = self.rollout(q[1])
        if q[0] == "order":
            return [r.languages[STEPS]["order"]]
        if q[0] == "say":
            return [text_of(u) for u in utterances_for(r.languages[STEPS], q[2], q[3])] or ["none"]
        language = r.languages[q[2]]
        if q[0] == "word":
            return [text_of(language["lexicon"].get(q[3]) or language["properties"].get(q[3]))]
        return meanings_of(language, q[3]) or ["none"]

    def truth(self, prompt: str) -> float | int | str | None:
        """The simulation's own value for a question about a metric, or ``None``."""
        q = self.parse(prompt)
        if q is None or q[0] not in ("step", "next", "final", "params"):
            return None
        r = self.rollout(q[1])
        if q[0] == "final":
            return r.summary[q[2]]
        if q[0] == "params":
            return r.params[q[2]]
        return r.steps[q[2] + (q[0] == "next")][q[3]]

    def check(self, prompt: str, answer: str) -> Verdict:
        """Replay the seed (parameters from :func:`random_params`) and compare: words, counts and
        utterances exactly, other numbers within 5 percent. A ``say`` question accepts every
        utterance the language has for the pair, a ``meaning`` question every meaning of the
        utterance."""
        q = self.parse(prompt)
        if q is None:
            return Verdict(False, None, "not my question")
        accepted = self.answers(prompt)
        if accepted is not None:
            ok = answer in accepted
            return Verdict(ok, accepted[0], "what the settled language gives" if ok else "not in the language")
        value = self.truth(prompt)
        key = q[-1]
        expected = param_value(key, value) if q[0] == "params" else dense_value(value)
        if key in WORD_KEYS:
            ok = answer == value
            return Verdict(ok, expected, "exact word" if ok else "wrong word")
        got = parse_num(answer)
        if not is_finite(got):
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


def _taught(language: dict, p: dict) -> tuple[int, float]:
    """The pairs of stage 3 in the first group (things that have a word, times properties), and
    the share of the taught ones a learner is expected to hear: both 0 when stage 3 was not
    reached, the share 1 without a bottleneck."""
    if not language["things"]:
        return 0, 0.0
    pairs = holistic_words(len(language["things"]), p["properties"])
    return pairs, coverage(pairs - len(language["held"]), p["bottleneck"]) if p["bottleneck"] else 1.0


def _sizes(population: int, split: int, ears_voice: float) -> list[tuple[int, int]]:
    """Per group: its agents, and how many of them have ears and a voice (rounded half up)."""
    heads = [(population + 1) // 2, population // 2] if split else [population]
    return [(n, int(n * ears_voice + 0.5)) for n in heads]


def _digits_end(words: list[str], start: int) -> int:
    """Index after the run of single-digit tokens that begins at ``start``."""
    i = start
    while i < len(words) and words[i] in _DIGITS:
        i += 1
    return i


def _form_fault(form, legal: frozenset | set) -> bool:
    return not isinstance(form, tuple) or not 1 <= len(form) <= MAX_SYLLABLES or any(s not in legal for s in form)


def _state_fault(state: dict, world: dict, p: dict, talkers: int) -> str | None:
    """What is wrong with one group's state, or ``None``."""
    mode = state.get("mode")
    named, legal = world["named"], set(world["syllables"])
    if mode == "silent":
        return None if talkers < 2 or not _can_signal(world) else "a group that can signal is silent"
    if talkers < 2 or not _can_signal(world):
        return "signals without two talkers and two calls"
    if mode == "calls":
        A = state["A"]
        if A.shape != (talkers, len(world["meanings"]), world["calls"]):
            return "the weights are not one table of meanings and calls per talker"
        if not np.isfinite(A).all() or A.min() < 0:
            return "a negative association weight"
        if (A.sum(axis=2) <= 0).any() or (A.sum(axis=1) <= 0).any():
            return "a meaning or a call without any weight"
        return None
    if p["capacity"] < LEXICON_MIN:
        return "words in a brain that only holds calls"
    if mode == "naming":
        inventories = state["inventories"]
        if len(inventories) != talkers:
            return "not one inventory per talker"
        circulation: dict[int, set] = {}
        for inv in inventories:
            held: set = set()
            for m, words in inv.items():
                if m not in named or not words:
                    return "a word for a meaning that cannot be named"
                for form in words:
                    if _form_fault(form, legal):
                        return "a word that the inventory cannot form"
                    if form in held:
                        return "a talker uses one form for two meanings"
                    held.add(form)
                circulation.setdefault(m, set()).update(words)
        if any(len(forms) > naming_max_words(talkers) for forms in circulation.values()):
            return "more words for a meaning in circulation than half the talkers"
        return None
    if mode != "grammar":
        return "an unknown game"
    lexicon, things, agents = state["lexicon"], state["things"], state["agents"]
    n_prop = len(world["properties"])
    if len(agents) != talkers or len(things) < 2 or any(t not in lexicon for t in things):
        return "a thing without a word, or not one learner per talker"
    if any(m not in named or _form_fault(form, legal) for m, form in lexicon.items()) \
            or len(set(lexicon.values())) != len(lexicon):
        return "the lexicon holds a form twice or a word the inventory cannot form"
    room = p["capacity"] - (len(lexicon) - len(things)) - compositional_words(len(things), n_prop)
    if state["room"] != room or room < 0:
        return "the room for whole phrases does not follow from the capacity"
    m = holistic_words(len(things), n_prop)
    held_out = state["held"]
    if len(held_out) != (1 if m < 16 else 2) or any(not 0 <= x < m for x in held_out):
        return "the pairs that are never taught are not 1 or 2 of the pairs"
    for memory, morph, order in agents:
        if order not in ORDERS or len(memory) > room:
            return "a learner remembers more wholes than it has room for"
        if any(x in held_out or not 0 <= x < m or _form_fault(form, legal) for x, form in memory.items()):
            return "a learner remembers a pair that is never taught, or a whole that is no word"
        if any(not 0 <= prop < n_prop or _form_fault(form, legal) for prop, form in morph.items()):
            return "a property word that the inventory cannot form"
    return None


def _language_fault(language: dict, state: dict, world: dict) -> str | None:
    """Is the language the majority's, and do lexicon, grammar and phrases fit each other?"""
    names, named = world["meanings"], world["named"]
    mode = state["mode"]
    # the lexicon is the word of the majority, counted again talker by talker
    if mode in ("calls", "naming"):
        for m in named:
            votes: dict[Form, int] = {}
            if mode == "calls":
                talkers = state["A"].shape[0]
                for i in range(talkers):
                    weights = state["A"][i, m]
                    for call in range(weights.shape[0]):
                        if 2.0 * weights[call] > weights.sum():
                            form = (world["syllables"][call],)
                            votes[form] = votes.get(form, 0) + 1
            else:
                talkers = len(state["inventories"])
                for inv in state["inventories"]:
                    if m in inv:
                        form = min(inv[m], key=len)
                        votes[form] = votes.get(form, 0) + 1
            winner = [form for form, c in votes.items() if 2 * c > talkers]
            word = language["lexicon"].get(names[m])
            if word is not None and winner != [word]:
                return f"the word for {names[m]} is not the word of the majority"
            if word is None and winner and mode == "naming":
                return f"the majority has a word for {names[m]} that the lexicon lacks"
    elif mode == "grammar":
        if language["lexicon"] != {names[m]: form for m, form in state["lexicon"].items()}:
            return "the lexicon is not the one the learners hold"
    order, properties = language["order"], language["properties"]
    if order == "none" and (properties or language["rule"] or language["compositional"]):
        return "a rule without an order"
    regular = 0
    for (thing, prop), utterance in language["rule"].items():
        if utterance != _compose(order, language["lexicon"][thing], properties[prop]):
            return "the rule is not the word of the thing with the word of the property in the order"
    for key, utterance in language["say"].items():
        if f"{key[0]} {key[1]}" not in meanings_of(language, utterance):
            return "an utterance does not lead back to its meaning"
        if key not in language["held"] and language["rule"].get(key) == utterance:
            regular += 1
    if mode == "grammar":
        taught = holistic_words(len(language["things"]), len(world["properties"])) - len(language["held"])
        agents = state["agents"]
        if any(key in language["held"] and language["rule"].get(key) != utterance
               for key, utterance in language["say"].items()):
            return "an utterance for a pair that was never taught does not come from the rule"
        if abs(language["compositional"] - regular / taught) > 1e-12:
            return "the compositional share is not the share of the taught pairs that follow the rule"
        for prop, form in properties.items():
            index = world["properties"].index(prop)
            if 2 * sum(1 for _, morph, _ in agents if morph.get(index) == form) <= len(agents):
                return f"the word for {prop} is not the word of the majority"
    elif regular or language["say"]:
        return "phrases before the third stage"
    return None


# ---------------------------------------------------------------------------------------------
# lines


def params_line(r: Rollout) -> Line:
    """``signals seed 7 params. meanings 8. consonants 4. ... coverage 0 point 6 8 1.``"""
    fields = " ".join(f"{k} {param_value(k, r.params[k])}." for k in PARAM_KEYS)
    return Line(f"{r.sim} seed {num(r.seed)} params. {fields}", topic=r.sim, kind="record")


def world_lines(r: SignalsRollout) -> list[Line]:
    """The phonemes of the run and its meanings: which can be named (most talked about first),
    which cannot, the things and properties of stage 3 and the pairs that are never taught."""
    w = r.world
    head = f"{r.sim} seed {num(r.seed)}"
    out = [Line(f"{head} inventory. consonants {' '.join(w['consonants'])}. vowels {' '.join(w['vowels'])}.",
                topic=r.sim, kind="record")]
    named = [w["meanings"][m] for m in w["named"]]
    fields = [("named", named), ("unnamed", [m for m in w["meanings"] if m not in named]),
              ("things", list(w["talked"])), ("properties", list(w["properties"])),
              ("unseen", [word for pair in w["unseen"] for word in pair])]
    text = " ".join(f"{key} {' '.join(words)}." for key, words in fields if words)
    return out + [Line(f"{head} meanings. {text}", topic=r.sim, kind="record")]


def language_lines(r: SignalsRollout, step: int = STEPS) -> list[Line]:
    """The language of the first group at a snapshot step: lexicon, grammar and irregular phrases
    as records, and the questions ``word``, ``meaning``, ``say`` and ``order``. The last snapshot
    is the language of the run and is asked without a step."""
    language, w = r.languages[step], r.world
    final = step == STEPS
    head = f"{r.sim} seed {num(r.seed)}" + ("" if final else f" step {num(step)}")
    lexicon, properties = language["lexicon"], language["properties"]
    known = [(m, lexicon[m]) for m in w["meanings"] if m in lexicon] \
        + [(p, properties[p]) for p in w["properties"] if p in properties]
    out = [Line(f"{head} lexicon. " + " ".join(f"{name} {text_of(form)}." for name, form in known[i:i + 8]),
                topic=r.sim, kind="record") for i in range(0, len(known), 8)]
    pairs = [(thing, prop) for thing in language["things"] for prop in w["properties"]]
    if final:
        out.append(Line(f"{head} grammar. order {language['order']}. "
                        f"compositional {dense_value(sig(language['compositional']))}.", topic=r.sim, kind="record"))
        whole = [key for key in pairs if key in language["say"] and language["rule"].get(key) != language["say"][key]]
        out += [Line(f"{head} phrases. " + " ".join(f"{thing} {prop} {text_of(language['say'][(thing, prop)])}."
                                                    for thing, prop in whole[i:i + 6]), topic=r.sim, kind="record")
                for i in range(0, len(whole), 6)]
    for name in list(w["meanings"]) + (list(w["properties"]) if pairs else []):
        out.append(Line(f"{head} word {name}", text_of(lexicon.get(name) or properties.get(name)), r.sim, "fact"))
    forms = [form for _, form in known]
    for key in pairs:
        forms += utterances_for(language, *key)
    for form in dict.fromkeys(forms):
        out.append(Line(f"{head} meaning {text_of(form)}", meanings_of(language, form)[0], r.sim, "fact"))
    if final:
        for thing, prop in pairs:
            said = utterances_for(language, thing, prop)
            unseen = (thing, prop) in language["held"]
            out.append(Line(f"{head} say {thing} {prop}", text_of(said[0] if said else None), r.sim,
                            "calc" if unseen else "fact", {"unseen": unseen}))
        out.append(Line(f"{head} order", language["order"], r.sim, "fact"))
    return out


def lines(r: SignalsRollout, every: int = 1) -> list[Line]:
    """The parameter and world records, a state record and questions for every ``every``-th step
    (now and next), the final questions, and the language at the three snapshot steps."""
    keys = list(STATE_KEYS)
    out = [params_line(r)] + world_lines(r)
    for t in range(0, len(r.steps), every):
        out.append(state_line(r, t, keys))
        out += query_lines(r, t, keys)
    out += summary_lines(r)
    for step in SNAPSHOTS:
        out += language_lines(r, step)
    return out


_SIM = Signals()


def simulation() -> Signals:
    return _SIM


def run(seed: int, **params) -> SignalsRollout:
    return _SIM.run(seed, **params)


def conserved(r: Rollout) -> Verdict:
    return _SIM.conserved(r)


def owns(prompt: str) -> bool:
    """True for a question about a seed of this level and for one of its lessons."""
    return _SIM.owns(prompt) or LESSONS.owns(prompt)


def check(prompt: str, answer: str) -> Verdict:
    """Judge an answer to a question about a seed or to a lesson of this level."""
    if LESSONS.owns(prompt):
        return LESSONS.check(prompt, answer)
    return _SIM.check(prompt, answer)
