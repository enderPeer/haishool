"""Level 8 of the toy universe: from naked replicators to cells with genomes.

Level 5 ends with molecules that copy themselves. Here they become genes: enclosed in
compartments, competing inside them, shared out at random when a compartment divides, and kept
in balance only because compartments with a bad mix of genes fall behind. That is the
**stochastic corrector** (after Szathmary and Demeter 1987), with **Eigen's error threshold**
(after Eigen 1971) deciding how long a gene may be, and four heritable innovations on top:
chromosomes, dna, an engulfed partner and a larger genome.

**The vessel** holds at most 400 protocells. A cell needs ``gene_types`` kinds of gene (2 to 6:
think replicase, metabolism, membrane) and holds several copies of each. At generation 0 the
``replicators`` handed over by level 5 are enclosed at random, ``threshold // 2`` to a
compartment (the size of a newborn cell), each of a random type. Then, generation by generation
(one output step is 5 generations, 60 steps, 300 generations):

1. **Inflow.** ``inflow`` bases of nutrient enter and are offered in equal shares,
   ``inflow // cells`` to each cell (:func:`nutrient_share`).
2. **Uptake.** A cell takes of its share what its metabolism can use: the share times its
   **balance** times its **efficiency**. The balance is the harmonic mean of its gene counts
   over their arithmetic mean (:func:`balance`): a chain of steps, one gene type for each,
   runs at the pace 1 / (sum of 1 / count); it is 1 when all types are equally common and 0
   when one is missing, so a cell without a full set takes up nothing and stops growing. The
   efficiency is genes / (genes + 8) for its number of gene types (:func:`efficiency`).
3. **Copying.** The cell makes ``rate`` times the smaller of two numbers of copy attempts: the
   copies its nutrient pays for (a copy costs its length in bases), and its templates times
   its balance (no template is tried twice in a generation). A fraction of an attempt is made
   with that chance. Less than one copy's worth of nutrient is carried over, the rest flows
   out. ``rate`` is how fast the cell runs its genes, e / (e + 1) for e units of energy per
   gene type (:func:`expression`, :func:`energy_per_gene`: 10 units shared among the gene
   types), times the pace its innovations leave it. A copy has no error with chance
   fidelity ^ length (:func:`copy_success`); a copy with an error is waste
   (:func:`expected_waste`). Which gene a good copy is, is drawn in proportion to the copies
   at hand, the first gene type counting 1 + ``selfish`` times: the **selfish gene** that
   copies faster and helps no more.
4. **Division.** A cell with a full set and ``threshold`` gene copies divides; each copy goes
   to either daughter by the toss of a coin (:func:`full_set` is the chance that a daughter of
   a balanced cell gets every type). A daughter without any gene is no cell.
5. **Death.** A cell with a full set dies with chance 0.1 per generation, one without with
   chance 0.5; above 400 cells the surplus is washed out at random.

Selection among cells is all that keeps the genes in balance: inside a cell the selfish gene
gains, at division chance makes some daughters better balanced than their mother, and those
grow faster. Without it (``run(..., corrector=False)``, the control) no cell keeps a full set.

**The error threshold.** At the very best the genes of a cell grow by the share
``rate`` * fidelity ^ length per generation while a tenth of the cells die. They hold their own
only if (1 + rate * fidelity ^ length) * 0.9 > 1, that is if sigma * fidelity ^ length > 1 with
sigma = 9 * rate (:func:`superiority`). Read as a count, sigma is the number of copies a
template would leave in its lifetime if no copy had an error: ``rate`` attempts in each of the
10 generations a cell lives on average, a copy still being there after the generation it was
made in with chance 0.9. A template must leave at least one good copy among them. With
ln fidelity about fidelity - 1 the condition has the form of Eigen's threshold: the piece that
is copied in one go must be shorter than L_max = ln(sigma) / (1 - fidelity)
(:func:`error_threshold`, :func:`under_threshold`). A longer piece is lost however good
everything else is. Shorter pieces can still fail: assortment, imbalance and hunger come on
top, so L_max is a ceiling, not the edge itself.

What is Eigen's and what is not: in Eigen's model sigma is the superiority of the best
sequence over its mutants, the mutants still replicate, and beyond the threshold the
population drifts away from the best sequence while it goes on replicating. Here a copy with
an error is simply waste (as if every mutation were lethal), there is no cloud of mutants,
the death of cells takes the place of the mutants' competition in sigma, and beyond the
threshold the genes die out. The formula is Eigen's; what happens beyond it is this toy's
simplification.

**Innovations** arise in a daughter with chance ``innovation`` each (an engulfment with a
twentieth of it) and are inherited; none is ever lost again:

* **chromosome** (after Maynard Smith and Szathmary 1993): the genes are linked, one of each
  type per chromosome (copies without partners are lost when they join). Daughters always get
  full sets (chromosomes are shared out evenly, a cell divides with ``threshold`` genes and at
  least 2 chromosomes), the balance is 1 and the selfish gene gains nothing. The price: the
  whole chromosome is copied in one piece, so one error spoils all of it
  (fidelity ^ (types * length)), and the pace is 0.7.
* **dna** (a toy stand-in for a more faithful way of keeping and copying genes): the error
  rate per base is divided by 100, so pieces nearly a hundred times longer pass the threshold
  (nearly: the pace lowers sigma a little). The price: pace 0.9. The factor 100 is invented;
  real dna copying with proofreading and repair is more accurate than rna copying by far more.
* **partner** (an engulfed cell; the idea that it lifts the energy per gene after Lane and
  Martin 2010, every number a toy): the energy is multiplied by 1 + 15 * fidelity ^ 10000
  (:func:`partner_gain`): the partner has 10000 bases of its own, copied with the host's
  fidelity, and only works when they come out right, which without dna they practically never
  do (at fidelity 0.999 with chance 0.00005). That the partner needs a faithful copy of its own
  genome, and so comes after dna, is this toy's rule, not Lane and Martin's (in their argument
  the partner's genome shrinks to a few genes).
  More energy per gene runs the genes faster and pays for more of them. The price: pace 0.8.
* **larger genome** (the idea of a duplicated gene that takes on a new task, after Ohno 1970;
  the rule itself is a toy): a linked cell about to divide breaks up its chromosomes and
  rebuilds them with one more gene type (what does not fill a set is lost) instead of
  dividing. More gene types take up more of the food; they need energy (the same energy is
  shared among more genes) and fidelity (a longer chromosome), and at most 32 fit.

**Parameters** (:func:`random_params` draws them from short lists): ``gene_types`` 2 to 6,
``gene_length`` 20 to 640 bases, ``fidelity`` 0.95 to 0.999 per base, ``selfish`` 0.02 to 0.2,
``inflow`` 200000 to 5 million bases per generation, ``threshold`` 6 to 16 copies for each gene
type, ``innovation`` 0 to 0.02, ``replicators`` 100 to 3000.

**Stages**, from the counts of a step: ``collapse`` (no cell), ``soup`` (generation 0, the
replicators just enclosed and nothing selected yet, or fewer than half the compartments with a
full set), ``complex_cells`` (half the cells have a partner and dna), ``chromosomes`` (half are
linked), ``cells`` (half have dna), else ``protocells``. Outcomes: ``collapse``,
``protocells`` (also a soup), ``cells`` (also chromosomes), ``complex_cells``.

**Lines** (seed 33 verbatim; numbers as digits, measured numbers to 3 significant digits, the
fidelity as 1 minus an error rate of 3 digits; a line has at most 97 tokens in seeds 1 to 400,
a rollout that does not collapse 1658 lines; after the first ``collapse`` step the
steps are left out, the vessel stays empty):

    cells seed 3 3 params. gene_types 6. gene_length 4 0. fidelity 0 point 9 9 9. selfish 0 point 0 5.
        inflow 1 0 0 0 0 0 0. threshold 7 2. innovation 0 point 0 2. replicators 3 0 0.
    cells seed 3 3 derived. copy_success 0 point 9 6 1. sigma 5 point 6 2. l_max 1 7 3 0.
        full_set 0 point 9 9 9. copy_waste 1 point 5 7. doubling 1 point 4 7. takeover 8 5 point 5.
    cells seed 3 3 step 0. cells 8. genes_per_cell 3 6. gene_types 6. genome_length 2 4 0. fidelity 0 point 9 9 9.
        error_load 0. viable 1. energy_per_cell 1 0. linked 0. dna 0. complex 0. selfish 0 point 1 5 3. stage soup.
    cells seed 3 3 step 3 1. cells 4 0 0. genes_per_cell 4 8 point 3. gene_types 7 point 2 8. genome_length 2 9 1.
        fidelity 0 point 9 9 9 9 8 2 6. error_load 0 point 0 0 6 0 2. viable 1. energy_per_cell 8 1 point 9.
        linked 1. dna 0 point 9 9 3. complex 0 point 5 3. selfish 0 point 1 4. stage complex_cells.
    q cells seed 3 3 step 3 1 energy_per_cell. a 8 1 point 9.
    q cells seed 3 3 step 3 0 next stage. a complex_cells.          (one step, 5 generations, later)
    q cells seed 3 3 final first_chromosome. a 2 5.                 (or: a never.)
    q cells seed 3 3 final outcome. a complex_cells.
    q cells seed 3 3 param threshold. a 7 2.
    q cells seed 3 3 derived l_max. a 1 7 3 0.

The ``derived`` record is what the lesson rules make of the parameters alone: the chance of a
good copy, sigma and L_max of a protocell at the start, the chance of a full set when
``threshold // gene_types`` copies of each type are split, the mean waste per copy attempt, the
generations in which a fed, balanced protocell doubles its genes, ln 2 / ln(1 + rate *
copy_success), and the generations the selfish gene would need to fill a cell unchecked, by the
toy rule ln(threshold) / selfish.

**Lessons** (:data:`LESSONS`, topic ``predict_cells``) carry their inputs; each is a function
the simulation itself calls (13 rules; the lines below are among the 400 of
``LESSONS.generate(random.Random(1), 400)``):

    q cells predict copy_success fidelity 0 point 9 9 4 8 6 length 5 1 3. a 0 point 0 7 1 1.
    q cells predict error_threshold sigma 2 4 point 2 fidelity 0 point 9 2 1 9. a 4 0 point 8.
    q cells predict under_threshold length 1 0 3 sigma 5 point 9 fidelity 0 point 9 9 7 1. a yes.
    q cells predict full_set copies 6 types 2 3. a 0 point 6 9 6 1.
    q cells predict expected_waste fidelity 0 point 9 9 0 1 5 length 1 9 5. a 1 6 6 point 7.
    q cells predict nutrient_share inflow 3 6 3 3 9 3 4 cells 2 1 7. a 1 6 7 4 6.
    q cells predict energy_per_gene energy 2 9 point 1 genes 6 4 partner 1 gain 5 9 point 3. a 2 6 point 9 6.
    q cells predict generations_to_fix population 1 3 4 0 1 advantage 1 point 8. a 5 point 2 7 9.
    q cells predict doubling_time rate 0 point 5 7 4. a 1 point 5 2 8.

and ``balance`` (three gene counts), ``efficiency`` (gene types), ``expression`` (energy per
gene) and ``partner_gain`` (fidelity). ``copy_success``, ``nutrient_share``, ``efficiency``,
``expression``, ``energy_per_gene`` and ``partner_gain`` are steps of every generation (the
last five through the tables of a run, :func:`tables`); ``balance`` is the three-type case of
:func:`_balance`, which the simulation calls on all cells at once; ``error_threshold`` and
``under_threshold`` decide in the ledger which cells could keep their information and give
``l_max``; ``full_set`` and ``expected_waste`` are exact expectations of the coin tosses at
division and of a copy attempt; ``doubling_time`` and ``generations_to_fix`` give ``doubling``
and ``takeover``. The last is a toy rule and says so: a type that grows faster by the factor
e^s takes ln(N) / s generations from one copy to N; a real sweep in a finite population takes
about twice as long and may fail by chance.

The inputs are drawn so that no answer dominates (2000 lessons of seeds 1 to 4): the most
common answer of a number rule has a share of at most 0.14 (``balance`` 0, a missing gene
type), ``under_threshold`` says yes in about half of its lessons (0.48), and ``expected_waste``
equals its own ``length`` input (a copy that practically always fails) in fewer than 1 of 20.

**Gates.** :meth:`Cells.conserved` checks at every step: the nutrient ledger, exactly, in whole
bases: ``inflow * generation == built + waste + unused + stored`` (built into good copies,
spent on failed ones, flowed out, carried over by living cells); the genes: ``bases == bases at
step 0 + built - lost``, exactly; no count negative, no ledger entry ever falling; the cells
themselves (kept per step in ``snaps``) give the same counts, none holds a negative number of
genes, a linked cell holds the same number of each; viable cells within cells within 400, the
shares are the counts over the cells; stage, summary and derived record follow from the
numbers. **Error catastrophe**: the ledger counts the cell generations lived with a piece
shorter than the cell's own L_max (``keeper_gens``) and the best growth any cell could have
(``best_growth``, rate * copy success). Where no cell was ever under its threshold the genes
shrink on average at least by the factor (1 + best_growth) * 0.9 < 1 per generation, and the
gate fails a run that keeps more than a billion times what this allows (Markov's inequality: a
true run does so less than once in a million; in seeds 1 to 400 the 104 such runs all end
without a cell). The gate cannot tell wasted from unused nutrient without replaying the run;
it holds their sum. :meth:`Cells.check` replays the seed (parameters from :func:`random_params` with
``random.Random(seed)``) and compares words, counts and parameters exactly, the fidelity by its
error rate and other numbers within 5 percent.

**Hand-off.** In (:func:`handoff_in`, from level 5): ``replicators`` as they are; ``fidelity``
= 1 - ``mu``; ``inflow`` = 12500 bases for each monomer of life's inflow per step (the cells
are a far bigger vessel than life's; the factor is this level's choice of scale and nothing
measured; a closed vessel, inflow 0, feeds no cells). Out (the summary, to level 9, bodies):
``cells``, ``stage`` and ``outcome`` (``complex_cells`` when complex cells exist), ``complex``
(their share), ``genome_length`` (bases), ``energy_per_cell`` (toy units, 10 without a working
partner), ``first_chromosome``.

**Plausibility targets.** These are properties of the toy, which the tests hold; none is a
finding about real cells. Measured on the canonical seeds 1 to 400 unless said otherwise
(``tests/test_evo_cells.py`` holds each with a margin):

* outcomes: collapse 176, cells 133, complex_cells 48, protocells 43; every ledger exact;
* low fidelity: of the 107 runs whose gene is longer than L_max 105 collapse (in the other 2
  the gene, 40 bases at L_max 36 and 37, is only just too long, the decline is slow, and dna
  arises in time); without innovations all 24 do. At gene length 160 with 3 gene types and no
  innovations, fidelity 0.999 and 0.995 keep protocells, 0.99, 0.98 and 0.95 collapse (8 seeds
  each; at 0.99 the gene is still below L_max 193: the ceiling is not the edge);
* chromosomes spread when the assortment load is high: 6 gene types with 6 copies each at
  division (``full_set`` 0.91; in practice 0.61 to 0.75 of the cells are viable): every
  population that lives has chromosomes within 15 to 30 generations (5 of 10 seeds, the other
  5 collapse before one arises); 2 gene types with 16 copies each (``full_set`` 1, 0.98 of the
  cells viable): never, the linked share stays below 0.25 in 10 of 10 seeds;
* complex cells only after dna and in a minority: 49 of 400 runs reach ``complex_cells``, at
  generation 100 to 300 (median 190); in no population of 20 or more cells does the share with
  a partner reach one half before the share with dna did; final genomes of complex cells have
  gained 2.6 gene types on average, those of runs that end at ``chromosomes`` 0.8;
* the selfish gene is held in check: in populations of protocells (20 or more cells, half of
  them viable) its share stays within 0.29 of the fair share 1 / gene_types (median 0.05).
  In the control without selection among cells no cell is left with a full set after 300
  generations (6 of 6 seeds) and the fast gene's share has passed 0.55 by generation 100
  (``selfish`` 0.2).

**What is taken from where, and what is invented.** From Szathmary and Demeter: templates of
several kinds in a compartment that needs all of them, copied at unequal rates, shared out at
random at division, with selection among compartments correcting what selection inside them
spoils. From Eigen: the error threshold L_max = ln(sigma) / (1 - q). From Maynard Smith and
Szathmary: linkage ends assortment loss and competition between genes of one cell and costs
replication speed. From Lane and Martin: the argument that the energy per gene limits a
genome and that an engulfed partner lifts the limit (an argument that is disputed, for one by
Lynch and Marinov 2015; the toy takes it as its rule and does not test it). From Ohno: a new
gene as a changed duplicate. Invented for the toy, and tuned so that something happens within
300 generations: every number (death rates, paces, the factor 100 for dna, 16 for the partner,
its 10000 bases, 10 units of energy, genes / (genes + 8), the vessel of 400); the
harmonic-mean metabolism; every copying error being waste, so that genes never change and
nothing evolves inside a gene; sigma as the copies of a lifetime; one synchronous generation
for everything; innovations as single heritable switches that are never lost (the real
chromosomes, dna and engulfed partners have long histories that are known only in part, and
the toy says nothing about them); the order of the stages, which follows from the toy's
prices and gains and is no claim about the order in which real cells got them; equal shares
of food. Nothing here is a measured fact about early life: nobody knows how cells arose, and
a run's numbers tell the toy's rules.

**Reproducibility.** All randomness is the raw PCG64 stream of the seed (the bits numpy keeps
fixed across versions), turned into uniform numbers here; every draw is a comparison of such a
number with a threshold; counts are integers; the float steps are sums, products, quotients,
``floor`` and comparisons in a fixed order, and ``copy_success`` multiplies. Logarithms enter
only the error threshold (which cells the ledger counts as keepers, never the dynamics), the
derived record and the rounding of a metric to 3 digits. ``tests/test_evo_cells.py`` pins the
stream and three rollouts by digest, so a numpy or machine that computes anything else fails
loudly; it also checks, for every parameter set :func:`random_params` can draw, that no derived
number that takes a logarithm (``l_max``, ``doubling``, ``takeover``) lies within a millionth
of a rounding edge and no piece within a millionth of its error threshold, so a logarithm
that differs in its last bit between machines gives the same digits and the same ledger. One
run takes about 0.07 to 0.14 seconds on the machine this was measured on, depending on its
load (the slowest of seeds 1 to 400 under 0.5).
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

import numpy as np

from haishool.cosmos import Rollout, close, query_lines, seeds, state_line, summary_lines
from haishool.evo import Input, LessonGate, Rule, sig
from haishool.truth import Line, Verdict, is_finite, num, parse_num

SIM = "cells"

#: the vessel holds at most this many cells; surplus cells are washed out at random
CAP = 400
STEPS = 60
#: generations between two output steps
GENS_PER_STEP = 5
GENERATIONS = STEPS * GENS_PER_STEP
#: most gene types a genome can reach (the width of the gene table)
GMAX = 32
#: chance per generation that a cell with a full gene set dies
DEATH = 0.1
#: chance per generation that a cell without a full set dies
DEATH_BROKEN = 0.5
#: (1 - DEATH) / DEATH: the 10 generations a cell lives on average, times the chance 0.9 that a
#: copy outlives the generation it was made in
SURVIVAL_ODDS = 9.0
#: dna divides the error rate per base by this (toy value; real proofreading gains more)
DNA_GAIN = 100.0
#: share of its copy attempts a cell still makes when it carries the innovation (the price)
LINK_RATE, DNA_RATE, ENDO_RATE = 0.7, 0.9, 0.8
#: a linked cell divides only when it holds at least this many chromosomes
MIN_CHROMOSOMES = 2
#: energy of a cell without a working partner per generation, toy units
ENERGY = 10.0
#: energy per gene type at which the genes run at half their fastest
ENERGY_HALF = 1.0
#: gene types with which a cell takes up half of its nutrient share
FUNC_HALF = 8.0
#: a working partner multiplies the energy by this (toy value, after Lane and Martin 2010)
ENDO_GAIN = 16.0
#: bases of the partner's own genome; it is copied with the host's fidelity
ENDO_LENGTH = 10000
#: an engulfment arises with this share of the chance of the other innovations
ENDO_RARITY = 0.05
#: bases of inflow here for one monomer of inflow per step at level 5 (hand-off)
INFLOW_SCALE = 12500
_TWO_M53 = 1.0 / 9007199254740992.0
_DIGITS = frozenset("0123456789")
_COLS = np.arange(GMAX, dtype=np.int64)

STAGES = ("soup", "protocells", "cells", "chromosomes", "complex_cells", "collapse")
OUTCOMES = ("collapse", "protocells", "cells", "complex_cells")

PARAM_KEYS: dict[str, str] = {
    "gene_types": "gene types a cell needs at the start, 2 to 6",
    "gene_length": "bases of one gene, 20 to 640",
    "fidelity": "chance that one base is copied right before dna, 0.95 to 0.999 (level 5: 1 - mu)",
    "selfish": "extra copying speed of the first gene type inside a cell, 0.1 is ten percent",
    "inflow": "nutrient entering per generation, bases",
    "threshold": "gene copies at which a cell divides, 6 to 16 for each gene type",
    "innovation": "chance per daughter of each innovation, 0 to 0.02 (an engulfment a twentieth of it)",
    "replicators": "naked replicators handed over by level 5, enclosed threshold // 2 at a time",
}
INT_PARAMS = frozenset({"gene_types", "gene_length", "inflow", "threshold", "replicators"})
DERIVED_KEYS: dict[str, str] = {
    "copy_success": "chance that a copy of one gene has no error: fidelity to the power gene_length",
    "sigma": "copies a gene of a protocell would leave in its lifetime if no copy had an error: "
             "9 times its copying rate",
    "l_max": "error threshold of a protocell, bases: ln sigma over 1 minus fidelity",
    "full_set": "chance that a daughter gets every type when threshold // gene_types copies of each are split",
    "copy_waste": "bases wasted per copy attempt on average: gene_length times 1 minus copy_success",
    "doubling": "generations in which a fed, balanced protocell doubles its genes: ln 2 over ln(1 + rate * copy_success)",
    "takeover": "generations the fast gene needs to fill a cell unchecked, toy rule ln threshold over selfish (never: selfish 0)",
}
STATE_KEYS = ("cells", "genes_per_cell", "gene_types", "genome_length", "fidelity", "error_load", "viable",
              "energy_per_cell", "linked", "dna", "complex", "selfish", "stage")
LEDGER_KEYS = ("n_viable", "n_linked", "n_dna", "n_complex", "n_working", "genes", "types_sum", "selfish_genes",
               "stored", "bases", "generation", "inflow_total", "built", "waste", "unused", "lost", "attempts",
               "failures", "keeper_gens", "best_growth")
SUMMARY_KEYS: dict[str, str] = {
    "stage": "stage at the last step",
    "cells": "cells at the last step",
    "genome_length": "mean genome length at the last step, bases",
    "complex": "share of cells with a partner at the last step",
    "energy_per_cell": "mean energy per cell at the last step, toy units",
    "first_chromosome": "generation of the first step at which half the cells had chromosomes, or never",
    "outcome": "collapse, protocells, cells or complex_cells",
}
#: what the summary hands upward (level 9, bodies)
HANDOFF_OUT = ("cells", "stage", "outcome", "complex", "genome_length", "energy_per_cell", "first_chromosome")
#: exact whole numbers in the lines (``first_chromosome`` may also be the word ``never``)
COUNT_KEYS = frozenset({"cells", "first_chromosome"})
WORD_KEYS = frozenset({"stage", "outcome"})

KEYS: dict[str, str] = {
    "cells": "living protocells or cells (count, at most 400)",
    "genes_per_cell": "mean gene copies per cell",
    "gene_types": "mean number of distinct gene types per cell",
    "genome_length": "mean genome length, bases: distinct gene types times gene_length",
    "fidelity": "mean chance that one base is copied right (written as 1 minus the error rate to 3 digits)",
    "error_load": "share of the copy attempts since the last step that failed",
    "viable": "share of cells that hold every gene type they need",
    "energy_per_cell": "mean energy per cell and generation, toy units (10 without a working partner)",
    "linked": "share of cells whose genes are linked in chromosomes",
    "dna": "share of cells that keep their genes as dna",
    "complex": "share of cells with an engulfed partner",
    "selfish": "share of all gene copies that are of the first, fast gene type",
    "stage": "soup, protocells, cells, chromosomes, complex_cells or collapse",
    "generation": "ledger: generations so far, 5 per step",
    "n_viable": "ledger: cells with a full set",
    "n_linked": "ledger: cells with chromosomes",
    "n_dna": "ledger: cells with dna",
    "n_complex": "ledger: cells with a partner",
    "n_working": "ledger: cells with a partner and dna",
    "genes": "ledger: gene copies in living cells",
    "types_sum": "ledger: distinct gene types summed over the cells",
    "selfish_genes": "ledger: copies of the first gene type in living cells",
    "bases": "ledger: bases in the genes of living cells",
    "stored": "ledger: nutrient taken up by living cells and not yet used, bases",
    "inflow_total": "ledger: nutrient that has entered so far, bases",
    "built": "ledger: bases built into good copies so far",
    "waste": "ledger: bases spent on failed copies so far",
    "unused": "ledger: bases that flowed out unused so far",
    "lost": "ledger: bases of genes lost with dead and washed out cells and when genes were joined, so far",
    "attempts": "ledger: copy attempts so far",
    "failures": "ledger: failed copy attempts so far",
    "keeper_gens": "ledger: cell generations lived with a copied piece shorter than the cell's error threshold",
    "best_growth": "ledger: the largest share by which any cell could grow its genes per generation so far",
}


# ---------------------------------------------------------------- the rules (also the lessons)

def copy_success(fidelity: float, length: int) -> float:
    """Chance that a copy of ``length`` bases has no error: ``fidelity ** length``, by repeated
    squaring (only multiplications, so every machine gets the same bits).

    >>> copy_success(0.99, 100), copy_success(1.0, 500), copy_success(0.5, 3)
    (0.3660323412732289, 1.0, 0.125)
    """
    n = int(length)
    if n < 0:
        raise ValueError("length must not be negative")
    result, base = 1.0, float(fidelity)
    while n:
        if n & 1:
            result *= base
        base *= base
        n >>= 1
    return result


def error_threshold(sigma: float, fidelity: float) -> float:
    """Eigen's error threshold: the longest genome, in bases, that selection of strength
    ``sigma`` keeps free of errors at a given per-base ``fidelity``: ln(sigma) / (1 - fidelity)."""
    return math.log(sigma) / (1.0 - fidelity)


def under_threshold(length: int, sigma: float, fidelity: float) -> str:
    """``yes`` when a piece of ``length`` bases, copied in one go, is shorter than the error
    threshold ``error_threshold(sigma, fidelity)`` and so can keep its information, else ``no``
    (also when sigma is not above 1: such a template leaves less than one copy even without errors)."""
    return "yes" if sigma > 1.0 and length < error_threshold(sigma, fidelity) else "no"


def full_set(copies: int, types: int) -> float:
    """Chance that one daughter gets at least one copy of each of ``types`` gene types when
    ``copies`` copies of each go to either daughter with chance one half: (1 - 0.5^copies)^types."""
    return copy_success(1.0 - copy_success(0.5, copies), types)


def expected_waste(fidelity: float, length: int) -> float:
    """Bases wasted per copy attempt on average: an attempt costs ``length`` bases and fails
    with chance 1 - fidelity^length."""
    return length * (1.0 - copy_success(fidelity, length))


def nutrient_share(inflow: int, cells: int) -> int:
    """Bases each cell is offered per generation: the inflow shared equally, whole bases."""
    return int(inflow) // int(cells)


def energy_per_gene(energy: float, genes: int, partner: int, gain: float) -> float:
    """Energy for each gene type: the cell's energy, times ``gain`` when it has a partner,
    divided by its number of gene types."""
    return energy * (gain if partner else 1.0) / genes


def generations_to_fix(population: int, advantage: float) -> float:
    """Toy rule for a sweep: a type that multiplies faster by the factor e^advantage per
    generation grows from one copy to ``population`` copies in ln(population) / advantage."""
    return math.log(population) / advantage


def doubling_time(rate: float) -> float:
    """Generations until something that grows by the share ``rate`` per generation has doubled:
    ln 2 / ln(1 + rate)."""
    return math.log(2.0) / math.log1p(rate)


def _balance(genes: np.ndarray, types: np.ndarray) -> np.ndarray:
    """Per row: the harmonic mean of the first ``types`` gene counts over their arithmetic mean,
    ``types^2 / (sum of the counts * sum of 1 / count)``. 1 when all types are equally common,
    0 when one is missing. Only sums, products and quotients, column by column in a fixed order."""
    width = genes.shape[1]
    total = np.zeros(len(genes), dtype=np.float64)
    inverse = np.zeros(len(genes), dtype=np.float64)
    missing = np.zeros(len(genes), dtype=bool)
    for k in range(width):
        needed = k < types
        count = genes[:, k].astype(np.float64)
        missing |= needed & (genes[:, k] <= 0)
        total = total + np.where(needed, count, 0.0)
        inverse = inverse + np.where(needed & (genes[:, k] > 0), 1.0 / np.maximum(count, 1.0), 0.0)
    kinds = types.astype(np.float64)
    return np.where(missing, 0.0, np.minimum(1.0, kinds * kinds / np.maximum(total * inverse, 1.0)))


def balance(first: int, second: int, third: int) -> float:
    """Metabolism of a cell that needs three gene types and holds ``first``, ``second`` and
    ``third`` copies: 9 / ((first + second + third) * (1 / first + 1 / second + 1 / third)),
    their harmonic mean over their arithmetic mean; 0 when a type is missing."""
    return float(_balance(np.array([[first, second, third]], dtype=np.int64), np.array([3], dtype=np.int64))[0])


def efficiency(genes: int) -> float:
    """Share of its nutrient share that a balanced cell with ``genes`` gene types can take up:
    genes / (genes + 8). More gene types, more of the food can be used."""
    return genes / (genes + FUNC_HALF)


def expression(energy: float) -> float:
    """How fast a cell runs its genes, as a share of the fastest: e / (e + 1) for ``energy`` e
    per gene type (the lesson names this input ``energy_per_gene``)."""
    return energy / (energy + ENERGY_HALF)


def superiority(rate: float) -> float:
    """The sigma of the error threshold: the copies a template would leave in its lifetime if
    no copy had an error. A template is tried ``rate`` times per generation, its cell lives
    1 / DEATH = 10 generations on average, and a copy is still there after the generation it was
    made in with chance 1 - DEATH: sigma = rate * (1 - DEATH) / DEATH = 9 * rate. Said the other
    way round: genes that grow by rate * copy_success per generation while the share DEATH of
    the cells dies hold their own if (1 + rate * copy_success) * (1 - DEATH) > 1, which is
    sigma * copy_success > 1."""
    return rate * SURVIVAL_ODDS


def dna_error(error: float) -> float:
    """Error rate per base once the genes are kept as dna: a hundredth of the rate before."""
    return error / DNA_GAIN


def partner_gain(fidelity: float) -> float:
    """Factor by which a partner multiplies the energy: 1 + 15 times the chance that the
    partner's own 10000 bases are copied without error."""
    return 1.0 + (ENDO_GAIN - 1.0) * copy_success(fidelity, ENDO_LENGTH)


RULES: dict[str, Rule] = {
    "copy_success": Rule(
        (Input("fidelity", 0.98, 0.99999, places=5), Input("length", 1, 640, True)), copy_success,
        "chance that a copy of length bases has no error. fidelity to the power length"),
    "error_threshold": Rule(
        (Input("sigma", 1.5, 100, places=1), Input("fidelity", 0.9, 0.9999, places=4)), error_threshold,
        "longest piece in bases that keeps its information when it is copied in one go. the error threshold "
        "after eigen. natural log of sigma over 1 minus fidelity. in this toy sigma is the number of copies "
        "a template would leave in its lifetime if no copy had an error"),
    "under_threshold": Rule(
        (Input("length", 1, 400, True), Input("sigma", 1.1, 10, places=2),
         Input("fidelity", 0.98, 0.9999, places=4)), under_threshold,
        "yes when a piece of length bases is shorter than the error threshold and can keep its information. "
        "length below natural log of sigma over 1 minus fidelity. else no"),
    "full_set": Rule(
        (Input("copies", 1, 12, True), Input("types", 1, GMAX, True)), full_set,
        "chance that a daughter gets at least one copy of every type when copies of each type go to "
        "either daughter with chance one half. 1 minus one half to the power copies. all to the power types"),
    "expected_waste": Rule(
        (Input("fidelity", 0.98, 0.99999, places=5), Input("length", 1, 640, True)), expected_waste,
        "bases wasted per copy attempt on average. length times 1 minus fidelity to the power length"),
    "nutrient_share": Rule(
        (Input("inflow", 0, 10000000, True), Input("cells", 1, CAP, True)), nutrient_share,
        "bases offered to each cell per generation. inflow divided by cells. whole bases rounded down"),
    "energy_per_gene": Rule(
        (Input("energy", 1, 100, places=1), Input("genes", 1, 64, True), Input("partner", 0, 1, True),
         Input("gain", 1, 100, places=1)), energy_per_gene,
        "energy for each gene type. energy over genes. times gain when partner is 1. "
        "toy units. the idea that the energy per gene limits a genome is after lane and martin"),
    "generations_to_fix": Rule(
        (Input("population", 2, 100000, True), Input("advantage", 0.01, 2, places=2)), generations_to_fix,
        "toy rule for a sweep. generations for a type that grows faster by the factor e to the power advantage "
        "to grow unchecked from one copy to population copies. natural log of population over advantage. "
        "a real sweep in a finite population takes about twice as long and may fail by chance"),
    "doubling_time": Rule(
        (Input("rate", 0.01, 1, places=3),), doubling_time,
        "generations to double at growth share rate per generation. natural log of 2 over natural log of 1 plus rate"),
    "balance": Rule(
        (Input("first", 0, 20, True), Input("second", 0, 20, True), Input("third", 0, 20, True)), balance,
        "toy rule. metabolism of a cell that needs three gene types. 9 over the sum of its three gene counts times the "
        "sum of 1 over each count. the harmonic mean of the counts over their mean. 1 when balanced and 0 when "
        "a type is missing"),
    "efficiency": Rule(
        (Input("genes", 1, 64, True),), efficiency,
        "toy rule. share of its nutrient share a balanced cell with genes gene types can take up. "
        "genes over genes plus 8"),
    "expression": Rule(
        (Input("energy_per_gene", 0.05, 50, places=2),), expression,
        "toy rule. how fast a cell runs its genes as a share of the fastest. "
        "energy_per_gene over energy_per_gene plus 1"),
    "partner_gain": Rule(
        (Input("fidelity", 0.9995, 0.99999, places=6),), partner_gain,
        "toy rule. factor by which an engulfed partner multiplies the energy of a cell. 1 plus 15 times "
        "fidelity to the power 1 0 0 0 0. the partner has 1 0 0 0 0 bases of its own and works only when they are copied "
        "without error"),
}
LESSONS = LessonGate(SIM, RULES)


def lesson_gate() -> LessonGate:
    return LESSONS


# ---------------------------------------------------------------- parameters

def random_params(rng: random.Random, rules: int = 7) -> dict[str, float | int]:
    """Seeded parameters from short lists, so the model sees each value many times."""
    if rules != 7:
        raise ValueError("cells is a round 7 level: rules must be 7")
    types = rng.randint(2, 6)
    return {
        "gene_types": types,
        "gene_length": rng.choice([20, 40, 80, 160, 320, 640]),
        "fidelity": rng.choice([0.95, 0.98, 0.99, 0.995, 0.998, 0.999]),
        "selfish": rng.choice([0.02, 0.05, 0.1, 0.2]),
        "inflow": rng.choice([200000, 500000, 1000000, 2000000, 5000000]),
        "threshold": types * rng.choice([6, 8, 10, 12, 16]),
        "innovation": rng.choice([0.0, 0.002, 0.005, 0.01, 0.02]),
        "replicators": rng.choice([100, 300, 1000, 3000]),
    }


DEFAULTS: dict[str, float | int] = {"gene_types": 4, "gene_length": 80, "fidelity": 0.995, "selfish": 0.1,
                                    "inflow": 500000, "threshold": 24, "innovation": 0.01, "replicators": 800}


def _params(given: dict) -> dict[str, float | int]:
    unknown = set(given) - set(PARAM_KEYS)
    if unknown:
        raise ValueError(f"unknown parameter: {', '.join(sorted(unknown))}")
    p = {**DEFAULTS, **given}
    for k in INT_PARAMS:
        if isinstance(p[k], bool) or not is_finite(p[k]) or int(p[k]) != p[k]:
            raise ValueError(f"{k} must be a whole number")
        p[k] = int(p[k])
    for k in ("fidelity", "selfish", "innovation"):
        if isinstance(p[k], bool) or not is_finite(p[k]):
            raise ValueError(f"{k} must be a number")
        p[k] = float(p[k])
    if not 1 <= p["gene_types"] <= GMAX // 2:
        raise ValueError(f"gene_types must be 1 to {GMAX // 2}")
    if not 1 <= p["gene_length"] <= 100000:
        raise ValueError("gene_length must be 1 to 100000 bases")
    if not 0 < p["fidelity"] < 1 or copy_success(p["fidelity"], p["gene_length"]) < 1e-300:
        raise ValueError("fidelity must lie between 0 and 1 and allow a copy at all")
    if not 0 <= p["selfish"] <= 10 or not 0 <= p["innovation"] <= 1:
        raise ValueError("selfish must be 0 to 10, innovation 0 to 1")
    if not 0 <= p["inflow"] <= 10 ** 12 or not 0 <= p["replicators"] <= 10 ** 9:
        raise ValueError("inflow and replicators must not be negative")
    if not max(2, p["gene_types"]) <= p["threshold"] <= 1000:
        raise ValueError("threshold must be at least gene_types (and 2) and at most 1000")
    return {k: p[k] for k in PARAM_KEYS}


def start_rate(p: dict) -> float:
    """Copy attempts per template and generation of a protocell at the start: how fast it runs
    its ``gene_types`` genes on the energy of a cell without a partner."""
    return expression(energy_per_gene(ENERGY, p["gene_types"], 0, 1.0))


def derived_exact(p: dict) -> dict[str, float | str]:
    """What follows from the parameters alone, by the lesson rules, before rounding."""
    q = copy_success(p["fidelity"], p["gene_length"])
    sigma = superiority(start_rate(p))
    return {
        "copy_success": q,
        "sigma": sigma,
        "l_max": error_threshold(sigma, p["fidelity"]),
        "full_set": full_set(p["threshold"] // p["gene_types"], p["gene_types"]),
        "copy_waste": expected_waste(p["fidelity"], p["gene_length"]),
        "doubling": doubling_time(start_rate(p) * q),
        "takeover": generations_to_fix(p["threshold"], p["selfish"]) if p["selfish"] > 0 else "never",
    }


def derived(p: dict) -> dict[str, float | str]:
    """What follows from the parameters alone, by the lesson rules, to 3 significant digits."""
    return {k: v if isinstance(v, str) else sig(v) for k, v in derived_exact(p).items()}


def handoff_in(below: dict) -> dict[str, float | int]:
    """Parameters of this level from the level below (level 5, life: its summary and parameters).

    ``replicators``: the replicators at life's last step, as they are (none: nothing to enclose).
    ``fidelity``: 1 - ``mu``, life's chance that a letter of a copy is changed (or ``fidelity``
    when the level below already names it). ``inflow``: life's monomers per step times
    :data:`INFLOW_SCALE` (a closed vessel, inflow 0, feeds no cells). Keys that ``below`` does
    not hold are left out, so the caller's own parameters stand.
    """
    out: dict[str, float | int] = {}
    if "replicators" in below:
        out["replicators"] = max(0, int(below["replicators"]))
    if "fidelity" in below:
        out["fidelity"] = round(float(below["fidelity"]), 6)
    elif "mu" in below:
        out["fidelity"] = round(1.0 - float(below["mu"]), 6)
    if "fidelity" in out:
        out["fidelity"] = min(0.999999, max(0.5, out["fidelity"]))
    if "inflow" in below:
        out["inflow"] = INFLOW_SCALE * max(0, int(below["inflow"]))
    return out


# ---------------------------------------------------------------- the simulation

class _Rng:
    """Uniform numbers straight from the PCG64 bit stream (numpy keeps the bits of a seeded
    ``PCG64`` fixed across versions; the conversion to fractions is done here)."""

    def __init__(self, seed: int) -> None:
        self._bits = np.random.PCG64(seed)

    def uniform(self, n: int) -> np.ndarray:
        if n <= 0:
            return np.empty(0, dtype=np.float64)
        return (self._bits.random_raw(n) >> np.uint64(11)).astype(np.float64) * _TWO_M53


@dataclass
class _Tables:
    """Per kind of cell, from the lesson rules. A kind is ``linked * 4 + dna * 2 + partner``."""
    copy: np.ndarray  # [dna, genes per unit]: chance that a copy of the unit has no error
    eff: np.ndarray  # [gene types]: share of the nutrient share a balanced cell takes up
    rate: np.ndarray  # [kind, gene types]: copy attempts per template and generation
    l_max: np.ndarray  # [kind, gene types]: error threshold of that kind of cell, bases
    keeper: np.ndarray  # [kind, gene types]: is the piece such a cell copies under its error threshold
    fidelity: tuple[float, float]
    gain: tuple[float, float]


def tables(fidelity: float, gene_length: int) -> _Tables:
    q = (fidelity, 1.0 - dna_error(1.0 - fidelity))
    gain = (partner_gain(q[0]), partner_gain(q[1]))
    copy = np.array([[copy_success(q[d], gene_length * k) for k in range(GMAX + 1)] for d in (0, 1)])
    eff = np.array([0.0] + [efficiency(g) for g in range(1, GMAX + 1)])
    rate = np.zeros((8, GMAX + 1))
    l_max = np.zeros((8, GMAX + 1))
    keeper = np.zeros((8, GMAX + 1), dtype=bool)
    for kind in range(8):
        linked, dna, partner = kind >> 2, (kind >> 1) & 1, kind & 1
        pace = (LINK_RATE if linked else 1.0) * (DNA_RATE if dna else 1.0) * (ENDO_RATE if partner else 1.0)
        for g in range(1, GMAX + 1):
            rate[kind, g] = pace * expression(energy_per_gene(ENERGY, g, partner, gain[dna]))
            sigma = superiority(rate[kind, g])
            l_max[kind, g] = error_threshold(sigma, q[dna]) if sigma > 1.0 else 0.0
            piece = gene_length * (g if linked else 1)  # a chromosome is copied in one piece
            keeper[kind, g] = under_threshold(piece, sigma, q[dna]) == "yes"
    return _Tables(copy, eff, rate, l_max, keeper, q, gain)


class _Vessel:
    """The mutable state: one row per cell."""

    def __init__(self, rng: _Rng, p: dict, corrector: bool = True) -> None:
        self.p, self.rng, self.corrector = p, rng, corrector
        self.t = tables(p["fidelity"], p["gene_length"])
        g0, k0 = p["gene_types"], p["threshold"] // 2
        n = min(CAP, p["replicators"] // k0)
        self.genes = np.zeros((n, GMAX), dtype=np.int64)
        if n:  # each compartment encloses as many random replicators as a newborn cell holds genes
            kinds = np.minimum((rng.uniform(n * k0) * g0).astype(np.int64), g0 - 1)
            np.add.at(self.genes, (np.repeat(np.arange(n), k0), kinds), 1)
        self.types = np.full(n, g0, dtype=np.int64)
        self.linked = np.zeros(n, dtype=bool)
        self.dna = np.zeros(n, dtype=bool)
        self.endo = np.zeros(n, dtype=bool)
        self.store = np.zeros(n, dtype=np.int64)
        self.led: dict[str, float | int] = {"generation": 0, "inflow_total": 0, "built": 0, "waste": 0, "unused": 0,
                                            "lost": 0, "attempts": 0, "failures": 0, "keeper_gens": 0,
                                            "best_growth": 0.0}

    def _full(self) -> np.ndarray:
        """Which cells hold every gene type they need (without the corrector: every cell counts)."""
        if not self.corrector:
            return np.ones(len(self.types), dtype=bool)
        return (self.genes > 0).sum(axis=1) == self.types

    def _remove(self, gone: np.ndarray) -> None:
        """Dead or washed out cells: their genes are lost, their stored nutrient flows out."""
        if gone.any():
            self.led["lost"] += int(self.genes[gone].sum()) * self.p["gene_length"]
            self.led["unused"] += int(self.store[gone].sum())
            for name in ("genes", "types", "linked", "dna", "endo", "store"):
                setattr(self, name, getattr(self, name)[~gone])

    def advance(self) -> None:
        """One generation: inflow, uptake, copying, division with innovations, death, washout."""
        p, led, t, rng = self.p, self.led, self.t, self.rng
        length = p["gene_length"]
        led["generation"] += 1
        led["inflow_total"] += p["inflow"]
        n = len(self.types)
        if n == 0:
            led["unused"] += p["inflow"]
            return
        width = int(self.types.max())
        g = self.genes[:, :width]  # a view: columns beyond a cell's gene types are zero
        types, linked = self.types, self.linked
        total = g.sum(axis=1)
        full = self._full()
        kind = linked * 4 + self.dna * 2 + self.endo
        unit = np.where(linked, types, 1)  # genes copied in one piece
        cost = length * unit

        # the ledger of who could keep its information at all
        q_copy = t.copy[self.dna.astype(np.int64), unit]
        rate = t.rate[kind, types]
        led["keeper_gens"] += int(t.keeper[kind, types].sum())
        led["best_growth"] = max(led["best_growth"], float((rate * q_copy).max()))

        # 1. uptake: an equal share, of which a cell takes what its metabolism can use
        share = nutrient_share(p["inflow"], n)
        m = _balance(g, types) if self.corrector else np.ones(n, dtype=np.float64)
        uptake = np.floor(share * m * t.eff[types]).astype(np.int64)
        avail = self.store + uptake

        # 2. copying: per template at most rate * balance attempts, as far as the nutrient reaches
        templates = np.where(linked, g[:, 0], total)
        want = rate * np.minimum(avail // cost, m * templates)
        whole = np.floor(want)
        attempts = whole.astype(np.int64) + (rng.uniform(n) < (want - whole))
        n_att = int(attempts.sum())
        good = np.zeros(n, dtype=np.int64)
        if n_att:
            cell = np.repeat(np.arange(n), attempts)
            ok = rng.uniform(n_att) < q_copy[cell]
            good = np.bincount(cell[ok], minlength=n)
            loose = cell[ok & ~linked[cell]]
            if len(loose):  # which gene type each good copy of a loose gene is: the fast type weighs more
                weight = g.astype(np.float64)
                weight[:, 0] *= 1.0 + p["selfish"]
                cum = np.cumsum(weight, axis=1)
                pick = rng.uniform(len(loose)) * cum[loose, -1]
                last = width - 1 - np.argmax(g[:, ::-1] > 0, axis=1)  # the last type a cell holds
                kind_of = np.minimum((cum[loose] <= pick[:, None]).sum(axis=1), last[loose])
                np.add.at(self.genes, (loose, kind_of), 1)
            if linked.any():
                g += (good * linked)[:, None] * (_COLS[None, :width] < types[:, None])
        led["attempts"] += n_att
        led["failures"] += n_att - int(good.sum())
        led["built"] += int((good * cost).sum())
        led["waste"] += int(((attempts - good) * cost).sum())
        left = avail - attempts * cost
        self.store = np.minimum(left, cost - 1)  # an unfinished copy's worth is kept, the rest flows out
        led["unused"] += p["inflow"] - share * n + int((share - uptake).sum()) + int((left - self.store).sum())

        # 3. a cell that has grown to the threshold divides; a linked one may enlarge its genome instead
        need = np.maximum(MIN_CHROMOSOMES, -(-p["threshold"] // types))
        ready = full & np.where(linked, g[:, 0] >= need, g.sum(axis=1) >= p["threshold"])
        if p["innovation"] > 0 and linked.any():
            ready &= ~self._enlarge(np.flatnonzero(ready & linked & (types < GMAX)))
        mothers = np.flatnonzero(ready)
        if len(mothers):
            self._divide(mothers, width)

        # 4. death, then the vessel's limit
        self._remove(rng.uniform(len(self.types)) < np.where(self._full(), DEATH, DEATH_BROKEN))
        n = len(self.types)
        if n > CAP:  # washed out: all but CAP cells drawn at random
            gone = np.ones(n, dtype=bool)
            gone[np.argsort(rng.uniform(n), kind="stable")[:CAP]] = False
            self._remove(gone)

    def _enlarge(self, rows: np.ndarray) -> np.ndarray:
        """Larger genome: in a linked cell about to divide, with chance ``innovation``, the
        chromosomes are broken up and rebuilt with one more gene type (a changed duplicate):
        as many full sets of the longer chromosome as the genes at hand allow, the rest is lost.
        Returns which cells did so (they do not divide in this generation)."""
        done = np.zeros(len(self.types), dtype=bool)
        hit = rows[self.rng.uniform(len(rows)) < self.p["innovation"]]
        if len(hit):
            types = self.types[hit]
            sets = self.genes[hit, 0] * types // (types + 1)
            new = np.where(_COLS[None, :] < (types + 1)[:, None], sets[:, None], 0)
            self.led["lost"] += int(self.genes[hit].sum() - new.sum()) * self.p["gene_length"]
            self.genes[hit] = new
            self.types[hit] = types + 1
            done[hit] = True
        return done

    def _divide(self, mothers: np.ndarray, width: int) -> None:
        p, led, rng = self.p, self.led, self.rng
        n, nd = len(self.types), len(mothers)
        whole = self.genes[mothers]
        first = (whole + 1) // 2  # chromosomes are shared out evenly
        loose = ~self.linked[mothers]
        if loose.any():  # loose genes go to either daughter by the toss of a coin
            first[loose] = 0
            first[np.flatnonzero(loose)[:, None], _COLS[None, :width]] = split(rng, whole[loose][:, :width])
        store_first = self.store[mothers] // 2
        self.genes[mothers] = first
        self.genes = np.concatenate([self.genes, whole - first])
        self.store = np.concatenate([self.store, self.store[mothers] - store_first])
        self.store[mothers] = store_first
        for name in ("types", "linked", "dna", "endo"):
            arr = getattr(self, name)
            setattr(self, name, np.concatenate([arr, arr[mothers]]))

        # innovations, per daughter
        rows = np.concatenate([mothers, np.arange(n, n + nd)])
        chance = p["innovation"]
        if chance > 0:
            u = rng.uniform(3 * len(rows)).reshape(len(rows), 3)
            types = self.types[rows]
            genes = self.genes[rows]
            mask = _COLS[None, :] < types[:, None]
            # chromosome: one gene of each type per chromosome, copies without partners are lost
            link = ~self.linked[rows] & ((genes > 0).sum(axis=1) == types) & (u[:, 0] < chance)
            if link.any():
                sets = np.where(mask, genes, np.iinfo(np.int64).max).min(axis=1)
                new = np.where(mask, sets[:, None], 0)
                led["lost"] += int((genes[link] - new[link]).sum()) * p["gene_length"]
                self.genes[rows[link]] = new[link]
                self.linked[rows[link]] = True
            self.dna[rows[~self.dna[rows] & (u[:, 1] < chance)]] = True
            self.endo[rows[~self.endo[rows] & (u[:, 2] < chance * ENDO_RARITY)]] = True
        self._remove(self.genes.sum(axis=1) == 0)  # a daughter without any gene is no cell

    def snapshot(self) -> list[list[int]]:
        """Per cell: gene types needed, linked, dna, partner, stored bases, then its gene counts."""
        if not len(self.types):
            return []
        head = np.stack([self.types, self.linked, self.dna, self.endo, self.store], axis=1)
        return [row[:5 + row[0]] for row in np.concatenate([head, self.genes], axis=1).tolist()]


def split(rng: _Rng, counts: np.ndarray) -> np.ndarray:
    """What the first daughter gets of each gene type: every copy goes to either daughter with
    chance one half (one uniform number per copy, in the order of the table)."""
    flat = counts.ravel()
    heads = rng.uniform(int(flat.sum())) < 0.5
    return np.bincount(np.repeat(np.arange(flat.size), flat)[heads], minlength=flat.size).reshape(counts.shape)


def census(snap: list[list[int]]) -> dict[str, int]:
    """The gate's counts of one step, from the cells themselves."""
    return {
        "cells": len(snap),
        "n_viable": sum(1 for c in snap if min(c[5:]) > 0),
        "n_linked": sum(c[1] for c in snap),
        "n_dna": sum(c[2] for c in snap),
        "n_complex": sum(c[3] for c in snap),
        "n_working": sum(1 for c in snap if c[2] and c[3]),
        "genes": sum(sum(c[5:]) for c in snap),
        "types_sum": sum(sum(1 for x in c[5:] if x > 0) for c in snap),
        "selfish_genes": sum(c[5] for c in snap),
        "stored": sum(c[4] for c in snap),
    }


def measure(counts: dict[str, int], p: dict, led: dict, before: dict | None) -> dict[str, float | int | str]:
    """One output step: the metrics (3 significant digits) from the counts and the ledger."""
    n, n_dna, n_complex, n_working = counts["cells"], counts["n_dna"], counts["n_complex"], counts["n_working"]
    q0 = p["fidelity"]
    q1 = 1.0 - dna_error(1.0 - q0)
    attempts = led["attempts"] - (before["attempts"] if before else 0)
    failures = led["failures"] - (before["failures"] if before else 0)
    error = ((n - n_dna) * (1.0 - q0) + n_dna * (1.0 - q1)) / n if n else 1.0
    energy = ENERGY * ((n - n_complex) + (n_complex - n_working) * partner_gain(q0)
                       + n_working * partner_gain(q1)) / n if n else 0.0
    step: dict[str, float | int | str] = {
        "cells": n,
        "genes_per_cell": sig(counts["genes"] / n) if n else 0.0,
        "gene_types": sig(counts["types_sum"] / n) if n else 0.0,
        "genome_length": sig(counts["types_sum"] * p["gene_length"] / n) if n else 0.0,
        "fidelity": 1.0 - sig(error),
        "error_load": sig(failures / attempts) if attempts else 0.0,
        "viable": sig(counts["n_viable"] / n) if n else 0.0,
        "energy_per_cell": sig(energy),
        "linked": sig(counts["n_linked"] / n) if n else 0.0,
        "dna": sig(n_dna / n) if n else 0.0,
        "complex": sig(n_complex / n) if n else 0.0,
        "selfish": sig(counts["selfish_genes"] / counts["genes"]) if counts["genes"] else 0.0,
        "stage": stage_of(led["generation"], n, counts["n_viable"], counts["n_linked"], n_dna, n_working),
    }
    step.update({k: v for k, v in counts.items() if k != "cells"})
    step["bases"] = counts["genes"] * p["gene_length"]
    step.update(led)
    return step


def stage_of(generation: int, cells: int, viable: int, linked: int, dna: int, working: int) -> str:
    """The stage follows from the counts: ``collapse`` without a cell; ``soup`` at generation 0
    (replicators enclosed at random, nothing selected yet) and whenever fewer than half the
    compartments hold a full set; then by what half the cells have reached: a partner together
    with dna (``complex_cells``), chromosomes, dna (``cells``), or none of these (``protocells``)."""
    if cells == 0:
        return "collapse"
    if generation == 0 or 2 * viable < cells:
        return "soup"
    if 2 * working >= cells:
        return "complex_cells"
    if 2 * linked >= cells:
        return "chromosomes"
    return "cells" if 2 * dna >= cells else "protocells"


def summarise(steps: list[dict]) -> dict[str, float | int | str]:
    """The summary follows from the steps (and is what the level hands upward)."""
    last = steps[-1]
    first = next((s["generation"] for s in steps if s["cells"] and 2 * s["n_linked"] >= s["cells"]), "never")
    outcome = {"collapse": "collapse", "soup": "protocells", "protocells": "protocells", "cells": "cells",
               "chromosomes": "cells", "complex_cells": "complex_cells"}[last["stage"]]
    return {"stage": last["stage"], "cells": last["cells"], "genome_length": last["genome_length"],
            "complex": last["complex"], "energy_per_cell": last["energy_per_cell"],
            "first_chromosome": first, "outcome": outcome}


@dataclass
class CellRollout(Rollout):
    """A :class:`Rollout` that also keeps what follows from the parameters and every cell at
    every output step."""
    derived: dict[str, float | str] = field(default_factory=dict)
    #: per step, per cell: ``[gene types needed, linked, dna, partner, stored bases, copies of type 1, 2, ...]``
    snaps: list[list[list[int]]] = field(default_factory=list)

    def value(self, text: float | int | str) -> str:
        return dense_value(text)


def dense_value(value: float | int | str) -> str:
    """A metric as it stands in a line: a word, an exact count, or a number that was rounded to
    3 significant digits when it was measured (up to 9 digits are written, so a fidelity of
    0.99999, one minus an error rate of 3 digits, keeps its nines).

    >>> dense_value(0.99999), dense_value(1230.0), dense_value(0.000123), dense_value(7), dense_value("soup")
    ('0 point 9 9 9 9 9', '1 2 3 0', '1 point 2 3 e minus 4', '7', 'soup')
    """
    if isinstance(value, str):
        return value
    return num(value, sig=9)


def run(seed: int, rules: int = 7, corrector: bool = True, **params: float | int) -> CellRollout:
    """Deterministic for a given seed and parameters (the PCG64 stream of ``seed``).

    Parameters not given take :data:`DEFAULTS`. ``corrector=False`` is the control experiment
    without selection among cells: a cell then grows, divides and dies the same whatever genes
    it holds (not a parameter of the canonical runs, never in a line)."""
    if rules != 7:
        raise ValueError("cells is a round 7 level: rules must be 7")
    p = _params(params)
    vessel = _Vessel(_Rng(seed), p, corrector)
    r = CellRollout(SIM, int(seed), p, [], derived=derived(p))
    for t in range(STEPS + 1):
        if t:
            for _ in range(GENS_PER_STEP):
                vessel.advance()
        snap = vessel.snapshot()
        r.snaps.append(snap)
        r.steps.append(measure(census(snap), p, vessel.led, r.steps[-1] if t else None))
    r.summary = summarise(r.steps)
    return r


def _digits_end(words: list[str], start: int) -> int:
    """Index after the run of single-digit tokens that begins at ``start``."""
    i = start
    while i < len(words) and words[i] in _DIGITS:
        i += 1
    return i


def _whole(words: list[str]) -> int | None:
    """The whole number written by digit tokens, in its one spelling (no leading zeros)."""
    value = parse_num(words)
    return value if isinstance(value, int) and value >= 0 and num(value).split() == words else None


class Cells:
    """The simulation and its gate. Rollouts replayed by :meth:`check` are cached per seed
    (without their cells: a question never asks for one cell)."""

    sim = SIM
    topic = SIM
    KEYS = KEYS

    def __init__(self) -> None:
        self._cache: dict[int, CellRollout] = {}

    def run(self, seed: int, rules: int = 7, **params: float | int) -> CellRollout:
        return run(seed, rules, **params)

    def rollout(self, seed: int) -> CellRollout:
        """The canonical run of a seed: parameters from ``random_params(random.Random(seed))``.
        Cached without its cells (``snaps`` is empty); :func:`run` gives them."""
        if seed not in self._cache:
            if len(self._cache) >= 256:
                self._cache.clear()
            r = run(seed, **random_params(random.Random(seed)))
            r.snaps = []
            self._cache[seed] = r
        return self._cache[seed]

    # -- the gate of the simulation itself
    def conserved(self, r: Rollout) -> Verdict:
        """The ledger and what must follow from it, at every step:

        * nutrient: ``inflow_total == inflow * generation == built + waste + unused + stored``, exactly;
        * genes: ``bases == bases at step 0 + built - lost``, exactly; no count is negative and no
          ledger entry ever falls; a failed copy wastes and a good one builds at least one gene;
        * the cells themselves (``snaps``) give the same counts, no gene count is negative, a
          linked cell holds the same number of every gene, a cell stores less than one copy costs;
        * ``n_viable <= cells <= 400``, a linked cell is viable, the shares are the counts over
          the cells, the stage follows from the counts, the summary from the steps, the derived
          numbers from the parameters;
        * error catastrophe: where no cell ever lived under its error threshold, what is left of
          the genes may not exceed what the best growth any cell had allows (see the module text).
        """
        try:
            p = _params({k: r.params[k] for k in PARAM_KEYS})
        except (KeyError, ValueError) as err:
            return Verdict(False, None, f"parameters: {err}")
        if dict(r.params) != p:
            return Verdict(False, None, "parameters are not in their plain form")
        if len(r.steps) != STEPS + 1:
            return Verdict(False, num(STEPS + 1), f"{len(r.steps)} steps")
        snaps = getattr(r, "snaps", None) or None
        if snaps is not None and len(snaps) != len(r.steps):
            return Verdict(False, None, "one snapshot per step is needed")
        t = tables(p["fidelity"], p["gene_length"])
        length = p["gene_length"]
        first = r.steps[0]
        for i, s in enumerate(r.steps):
            before = r.steps[i - 1] if i else None
            if set(s) != set(STATE_KEYS) | set(LEDGER_KEYS):
                return Verdict(False, None, f"step {i}: keys do not fit")
            if any(isinstance(s[k], (str, bool)) or not is_finite(s[k]) or s[k] < 0
                   for k in STATE_KEYS + LEDGER_KEYS if k != "stage"):
                return Verdict(False, None, f"step {i}: a negative number or none at all")
            if s["generation"] != i * GENS_PER_STEP or s["inflow_total"] != p["inflow"] * s["generation"]:
                return Verdict(False, num(p["inflow"] * i * GENS_PER_STEP), f"step {i}: generation or inflow_total")
            spent = s["built"] + s["waste"] + s["unused"] + s["stored"]
            if spent != s["inflow_total"]:
                return Verdict(False, num(s["inflow_total"]), f"step {i}: built + waste + unused + stored is {spent}")
            if s["bases"] != first["bases"] + s["built"] - s["lost"] or s["bases"] != s["genes"] * length:
                return Verdict(False, num(first["bases"] + s["built"] - s["lost"]),
                               f"step {i}: bases {s['bases']} do not fit the genes built and lost")
            if not s["failures"] <= s["attempts"] or s["waste"] < s["failures"] * length \
                    or s["built"] < (s["attempts"] - s["failures"]) * length:
                return Verdict(False, None, f"step {i}: copies, waste and built do not fit")
            if before is not None and any(s[k] < before[k] for k in MONOTONE_KEYS):
                return Verdict(False, None, f"step {i}: a ledger entry fell")
            if not s["n_linked"] <= s["n_viable"] <= s["cells"] <= CAP \
                    or not s["n_working"] <= min(s["n_dna"], s["n_complex"]) \
                    or not max(s["n_dna"], s["n_complex"]) <= s["cells"] \
                    or not s["cells"] <= s["types_sum"] <= s["genes"] or not s["selfish_genes"] <= s["genes"]:
                return Verdict(False, None, f"step {i}: the counts do not fit each other")
            counts = {k: s[k] for k in CENSUS_KEYS}
            if snaps is not None:
                bad = _bad_cell(snaps[i], length)
                if bad:
                    return Verdict(False, None, f"step {i}: {bad}")
                if census(snaps[i]) != counts:
                    return Verdict(False, None, f"step {i}: the counts are not those of the cells")
            again = measure(counts, p, {k: s[k] for k in LED_KEYS}, before)
            wrong = next((k for k in STATE_KEYS if again[k] != s[k]), None)
            if wrong:
                return Verdict(False, dense_value(again[wrong]), f"step {i}: {wrong} does not follow from the counts")
        if r.summary != summarise(r.steps):
            return Verdict(False, None, "the summary does not follow from the steps")
        if getattr(r, "derived", None) not in (None, derived(p)):
            return Verdict(False, None, "the derived numbers do not follow from the parameters")
        return self._catastrophe(r, p, snaps, t)

    @staticmethod
    def _catastrophe(r: Rollout, p: dict, snaps: list | None, t: _Tables) -> Verdict:
        """No lineage beyond its error threshold keeps its information. A cell is beyond it when
        the piece it copies is longer than ``error_threshold(9 * rate, fidelity)`` of its kind;
        its genes then grow by less than they are lost, on average by the factor
        ``(1 + best_growth) * (1 - DEATH) < 1`` per generation at the very best. Where every cell
        of a run was beyond its threshold, the bases left may not exceed a billion times that
        bound (by Markov's inequality a true run fails this less than once in a million)."""
        last, length = r.steps[-1], p["gene_length"]
        if last["keeper_gens"] > 0:
            return Verdict(True, None, "ledger, counts, stages and summary are consistent")
        if snaps is not None and any(t.keeper[c[1] * 4 + c[2] * 2 + c[3], c[0]] for snap in snaps for c in snap):
            return Verdict(False, None, "a cell under its error threshold, but none in the ledger")
        if last["best_growth"] > (1.0 + 1e-9) / SURVIVAL_ODDS:
            return Verdict(False, None, "no cell under its error threshold, yet one grew faster than cells die")
        bound = r.steps[0]["bases"] * ((1.0 + last["best_growth"]) * (1.0 - DEATH)) ** last["generation"]
        if last["bases"] > 1e9 * bound:
            return Verdict(False, num(0), f"error catastrophe: {last['bases']} bases kept beyond the error threshold")
        return Verdict(True, None, "ledger, counts, stages and summary are consistent; beyond the error threshold")

    # -- the gate of the lines
    def parse(self, prompt: str) -> tuple[int, str, int, str] | None:
        """``(seed, where, step, key)`` with ``where`` in ``step``, ``next``, ``final``, ``param``, ``derived``."""
        w = prompt.split(" ")
        if len(w) < 5 or w[0] != SIM or w[1] != "seed":
            return None
        i = _digits_end(w, 2)
        seed = _whole(w[2:i])
        if seed is None or i >= len(w):
            return None
        where, rest = w[i], w[i + 1:]
        if where in ("final", "param", "derived"):
            keys = {"final": SUMMARY_KEYS, "param": PARAM_KEYS, "derived": DERIVED_KEYS}[where]
            return (seed, where, -1, rest[0]) if len(rest) == 1 and rest[0] in keys else None
        if where != "step":
            return None
        j = _digits_end(rest, 0)
        step, rest = _whole(rest[:j]), rest[j:]
        if step is None or step > STEPS:
            return None
        if len(rest) == 2 and rest[0] == "next" and rest[1] in STATE_KEYS and step < STEPS:
            return seed, "next", step, rest[1]
        if len(rest) == 1 and rest[0] in STATE_KEYS:
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
        if where == "param":
            return r.params[key]
        if where == "derived":
            return r.derived[key]
        return r.steps[step + (where == "next")][key]

    def check(self, prompt: str, answer: str) -> Verdict:
        """Replay the seed and compare: words, counts and parameters exactly; the fidelity by its
        error rate and every other number within 5 percent (a 0 only by 0)."""
        value = self.truth(prompt)
        if value is None:
            return Verdict(False, None, "not my question")
        words = prompt.split(" ")
        key, expected = words[-1], dense_value(value)
        if isinstance(value, str):
            ok = answer.strip() == value
            return Verdict(ok, expected, "exact word" if ok else "wrong word")
        got = parse_num(answer)
        if not is_finite(got):  # "1 e 9 9 9" parses to infinity, a word to nothing
            return Verdict(False, expected, "not a number")
        if isinstance(value, int) or words[-2] == "param":
            ok = got == value
            return Verdict(ok, expected, "exact" if ok else "not the exact number")
        if key == "fidelity":
            ok = close(1.0 - float(got), 1.0 - value, rel=0.05, abs_=1e-12)
            return Verdict(ok, expected, "error rate within 5 percent" if ok else "error rate more than 5 percent off")
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


#: the counts :func:`census` takes from the cells
CENSUS_KEYS = ("cells", "n_viable", "n_linked", "n_dna", "n_complex", "n_working", "genes", "types_sum",
               "selfish_genes", "stored")
#: the running ledger of the vessel
LED_KEYS = ("generation", "inflow_total", "built", "waste", "unused", "lost", "attempts", "failures",
            "keeper_gens", "best_growth")
MONOTONE_KEYS = ("built", "waste", "unused", "lost", "attempts", "failures", "keeper_gens", "best_growth")


def _bad_cell(snap: list[list[int]], length: int) -> str:
    """What is wrong with the cells of one step, or the empty string."""
    for c in snap:
        if len(c) < 6 or any(isinstance(x, bool) or not isinstance(x, int) for x in c):
            return "a cell that is not a row of whole numbers"
        types, linked, dna, partner, stored = c[:5]
        genes = c[5:]
        if not 1 <= types <= GMAX or len(genes) != types or {linked, dna, partner} - {0, 1}:
            return "a cell with an impossible genome"
        if min(genes) < 0 or sum(genes) == 0:
            return "a negative gene count or a cell without genes"
        if linked and min(genes) != max(genes):
            return "a linked cell with unequal gene counts"
        if not 0 <= stored < length * (types if linked else 1):
            return "a cell stores more than one copy costs"
    return ""


def params_line(r: Rollout) -> Line:
    """``cells seed 7 params. gene_types 4. gene_length 8 0. fidelity 0 point 9 9 5. ...``"""
    fields = " ".join(f"{k} {dense_value(r.params[k])}." for k in PARAM_KEYS)
    return Line(f"{SIM} seed {num(r.seed)} params. {fields}", topic=SIM, kind="record")


def derived_line(r: CellRollout) -> Line:
    """``cells seed 7 derived. copy_success 0 point 6 7. sigma 6 point 4 3. l_max 3 7 2. ...``"""
    fields = " ".join(f"{k} {dense_value(r.derived[k])}." for k in DERIVED_KEYS)
    return Line(f"{SIM} seed {num(r.seed)} derived. {fields}", topic=SIM, kind="record")


def lines(r: CellRollout, every: int = 1) -> list[Line]:
    """The parameter and derived records and their questions, a state record and questions for
    every ``every``-th step (now and one step on), and the final questions. A vessel without a
    cell stays empty, so the steps after the first ``collapse`` are left out. The ledger is left
    out too: it is the gate's, not the model's."""
    keys = list(STATE_KEYS)
    head = f"{SIM} seed {num(r.seed)}"
    out = [params_line(r), derived_line(r)]
    out += [Line(f"{head} param {k}", dense_value(r.params[k]), SIM, "fact") for k in PARAM_KEYS]
    out += [Line(f"{head} derived {k}", dense_value(r.derived[k]), SIM, "calc") for k in DERIVED_KEYS]
    for t in range(0, len(r.steps), every):
        out.append(state_line(r, t, keys))
        out += query_lines(r, t, keys)
        if r.steps[t]["stage"] == "collapse":
            break
    return out + summary_lines(r)


_SIM = Cells()


def simulation() -> Cells:
    return _SIM


def conserved(rollout: Rollout) -> Verdict:
    return _SIM.conserved(rollout)


def check(prompt: str, answer: str) -> Verdict:
    return _SIM.check(prompt, answer)


def owns(prompt: str) -> bool:
    return _SIM.owns(prompt)
