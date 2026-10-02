"""Round 6, level 4: a toy chemistry of a cooling cloud or planet surface.

A pool of about twenty thousand atoms, drawn from one of four element mixes, cools from a start
temperature (3000-6000 K) to an end temperature (100-1500 K) over about thirty steps. Atoms bond
by valence: every element holds a fixed number of bonds (H 1, O 2, C 4, N 3, ...; He and Ne none,
they are noble), and every pair of elements has a *stability temperature* from a small table: the
bond forms when the gas is cooler than that and breaks when it is hotter. So a hot gas is atomic,
a cool gas is molecular, and the strongest bonds (N-N, Si-O, C-O) form first.

Each step, at the step's temperature (which falls by a constant factor per step):

1. every bond hotter than its stability temperature breaks;
2. in each of :data:`ROUNDS` rounds, the atoms that still have a free valence are shuffled
   (seeded) and paired off; a pair whose bond is stable at this temperature bonds and shares as
   many valences as both have free, up to three. So O meets O gives O=O, N meets N gives a
   triple bond, Si meets a fresh O gives Si=O and then O=Si=O, and H meets O gives OH and then
   H2O. Two atoms that already share a bond do not bond again. Nothing else happens.

Molecules are the connected groups of two or more atoms, counted by composition: ``h2o`` is any
molecule with exactly two H and one O, ``organic`` any with more than two carbons. ``free_atoms``
is the fraction of all atoms with no bond, noble gases included.

Mixes (fractions of atoms; a toy pattern, see ``MIXES``):

    cosmic   H 90 %, He 9 %, traces of O, C, N, Ne, Mg, Si, S, Fe in roughly solar proportions
    rocky    O 55 %, Si 14 %, Mg 10 %, Fe 6 %, then Al, Ca, Na, K, H, C, N, S, Cl
    ocean    H 59 %, O 33 %, then Na, Cl, C, S, N, Mg
    carbon   H 70 %, C 16 %, O 6 %, N 4 %, Si, S: more carbon than oxygen, like a carbon star's
             wind, the one mix where carbon is left over to make chains

What is real and what is toy
----------------------------
Real: atoms are neither made nor lost; the noble gases do not bond; the school valences give
the right formulas (H2, H2O, CH4, NH3, CO2, N2, O2, NaCl, SiO2, MgO); a hot gas is atomic and a
cooling one turns molecular, strong bonds first; cool gas of solar make-up is H2 with water,
methane and ammonia; free O2 hardly ever forms (on a planet, much free O2 is read as a possible
sign of life).

Toy, and not to be read as chemistry:

- the stability temperatures are invented numbers in a believable order, not measurements; a
  real bond does not break at one temperature but gradually, depending on pressure;
- there is no energy, no pressure, no reaction rate or barrier, no shape, no charge. Bonded
  atoms never swap partners, so the end state records who met whom, it is not an equilibrium;
- nothing condenses: ``frozen`` only means that no bond breaks any more. Real rock is a lattice
  of silicates and real sea salt is dissolved ions; here they are loose SiO2, MgO, FeO and NaCl
  units, and real iron in such a cloud is mostly metal, not FeO;
- carbon monoxide: in the real universe CO is the commonest molecule after H2. Here a C=O keeps
  two free valences on the carbon and goes on to CO2 or H2CO, so ``co`` rises and falls again
  and ends at zero or close to it;
- odd leftovers are counted as molecules like any other (OH, H2SO2, a carbon chain with open
  ends); ``organic`` means only "more than two carbons";
- the mixes are round toy fractions and iron's valence is fixed at 2 (see ``NOTES``).

Plausibility targets
--------------------
Tested in ``tests/test_cosmos_chem.py``; the numbers are measured on seeds 1-40, each seed's
own parameters with each of the four mixes (160 runs):

- at 4000 K and above the gas is atomic: free_atoms 1 and no molecule on all 492 such steps;
- the temperature only falls and the stages come in order plasma, atomic, forming, molecular,
  frozen; the stage always follows from the temperature and free_atoms shown in the same line;
- cosmic: ends as H2 (98 % of all molecules); water is the most common molecule with oxygen
  and holds 92-100 % of the oxygen; helium and neon stay free (free_atoms 0.091);
- rocky: the three most common molecules are SiO2, MgO, FeO in that order, in every run
  (82-100 % of the silicon, 84-97 % of the magnesium, 53-78 % of the iron); no H2;
- ocean: water is the most common molecule, water_fraction 0.83-0.89; NaCl is the second most
  common whenever the gas ends below 1500 K (it forms only below that; at 1500 K there is none);
- carbon: H2 is the most common molecule, then CH4 (277-890) and 231-420 organic chains, the
  biggest of 17-54 atoms; the other three mixes make at most two organic molecules.

A seed's own run always cools; ``run`` also takes an end temperature above the start, and then
the molecules break up again in the reverse order.

Lines (the dense alphabet of round 5; the model only ever sees these):

    chem seed 4 2 step 1 2. mix cosmic. temperature 2 0 8 8. stage atomic. free_atoms 0 point 9 7 5.
        molecules 1 2 6. h2 0. h2o 6 6. ch4 4 2. nh3 1 7. nacl 0. sio2 0. organic 0. biggest 5.
    q chem seed 4 2 step 1 2 ch4. a 4 2.
    q chem seed 4 2 step 1 2 next temperature. a 2 0 2 6.     (one step later)
    q chem seed 4 2 final water_fraction. a 0 point 0 0 7 7 7.
    q chem seed 4 2 final first_water_step. a 1.
    chem valence. h 1. he 0. o 2. c 4. ...
    chem bond h o. stable_below_k 3 0 0 0.
    chem molecule h2o. formula h 2 o 1. atoms 3.
    chem mix cosmic. main h. elements h he o c ne n mg si s fe.
    q chem valence c. a 4.
    q chem bond h he stable_below_k. a never.

Gate: a prompt names a seed; the parameters follow from the seed (:func:`random_params`), the
run is replayed and compared: a count may be off by one or by 5 %, other numbers by 5 %, words
must be exact, and an answer must be a finite number. :meth:`Chem.conserved` replays a rollout
and checks that every element's atom count stays the same (free plus bonded), that no atom holds
more bonds than its valence, that no two atoms are bonded twice, that noble gases never bond,
that no count is negative, that the molecules a step counts fit into the atoms the mix has
(:func:`budget`) and that the rollout's numbers are the replay's.

The same seed gives the same rollout on any machine: the shuffle is a stable sort of raw PCG64
draws, the state is whole numbers, the temperature is rounded with whole-number arithmetic
(:func:`temperature_at`), and the only floats are single divisions rounded to three digits.

Rules 7
-------
Round 7 reviewed this level and found rules that were wrong or arbitrary. The corrections are an
option: ``run(seed, rules=7)`` (also ``random_params(rng, rules=7)``, ``lines``, ``generate``,
``records`` and ``table_lines`` with ``rules=7``). Without ``rules`` every rollout and every line
is round 6's, bit for bit; the version-5 models were trained on those. A rules-7 rollout carries
``"rules": 7`` in its parameters and every one of its lines says so, because the same seed has
another story under each rule set:

    chem seed 5 rules 7 params. mix ocean. t_start 5 3 0 0. t_end 4 4 0. steps 3 5. atoms 2 2 0 0 0. sparks 0.
    chem seed 5 rules 7 step 3 0. mix ocean. temperature 6 2 8. stage molecular. water vapor.
        solids corundum silicate iron iron_sulfide. spark 0. hits 0. free_atoms 0. molecules 6 9 3 9. h2 0.
        h2o 5 9 5 2. ch4 0. nh3 0. co 0. co2 5 0. o2 3 0. n2 3 9. nacl 2 9 6. h2s 1. h2co 0. other 5 7 1.
    chem seed 5 rules 7 step 3 0 census. organic 3 6. precursors 3 6. chains 2 0. biggest 8 0. rock 0.
        grains 0. si_o 0. mg_o 2 9 7. fe_o 0. fe_metal 0.
    q chem seed 5 rules 7 step 3 0 h2o. a 5 9 5 2.
    q chem seed 5 rules 7 step 3 0 next o2. a 2 9.
    q chem seed 5 rules 7 final water_share. a 0 point 8 6 5.
    q chem seed 5 rules 7 param atoms. a 2 2 0 0 0.
    chem rules 7 compound h2o. formula h 2 o 1. atoms_kj 6 8 5 point 2. formation_kj minus 2 4 1 point 8.
        bond h o. order 1. units 2. unit_kj 4 6 4.
    chem rules 7 bond h o order 1. enthalpy_kj 4 6 4. stable_below_k 2 2 2 7. source h2o.
    q chem rules 7 bond c o order 2 enthalpy_kj. a 8 0 4.
    q chem rules 7 mix ocean share o. a 0 point 3 3.
    q chem rules 7 molecule h2o name. a water.
    q chem predict stable_below_k enthalpy_kj 4 6 4. a 2 2 2 7.          (a lesson: the inputs are in the question)

The widest rules-7 line measured has 107 tokens (limit 128); a state is two records for that reason.

What changed, the real value each change follows, and what stays toy:

1. **Bond enthalpies instead of invented stability temperatures.** The table is computed in code
   by Hess's law from standard enthalpies of formation (:data:`FORMATION_KJ`, :data:`ATOM_KJ`):
   the enthalpy of the free atoms minus the enthalpy of formation of the compound, divided by
   the valence units bonded (:func:`unit_enthalpy`): O-H 464 kJ/mol from water, C-H 416 from
   methane, the C=O of carbon dioxide 804, the N-N triple bond 945. Bonds that no single compound
   isolates are textbook mean bond enthalpies (:data:`MEAN_KJ`: C-C 346, C=C 614, C-N 305, O-O
   146). Two metals share the atomisation enthalpy of each per valence (Fe-Fe 416). Any other
   pair gets :data:`DEFAULT_KJ` 200, printed as ``bond default`` and never asked as a fact about
   a pair; 24 of the 91 pairs are default (silicides, carbides and nitrides of the metals, Si-N,
   Si-S, N-S, Cl-N, Cl-O, Cl-S). All enthalpies were typed from memory of the usual tables (CRC
   handbook, NIST-JANAF) and were not checked against a handbook: see ``NOTES7``. Toy: for a solid the
   number is the atomisation of the lattice spread over the valences, which is more than the
   bond of the gaseous molecule (Na-Cl 640 against 412 for gaseous NaCl), so toy salt "holds"
   up to 3072 K, which is no temperature of real salt.
2. **One rule for the temperature.** A bond holds below 4.8 K per kJ/mol of its enthalpy
   (:func:`stable_below_k`): the rule of thumb that a thin gas is half dissociated where the bond
   enthalpy is about 25 R T. H2 gives 2092 K, water 2227 K, N2 4536 K, which is :data:`HOT_K`:
   above it nothing bonds. For H2 itself the rule is right near 1e-5 bar; at the 1e-4 bar of the
   condensation table below, H2 is half dissociated near 2300 K, about 22 R T (equilibrium
   constants of the JANAF tables, from memory). Toy: one temperature per bond, no pressure, and
   the factor is the same for every bond. The strongest real bond, the triple bond of carbon
   monoxide (1072 kJ/mol), is not in the table: oxygen holds two bonds here.
3. **Bond orders.** Only C-C, C-N, N-N can be triple and C-O, O-O, N-O, O-S, C-S, S-S double
   (:data:`HIGHEST_ORDER`); every other pair bonds singly, the double-bond rule (after Pitzer
   1948). Si-O, Mg-O and Fe-O are then single bonds, each oxygen bridges two atoms, and rock
   condenses by itself into one network: ``rock`` (atoms in groups of 100 or more atoms),
   ``grains``, and the bond counts ``si_o``, ``mg_o``, ``fe_o`` replace round 6's loose SiO2,
   MgO and FeO units.
4. **Exchange: bonded atoms can swap partners when that gives off heat.** After each meeting
   round all bonding atoms pair off again (:meth:`Pool7.exchange`). The pair's bond rises by one
   order if the order exists and holds at this temperature. An atom with no free valence pays
   with the cheapest valence it shares with another neighbour; if both atoms pay, their two
   released neighbours bond to each other when they can (a double displacement: HCl + NaOH gives
   NaCl + H2O). The move is made only if the enthalpy of the bonds made is above that of the
   bonds broken (:func:`move_allowed`). What is left of a multiple bond breaks at once if it
   cannot hold at this temperature. Invented for the toy: the pairing, one order per move, no
   barrier (every exothermic exchange happens as soon as the atoms meet), and heat alone decides.
   Real: at low temperature the direction of a reaction is set mostly by the heat it gives off;
   the full criterion is the free energy, which also counts entropy (T times delta S), and that
   term the toy leaves out. The end state is then close to the low-temperature products instead
   of a record of who met whom. Without the exchange round the same table leaves an
   ocean with less than half the salt and a third of its hydrogen as H2 beside free O2
   (measured on 6000 atoms, seeds 1-3: water 1105-1130 against 1631-1648 molecules, NaCl 28-41
   against 83-89, H2 549-582 and O2 269-281 against 0 and 4-8).
5. **Sparks: an energy source.** The parameter ``sparks`` (0 to 20) adds that many steps at the
   end temperature. A spark takes apart :data:`SPARK_PPM` of a million groups of 2 to 64 atoms
   (:func:`expected_hits`), then the atoms meet and exchange again. Real: Miller's 1953 experiment
   needed a reducing gas (CH4, NH3, H2, H2O) and days of sparks; in neutral CO2-N2-H2O gas the
   yields are far lower (after Schlesinger and Miller 1983), and the early atmosphere is now
   thought to have been of that neutral kind (after Kasting 1993). The fifth mix ``reducing``
   puts the contrast into the data. Toy: the 2 percent, the size limit, no energy is counted;
   the reducing mix makes about 7 times the precursors of the ocean without sparks (5.6 to 10
   seed by seed) and about 2 times after ten, not the real orders of magnitude. Cooling from atoms is itself an energy
   source (the gas of a lightning channel or an impact plume cools like this); a run with
   ``sparks`` 0 is "quench only".
6. **organic means a carbon-hydrogen bond.** ``organic`` counts the groups with at least one C-H
   bond (:func:`ch_molecules`, the usual textbook criterion), ``precursors`` those that also hold
   a C-C, C-N or C-O bond (formaldehyde, methanol, methylamine), ``chains`` the groups with more
   than two carbon atoms (what round 6 called organic). Real: formaldehyde and other aldehydes
   are, with hydrogen cyanide and ammonia, what the Strecker route turned into amino acids in
   Miller's experiment; methanol and methylamine are not on that route, and the toy keeps no
   hydrogen cyanide. ``precursors`` is what world7 hands to the cells level.
7. **Stage words, water and solids.** The stages are ``hot`` (at or above HOT_K), ``atomic``,
   ``forming``, ``molecular``; ``plasma`` (a 4000 K gas is neutral atoms, not ionised) and
   ``frozen`` (which labelled 650 K steam) are gone. ``water`` is what water is at the step's
   temperature and 1 bar (:func:`water_state`: ice below 273 K, liquid to 373 K, vapor above).
   ``solids`` lists the substances already condensed in a nebula near 1e-4 bar from a table
   (:data:`CONDENSE_K`: the temperatures at which each starts to condense in a gas of solar
   make-up, after Lodders 2003, from memory and rounded: corundum 1680 K, silicate (forsterite)
   and iron metal 1350 K, iron sulfide 700 K, water ice 180 K, ammonia ice 130 K, methane ice
   80 K; the last two are the hydrates of ammonia and methane, pure methane ice needs about
   40 K; half of an element is condensed some tens of kelvin lower), which agrees with level
   3's frost line (169 K). Toy: the table is read and is not what the pool does. The pool's
   own network (``rock``) forms as soon as its bonds hold, at 2180-2225 K in the 7 rocky runs
   of seeds 1-40 and at 1563-1652 K in the 7 carbon runs, while the table has no solid above
   1680 K and silicate only below 1350 K: one temperature per bond knows nothing of the entropy
   a gas loses when it condenses. Where the two disagree, ``solids`` is the real-world reading
   and ``rock`` the toy's. In such a nebula (near 1e-4 bar; the boundaries move with pressure)
   real carbon is mostly CO above about 650 K and real nitrogen mostly N2 above about 330 K
   (after Lewis and Prinn 1980); the toy shows the cold end state at every temperature and ends
   without CO.
8. **Sulfur opens valences toward oxygen.** A sulfur atom with no free valence that meets an
   oxygen atom with a free one opens two more valences (up to 6) and bonds it at once; two
   opened valences that are free again close; and with anything but oxygen sulfur never shares
   more than its 2. Real: sulfur is 2 toward hydrogen and metals and 4 or 6 toward oxygen.
   Measured: in pure S and O the rule gives SO3; in the 8 ocean runs of seeds 1-40, 11-27
   percent of the sulfur atoms end with 4 bonds and 0-12 percent with 6, but free SO2 and H2SO4
   are rare (0-2 and 0-1 molecules a run) and SO3 does not occur, because nothing protects an
   S=O bond from hydrogen (no barriers), so they are not counted keys. Iron stays 2 (FeO in a
   mantle, FeS; Fe2O3 needs free oxygen).
9. **Mixes that say what they are** (:data:`MIXES7`): ``cosmic`` has the sun's ratios times a
   declared ``metal_boost`` 8, ``rocky`` follows the silicate earth (more magnesium than
   silicon), ``ocean`` and ``carbon`` keep round 6's fractions (``salt_boost`` 7 is printed),
   ``reducing`` is new. A mix record prints the fractions. A seed's own ocean or reducing run
   ends at 200 to 500 K, so its water is ice, liquid or vapor.
10. **Records that add up.** Every counted key is in a state record and asked; ``other`` closes
    the line (molecules minus the named ones); there is a ``params`` record and ``param``
    questions; ``valence`` is called ``bonds`` (round 5 uses valence for oxidation states) and
    elements and molecules carry their round-5 names. ``water_share`` replaces ``water_fraction``
    and is always a float, ``water_atoms`` is the share of the atoms in water, and
    ``first_water_step`` must be exact.

Still toy under rules 7, and not to be read as chemistry: no pressure, no entropy beyond the one
temperature per bond, no reaction barriers, no charge, no shape; the pool is well mixed. Because
there are no barriers, hydrogen takes everything it can: HCN and CO do not survive, carbon in
hydrogen-rich gas ends as methane. An ocean run without sparks keeps 14 to 29 O2 molecules
(the mix has less hydrogen than its oxygen can hold), and sparks split water into H2, O2 and
H2O2 that do not all find each other again (the products of the radiolysis of water, but the
amounts are toy). In the cosmic mix the rock-forming atoms are too few (16 in 20000) to build
grains. In rocky runs nearly all iron is oxidised (95-99 % of its valences, ``fe_metal`` 0 or 1):
the toy has no core, and with less oxygen silicon is left without it as readily as iron. A mean
bond enthalpy hides that the first bond of a molecule is often stronger than the mean (H2N-H is
450 kJ/mol against the mean 391): here an NH2 group cannot take a hydrogen atom from H2 (436),
so a cosmic run without a spark keeps NH2 beside H2 where real gas would make ammonia. A run that
starts below HOT_K (the seed's own start is 3000 to 6000 K) starts as atoms colder than their
bonds, and every bond that holds forms in the first step.

Measured under rules 7 (tests/test_cosmos_chem_rules7.py; seeds 1-40 with their own parameters:
10 cosmic, 7 rocky, 8 ocean, 7 carbon, 8 reducing runs):

- at HOT_K and above the gas is atomic on every step; the stages come in the order hot, atomic,
  forming, molecular in all 40 runs;
- cosmic: H2 is 98.6-98.7 % of the molecules; water holds 81-94 % of the oxygen, methane 86-98 %
  of the carbon, ammonia 73-100 % of the nitrogen in the 9 runs with a spark (1 of 9 atoms in
  the run without: hydrogen pairs into H2 at 2092 K, before N-H holds at 1876 K, and 4 of the 9
  end as NH2); free_atoms 0.0886-0.0888 (helium and neon);
- rocky: one grain holds 97.7-99.1 % of all atoms; 96-99 % of the silicon valences, 99.9 % of
  the magnesium valences and 95-99 % of the iron valences are bonds to oxygen; 69-175 loose
  molecules beside it, among them 1-31 O2 and 4-12 H2;
- ocean: water is the most common molecule (water_share 0.79-0.87) and holds 75-83 % of all
  atoms, NaCl the second (61-73 % of the sodium); carbon ends as CO2, carbonic acid and a tar
  (the largest group has 33-337 atoms);
- reducing: water first, then 592-1383 H2, 190-316 NH3, 62-204 CH4, 65-134 formaldehyde;
- precursors: ocean 9 and 17 in the two runs without sparks, 37-76 in the six with 2 to 20;
  reducing 91-202. With fixed parameters (4600 to 290 K, 30 steps, 20000 atoms, seeds 1-5) and
  0, 5, 10, 20 sparks: ocean 14-21, 55-69, 70-90, 72-87 (sparks also destroy); reducing
  117-144, 155-182, 165-181, 163-178; the ocean's water falls from 5475-5492 to 4982-5035
  molecules;
- carbon: methane is the most common molecule (1476-1755), then ammonia or water; 1780-2126
  organic groups, 136-207 chains, and a grain of 1362-3512 atoms (soot, silicon carbide and
  silica);
- a seed's own run takes 0.3-1.2 s (best of two, mean 0.6 s, measured with other jobs running on
  the machine; the slowest is a carbon run with 16 sparks); the largest (20 sparks, 35 steps,
  22000 atoms) 1.1-1.5 s. A round-6 run takes about 0.15 s.

Gate (rules 7): :func:`conserved7` replays the rollout and checks at every step that every
element's atom count is the same, that no atom holds more bonds than its valence (sulfur's opened
ones included), that noble gases never bond, that no bond has an order its pair does not have or
cannot hold at the step's temperature, that every exchange gave off heat and each exchange
round changed the pool's bond enthalpy by exactly the heat of its moves less the multiple bonds
that fell apart, that the counted molecules and bonds fit into the atoms and valences of the mix
(:func:`budget7`), that stage, water and solids follow from the numbers in the same record, and
that the sparks are counted in order. :func:`check7` replays a seed under rules 7: a count may be
off by one or 5 %, other numbers by 5 %, ordinals (``spark``, ``first_water_step``), parameters
and words must be exact.

Lessons (:data:`LESSONS`, ``chem predict <rule> <input> <value> ...``): thirteen rules that
carry their inputs, each the function the rules-7 simulation or its gate calls:
schedule_temperature, unit_enthalpy, stable_below_k, bond_order, free_valence, reaction_heat,
move_allowed, stage, water_state, condensed, expected_hits, water_share, water_atoms.

The same seed gives the same rules-7 rollout on any machine: whole numbers everywhere, shuffles
by a stable sort of raw PCG64 draws, pairs judged in a fixed order, groups numbered by their
lowest atom, and the only floats are single divisions rounded to three digits. The numpy
prefilter of the exchange round is exact (``Pool7(..., prefilter=False)`` judges every pair and
gives the same pool; the tests compare).
"""

from __future__ import annotations

import heapq
import math
import random
import re
from collections.abc import Iterator
from functools import lru_cache

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from haishool.cosmos import Rollout, close, query_lines, seeds, state_line, summary_lines
from haishool.evo import Input, LessonGate, Rule
from haishool.truth import Line, Verdict, dedupe, is_finite, num, parse_num
from haishool.truth.formula import dense, parse

SIM = "chem"

#: element -> bonds it can hold; 0 for the noble gases, which never bond
VALENCE: dict[str, int] = {"h": 1, "he": 0, "o": 2, "c": 4, "n": 3, "ne": 0, "mg": 2, "si": 4, "s": 2,
                           "fe": 2, "al": 3, "ca": 2, "na": 1, "k": 1, "cl": 1}
ELEMENTS = tuple(VALENCE)
_IX = {e: i for i, e in enumerate(ELEMENTS)}
METALS = frozenset({"mg", "fe", "al", "ca", "na", "k"})

#: (element, element), sorted -> temperature in kelvin below which the bond holds: the table
#: this level was specified with
GIVEN_BONDS: dict[tuple[str, str], int] = {
    ("h", "h"): 2000, ("h", "o"): 3000, ("c", "h"): 2500, ("c", "o"): 3500, ("n", "n"): 4000,
    ("o", "si"): 3800, ("cl", "na"): 1500, ("mg", "o"): 3000, ("fe", "o"): 2500, ("c", "c"): 2800,
    ("h", "n"): 2200, ("h", "s"): 1800, ("o", "s"): 2600, ("o", "o"): 1200,
}
#: pairs the mixes also bring together, with values chosen here so that the usual compounds win:
#: Al and Ca oxides form like those of Si and Mg, the alkali metals go to chlorine before
#: hydrogen or oxygen (salt before acid, lye or hydride), the weak Cl-Cl and Cl-O bonds come last
ADDED_BONDS: dict[tuple[str, str], int] = {
    ("al", "o"): 3200, ("ca", "o"): 3000, ("cl", "k"): 1500, ("cl", "mg"): 1400, ("ca", "cl"): 1400,
    ("cl", "h"): 1300, ("cl", "cl"): 1300, ("cl", "o"): 1300, ("na", "o"): 1300, ("k", "o"): 1300,
    ("h", "na"): 1000, ("h", "k"): 1000,
}
BONDS: dict[tuple[str, str], int] = {**GIVEN_BONDS, **ADDED_BONDS}
#: any other pair of bonding elements
DEFAULT_K = 2000
#: any other pair of two metals (clusters only when cold)
METAL_METAL_K = 1000
#: where each table comes from, for a checker
NOTES: dict[str, str] = {
    "valence": "the usual school valences; iron is taken as 2 although it is also 3",
    "given_bonds": "toy stability temperatures from the specification of this level, ordered roughly "
                   "like bond strengths; not measured values",
    "added_bonds": "chosen in this module, not measured and not from a source: only their order matters. "
                   "real h-cl is a strong bond; it is put below na-cl so that the ocean mix ends in salt",
    "metal_metal": "chosen in this module: two metals only cluster below 1000 k",
    "mixes": "toy fractions chosen in this module; the cosmic traces follow the solar order from memory "
             "(o, c, ne, n, then mg, si, s, fe) with helium at 9 percent; rocky, ocean and carbon are "
             "tuned so that oxygen roughly matches what silicon, the metals and hydrogen can hold",
}
#: at or above this nothing bonds (the strongest bond's stability); below FROZEN_K every bond holds
PLASMA_K = max(BONDS.values())
FROZEN_K = min(min(BONDS.values()), METAL_METAL_K)
#: meeting rounds per step, and the largest bond order
ROUNDS = 3
MAX_BOND = 3

#: mix -> element -> fraction of atoms, in falling order; the first element takes the rounding rest
MIXES: dict[str, dict[str, float]] = {
    "cosmic": {"h": 0.90, "he": 0.09, "o": 0.0035, "c": 0.0022, "ne": 0.0012, "n": 0.0010,
               "mg": 0.0005, "si": 0.0005, "s": 0.0004, "fe": 0.0004},
    "rocky": {"o": 0.555, "si": 0.14, "mg": 0.10, "fe": 0.06, "h": 0.03, "al": 0.02, "ca": 0.02,
              "c": 0.02, "na": 0.015, "s": 0.015, "n": 0.01, "cl": 0.01, "k": 0.005},
    "ocean": {"h": 0.59, "o": 0.33, "na": 0.02, "cl": 0.02, "c": 0.012, "s": 0.01, "n": 0.01, "mg": 0.008},
    "carbon": {"h": 0.70, "c": 0.16, "o": 0.06, "n": 0.04, "si": 0.02, "s": 0.02},
}

#: metric name -> formula of the molecules it counts (exact composition)
MOLECULES: dict[str, str] = {"h2": "H2", "h2o": "H2O", "ch4": "CH4", "nh3": "NH3", "co": "CO", "co2": "CO2",
                             "o2": "O2", "n2": "N2", "nacl": "NaCl", "sio2": "SiO2", "mgo": "MgO", "feo": "FeO"}

KEYS: dict[str, str] = {
    "mix": "the element mix: cosmic, rocky, ocean or carbon",
    "temperature": "gas temperature in kelvin",
    "stage": "plasma (4000 k or more, nothing bonds), atomic (9 in 10 atoms free), forming, "
             "molecular (under half free), frozen (under 1000 k, every bond holds)",
    "free_atoms": "fraction of all atoms with no bond, noble gases included",
    "molecules": "number of bonded groups of two or more atoms",
    **{k: f"molecules with exactly the composition {dense(f)}" for k, f in MOLECULES.items()},
    "organic": "molecules with more than two carbon atoms",
    "biggest": "atoms in the largest molecule (1 while the gas is atomic)",
    "water_fraction": "h2o molecules divided by all molecules at the end",
    "first_water_step": "first step with an h2o molecule, or never",
}
STATE_KEYS = ["mix", "temperature", "stage", "free_atoms", "molecules", "h2", "h2o", "ch4", "nh3", "nacl",
              "sio2", "organic", "biggest"]
QUERY_KEYS = ["temperature", "stage", "free_atoms", "molecules", *MOLECULES, "organic", "biggest"]
COUNT_KEYS = frozenset(["molecules", *MOLECULES, "organic", "biggest", "first_water_step"])
STAGES = ("plasma", "atomic", "forming", "molecular", "frozen")


def stability(a: str, b: str) -> int:
    """Kelvin below which a bond between elements ``a`` and ``b`` holds; 0 when one is noble."""
    if VALENCE[a] == 0 or VALENCE[b] == 0:
        return 0
    key = (a, b) if a <= b else (b, a)
    if key in BONDS:
        return BONDS[key]
    return METAL_METAL_K if a in METALS and b in METALS else DEFAULT_K


_VAL = np.array([VALENCE[e] for e in ELEMENTS], dtype=np.int64)
_STAB = np.array([[stability(a, b) for b in ELEMENTS] for a in ELEMENTS], dtype=np.int64)
_TARGETS = {k: np.array([parse(f).get(e.capitalize(), 0) for e in ELEMENTS], dtype=np.int64)
            for k, f in MOLECULES.items()}


def element_counts(mix: str, atoms: int, rules: int = 6) -> dict[str, int]:
    """Atoms of each element in a pool of ``atoms`` atoms: every share rounded, the first element
    takes the rest. ``rules=7`` reads the mixes of rules 7 (:data:`MIXES7`)."""
    counts = {e: int(round(f * atoms)) for e, f in (MIXES7 if rules == 7 else MIXES)[mix].items()}
    first = next(iter(counts))
    counts[first] += atoms - sum(counts.values())
    return counts


def stage(temperature: int, free_atoms: float) -> str:
    if temperature >= PLASMA_K:
        return "plasma"
    if temperature < FROZEN_K:
        return "frozen"
    return "atomic" if free_atoms >= 0.9 else "forming" if free_atoms >= 0.5 else "molecular"


def sig3(x: float) -> float:
    return float(f"{x:.3g}")


def temperature_at(t_start: int, t_end: int, t: int, n_steps: int) -> int:
    """Kelvin at step ``t`` of ``n_steps``: ``t_start * (t_end / t_start) ** (t / n_steps)``,
    rounded to the nearest whole number. The float power only proposes; whole numbers decide
    (``2 k - 1 <= 2 x < 2 k + 1``, both sides to the power ``n_steps``), so the result does not
    depend on a platform's ``pow``. An exact half cannot occur: a root of a whole number is
    whole or irrational."""
    k = int(round(t_start * (t_end / t_start) ** (t / n_steps)))
    target = 2 ** n_steps * t_start ** (n_steps - t) * t_end ** t
    while (2 * k + 1) ** n_steps <= target:
        k += 1
    while k > 0 and (2 * k - 1) ** n_steps > target:
        k -= 1
    return k


class Pool:
    """The atoms (an element index each), their free valences and their bonds (atom, atom, order)."""

    def __init__(self, counts: dict[str, int], rng: np.random.Generator) -> None:
        self.el = np.repeat(np.array([_IX[e] for e in counts], dtype=np.int64), list(counts.values()))
        self.valence = _VAL[self.el]
        self.free = self.valence.copy()
        self.bond_a = np.zeros(0, dtype=np.int64)
        self.bond_b = np.zeros(0, dtype=np.int64)
        self.bond_m = np.zeros(0, dtype=np.int64)
        self.rng = rng
        self.formed = self.broken = 0

    def step(self, temperature: int) -> None:
        self.broken = self.break_bonds(temperature)
        self.formed = sum(self.meet(temperature) for _ in range(ROUNDS))

    def break_bonds(self, temperature: int) -> int:
        """Every bond hotter than its stability temperature breaks; returns how many."""
        gone = _STAB[self.el[self.bond_a], self.el[self.bond_b]] < temperature
        if gone.any():
            np.add.at(self.free, self.bond_a[gone], self.bond_m[gone])
            np.add.at(self.free, self.bond_b[gone], self.bond_m[gone])
            keep = ~gone
            self.bond_a, self.bond_b, self.bond_m = self.bond_a[keep], self.bond_b[keep], self.bond_m[keep]
        return int(gone.sum())

    def meet(self, temperature: int) -> int:
        """One round: atoms with a free valence pair off at random and bond where stable."""
        idx = np.flatnonzero(self.free > 0)
        if len(idx) < 2:
            return 0
        # shuffled by the raw bit stream, which numpy keeps stable across versions (permutation() it does not)
        idx = idx[np.argsort(self.rng.bit_generator.random_raw(len(idx)), kind="stable")]
        n = len(idx) - len(idx) % 2
        a, b = idx[0:n:2], idx[1:n:2]
        ok = _STAB[self.el[a], self.el[b]] > temperature
        # two atoms that already share a bond keep it as it is: a triple bond never becomes a fourth
        again = (self.free[self.bond_a] > 0) & (self.free[self.bond_b] > 0)
        if again.any():
            ok &= ~np.isin(self.pair(a, b), self.pair(self.bond_a[again], self.bond_b[again]))
        a, b = a[ok], b[ok]
        m = np.minimum(np.minimum(self.free[a], self.free[b]), MAX_BOND)
        self.free[a] -= m  # every atom is in at most one pair per round, so plain indexing is safe
        self.free[b] -= m
        self.bond_a = np.concatenate([self.bond_a, a])
        self.bond_b = np.concatenate([self.bond_b, b])
        self.bond_m = np.concatenate([self.bond_m, m])
        return len(a)

    def pair(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """One number per unordered pair of atoms."""
        return np.minimum(a, b) * len(self.el) + np.maximum(a, b)

    def census(self) -> tuple[np.ndarray, np.ndarray]:
        """Composition of every connected group (a row over ELEMENTS) and its size in atoms."""
        n = len(self.el)
        graph = coo_matrix((np.ones(len(self.bond_a)), (self.bond_a, self.bond_b)), shape=(n, n))
        k, labels = connected_components(graph, directed=False)
        comp = np.bincount(labels * len(ELEMENTS) + self.el, minlength=k * len(ELEMENTS)).reshape(k, len(ELEMENTS))
        return comp, comp.sum(1)

    def audit(self, totals: np.ndarray) -> list[str]:
        """The laws: atoms conserved per element, valence never exceeded, nobles never bonded."""
        problems = []
        n = len(self.el)
        held = (np.bincount(self.bond_a, self.bond_m, minlength=n) + np.bincount(self.bond_b, self.bond_m, minlength=n))
        if (self.free < 0).any() or not np.array_equal(held + self.free, self.valence):
            problems.append("an atom holds more bonds than its valence")
        if (self.bond_m < 1).any() or (self.bond_m > MAX_BOND).any() or (self.bond_a == self.bond_b).any():
            problems.append("a bond of impossible order or an atom bonded to itself")
        if len(np.unique(self.pair(self.bond_a, self.bond_b))) != len(self.bond_a):
            problems.append("two atoms are bonded twice")
        if (self.valence[self.bond_a] == 0).any() or (self.valence[self.bond_b] == 0).any():
            problems.append("a noble gas is bonded")
        comp, size = self.census()
        if not np.array_equal(comp[size == 1].sum(0) + comp[size >= 2].sum(0), totals):
            problems.append("an element's atom count changed")
        return problems


def metrics(pool: Pool, temperature: int) -> dict[str, float | int | str]:
    comp, size = pool.census()
    free_atoms = sig3(int((size == 1).sum()) / len(pool.el))  # the stage follows the number the line shows
    out: dict[str, float | int | str] = {"temperature": temperature, "stage": stage(temperature, free_atoms),
                                         "free_atoms": free_atoms, "molecules": int((size >= 2).sum())}
    for name, target in _TARGETS.items():
        out[name] = int((comp == target).all(1).sum())
    out["organic"] = int((comp[:, _IX["c"]] > 2).sum())
    out["biggest"] = int(size.max())
    return out


def molecule_counts(pool: Pool) -> dict[str, int]:
    """Dense formula (``h 2 o 1``) -> how many molecules of that composition, most common first."""
    comp, size = pool.census()
    rows, counts = np.unique(comp[size >= 2], axis=0, return_counts=True)
    order = np.argsort(-counts, kind="stable")
    return {dense("".join(f"{e.capitalize()}{n}" for e, n in zip(ELEMENTS, rows[i]) if n)): int(counts[i]) for i in order}


def random_params(rng: random.Random, rules: int = 6) -> dict[str, int | str]:
    """The parameters a seed stands for: ``check`` replays a seed with exactly these.

    ``rules=7`` makes the same five draws first and then three more: ``sparks`` (0 to
    :data:`MAX_SPARKS`); one chance in five that the mix becomes ``reducing``; and for the two
    watery mixes (``ocean``, ``reducing``) an end temperature of 200 to 500 K, so that the
    water of a seed's own run is ice, liquid or vapor and never steam at 1400 K."""
    params: dict[str, int | str] = {
            "mix": rng.choice(tuple(MIXES)), "t_start": rng.randrange(3000, 6001, 100),
            "t_end": rng.randrange(100, 1501, 50), "steps": rng.randint(25, 35),
            "atoms": rng.randrange(18000, 22001, 1000)}
    if _rules(rules) == 6:
        return params
    params["sparks"] = rng.randint(0, MAX_SPARKS)
    if rng.randrange(len(MIXES7)) == 0:
        params["mix"] = "reducing"
    if params["mix"] in WATERY_MIXES:
        params["t_end"] = rng.randrange(200, 501, 10)
    return params


def _rules(rules: int) -> int:
    if isinstance(rules, bool) or rules not in (6, 7):
        raise ValueError(f"rules must be 6 (round 6, the default) or 7, not {rules!r}")
    return int(rules)


def is_rules7(r: Rollout) -> bool:
    """True for a rollout made with ``run(seed, rules=7)``."""
    return r.params.get("rules") == 7


def bad_params(params: dict) -> str:
    """What is wrong with a run's parameters; empty when they can be simulated."""
    if "rules" in params:
        return bad_params7(params)
    if params.get("mix") not in MIXES:
        return f"unknown mix {params.get('mix')!r}"
    for key, least in (("t_start", 1), ("t_end", 1), ("steps", 1), ("atoms", 1)):
        value = params.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < least:
            return f"{key} must be a whole number from {least} on, not {value!r}"
    if min(element_counts(str(params["mix"]), int(params["atoms"])).values()) < 0:
        return "too few atoms for this mix"
    return ""


def budget(step: dict[str, float | int | str], counts: dict[str, int]) -> list[str]:
    """Conservation read off one step's numbers alone: the counted molecules may not hold more
    atoms of an element than the mix has, nor more atoms than are bonded at all."""
    problems = []
    n = {k: int(step.get(k, 0)) for k in (*MOLECULES, "organic", "molecules")}
    for e, have in counts.items():
        used = sum(n[k] * int(_TARGETS[k][_IX[e]]) for k in MOLECULES) + (3 * n["organic"] if e == "c" else 0)
        if used > have:
            problems.append(f"more {e} in the counted molecules than the mix holds")
    atoms = sum(counts.values())
    other = n["molecules"] - sum(n[k] for k in (*MOLECULES, "organic"))
    bonded = sum(n[k] * int(_TARGETS[k].sum()) for k in MOLECULES) + 3 * n["organic"] + 2 * max(other, 0)
    if bonded > atoms * (1 - float(step.get("free_atoms", 0))) + 0.005 * atoms + 1:  # free_atoms has 3 digits
        problems.append("more atoms in molecules than are bonded")
    return problems


def cool(seed: int, params: dict[str, int | str]) -> Iterator[tuple[int, int, Pool]]:
    """(step, temperature in kelvin, the pool after that step), from the pristine pool at the
    start temperature down to the end temperature, the temperature falling by a constant factor."""
    if "rules" in params:  # rules 7: a Pool7, and one more step per spark
        yield from cool7(seed, params)
        return
    mix, t_start, t_end = str(params["mix"]), int(params["t_start"]), int(params["t_end"])
    n_steps = int(params["steps"])
    # PCG64 by name: default_rng() may pick another generator one day, and a seed must stay its rollout
    pool = Pool(element_counts(mix, int(params["atoms"])), np.random.Generator(np.random.PCG64(seed)))
    for t in range(n_steps + 1):
        temperature = temperature_at(t_start, t_end, t, n_steps)
        if t:
            pool.step(temperature)
        yield t, temperature, pool


def simulate(seed: int, params: dict[str, int | str], audit: bool = False) -> tuple[Rollout, list[str]]:
    """The rollout and, with ``audit``, every broken law found along the way."""
    if "rules" in params:
        return simulate7(seed, params, audit)
    mix = str(params["mix"])
    counts = element_counts(mix, int(params["atoms"]))
    totals = np.array([counts.get(e, 0) for e in ELEMENTS])
    steps: list[dict[str, float | int | str]] = []
    problems: list[str] = []
    for t, temperature, pool in cool(seed, params):
        steps.append({"mix": mix, **metrics(pool, temperature)})
        if audit:
            problems += [f"step {t}: {p}" for p in pool.audit(totals)]
    last = steps[-1]
    watery = [t for t, s in enumerate(steps) if s["h2o"]]
    summary: dict[str, float | int | str] = {"mix": mix}
    summary.update({k: last[k] for k in ("h2", "h2o", "ch4", "nh3", "nacl", "sio2", "organic", "molecules", "biggest")})
    summary["water_fraction"] = sig3(last["h2o"] / last["molecules"]) if last["molecules"] else 0
    summary["first_water_step"] = watery[0] if watery else "never"
    return Rollout(SIM, seed, dict(params), steps, summary), problems


_N = r"(0|[1-9](?: \d)*)"  # digit tokens, no leading zero: one spelling per number
_SIM_Q = re.compile(rf"^chem seed {_N} (?:step {_N} (next )?|final )([a-z][a-z0-9_]*)$")
_VALENCE_Q = re.compile(r"^chem valence ([a-z]{1,2})$")
_BOND_Q = re.compile(r"^chem bond ([a-z_]{1,11}) (?:([a-z]{1,2}) )?stable_below_k$")
_MOLECULE_Q = re.compile(r"^chem molecule ([a-z0-9]+) (formula|atoms)$")
_MIX_Q = re.compile(r"^chem mix ([a-z]+) (main|elements)$")


class Chem:
    sim = SIM
    KEYS = KEYS

    def run(self, seed: int, rules: int = 6, **params) -> Rollout:
        """``run(seed)`` is the seed's own rollout; named parameters (``mix``, ``t_start``,
        ``t_end``, ``steps``, ``atoms``) override the seed's. ``rules=7`` runs the corrected
        chemistry (section "Rules 7" of the module docstring): the seed's parameters are those of
        ``random_params(rng, rules=7)`` (``sparks`` is one more), the metrics are :data:`KEYS7`
        and the rollout's parameters carry ``"rules": 7``."""
        if _rules(rules) == 7:
            params = {**random_params(random.Random(seed), rules=7), **params, "rules": 7}
        else:
            params = {**random_params(random.Random(seed)), **params}
        problem = bad_params(params)
        if problem:
            raise ValueError(problem)
        return simulate(seed, params)[0]

    def conserved(self, r: Rollout) -> Verdict:
        """The laws of a rollout of either rule set (a ``rules=7`` rollout: :func:`conserved7`)."""
        if r.sim == SIM and is_rules7(r):
            return conserved7(r)
        if r.sim != SIM or bad_params(r.params):
            return Verdict(False, None, "not a chem rollout")
        replay, problems = simulate(r.seed, r.params, audit=True)
        counts = element_counts(str(r.params["mix"]), int(r.params["atoms"]))
        for t, s in enumerate(r.steps):
            if any(isinstance(v, (int, float)) and v < 0 for v in s.values()):
                problems.append(f"step {t}: a negative count")
            named = sum(int(s.get(k, 0)) for k in (*MOLECULES, "organic"))
            if int(s.get("molecules", 0)) < named:
                problems.append(f"step {t}: more named molecules than molecules")
            problems += [f"step {t}: {p}" for p in budget(s, counts)]
        if replay.steps != r.steps or replay.summary != r.summary:
            problems.append("the rollout is not what its seed and parameters give")
        return Verdict(not problems, None, "; ".join(problems[:5]))

    def check(self, prompt: str, answer: str) -> Verdict:
        if _RULES7_Q.match(prompt):  # ``chem seed 5 rules 7 ...`` and ``chem rules 7 ...``
            return check7(prompt, answer)
        m = _SIM_Q.match(prompt)
        if m:
            return self._check_run(int(parse_num(m[1])), m[2], bool(m[3]), m[4], answer)
        m = _VALENCE_Q.match(prompt)
        if m:
            if m[1] not in VALENCE:
                return Verdict(False, None, "unknown element")
            return _judge(num(VALENCE[m[1]]), answer)
        m = _BOND_Q.match(prompt)
        if m:
            if m[2] is None:
                special = {"default": DEFAULT_K, "metal_metal": METAL_METAL_K}
                if m[1] not in special:
                    return Verdict(False, None, "unknown bond")
                return _judge(num(special[m[1]]), answer)
            if m[1] not in VALENCE or m[2] not in VALENCE:
                return Verdict(False, None, "unknown element")
            k = stability(m[1], m[2])
            return _judge("never" if not k else num(k), answer)
        m = _MOLECULE_Q.match(prompt)
        if m:
            if m[1] not in MOLECULES:
                return Verdict(False, None, "unknown molecule")
            formula = MOLECULES[m[1]]
            return _judge(dense(formula) if m[2] == "formula" else num(sum(parse(formula).values())), answer)
        m = _MIX_Q.match(prompt)
        if m:
            if m[1] not in MIXES:
                return Verdict(False, None, "unknown mix")
            elements = list(MIXES[m[1]])
            return _judge(elements[0] if m[2] == "main" else " ".join(elements), answer)
        return Verdict(False, None, "not my question")

    def _check_run(self, seed: int, step: str | None, nxt: bool, key: str, answer: str) -> Verdict:
        r = _rollout(seed)
        if step is None:
            if key not in r.summary:
                return Verdict(False, None, "unknown summary key")
            expected = r.summary[key]
        else:
            t = int(parse_num(step)) + (1 if nxt else 0)
            if t >= len(r.steps):
                return Verdict(False, None, "no such step")
            if key not in r.steps[t]:
                return Verdict(False, None, "unknown key")
            expected = r.steps[t][key]
        text = r.value(expected)
        if isinstance(expected, str):
            return _judge(text, answer)
        got = parse_num(answer)
        if not is_finite(got):  # 1 e 9 9 9 parses as infinity, which is close to anything
            return Verdict(False, text, "not a number")
        if key in COUNT_KEYS and abs(got - expected) <= 1:
            return Verdict(True, text)
        return Verdict(close(got, expected, rel=0.05), text)

    def owns(self, prompt: str) -> bool:
        return any(p.match(prompt) for p in (_SIM_Q, _VALENCE_Q, _BOND_Q, _MOLECULE_Q, _MIX_Q)) or owns7(prompt)

    def rollout(self, seed: int, rules: int = 6) -> Rollout:
        """The rollout of ``seed`` with its seeded parameters, cached (do not change it)."""
        return _rollout7(seed) if _rules(rules) == 7 else _rollout(seed)

    def records(self, rules: int = 6) -> list[Line]:
        """The toy world's tables: valences, bond stabilities, the counted molecules, the mixes.
        ``rules=7``: the tables of rules 7 (:func:`records7`)."""
        if _rules(rules) == 7:
            return records7()
        out = [Line("chem valence. " + " ".join(f"{e} {num(v)}." for e, v in VALENCE.items()), topic=SIM, kind="record")]
        for (a, b), k in BONDS.items():
            out.append(Line(f"chem bond {a} {b}. stable_below_k {num(k)}.", topic=SIM, kind="record"))
        out.append(Line(f"chem bond default. stable_below_k {num(DEFAULT_K)}.", topic=SIM, kind="record"))
        out.append(Line(f"chem bond metal_metal. stable_below_k {num(METAL_METAL_K)}.", topic=SIM, kind="record"))
        for name, formula in MOLECULES.items():
            atoms = sum(parse(formula).values())
            out.append(Line(f"chem molecule {name}. formula {dense(formula)}. atoms {num(atoms)}.", topic=SIM, kind="record"))
        for mix, fractions in MIXES.items():
            out.append(Line(f"chem mix {mix}. main {next(iter(fractions))}. elements {' '.join(fractions)}.",
                            topic=SIM, kind="record"))
        return out

    def table_lines(self, rules: int = 6) -> list[Line]:
        """Every question the tables answer (fixed; no seed involved). ``rules=7``: the questions
        of the rules-7 tables (:func:`table_lines7`)."""
        if _rules(rules) == 7:
            return table_lines7()
        out = [Line(f"chem valence {e}", num(v), SIM, "fact") for e, v in VALENCE.items()]
        for a in ELEMENTS:
            for b in ELEMENTS:
                if a <= b:
                    k = stability(a, b)
                    out.append(Line(f"chem bond {a} {b} stable_below_k", "never" if not k else num(k), SIM, "fact"))
        out.append(Line("chem bond default stable_below_k", num(DEFAULT_K), SIM, "fact"))
        out.append(Line("chem bond metal_metal stable_below_k", num(METAL_METAL_K), SIM, "fact"))
        for name, formula in MOLECULES.items():
            out.append(Line(f"chem molecule {name} formula", dense(formula), SIM, "fact"))
            out.append(Line(f"chem molecule {name} atoms", num(sum(parse(formula).values())), SIM, "fact"))
        for mix, fractions in MIXES.items():
            out.append(Line(f"chem mix {mix} main", next(iter(fractions)), SIM, "fact"))
            out.append(Line(f"chem mix {mix} elements", " ".join(fractions), SIM, "fact"))
        return out


def _judge(expected: str, answer: str) -> Verdict:
    return Verdict(answer.strip() == expected, expected)


@lru_cache(maxsize=256)
def _rollout(seed: int) -> Rollout:
    return simulation().run(seed)


_SIM = Chem()


def simulation() -> Chem:
    return _SIM


def lines(r: Rollout, every: int = 1, rules: int = 6) -> list[Line]:
    """A rollout as training lines: one state line per step, every metric asked (now and next)
    at every ``every``-th step, and the summary. A rollout made with ``rules=7`` gives the
    ``chem seed 5 rules 7 ...`` lines of :func:`lines7` whatever ``rules`` says here; asking a
    round-6 rollout for ``rules=7`` lines is an error (one rollout has one set of rules)."""
    if is_rules7(r):
        return lines7(r, every)
    if _rules(rules) == 7:
        raise ValueError("this rollout was made with the round-6 rules; run(seed, rules=7) makes the other")
    out = [state_line(r, t, STATE_KEYS) for t in range(len(r.steps))]
    for t in range(0, len(r.steps), every):
        out += query_lines(r, t, QUERY_KEYS)
    out += summary_lines(r)
    return dedupe(out)


def generate(rng: random.Random, n: int, every: int = 1, rules: int = 6) -> list[Line]:
    """The table questions plus the lines of ``n`` rollouts on seeds drawn from ``rng``
    (``rules=7``: the rules-7 tables and rollouts)."""
    sim = simulation()
    out = sim.table_lines(_rules(rules))
    for seed in seeds(rng, n):
        out += lines(sim.run(seed, rules=rules), every)
    return dedupe(out)


# ---------------------------------------------------------------------------------------------
# rules 7: the corrected chemistry (see "Rules 7" in the module docstring). Nothing above calls
# into this part unless a rollout is made with ``rules=7``; the tables, the pool and every line
# above stay as round 6 wrote them.
# ---------------------------------------------------------------------------------------------

#: most sparks a seed's own run draws (``run`` takes up to :data:`SPARKS_LIMIT`)
MAX_SPARKS = 20
SPARKS_LIMIT = 200
#: toy: a spark takes apart this many of a million small molecules
SPARK_PPM = 20000
#: toy: a spark does not reach groups of more atoms than this (a grain is not atomised by lightning)
SPARK_MAX_ATOMS = 64
#: toy: a group of this many atoms or more counts as a network solid (``rock``, ``grains``)
ROCK_ATOMS = 100
#: toy: the cosmic mix holds this many times more of every metal than the sun
METAL_BOOST = 8
#: toy: the ocean mix holds about this many times more sodium and chlorine than sea water
SALT_BOOST = 7

#: mix -> element -> fraction of atoms under rules 7, in falling order; the first element takes the rounding rest
MIXES7: dict[str, dict[str, float]] = {
    "cosmic": {"h": 0.90442, "he": 0.0880, "o": 0.0036, "c": 0.0020, "ne": 0.0006, "n": 0.0005,
               "mg": 0.0003, "si": 0.00024, "fe": 0.00024, "s": 0.0001},
    "rocky": {"o": 0.56, "mg": 0.18, "si": 0.145, "fe": 0.045, "al": 0.02, "ca": 0.014, "h": 0.012,
              "na": 0.006, "c": 0.006, "s": 0.005, "n": 0.003, "cl": 0.003, "k": 0.001},
    "ocean": dict(MIXES["ocean"]),
    "carbon": dict(MIXES["carbon"]),
    "reducing": {"h": 0.66, "o": 0.26, "c": 0.02, "n": 0.02, "na": 0.015, "cl": 0.015, "s": 0.005, "mg": 0.005},
}
#: the mixes that stand for a planet's surface water; their own runs end at 200 to 500 K
WATERY_MIXES = ("ocean", "reducing")
#: mix -> the declared toy factor its record prints
MIX_BOOSTS7: dict[str, tuple[str, int]] = {"cosmic": ("metal_boost", METAL_BOOST), "ocean": ("salt_boost", SALT_BOOST)}

#: standard enthalpy of formation of the gaseous atom, kJ/mol at 298 K
ATOM_KJ: dict[str, float] = {"h": 218.0, "o": 249.2, "c": 716.7, "n": 472.7, "s": 277.2, "si": 450.0, "cl": 121.3,
                             "na": 107.3, "k": 89.0, "mg": 147.1, "ca": 177.8, "fe": 416.3, "al": 330.0}
#: compound -> (formula, standard enthalpy of formation in kJ/mol, the pair of elements whose bond
#: it isolates, the order of that bond, the valence units bonded in one formula unit). Gases for
#: the molecules, the solid for the salts, oxides, sulfides and hydrides of the metals and for
#: silica, silicon carbide and silicon itself.
FORMATION_KJ: dict[str, tuple[str, float, tuple[str, str], int, int]] = {
    "h2": ("H2", 0.0, ("h", "h"), 1, 1), "o2": ("O2", 0.0, ("o", "o"), 2, 2), "n2": ("N2", 0.0, ("n", "n"), 3, 3),
    "cl2": ("Cl2", 0.0, ("cl", "cl"), 1, 1), "si": ("Si", 0.0, ("si", "si"), 1, 2),
    "h2o": ("H2O", -241.8, ("h", "o"), 1, 2), "ch4": ("CH4", -74.6, ("c", "h"), 1, 4),
    "nh3": ("NH3", -45.9, ("h", "n"), 1, 3), "hcl": ("HCl", -92.3, ("cl", "h"), 1, 1),
    "h2s": ("H2S", -20.6, ("h", "s"), 1, 2), "sih4": ("SiH4", 34.3, ("h", "si"), 1, 4),
    "co2": ("CO2", -393.5, ("c", "o"), 2, 4), "sio2": ("SiO2", -910.9, ("o", "si"), 1, 4),
    "mgo": ("MgO", -601.6, ("mg", "o"), 1, 2), "cao": ("CaO", -634.9, ("ca", "o"), 1, 2),
    "feo": ("FeO", -272.0, ("fe", "o"), 1, 2), "al2o3": ("Al2O3", -1675.7, ("al", "o"), 1, 6),
    "na2o": ("Na2O", -414.2, ("na", "o"), 1, 2), "k2o": ("K2O", -361.5, ("k", "o"), 1, 2),
    "nacl": ("NaCl", -411.2, ("cl", "na"), 1, 1), "kcl": ("KCl", -436.5, ("cl", "k"), 1, 1),
    "mgcl2": ("MgCl2", -641.3, ("cl", "mg"), 1, 2), "cacl2": ("CaCl2", -795.4, ("ca", "cl"), 1, 2),
    "fecl2": ("FeCl2", -341.8, ("cl", "fe"), 1, 2), "alcl3": ("AlCl3", -704.2, ("al", "cl"), 1, 3),
    "so2": ("SO2", -296.8, ("o", "s"), 2, 4), "nah": ("NaH", -56.3, ("h", "na"), 1, 1),
    "kh": ("KH", -57.7, ("h", "k"), 1, 1), "mgh2": ("MgH2", -75.3, ("h", "mg"), 1, 2),
    "cah2": ("CaH2", -181.5, ("ca", "h"), 1, 2), "alh3": ("AlH3", -46.0, ("al", "h"), 1, 3),
    "fes": ("FeS", -100.0, ("fe", "s"), 1, 2), "mgs": ("MgS", -346.0, ("mg", "s"), 1, 2),
    "cas": ("CaS", -482.0, ("ca", "s"), 1, 2), "na2s": ("Na2S", -364.8, ("na", "s"), 1, 2),
    "k2s": ("K2S", -380.7, ("k", "s"), 1, 2), "al2s3": ("Al2S3", -724.0, ("al", "s"), 1, 6),
    "sic": ("SiC", -65.3, ("c", "si"), 1, 4), "cs2": ("CS2", 117.0, ("c", "s"), 2, 4),
    "ccl4": ("CCl4", -95.7, ("c", "cl"), 1, 4), "sicl4": ("SiCl4", -657.0, ("cl", "si"), 1, 4),
}
#: (element, element, order) -> mean bond enthalpy in kJ/mol, where no single compound isolates the bond
MEAN_KJ: dict[tuple[str, str, int], int] = {
    ("c", "c", 1): 346, ("c", "c", 2): 614, ("c", "c", 3): 839, ("c", "n", 1): 305, ("c", "n", 2): 615,
    ("c", "n", 3): 855, ("c", "o", 1): 358, ("n", "n", 1): 163, ("n", "n", 2): 418, ("o", "o", 1): 146,
    ("n", "o", 1): 201, ("n", "o", 2): 607, ("o", "s", 1): 265, ("c", "s", 1): 272, ("s", "s", 1): 266,
    ("s", "s", 2): 425, ("fe", "h", 1): 148,
}
#: pairs that can share more than one valence; every other pair bonds singly (the double-bond rule)
HIGHEST_ORDER: dict[tuple[str, str], int] = {("c", "c"): 3, ("c", "n"): 3, ("n", "n"): 3, ("c", "o"): 2, ("o", "o"): 2,
                                             ("n", "o"): 2, ("o", "s"): 2, ("c", "s"): 2, ("s", "s"): 2}
#: bond enthalpy of any pair of bonding elements that no entry covers, kJ/mol
DEFAULT_KJ = 200
#: a bond holds below this many tenths of a kelvin per kJ/mol of its enthalpy (4.8 K per kJ/mol)
KELVIN_TENTHS_PER_KJ = 48
#: element -> valences it can open beyond :data:`VALENCE`, two at a time, and only toward oxygen
EXTRA_VALENCE: dict[str, int] = {"s": 4}
#: water at 1 bar: ice below the first, vapor above the second (the same two numbers as
#: ``haishool.cosmos.world.CONSTANTS`` ``water_min_k`` and ``water_max_k``; the tests compare them)
WATER_MIN_K, WATER_MAX_K = 273, 373
WATER_STATES = ("ice", "liquid", "vapor")
#: substance -> kelvin below which it condenses in a gas of solar make-up near 1e-4 bar (where it
#: starts to; ``ammonia_ice`` and ``methane_ice`` are the hydrates of ammonia and methane)
CONDENSE_K: dict[str, int] = {"corundum": 1680, "silicate": 1350, "iron": 1350, "iron_sulfide": 700,
                              "water_ice": 180, "ammonia_ice": 130, "methane_ice": 80}
STAGES7 = ("hot", "atomic", "forming", "molecular")

#: where the rules-7 tables come from, for a checker
NOTES7: dict[str, str] = {
    "atom_kj": "standard enthalpies of formation of the gaseous atoms at 298 k as printed in the crc handbook and the "
               "nist-janaf tables, typed from memory and not checked against a handbook in this session",
    "formation_kj": "standard enthalpies of formation at 298 k (crc handbook, nist-janaf), typed from memory and not "
                    "checked against a handbook in this session; a value a few kj off moves a stability "
                    "temperature by a few kelvin per kj. for a solid the unit enthalpy is the atomisation of the "
                    "lattice spread over the valences, which is more than the bond of the gaseous molecule "
                    "(na-cl 640 against 412 kj/mol)",
    "mean_kj": "mean bond enthalpies as printed in general chemistry textbooks, from memory; c-n triple is "
               "hydrogen cyanide less one c-h bond (855; the usual textbook mean, from nitriles, is about 891); fe-h is the diatomic molecule",
    "metal_metal": "two metals: the atomisation enthalpy of each divided by its valence, added (a lattice in which "
                   "every valence is one bond); a toy reading of a metal, real metal dimers are far weaker",
    "default": "toy: 200 kj/mol for any pair no entry covers: weaker than every bond the table takes from a "
               "compound, about as strong as the weakest single bonds (n-o 201, n-n 163, o-o 146) and the bonds "
               "between two light metals; the records print it as bond default and never as a fact about a pair",
    "kelvin_per_kj": "a bond holds below 4.8 k per kj/mol of its enthalpy: the rule of thumb that a thin gas is "
                     "half dissociated where the enthalpy is about 25 r t. h2 gives 2092 k, which is right near "
                     "1e-5 bar; at 1e-4 bar h2 is half dissociated near 2300 k, about 22 r t (janaf equilibrium "
                     "constants, from memory). one temperature per bond is still toy: real dissociation is "
                     "gradual, depends on pressure and has another factor for every bond",
    "highest_order": "the double-bond rule: stable multiple bonds exist among c, n and o and for s=o, c=s and s=s; "
                     "si-o, mg-o and fe-o are single bonds of a network. carbon monoxide's real triple bond "
                     "(1072 kj/mol) is left out: oxygen holds two bonds here",
    "extra_valence": "sulfur is 2 toward hydrogen and metals and 4 or 6 toward oxygen; iron stays 2 (feo in a mantle, "
                     "fes; fe2o3 needs free oxygen)",
    "condense_k": "temperatures at which each substance starts to condense in a gas of solar make-up near 1e-4 bar "
                  "after lodders 2003, from memory and rounded (corundum 1677, forsterite 1354, iron metal 1357, "
                  "troilite 704, water ice 182, ammonia hydrate 131, methane hydrate 78 k; half of an element is "
                  "condensed some tens of kelvin lower, pure methane ice needs about 40 k). a table, not "
                  "simulated: the pool's own network (rock) forms where its bonds hold, near 2200 k in a rocky "
                  "mix, hotter than any solid of the table",
    "water": "273 and 373 k are the freezing and boiling points of water at 1 bar: a surface under an atmosphere, "
             "not a nebula",
    "mixes": "toy fractions. cosmic: the sun's ratios (o 4.5e-4, c 2.5e-4, ne 7.8e-5, n 6.2e-5, mg 3.7e-5, si 3.0e-5, "
             "fe 2.9e-5, s 1.2e-5 of all atoms, from memory) times metal_boost 8, helium a quarter of the mass. "
             "rocky: after the silicate earth (o 58.5, mg 20, si 16, fe 2.4, al 1.9, ca 1.3 percent of atoms, from "
             "memory) with more iron and traces of h, c, n, s, na, k, cl; oxygen is about 5 percent short of all "
             "valences. ocean: round 6's fractions; sodium and chlorine are about 7 times sea water's (salt_boost), "
             "magnesium 24 and sulfur 60 times, and h to o is 1.79 instead of 2. carbon: round 6's fractions; h to c "
             "is 4.4 against a thousand or more in a carbon star's wind (carbon only a little above oxygen). reducing: a hydrogen-rich surface gas with 2 "
             "percent carbon and nitrogen, the kind of gas of miller's 1953 experiment, with the ocean's salt",
    "sparks": "toy: each spark atomises 2 percent of the groups of 2 to 64 atoms; no energy is counted",
}


def unit_enthalpy(atoms_kj: float, formation_kj: float, units: int) -> int:
    """Hess's law for one bond: the enthalpy of the free atoms of a formula unit minus its enthalpy
    of formation, divided by the valence units bonded, in whole kJ/mol (halves round up). The two
    enthalpies are read in tenths of a kJ, so the result does not depend on float rounding.

    >>> unit_enthalpy(685.2, -241.8, 2), unit_enthalpy(436.0, 0.0, 1), unit_enthalpy(947.6, -910.9, 4)
    (464, 436, 465)
    """
    tenths = int(round(atoms_kj * 10)) - int(round(formation_kj * 10))
    return (2 * tenths + 10 * units) // (20 * units)


def atoms_enthalpy(formula: str) -> float:
    """Enthalpy of the free atoms of one formula unit, kJ/mol: the sum of :data:`ATOM_KJ`."""
    return sum(int(round(ATOM_KJ[e.lower()] * 10)) * n for e, n in parse(formula).items()) / 10


def stable_below_k(enthalpy_kj: int) -> int:
    """Kelvin below which a bond of this enthalpy holds: 4.8 K per kJ/mol, rounded down.

    >>> stable_below_k(436), stable_below_k(945), stable_below_k(200)
    (2092, 4536, 960)
    """
    return int(enthalpy_kj) * KELVIN_TENTHS_PER_KJ // 10


def reaction_heat(made_kj, broken_kj):
    """Heat an exchange gives off, kJ/mol: the enthalpy of the bonds made minus that of the bonds broken."""
    return made_kj - broken_kj


def move_allowed(made_kj: int, broken_kj: int) -> str:
    """``yes`` when an exchange gives off heat (it is exothermic), else ``no``."""
    return "yes" if reaction_heat(made_kj, broken_kj) > 0 else "no"


def bond_order(free_a: int, free_b: int, highest: int) -> int:
    """Valences two atoms share when they meet: as many as both have free, up to the pair's highest order."""
    return min(free_a, free_b, highest)


def free_valence(valence, held):
    """Valences an atom still has free: its valence minus the bond orders it holds."""
    return valence - held


def stage7(temperature: int, free_atoms: float) -> str:
    """The stage of a rules-7 state: ``hot`` at or above :data:`HOT_K` (nothing can bond), else by
    the free atoms: ``atomic`` from 0.9, ``forming`` from 0.5, ``molecular`` below."""
    if temperature >= HOT_K:
        return "hot"
    return "atomic" if free_atoms >= 0.9 else "forming" if free_atoms >= 0.5 else "molecular"


def water_state(temperature: int) -> str:
    """What water is at this temperature and 1 bar: ``ice`` below 273 K, ``liquid`` up to 373 K, else ``vapor``."""
    return "ice" if temperature < WATER_MIN_K else "liquid" if temperature <= WATER_MAX_K else "vapor"


def condensed(below_k: int, temperature: int) -> str:
    """``solid`` when the gas is cooler than a substance's condensation temperature, else ``vapor``."""
    return "solid" if temperature < below_k else "vapor"


def solids_at(temperature: int) -> str:
    """The substances of :data:`CONDENSE_K` that are solid at this temperature, or ``none``."""
    return " ".join(s for s, k in CONDENSE_K.items() if condensed(k, temperature) == "solid") or "none"


def expected_hits(molecules: int) -> float:
    """Groups one spark is expected to take apart among ``molecules`` groups of 2 to 64 atoms."""
    return molecules * SPARK_PPM / 1000000


def water_share(h2o: int, molecules: int) -> float:
    """h2o molecules divided by all molecules, 3 significant digits (0 without molecules)."""
    return sig3(h2o / molecules) if molecules else 0.0


def water_atoms(h2o: int, atoms: int) -> float:
    """Share of all atoms that sit in h2o molecules: three per molecule, 3 significant digits."""
    return sig3(3 * h2o / atoms)


def _bond_table() -> tuple[dict[tuple[str, str], tuple[int, ...]], dict[tuple[str, str, int], str]]:
    """(pair -> enthalpy of order 1, 2, ... in kJ/mol, (pair, order) -> where the number comes from)."""
    found: dict[tuple[str, str], dict[int, int]] = {}
    source: dict[tuple[str, str, int], str] = {}
    for name, (formula, formation, pair, order, units) in FORMATION_KJ.items():
        found.setdefault(pair, {})[order] = order * unit_enthalpy(atoms_enthalpy(formula), formation, units)
        source[(*pair, order)] = name
    for (a, b, order), kj in MEAN_KJ.items():
        if order in found.get((a, b), {}):
            raise ValueError(f"two sources for the bond {a} {b} order {order}")
        found.setdefault((a, b), {})[order] = kj
        source[(a, b, order)] = "mean"
    for a in sorted(METALS):
        for b in sorted(METALS):
            if a <= b:  # each metal's atomisation enthalpy per valence, added, in tenths and rounded half up
                va, vb = VALENCE[a], VALENCE[b]
                tenths = int(round(ATOM_KJ[a] * 10)) * vb + int(round(ATOM_KJ[b] * 10)) * va
                found.setdefault((a, b), {})[1] = (2 * tenths + 10 * va * vb) // (20 * va * vb)
                source[(a, b, 1)] = "metal"
    table: dict[tuple[str, str], tuple[int, ...]] = {}
    for a in ELEMENTS:
        for b in ELEMENTS:
            if a <= b and VALENCE[a] and VALENCE[b]:
                orders = found.get((a, b))
                if orders is None:
                    orders = {1: DEFAULT_KJ}
                    source[(a, b, 1)] = "default"
                top = HIGHEST_ORDER.get((a, b), 1)
                if sorted(orders) != list(range(1, top + 1)):
                    raise ValueError(f"the bond {a} {b} needs an enthalpy for each order up to {top}")
                table[(a, b)] = tuple(orders[m] for m in range(1, top + 1))
    return table, source


#: (element, element), sorted -> bond enthalpy in kJ/mol of order 1, 2, ... up to the pair's highest
BOND_KJ, BOND_SOURCE = _bond_table()


def highest_order(a: str, b: str) -> int:
    """Most valences a pair of elements can share under rules 7; 0 when one is noble."""
    return len(BOND_KJ.get((a, b) if a <= b else (b, a), ()))


def bond_enthalpy(a: str, b: str, order: int = 1) -> int:
    """Enthalpy in kJ/mol of the bond of this order between two elements; 0 when there is none."""
    kj = BOND_KJ.get((a, b) if a <= b else (b, a), ())
    return kj[order - 1] if 1 <= order <= len(kj) else 0


def stability7(a: str, b: str, order: int = 1) -> int:
    """Kelvin below which the bond of this order holds under rules 7; 0 when there is none."""
    return stable_below_k(bond_enthalpy(a, b, order))


_N_EL = len(ELEMENTS)
_D3 = np.zeros((_N_EL, _N_EL, 4), dtype=np.int64)  # enthalpy by (element, element, order); order 0 is no bond
for (_a, _b), _kjs in BOND_KJ.items():
    for _m, _kj in enumerate(_kjs, 1):
        _D3[_IX[_a], _IX[_b], _m] = _D3[_IX[_b], _IX[_a], _m] = _kj
_K3 = np.array([[[stable_below_k(int(kj)) for kj in row] for row in plane] for plane in _D3], dtype=np.int64)
_TOP = (_D3 > 0).sum(2)
_STEP3 = _D3[:, :, 1:] - _D3[:, :, :-1]  # what the m-th shared valence alone adds
_D = [[tuple(int(x) for x in _D3[i, j]) for j in range(_N_EL)] for i in range(_N_EL)]
_K = [[tuple(int(x) for x in _K3[i, j]) for j in range(_N_EL)] for i in range(_N_EL)]
_TOPS = [[int(_TOP[i, j]) for j in range(_N_EL)] for i in range(_N_EL)]
_VALS = [VALENCE[e] for e in ELEMENTS]
_OPENS = [EXTRA_VALENCE.get(e, 0) for e in ELEMENTS]
_EXTRA = np.array([EXTRA_VALENCE.get(e, 0) for e in ELEMENTS], dtype=np.int64)
_S, _O, _C, _H, _NI, _FE = (_IX[e] for e in ("s", "o", "c", "h", "n", "fe"))
#: at or above this nothing bonds: the stability of the strongest bond of the table (the N-N triple bond)
HOT_K = int(_K3.max())

#: metric name -> formula of the molecules it counts under rules 7 (exact composition)
MOLECULES7: dict[str, str] = {"h2": "H2", "h2o": "H2O", "ch4": "CH4", "nh3": "NH3", "co": "CO", "co2": "CO2",
                              "o2": "O2", "n2": "N2", "nacl": "NaCl", "h2s": "H2S", "h2co": "H2CO"}
#: the names round 5 gives these substances and elements
MOLECULE_NAMES: dict[str, str] = {
    "h2": "hydrogen", "h2o": "water", "ch4": "methane", "nh3": "ammonia", "co": "carbon_monoxide",
    "co2": "carbon_dioxide", "o2": "oxygen", "n2": "nitrogen", "nacl": "sodium_chloride",
    "h2s": "hydrogen_sulfide", "h2co": "formaldehyde"}
ELEMENT_NAMES: dict[str, str] = {
    "h": "hydrogen", "he": "helium", "o": "oxygen", "c": "carbon", "n": "nitrogen", "ne": "neon", "mg": "magnesium",
    "si": "silicon", "s": "sulfur", "fe": "iron", "al": "aluminum", "ca": "calcium", "na": "sodium",
    "k": "potassium", "cl": "chlorine"}
_TARGETS7 = {k: np.array([parse(f).get(e.capitalize(), 0) for e in ELEMENTS], dtype=np.int64)
             for k, f in MOLECULES7.items()}

KEYS7: dict[str, str] = {
    "mix": "the element mix: cosmic, rocky, ocean, carbon or reducing",
    "temperature": "gas temperature in kelvin. level 4 has no clock: a step is one cooling factor, not a time. "
                   "in a nebula near 1e-4 bar real carbon is mostly co above about 650 k and real nitrogen "
                   "mostly n2 above about 330 k; the toy shows the cold end state at every temperature",
    "stage": f"hot ({HOT_K} k or more, nothing can bond), atomic (9 in 10 atoms free), forming, molecular (under "
             "half free); follows from temperature and free_atoms",
    "water": f"what water is at this temperature and 1 bar: ice below {WATER_MIN_K} k, liquid up to {WATER_MAX_K} k, "
             "else vapor (a surface under an atmosphere, not a nebula)",
    "solids": "the substances of the condensation table that are solid at this temperature in a nebula near "
              "1e-4 bar, or none; read from the table, not simulated",
    "spark": "0 while the gas cools; after that the number of the spark this step applied, at the end temperature",
    "hits": "groups of 2 to 64 atoms this step's spark took apart (0 while the gas cools)",
    "free_atoms": "fraction of all atoms with no bond, noble gases included",
    "molecules": "number of bonded groups of two or more atoms, grains of rock included",
    **{k: f"molecules with exactly the composition {dense(f)}" for k, f in MOLECULES7.items()},
    "other": "molecules that are none of the named ones: molecules minus the eleven counts before it",
    "organic": "groups with at least one carbon-hydrogen bond (methane and formaldehyde are among them)",
    "precursors": "organic groups that also hold a c-c, c-n or c-o bond (formaldehyde, methanol, methylamine): "
                  "what world7 hands to the cells",
    "chains": "groups with more than two carbon atoms (round 6 called these organic)",
    "biggest": "atoms in the largest group (1 while the gas is atomic)",
    "rock": f"atoms in groups of {ROCK_ATOMS} or more atoms: a network solid (silicate rock in the rocky mix, soot and "
            "silicon carbide in the carbon mix, a tar of carbon, nitrogen and sulfur in an ocean). the toy's "
            "network forms where its bonds hold (near 2200 k in a rocky mix), hotter than real rock condenses; "
            "solids is the real-world reading",
    "grains": f"number of groups of {ROCK_ATOMS} or more atoms",
    "si_o": "silicon-oxygen bonds", "mg_o": "magnesium-oxygen bonds", "fe_o": "iron-oxygen bonds",
    "fe_metal": "iron atoms whose bonds are all to iron",
    "water_share": "h2o molecules divided by all molecules at the end (a share of molecules, not of mass)",
    "water_atoms": "share of all atoms that sit in h2o molecules at the end",
    "first_water_step": "first step with an h2o molecule, or never",
}
Chem.KEYS7 = KEYS7  # the metrics of a rules-7 rollout, beside round 6's ``Chem.KEYS``
#: the two records of a rules-7 state: ``chem seed 5 rules 7 step 3.`` and ``chem seed 5 rules 7 step 3 census.``
STATE_KEYS7 = ["mix", "temperature", "stage", "water", "solids", "spark", "hits", "free_atoms", "molecules",
               *MOLECULES7, "other"]
CENSUS_KEYS7 = ["organic", "precursors", "chains", "biggest", "rock", "grains", "si_o", "mg_o", "fe_o", "fe_metal"]
QUERY_KEYS7 = STATE_KEYS7 + CENSUS_KEYS7
#: counts: the gate lets them be off by one or by 5 percent
COUNT_KEYS7 = frozenset(["hits", "molecules", *MOLECULES7, "other", *CENSUS_KEYS7])
PARAM_KEYS7 = ("mix", "t_start", "t_end", "steps", "atoms", "sparks")
#: answers that must match exactly: ordinals and the parameters
EXACT_KEYS7 = frozenset(["spark", "first_water_step", *PARAM_KEYS7])
#: the constants a rules-7 record prints, asked as ``chem rules 7 constant hot_k``
CONSTANTS7: dict[str, float | int] = {
    "hot_k": HOT_K, "kelvin_per_kj": KELVIN_TENTHS_PER_KJ / 10, "default_kj": DEFAULT_KJ, "rounds": ROUNDS,
    "spark_ppm": SPARK_PPM, "spark_max_atoms": SPARK_MAX_ATOMS, "rock_atoms": ROCK_ATOMS,
    "water_min_k": WATER_MIN_K, "water_max_k": WATER_MAX_K}


class Pool7:
    """The rules-7 pool: the atoms, their valences (sulfur can open more toward oxygen) and their
    bonds, kept per atom (``nb[a]`` maps a partner to the bond order) because exchanges are
    judged one pair after the other. ``el``, ``valence``, ``free``, ``bond_a``, ``bond_b``,
    ``bond_m``, ``census`` and ``audit`` read like those of :class:`Pool`."""

    def __init__(self, counts: dict[str, int], rng: np.random.Generator, exchange: bool = True,
                 prefilter: bool = True, ledger: bool = False) -> None:
        self.el = np.repeat(np.array([_IX[e] for e in counts], dtype=np.int64), list(counts.values()))
        self.n = n = len(self.el)
        self.valence = _VAL[self.el].copy()
        self.free = self.valence.copy()
        self.extra = _EXTRA[self.el].copy()  # valences an atom can still open
        self.nb: list[dict[int, int]] = [{} for _ in range(n)]
        self.bonds: dict[int, int] = {}  # low atom * n + high atom -> order
        # plain lists for the pair-by-pair work, arrays for the whole-pool work
        self._el, self._free, self._extra_left = self.el.tolist(), self.free.tolist(), self.extra.tolist()
        # per atom, for the prefilter of exchange(): its cheapest shared valence, the partner and that bond's order
        self.cost = np.zeros(n, dtype=np.int64)
        self.cheapest = np.full(n, -1, dtype=np.int64)
        self.cheapest_order = np.zeros(n, dtype=np.int64)
        self._stale: list[int] = []
        self._is_stale = bytearray(n)
        self._arrays: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
        self.bonding = np.flatnonzero(self.valence > 0)
        self.rng = rng
        self.exchange_on, self.prefilter, self.ledger = exchange, prefilter, ledger
        self.temperature: int | None = None  # of the last step
        self.formed = self.broken = self.sparks = self.hits = self.eligible = 0
        self.moves = {"raise": 0, "displace": 0, "double": 0, "open": 0}
        self.heat_kj = self.thermal_kj = self.judged = 0
        self.problems: list[str] = []

    # the bonds as arrays, as Pool has them
    def _bond_arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self._arrays is None:
            keys = np.fromiter(self.bonds.keys(), dtype=np.int64, count=len(self.bonds))
            orders = np.fromiter(self.bonds.values(), dtype=np.int64, count=len(self.bonds))
            self._arrays = (keys // self.n, keys % self.n, orders)
        return self._arrays

    bond_a = property(lambda self: self._bond_arrays()[0])
    bond_b = property(lambda self: self._bond_arrays()[1])
    bond_m = property(lambda self: self._bond_arrays()[2])

    def pair(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """One number per unordered pair of atoms."""
        return np.minimum(a, b) * self.n + np.maximum(a, b)

    def shuffle(self, idx: np.ndarray) -> np.ndarray:
        # by the raw bit stream, which numpy keeps stable across versions (permutation() it does not)
        return idx[np.argsort(self.rng.bit_generator.random_raw(len(idx)), kind="stable")]

    def _set(self, a: int, b: int, order: int) -> None:
        """Make the bond between ``a`` and ``b`` of this order (0 removes it)."""
        nb = self.nb
        old = nb[a].get(b, 0)
        key = a * self.n + b if a < b else b * self.n + a
        if order:
            nb[a][b] = nb[b][a] = self.bonds[key] = order
        else:
            del nb[a][b], nb[b][a], self.bonds[key]
        self._free[a] += old - order
        self._free[b] += old - order
        self._arrays = None
        for x in (a, b):
            if not self._is_stale[x]:
                self._is_stale[x] = 1
                self._stale.append(x)

    def _sync(self) -> None:
        """Bring the per-atom arrays up to date. An atom that had opened extra valences and holds
        two of them free again closes them (sulfur falls back toward 2)."""
        el, free, nb = self._el, self._free, self.nb
        for a in self._stale:
            self._is_stale[a] = 0
            opened = int(_EXTRA[el[a]]) - self._extra_left[a]
            if opened and free[a] >= 2:
                closing = 2 * min(free[a] // 2, opened // 2)
                free[a] -= closing
                self._extra_left[a] += closing
                self.extra[a] += closing
                self.valence[a] -= closing
            self.free[a] = free[a]
            best = None
            row = _D[el[a]]
            for x, m in nb[a].items():
                kj = row[el[x]]
                offer = (kj[m] - kj[m - 1], x, m)
                if best is None or offer < best:
                    best = offer
            self.cost[a], self.cheapest[a], self.cheapest_order[a] = best or (0, -1, 0)
        self._stale = []

    def enthalpy(self) -> int:
        """Sum of the enthalpies of all bonds, kJ/mol of pools."""
        a, b, m = self._bond_arrays()
        return int(_D3[self.el[a], self.el[b], m].sum())

    def step(self, temperature: int) -> None:
        self.broken = self.break_bonds(temperature)
        self.rounds(temperature)

    def rounds(self, temperature: int) -> None:
        """:data:`ROUNDS` times: a meeting round, then an exchange round."""
        self.formed = 0
        for _ in range(ROUNDS):
            self.formed += self.meet(temperature)
            if self.exchange_on:
                self.exchange(temperature)
        self._sync()
        self.temperature = temperature

    def break_bonds(self, temperature: int) -> int:
        """Every bond hotter than its stability temperature breaks; returns how many. Nothing can
        break while the gas does not get hotter: every bond in the pool holds at the temperature
        of the step that left it."""
        if self.temperature is None or temperature <= self.temperature or not self.bonds:
            return 0
        a, b, m = self._bond_arrays()
        gone = np.flatnonzero(_K3[self.el[a], self.el[b], m] < temperature)
        for x, y in zip(a[gone].tolist(), b[gone].tolist()):
            self._set(x, y, 0)
        return len(gone)

    def meet(self, temperature: int) -> int:
        """One round: atoms with a free valence pair off at random; a pair that is not bonded yet
        shares :func:`bond_order` valences if that bond holds at this temperature."""
        self._sync()
        idx = np.flatnonzero(self.free > 0)
        if len(idx) < 2:
            return 0
        idx = self.shuffle(idx)
        n = len(idx) - len(idx) % 2
        a, b = idx[0:n:2], idx[1:n:2]
        ea, eb = self.el[a], self.el[b]
        order = np.minimum(np.minimum(self.free[a], self.free[b]), _TOP[ea, eb])
        ok = np.flatnonzero((order > 0) & (_K3[ea, eb, order] > temperature))
        made, nb, el = 0, self.nb, self._el
        for x, y in zip(a[ok].tolist(), b[ok].tolist()):  # every atom is in at most one pair per round
            if y not in nb[x]:
                shared = bond_order(self._room(x, y), self._room(y, x), _TOPS[el[x]][el[y]])
                if shared > 0 and _K[el[x]][el[y]][shared] > temperature:
                    self._set(x, y, shared)
                    made += 1
        return made

    def _other_held(self, a: int) -> int:
        """Bond orders ``a`` holds to atoms that are not oxygen."""
        el = self._el
        return sum(m for x, m in self.nb[a].items() if el[x] != _O)

    def _room(self, a: int, partner: int) -> int:
        """Free valences ``a`` can share with ``partner``: all of them, except that the valences an
        atom opened hold only oxygen, so toward anything else it never goes beyond :data:`VALENCE`."""
        el = self._el
        if not _OPENS[el[a]] or el[partner] == _O:
            return self._free[a]
        return min(self._free[a], _VALS[el[a]] - self._other_held(a))

    def _fits(self, a: int, partner: int, released: int | None) -> bool:
        """True when ``a`` may share one more valence with ``partner`` while it gives up one it
        shares with ``released`` (None: it uses a free one). Only sulfur can be refused: with
        anything but oxygen it shares at most :data:`VALENCE` valences."""
        el = self._el
        if not _OPENS[el[a]] or el[partner] == _O:
            return True
        held = self._other_held(a) - (1 if released is not None and el[released] != _O else 0)
        return held < _VALS[el[a]]

    def exchange(self, temperature: int) -> int:
        """One round: all bonding atoms pair off at random and the pairs are judged in order with
        the state the earlier pairs left (:meth:`judge`). Returns the number of moves.

        The prefilter skips, with whole-array arithmetic, the pairs that cannot move: those whose
        best possible heat (the largest step the pair's bond can take, plus a bond between the two
        cheapest partners, minus the two cheapest shared valences) is not above zero. It is exact:
        a skipped pair is judged after all whenever an earlier move of the round touched one of
        its atoms or their neighbours (``prefilter=False`` judges every pair; the tests compare)."""
        self._sync()
        idx = self.shuffle(self.bonding)
        n = len(idx) - len(idx) % 2
        a, b = idx[0:n:2], idx[1:n:2]
        if self.prefilter:
            ea, eb = self.el[a], self.el[b]
            fa, fb = self.free[a], self.free[b]
            xa, xb = self.cheapest[a], self.cheapest[b]
            gain = (_STEP3[ea, eb] * (_K3[ea, eb, 1:] > temperature)).max(1)
            cost = np.where(fa == 0, self.cost[a], 0) + np.where(fb == 0, self.cost[b], 0)
            swap = np.where((fa == 0) & (fb == 0), _D3[self.el[xa], self.el[xb], 1], 0)
            can = gain + swap - cost > 0
            can &= (fa == 0) | (fb == 0) | ((xa >= 0) & (xb >= 0))  # two free atoms move only if bonded to each other
            # the bound above reads each atom's cheapest valence; it is not the one given up when that
            # is the bond to the partner itself or a multiple bond, so those pairs are judged in full
            can |= ((fa == 0) & ((xa == b) | (self.cheapest_order[a] > 1))) \
                | ((fb == 0) & ((xb == a) | (self.cheapest_order[b] > 1)))
            can |= ((fa == 0) & (self.extra[a] >= 2) & (eb == _O) & (fb > 0)) \
                | ((fb == 0) & (self.extra[b] >= 2) & (ea == _O) & (fa > 0))  # sulfur opening toward oxygen
            todo = np.flatnonzero(can & (gain > 0)).tolist()
        else:
            todo = list(range(n // 2))
        if not todo:
            return 0
        first, second = a.tolist(), b.tolist()
        where = np.full(self.n, -1, dtype=np.int64)  # atom -> its pair of this round
        where[a] = where[b] = np.arange(n // 2)
        where = where.tolist()
        queued = bytearray(n // 2)
        for k in todo:
            queued[k] = 1
        before = (self.enthalpy(), self.heat_kj, self.thermal_kj) if self.ledger else None
        moves = 0
        while todo:  # a sorted list is a heap; pairs touched by a move join it and keep their place in the order
            k = heapq.heappop(todo)
            self.judged += 1
            touched = self.judge(first[k], second[k], temperature)
            if touched:
                moves += 1
                for atom in touched:
                    later = where[atom]
                    if later > k and not queued[later]:
                        queued[later] = 1
                        heapq.heappush(todo, later)
        if before and self.enthalpy() != before[0] + (self.heat_kj - before[1]) - (self.thermal_kj - before[2]):
            self.problems.append("the bond enthalpy of an exchange round is not what its moves gave off")
        return moves

    def _give_up(self, a: int, partner: int) -> tuple[int, int] | None:
        """(enthalpy, neighbour) of the cheapest shared valence ``a`` can give up, never one of the
        bond to ``partner``; ties go to the lowest neighbour."""
        el, best = self._el, None
        row = _D[el[a]]
        for x, m in self.nb[a].items():
            if x != partner:
                kj = row[el[x]]
                offer = (kj[m] - kj[m - 1], x)
                if best is None or offer < best:
                    best = offer
        return best

    def judge(self, a: int, b: int, temperature: int) -> list[int] | None:
        """The exchange rule for one pair. The bond between ``a`` and ``b`` rises by one order if
        that order exists and holds at this temperature and the move gives off heat. An atom with
        no free valence pays with the cheapest valence it shares with another neighbour; if both
        pay, their two released neighbours bond to each other when they can (a double
        displacement). Returns the atoms whose pairs must be judged (again) later in the round, or
        None when nothing moved."""
        el, free, nb = self._el, self._free, self.nb
        ea, eb = el[a], el[b]
        kj, kelvin, top = _D[ea][eb], _K[ea][eb], _TOPS[ea][eb]
        m = nb[a].get(b, 0)
        fa, fb = free[a], free[b]
        if (fa == 0) != (fb == 0):  # an atom that can open valences meets an oxygen atom with a free one
            full, other = (a, b) if fa == 0 else (b, a)
            if self._extra_left[full] >= 2 and el[other] == _O:
                order = m + bond_order(2, free[other], top - m)
                if order > m and kelvin[order] > temperature:
                    touched = [a, b, *nb[a], *nb[b]]
                    self._extra_left[full] -= 2
                    self.extra[full] -= 2
                    self.valence[full] += 2
                    free[full] += 2
                    self._set(a, b, order)
                    self.heat_kj += reaction_heat(kj[order], kj[m])
                    self.moves["open"] += 1
                    return touched
        if m >= top or not kelvin[m + 1] > temperature:
            return None
        if fa > 0 and fb > 0:
            if not m or not (self._fits(a, b, None) and self._fits(b, a, None)):
                return None  # two free atoms that are not bonded are the meeting round's business
            self._set(a, b, m + 1)
            self.heat_kj += reaction_heat(kj[m + 1], kj[m])
            self.moves["raise"] += 1
            return [a, b, *nb[a], *nb[b]]
        made, broken, x, y = kj[m + 1] - kj[m], 0, None, None
        if fa == 0:
            offer = self._give_up(a, b)
            if offer is None:
                return None
            broken, x = offer
        if fb == 0:
            offer = self._give_up(b, a)
            if offer is None:
                return None
            broken, y = broken + offer[0], offer[1]
        if not (self._fits(a, b, x) and self._fits(b, a, y)):
            return None
        swap = 0
        if x is not None and y is not None and x != y and y not in nb[x] and _K[el[x]][el[y]][1] > temperature \
                and self._fits(x, y, a) and self._fits(y, x, b):
            swap = _D[el[x]][el[y]][1]
        if move_allowed(made + swap, broken) != "yes":
            return None
        touched = [a, b, *nb[a], *nb[b]]
        if x is not None:
            touched += nb[x]
            self._lower(a, x, temperature)
        if y is not None:
            touched += nb[y]
            self._lower(b, y, temperature)
        self._set(a, b, m + 1)
        if swap:
            self._set(x, y, 1)
        self.heat_kj += reaction_heat(made + swap, broken)
        self.moves["double" if swap else "displace"] += 1
        return touched

    def _lower(self, a: int, x: int, temperature: int) -> None:
        """One order less between ``a`` and ``x``. What is left of a multiple bond breaks at once if
        it cannot hold at this temperature (the same law as at the start of a step)."""
        el = self._el
        order = self.nb[a][x] - 1
        if order and _K[el[a]][el[x]][order] < temperature:
            self.thermal_kj += _D[el[a]][el[x]][order]
            order = 0
        self._set(a, x, order)

    def spark(self, temperature: int) -> None:
        """One spark: every group of 2 to :data:`SPARK_MAX_ATOMS` atoms draws one raw number, in
        the order of the groups' lowest atoms, and loses all its bonds when the number modulo a
        million is below :data:`SPARK_PPM`; then the rounds of a step at this temperature."""
        self._sync()
        _, size, labels = self.groups()
        small = np.flatnonzero((size >= 2) & (size <= SPARK_MAX_ATOMS))
        raw = self.rng.bit_generator.random_raw(len(small))
        hit = small[raw % np.uint64(1000000) < np.uint64(SPARK_PPM)]
        self.sparks, self.eligible, self.hits = self.sparks + 1, len(small), len(hit)
        if len(hit):
            marked = np.zeros(len(size), dtype=bool)
            marked[hit] = True
            a, b, _ = self._bond_arrays()
            gone = np.flatnonzero(marked[labels[a]])
            for x, y in zip(a[gone].tolist(), b[gone].tolist()):
                self._set(x, y, 0)
        self.broken = 0
        self.rounds(temperature)

    def groups(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Composition (a row over ELEMENTS) and size of every connected group, and each atom's
        group. Groups are numbered by their lowest atom, whatever the graph library does."""
        n = self.n
        a, b, _ = self._bond_arrays()
        graph = coo_matrix((np.ones(len(a)), (a, b)), shape=(n, n))
        k, labels = connected_components(graph, directed=False)
        lowest = np.full(k, n, dtype=np.int64)
        np.minimum.at(lowest, labels, np.arange(n))
        rank = np.empty(k, dtype=np.int64)
        rank[np.argsort(lowest, kind="stable")] = np.arange(k)
        labels = rank[labels]
        comp = np.bincount(labels * _N_EL + self.el, minlength=k * _N_EL).reshape(k, _N_EL)
        return comp, comp.sum(1), labels

    def census(self) -> tuple[np.ndarray, np.ndarray]:
        """Composition of every connected group and its size in atoms, as :meth:`Pool.census`."""
        comp, size, _ = self.groups()
        return comp, size

    def audit(self, totals: np.ndarray) -> list[str]:
        """The laws: atoms conserved per element, valence never exceeded (and opened only as
        :data:`EXTRA_VALENCE` allows), nobles never bonded, no bond of an order the pair does not
        have, no bond that cannot hold at the pool's temperature, and what the ledger found."""
        self._sync()
        problems = list(self.problems)
        n = self.n
        a, b, m = self._bond_arrays()
        held = np.bincount(a, m, minlength=n).astype(np.int64) + np.bincount(b, m, minlength=n).astype(np.int64)
        if (self.free < 0).any() or not np.array_equal(free_valence(self.valence, held), self.free) \
                or self.free.tolist() != self._free:
            problems.append("an atom holds more bonds than its valence")
        if (self.extra < 0).any() or (self.extra % 2).any() \
                or not np.array_equal(self.valence + self.extra, _VAL[self.el] + _EXTRA[self.el]):
            problems.append("an atom opened valences it does not have")
        if (m < 1).any() or (m > _TOP[self.el[a], self.el[b]]).any() or (a == b).any():
            problems.append("a bond of impossible order or an atom bonded to itself")
        if len(self.bonds) != sum(len(x) for x in self.nb) // 2:
            problems.append("two atoms are bonded twice")
        other = np.zeros(n, dtype=np.int64)  # bond orders held to atoms that are not oxygen
        np.add.at(other, a[self.el[b] != _O], m[self.el[b] != _O])
        np.add.at(other, b[self.el[a] != _O], m[self.el[a] != _O])
        if (other > _VAL[self.el]).any():
            problems.append("an opened valence holds something other than oxygen")
        if (self.valence[a] == 0).any() or (self.valence[b] == 0).any():
            problems.append("a noble gas is bonded")
        if self.temperature is not None and (_K3[self.el[a], self.el[b], m] < self.temperature).any():
            problems.append("a bond that cannot hold at this temperature")
        comp, size = self.census()
        if not np.array_equal(comp[size == 1].sum(0) + comp[size >= 2].sum(0), totals):
            problems.append("an element's atom count changed")
        return problems


def _groups(pool: Pool | Pool7) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if isinstance(pool, Pool7):
        return pool.groups()
    n = len(pool.el)
    graph = coo_matrix((np.ones(len(pool.bond_a)), (pool.bond_a, pool.bond_b)), shape=(n, n))
    k, labels = connected_components(graph, directed=False)
    comp = np.bincount(labels * _N_EL + pool.el, minlength=k * _N_EL).reshape(k, _N_EL)
    return comp, comp.sum(1), labels


def _bonds_of(pool: Pool | Pool7, x: int, *partners: int) -> np.ndarray:
    """Mask over the pool's bonds: those between element ``x`` and one of ``partners``."""
    ea, eb = pool.el[pool.bond_a], pool.el[pool.bond_b]
    return ((ea == x) & np.isin(eb, partners)) | ((eb == x) & np.isin(ea, partners))


def ch_molecules(pool: Pool | Pool7) -> int:
    """Groups of a pool (of either rule set) that hold at least one carbon-hydrogen bond: ``organic``."""
    return _organic(pool, _groups(pool)[2])[0]


def precursor_molecules(pool: Pool | Pool7) -> int:
    """Groups with a carbon-hydrogen bond and also a C-C, C-N or C-O bond: ``precursors``."""
    return _organic(pool, _groups(pool)[2])[1]


def _organic(pool: Pool | Pool7, labels: np.ndarray) -> tuple[int, int]:
    with_ch = np.unique(labels[pool.bond_a[_bonds_of(pool, _C, _H)]])
    with_cx = np.unique(labels[pool.bond_a[_bonds_of(pool, _C, _C, _NI, _O)]])
    return len(with_ch), len(np.intersect1d(with_ch, with_cx))


def metrics7(pool: Pool7, temperature: int) -> dict[str, float | int | str]:
    """The numbers of one rules-7 state, all from one census of the pool."""
    comp, size, labels = pool.groups()
    n = pool.n
    free_atoms = sig3(int((size == 1).sum()) / n)  # the stage follows the number the line shows
    molecules = int((size >= 2).sum())
    out: dict[str, float | int | str] = {
        "temperature": temperature, "stage": stage7(temperature, free_atoms), "water": water_state(temperature),
        "solids": solids_at(temperature), "spark": pool.sparks, "hits": pool.hits if pool.sparks else 0,
        "free_atoms": free_atoms, "molecules": molecules}
    for name, target in _TARGETS7.items():
        out[name] = int((comp == target).all(1).sum())
    out["other"] = molecules - sum(int(out[k]) for k in MOLECULES7)
    out["organic"], out["precursors"] = _organic(pool, labels)
    out["chains"] = int((comp[:, _C] > 2).sum())
    out["biggest"] = int(size.max())
    big = size >= ROCK_ATOMS
    out["rock"], out["grains"] = int(size[big].sum()), int(big.sum())
    for key, metal in (("si_o", "si"), ("mg_o", "mg"), ("fe_o", "fe")):
        out[key] = int(_bonds_of(pool, _O, _IX[metal]).sum())
    a, b = pool.bond_a, pool.bond_b
    iron = _bonds_of(pool, _FE, _FE)
    partners = np.bincount(a, minlength=n) + np.bincount(b, minlength=n)
    iron_partners = np.bincount(a[iron], minlength=n) + np.bincount(b[iron], minlength=n)
    out["fe_metal"] = int(((pool.el == _FE) & (partners > 0) & (partners == iron_partners)).sum())
    return out


def bad_params7(params: dict) -> str:
    """What is wrong with the parameters of a rules-7 run; empty when they can be simulated."""
    if params.get("rules") != 7 or isinstance(params.get("rules"), bool):
        return f"rules must be 7 here, not {params.get('rules')!r}"
    if params.get("mix") not in MIXES7:
        return f"unknown mix {params.get('mix')!r}"
    for key, least in (("t_start", 1), ("t_end", 1), ("steps", 1), ("atoms", 1), ("sparks", 0)):
        value = params.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < least:
            return f"{key} must be a whole number from {least} on, not {value!r}"
    if int(params["sparks"]) > SPARKS_LIMIT:
        return f"at most {SPARKS_LIMIT} sparks"
    unknown = set(params) - {*PARAM_KEYS7, "rules"}
    if unknown:
        return f"unknown parameter under rules 7: {', '.join(sorted(unknown))}"
    if min(element_counts(str(params["mix"]), int(params["atoms"]), rules=7).values()) < 0:
        return "too few atoms for this mix"
    return ""


def budget7(step: dict[str, float | int | str], counts: dict[str, int]) -> list[str]:
    """Conservation read off one rules-7 step's numbers alone: the counted molecules and bonds may
    not hold more atoms or valences of an element than the mix has, nor more atoms than are
    bonded at all, and the counts must fit each other."""
    problems = []
    n = {k: int(step.get(k, 0)) for k in COUNT_KEYS7}
    have = {e: counts.get(e, 0) for e in ELEMENTS}
    # an organic group holds at least one carbon and one hydrogen atom, a chain three carbon atoms;
    # every methane and formaldehyde molecule is an organic group
    loose = max(0, n["organic"] - n["ch4"] - n["h2co"])
    used = {e: sum(n[k] * int(_TARGETS7[k][_IX[e]]) for k in MOLECULES7) for e in ELEMENTS}
    used["c"] += 3 * n["chains"] + max(0, loose - n["chains"])
    used["h"] += loose
    for e in ELEMENTS:
        if used[e] > have[e]:
            problems.append(f"more {e} in the counted molecules than the mix holds")
    if n["other"] != n["molecules"] - sum(n[k] for k in MOLECULES7) or n["other"] < 0:
        problems.append("other is not the molecules without a name")
    if not n["precursors"] <= n["organic"] <= n["molecules"] or n["chains"] > n["molecules"] \
            or n["organic"] < n["ch4"] + n["h2co"]:
        problems.append("organic, precursors and chains do not fit the molecules")
    if n["rock"] < ROCK_ATOMS * n["grains"] or (n["grains"] > 0) != (n["biggest"] >= ROCK_ATOMS) \
            or (n["grains"] > 0) != (n["rock"] > 0) or (n["grains"] and n["rock"] < n["biggest"]) \
            or n["grains"] > n["other"]:
        problems.append("rock, grains and biggest do not fit each other")
    if n["si_o"] > VALENCE["si"] * have["si"] or n["mg_o"] > VALENCE["mg"] * have["mg"] \
            or n["fe_o"] > VALENCE["fe"] * have["fe"] or n["si_o"] + n["mg_o"] + n["fe_o"] > VALENCE["o"] * have["o"]:
        problems.append("more bonds to oxygen than the valences allow")
    if n["fe_metal"] > have["fe"] or (n["fe_metal"] and n["fe_o"] > VALENCE["fe"] * (have["fe"] - n["fe_metal"])):
        problems.append("more iron in metal and oxide than the mix holds")
    atoms = sum(counts.values())
    bonded = sum(n[k] * int(_TARGETS7[k].sum()) for k in MOLECULES7) + 2 * (n["other"] - n["grains"]) + n["rock"]
    if bonded > atoms * (1 - float(step.get("free_atoms", 0))) + 0.005 * atoms + 1:  # free_atoms has 3 digits
        problems.append("more atoms in molecules than are bonded")
    return problems


def cool7(seed: int, params: dict[str, int | str], ledger: bool = False) -> Iterator[tuple[int, int, Pool7]]:
    """(step, temperature in kelvin, the rules-7 pool after that step): the cooling steps of
    :func:`cool`, then one step per spark at the end temperature."""
    mix, t_start, t_end = str(params["mix"]), int(params["t_start"]), int(params["t_end"])
    n_steps = int(params["steps"])
    pool = Pool7(element_counts(mix, int(params["atoms"]), rules=7), np.random.Generator(np.random.PCG64(seed)),
                 ledger=ledger)
    for t in range(n_steps + 1):
        temperature = temperature_at(t_start, t_end, t, n_steps)
        if t:
            pool.step(temperature)
        yield t, temperature, pool
    for k in range(1, int(params["sparks"]) + 1):
        pool.spark(t_end)
        yield n_steps + k, t_end, pool


def simulate7(seed: int, params: dict[str, int | str], audit: bool = False) -> tuple[Rollout, list[str]]:
    """The rules-7 rollout and, with ``audit``, every broken law found along the way."""
    mix, atoms = str(params["mix"]), int(params["atoms"])
    counts = element_counts(mix, atoms, rules=7)
    totals = np.array([counts.get(e, 0) for e in ELEMENTS])
    steps: list[dict[str, float | int | str]] = []
    problems: list[str] = []
    for t, temperature, pool in cool7(seed, params, ledger=audit):
        steps.append({"mix": mix, **metrics7(pool, temperature)})
        if audit:
            problems += [f"step {t}: {p}" for p in pool.audit(totals)]
            pool.problems.clear()
            if pool.sparks:  # the hits of a spark are a draw: within six standard deviations of the expectation
                mean = expected_hits(pool.eligible)
                if pool.hits > pool.eligible or abs(pool.hits - mean) > 6 * math.sqrt(mean) + 3:
                    problems.append(f"step {t}: the spark hit {pool.hits} of {pool.eligible} groups")
    last = steps[-1]
    watery = [t for t, s in enumerate(steps) if s["h2o"]]
    summary: dict[str, float | int | str] = {"mix": mix, "water": last["water"]}
    summary.update({k: last[k] for k in QUERY_KEYS7 if k in COUNT_KEYS7 and k != "hits"})
    summary["water_share"] = water_share(int(last["h2o"]), int(last["molecules"]))
    summary["water_atoms"] = water_atoms(int(last["h2o"]), atoms)
    summary["first_water_step"] = watery[0] if watery else "never"
    return Rollout(SIM, seed, dict(params), steps, summary), problems


#: the metrics of a rules-7 state that are words; ``free_atoms`` is a share, every other one a whole number
WORD_KEYS7 = frozenset(["mix", "stage", "water", "solids"])


def _well_typed7(step: dict) -> bool:
    """True when every metric of a rules-7 state is of its kind: a word, a finite share, a whole number."""
    for key in QUERY_KEYS7:
        value = step[key]
        if key in WORD_KEYS7:
            ok = isinstance(value, str)
        elif key == "free_atoms":
            ok = isinstance(value, (int, float)) and not isinstance(value, bool) and is_finite(value)
        else:
            ok = isinstance(value, int) and not isinstance(value, bool)
        if not ok:
            return False
    return True


def conserved7(r: Rollout) -> Verdict:
    """The gate of a rules-7 rollout: the replay with the pool's laws at every step (atoms per
    element, valences, nobles, no bond that cannot hold, the enthalpy ledger of every exchange
    round: each move gives off heat and the pool's bond enthalpy changes by exactly the heat of
    its moves less the multiple bonds that fell apart), the numbers of every step fitting the
    atoms the mix has (:func:`budget7`), the words following from the numbers, the sparks
    counted in order, and the rollout being what its seed and parameters give."""
    if r.sim != SIM or bad_params7(r.params) or isinstance(r.seed, bool) or not isinstance(r.seed, int) or r.seed < 0:
        return Verdict(False, None, "not a chem rollout")
    replay, problems = simulate7(r.seed, r.params, audit=True)
    counts = element_counts(str(r.params["mix"]), int(r.params["atoms"]), rules=7)
    n_steps, t_end = int(r.params["steps"]), int(r.params["t_end"])
    for t, s in enumerate(r.steps):
        if any(isinstance(v, (int, float)) and v < 0 for v in s.values()):
            problems.append(f"step {t}: a negative count")
        if set(QUERY_KEYS7) - set(s):
            problems.append(f"step {t}: a metric is missing")
            continue
        if not _well_typed7(s):  # a forged rollout is judged, the gate does not raise on it
            problems.append(f"step {t}: a value that is not of its metric's kind")
            continue
        problems += [f"step {t}: {p}" for p in budget7(s, counts)]
        temperature = s["temperature"]
        if s["stage"] != stage7(temperature, s["free_atoms"]) or s["water"] != water_state(temperature) \
                or s["solids"] != solids_at(temperature):
            problems.append(f"step {t}: stage, water or solids do not follow from the temperature")
        before = r.steps[t - 1].get("molecules") if t else 0
        if s["spark"] != max(0, t - n_steps) or (s["spark"] and temperature != t_end) \
                or (s["hits"] and (not s["spark"] or not isinstance(before, int) or s["hits"] > before)):
            problems.append(f"step {t}: the sparks are not counted in order")
    if replay.steps != r.steps or replay.summary != r.summary:
        problems.append("the rollout is not what its seed and parameters give")
    return Verdict(not problems, None, "; ".join(problems[:5]))


@lru_cache(maxsize=256)
def _rollout7(seed: int) -> Rollout:
    return simulation().run(seed, rules=7)


def _head7(r: Rollout) -> str:
    return f"{SIM} seed {num(r.seed)} rules 7"


def params_line7(r: Rollout) -> Line:
    """``chem seed 5 rules 7 params. mix ocean. t_start 5 3 0 0. t_end 4 4 0. steps 3 5. atoms 2 2 0 0 0. sparks 0.``"""
    fields = " ".join(f"{k} {r.value(r.params[k])}." for k in PARAM_KEYS7)
    return Line(f"{_head7(r)} params. {fields}", topic=SIM, kind="record")


def param_lines7(r: Rollout) -> list[Line]:
    return [Line(f"{_head7(r)} param {k}", r.value(r.params[k]), SIM, "fact") for k in PARAM_KEYS7]


def state_lines7(r: Rollout, t: int) -> list[Line]:
    """The two records of a state: the gas (``... step 3. mix ocean. temperature ...``) and the
    census of groups and bonds (``... step 3 census. organic 2 0. precursors 2 0. ...``)."""
    step, head = r.steps[t], f"{_head7(r)} step {num(t)}"
    return [Line(f"{head}{tag}. " + " ".join(f"{k} {r.value(step[k])}." for k in keys), topic=SIM, kind="record")
            for tag, keys in (("", STATE_KEYS7), (" census", CENSUS_KEYS7))]


def query_lines7(r: Rollout, t: int) -> list[Line]:
    """One question per metric at step ``t`` and, if there is a next step, what it is then."""
    head = f"{_head7(r)} step {num(t)}"
    out = [Line(f"{head} {k}", r.value(r.steps[t][k]), SIM, "fact") for k in QUERY_KEYS7]
    if t + 1 < len(r.steps):
        out += [Line(f"{head} next {k}", r.value(r.steps[t + 1][k]), SIM, "calc") for k in QUERY_KEYS7]
    return out


def summary_lines7(r: Rollout) -> list[Line]:
    return [Line(f"{_head7(r)} final {k}", r.value(v), SIM, "calc") for k, v in r.summary.items()]


def lines7(r: Rollout, every: int = 1) -> list[Line]:
    """Every line of one rules-7 rollout: the parameters, two records per step, every metric
    asked (now and next) at every ``every``-th step, the summary and the parameter questions."""
    if not is_rules7(r):
        raise ValueError("not a rules-7 rollout")
    out = [params_line7(r)]
    for t in range(len(r.steps)):
        out += state_lines7(r, t)
    for t in range(0, len(r.steps), every):
        out += query_lines7(r, t)
    return dedupe(out + summary_lines7(r) + param_lines7(r))


def _kj(x: float | int) -> str:
    return num(x, sig=6)  # an enthalpy with one decimal has up to five digits


def _bond_rows() -> list[tuple[str, str, int, int, str]]:
    """(a, b, order, enthalpy, source) of every bond the table holds from a source, not the default."""
    return [(a, b, m, kj, BOND_SOURCE[(a, b, m)]) for (a, b), kjs in BOND_KJ.items()
            for m, kj in enumerate(kjs, 1) if BOND_SOURCE[(a, b, m)] != "default"]


def records7() -> list[Line]:
    """The tables of rules 7: bond counts, elements, atom and formation enthalpies, the bond
    enthalpies computed from them, the counted molecules, the mixes with their fractions, the
    condensation table and the constants."""
    def rec(text: str) -> Line:
        return Line(f"{SIM} rules 7 {text}", topic=SIM, kind="record")

    out = [rec("bonds. " + " ".join(f"{e} {num(v)}." for e, v in VALENCE.items()))]
    for e, name in ELEMENT_NAMES.items():
        extra = f" extra_to_oxygen {num(EXTRA_VALENCE[e])}." if e in EXTRA_VALENCE else ""
        out.append(rec(f"element {e}. name {name}. bonds {num(VALENCE[e])}.{extra}"))
    for e, kj in ATOM_KJ.items():
        out.append(rec(f"atom {e}. enthalpy_kj {_kj(kj)}."))
    for name, (formula, formation, (a, b), order, units) in FORMATION_KJ.items():
        atoms = atoms_enthalpy(formula)
        out.append(rec(f"compound {name}. formula {dense(formula)}. atoms_kj {_kj(atoms)}. formation_kj {_kj(formation)}. "
                       f"bond {a} {b}. order {num(order)}. units {num(units)}. "
                       f"unit_kj {num(unit_enthalpy(atoms, formation, units))}."))
    for a, b, m, kj, source in _bond_rows():
        out.append(rec(f"bond {a} {b} order {num(m)}. enthalpy_kj {num(kj)}. stable_below_k {num(stable_below_k(kj))}. "
                       f"source {source}."))
    for (a, b), top in HIGHEST_ORDER.items():
        out.append(rec(f"bond {a} {b}. highest_order {num(top)}."))
    out.append(rec(f"bond default. enthalpy_kj {num(DEFAULT_KJ)}. stable_below_k {num(stable_below_k(DEFAULT_KJ))}. "
                   "highest_order 1."))
    for name, formula in MOLECULES7.items():
        out.append(rec(f"molecule {name}. name {MOLECULE_NAMES[name]}. formula {dense(formula)}. "
                       f"atoms {num(sum(parse(formula).values()))}."))
    for mix, fractions in MIXES7.items():
        boost = " {} {}.".format(MIX_BOOSTS7[mix][0], num(MIX_BOOSTS7[mix][1])) if mix in MIX_BOOSTS7 else ""
        out.append(rec(f"mix {mix}. " + " ".join(f"{e} {num(f)}." for e, f in fractions.items()) + boost))
    for substance, kelvin in CONDENSE_K.items():
        out.append(rec(f"condense {substance}. below_k {num(kelvin)}."))
    out.append(rec("constants. " + " ".join(f"{k} {num(v)}." for k, v in CONSTANTS7.items())))
    return out


def table_lines7() -> list[Line]:
    """Every question the rules-7 tables answer. Bonds are asked only where the table has a
    source, plus the default and one noble pair; the gate answers every pair."""
    def ask(prompt: str, answer: str) -> Line:
        return Line(f"{SIM} rules 7 {prompt}", answer, SIM, "fact")

    out = [ask(f"bonds {e}", num(v)) for e, v in VALENCE.items()]
    out += [ask(f"element {e} name", name) for e, name in ELEMENT_NAMES.items()]
    out += [ask(f"element {e} extra_to_oxygen", num(v)) for e, v in EXTRA_VALENCE.items()]
    out += [ask(f"atom {e} enthalpy_kj", _kj(kj)) for e, kj in ATOM_KJ.items()]
    for name, (formula, formation, (a, b), order, units) in FORMATION_KJ.items():
        atoms = atoms_enthalpy(formula)
        out += [ask(f"compound {name} formula", dense(formula)), ask(f"compound {name} atoms_kj", _kj(atoms)),
                ask(f"compound {name} formation_kj", _kj(formation)), ask(f"compound {name} bond", f"{a} {b}"),
                ask(f"compound {name} units", num(units)),
                ask(f"compound {name} unit_kj", num(unit_enthalpy(atoms, formation, units)))]
    for a, b, m, kj, source in _bond_rows():
        out += [ask(f"bond {a} {b} order {num(m)} enthalpy_kj", num(kj)),
                ask(f"bond {a} {b} order {num(m)} stable_below_k", num(stable_below_k(kj))),
                ask(f"bond {a} {b} order {num(m)} source", source)]
    out += [ask(f"bond {a} {b} highest_order", num(top)) for (a, b), top in HIGHEST_ORDER.items()]
    out += [ask("bond default enthalpy_kj", num(DEFAULT_KJ)),
            ask("bond default stable_below_k", num(stable_below_k(DEFAULT_KJ))),
            ask("bond h he highest_order", "0"), ask("bond h he order 1 stable_below_k", "never")]
    for name, formula in MOLECULES7.items():
        out += [ask(f"molecule {name} name", MOLECULE_NAMES[name]), ask(f"molecule {name} formula", dense(formula)),
                ask(f"molecule {name} atoms", num(sum(parse(formula).values())))]
    for mix, fractions in MIXES7.items():
        out.append(ask(f"mix {mix} main", next(iter(fractions))))
        out.append(ask(f"mix {mix} elements", " ".join(fractions)))
        out += [ask(f"mix {mix} share {e}", num(f)) for e, f in fractions.items()]
        if mix in MIX_BOOSTS7:
            out.append(ask(f"mix {mix} {MIX_BOOSTS7[mix][0]}", num(MIX_BOOSTS7[mix][1])))
    out += [ask(f"condense {substance} below_k", num(kelvin)) for substance, kelvin in CONDENSE_K.items()]
    out += [ask(f"constant {k}", num(v)) for k, v in CONSTANTS7.items()]
    return out


_E = r"([a-z]{1,2})"
_RULES7_Q = re.compile(rf"^chem (?:seed {_N} )?rules 7 ")
_SIM_Q7 = re.compile(rf"^chem seed {_N} rules 7 (?:step {_N} (next )?|(final|param) )([a-z][a-z0-9_]*)$")
_TABLE_Q7 = {
    "bonds": re.compile(rf"^chem rules 7 bonds {_E}$"),
    "element": re.compile(rf"^chem rules 7 element {_E} (name|bonds|extra_to_oxygen)$"),
    "atom": re.compile(rf"^chem rules 7 atom {_E} enthalpy_kj$"),
    "compound": re.compile(r"^chem rules 7 compound ([a-z0-9]+) (formula|atoms_kj|formation_kj|bond|order|units|unit_kj)$"),
    "bond": re.compile(rf"^chem rules 7 bond {_E} {_E} (?:order ([1-3]) (enthalpy_kj|stable_below_k|source)|(highest_order))$"),
    "default": re.compile(r"^chem rules 7 bond default (enthalpy_kj|stable_below_k|highest_order)$"),
    "molecule": re.compile(r"^chem rules 7 molecule ([a-z0-9]+) (name|formula|atoms)$"),
    "mix": re.compile(rf"^chem rules 7 mix ([a-z]+) (?:(main|elements|metal_boost|salt_boost)|share {_E})$"),
    "condense": re.compile(r"^chem rules 7 condense ([a-z_]+) below_k$"),
    "constant": re.compile(r"^chem rules 7 constant ([a-z_]+)$"),
}


def owns7(prompt: str) -> bool:
    """True for a question about a rules-7 rollout (``chem seed 5 rules 7 step 3 h2o``) or a
    rules-7 table (``chem rules 7 bond h o order 1 enthalpy_kj``)."""
    return bool(_SIM_Q7.match(prompt)) or any(p.match(prompt) for p in _TABLE_Q7.values())


def check7(prompt: str, answer: str) -> Verdict:
    """Judge a rules-7 question: a rollout question by replaying the seed under rules 7 (counts
    within one or 5 percent, ordinals and parameters exactly, other numbers within 5 percent,
    words exactly), a table question by the table."""
    m = _SIM_Q7.match(prompt)
    if m:
        return _check_run7(int(parse_num(m[1])), m[2], bool(m[3]), m[4], m[5], answer)
    expected = _table_answer7(prompt)
    if expected is None:
        return Verdict(False, None, "not my question")
    if not expected:
        return Verdict(False, None, "unknown name")
    return _judge(expected, answer)


def _check_run7(seed: int, step: str | None, nxt: bool, part: str | None, key: str, answer: str) -> Verdict:
    r = _rollout7(seed)
    if part == "final":
        if key not in r.summary:
            return Verdict(False, None, "unknown summary key")
        expected = r.summary[key]
    elif part == "param":
        if key not in PARAM_KEYS7:
            return Verdict(False, None, "unknown parameter")
        expected = r.params[key]
    else:
        t = int(parse_num(step)) + (1 if nxt else 0)
        if t >= len(r.steps):
            return Verdict(False, None, "no such step")
        if key not in r.steps[t]:
            return Verdict(False, None, "unknown key")
        expected = r.steps[t][key]
    text = r.value(expected)
    if isinstance(expected, str):
        return _judge(text, answer)
    got = parse_num(answer)
    if not is_finite(got):  # 1 e 9 9 9 parses as infinity, which is close to anything
        return Verdict(False, text, "not a number")
    if key in EXACT_KEYS7:
        return Verdict(got == expected, text)
    if key in COUNT_KEYS7 and abs(got - expected) <= 1:
        return Verdict(True, text)
    return Verdict(close(got, expected, rel=0.05), text)


def _table_answer7(prompt: str) -> str | None:
    """The answer a rules-7 table gives; ``""`` for a name the table does not have, ``None`` for
    a prompt that is no table question."""
    for kind, pattern in _TABLE_Q7.items():
        m = pattern.match(prompt)
        if m:
            break
    else:
        return None
    if kind == "bonds":
        return num(VALENCE[m[1]]) if m[1] in VALENCE else ""
    if kind == "element":
        if m[1] not in VALENCE:
            return ""
        return {"name": ELEMENT_NAMES[m[1]], "bonds": num(VALENCE[m[1]]),
                "extra_to_oxygen": num(EXTRA_VALENCE.get(m[1], 0))}[m[2]]
    if kind == "atom":
        return _kj(ATOM_KJ[m[1]]) if m[1] in ATOM_KJ else ""
    if kind == "compound":
        if m[1] not in FORMATION_KJ:
            return ""
        formula, formation, (a, b), order, units = FORMATION_KJ[m[1]]
        atoms = atoms_enthalpy(formula)
        return {"formula": dense(formula), "atoms_kj": _kj(atoms), "formation_kj": _kj(formation), "bond": f"{a} {b}",
                "order": num(order), "units": num(units), "unit_kj": num(unit_enthalpy(atoms, formation, units))}[m[2]]
    if kind == "bond":
        if m[1] not in VALENCE or m[2] not in VALENCE:
            return ""
        a, b = sorted((m[1], m[2]))
        if m[5]:
            return num(highest_order(a, b))
        order = int(m[3])
        kj = bond_enthalpy(a, b, order)
        if not kj:
            return "never" if not highest_order(a, b) else ""  # a noble gas never bonds; no such order otherwise
        return {"enthalpy_kj": num(kj), "stable_below_k": num(stable_below_k(kj)),
                "source": BOND_SOURCE[(a, b, order)]}[m[4]]
    if kind == "default":
        return {"enthalpy_kj": num(DEFAULT_KJ), "stable_below_k": num(stable_below_k(DEFAULT_KJ)),
                "highest_order": "1"}[m[1]]
    if kind == "molecule":
        if m[1] not in MOLECULES7:
            return ""
        formula = MOLECULES7[m[1]]
        return {"name": MOLECULE_NAMES[m[1]], "formula": dense(formula),
                "atoms": num(sum(parse(formula).values()))}[m[2]]
    if kind == "mix":
        if m[1] not in MIXES7:
            return ""
        fractions = MIXES7[m[1]]
        if m[3]:
            return num(fractions[m[3]]) if m[3] in fractions else ""
        if m[2] in ("main", "elements"):
            return next(iter(fractions)) if m[2] == "main" else " ".join(fractions)
        return num(MIX_BOOSTS7[m[1]][1]) if MIX_BOOSTS7.get(m[1], ("",))[0] == m[2] else ""
    if kind == "condense":
        return num(CONDENSE_K[m[1]]) if m[1] in CONDENSE_K else ""
    return num(CONSTANTS7[m[1]]) if m[1] in CONSTANTS7 else ""


def _schedule_ok(t_start: int, t_end: int, step: int, steps: int) -> bool:
    return step <= steps and t_end <= t_start


#: the lessons of this level: every rule is a function the rules-7 simulation or its gate calls
RULES: dict[str, Rule] = {
    "schedule_temperature": Rule((Input("t_start", 1, 10000, integer=True), Input("t_end", 1, 10000, integer=True),
                                  Input("step", 0, 60, integer=True), Input("steps", 1, 60, integer=True)),
                                 temperature_at,
                                 "temperature at step in cooling schedule. t_start times t_end over t_start raised "
                                 "to step over steps. rounded kelvin", valid=_schedule_ok),
    "unit_enthalpy": Rule((Input("atoms_kj", 100, 3500, places=1), Input("formation_kj", -1700, 250, places=1),
                           Input("units", 1, 12, integer=True)), unit_enthalpy,
                          "enthalpy of one bond by the law of hess. enthalpy of the free atoms minus enthalpy of "
                          "formation of the compound. divided by units the valence units bonded. rounded kj per mol",
                          valid=lambda atoms_kj, formation_kj, units: atoms_kj > formation_kj),
    "stable_below_k": Rule((Input("enthalpy_kj", 50, 1100, integer=True),), stable_below_k,
                           "temperature below which a bond holds. enthalpy_kj times 4 point 8 rounded down. kelvin"),
    "bond_order": Rule((Input("free_a", 1, 6, integer=True), Input("free_b", 1, 6, integer=True),
                        Input("highest", 1, 3, integer=True)), bond_order,
                       "valences two atoms share when they meet. the smallest of free_a and free_b and the highest "
                       "order of the pair"),
    "free_valence": Rule((Input("valence", 0, 6, integer=True), Input("held", 0, 6, integer=True)), free_valence,
                         "valences an atom has free. valence minus the bond orders it holds",
                         valid=lambda valence, held: held <= valence),
    "reaction_heat": Rule((Input("made_kj", 0, 2000, integer=True), Input("broken_kj", 0, 2000, integer=True)),
                          reaction_heat,
                          "heat an exchange gives off. enthalpy of the bonds made minus enthalpy of the bonds broken. "
                          "kj per mol"),
    "move_allowed": Rule((Input("made_kj", 0, 2000, integer=True), Input("broken_kj", 0, 2000, integer=True)),
                         move_allowed, "yes if an exchange gives off heat. made_kj above broken_kj. else no"),
    "stage": Rule((Input("temperature", 1, 10000, integer=True), Input("free_atoms", 0, 1)), stage7,
                  f"stage under rules 7. hot from {num(HOT_K)} kelvin on. else atomic if free_atoms is at least "
                  "0 point 9. forming if at least 0 point 5. else molecular"),
    "water_state": Rule((Input("temperature", 1, 10000, integer=True),), water_state,
                        f"water at 1 bar. ice below {num(WATER_MIN_K)} kelvin. liquid up to {num(WATER_MAX_K)}. "
                        "else vapor"),
    "condensed": Rule((Input("below_k", 1, 3000, integer=True), Input("temperature", 1, 10000, integer=True)), condensed,
                      "solid if temperature is below the condensation temperature below_k. else vapor"),
    "expected_hits": Rule((Input("molecules", 0, 20000, integer=True),), expected_hits,
                          f"groups one spark is expected to take apart. molecules times {num(SPARK_PPM)} over one million"),
    "water_share": Rule((Input("h2o", 0, 20000, integer=True), Input("molecules", 0, 20000, integer=True)), water_share,
                        "h2o molecules over all molecules. 3 significant digits. 0 without molecules",
                        valid=lambda h2o, molecules: h2o <= molecules),
    "water_atoms": Rule((Input("h2o", 0, 7400, integer=True), Input("atoms", 1000, 22000, integer=True)), water_atoms,
                        "share of all atoms in h2o molecules. 3 times h2o over atoms. 3 significant digits",
                        valid=lambda h2o, atoms: 3 * h2o <= atoms),
}


class ChemLessons(LessonGate):
    """The lesson gate with a draw that reaches every branch. Prompts, truth and judgement are
    those of :class:`haishool.evo.LessonGate`; only :meth:`generate` differs: uniform draws
    would hardly ever show liquid water or a real compound, and would mostly be refused where
    one input bounds another."""

    def draw(self, name: str, rng: random.Random) -> list[float | int]:
        if name == "schedule_temperature":
            a, b, steps = rng.randint(1, 10000), rng.randint(1, 10000), rng.randint(1, 60)
            return [max(a, b), min(a, b), rng.randint(0, steps), steps]
        if name == "unit_enthalpy" and rng.random() < 0.6:  # the inputs of a real table entry
            formula, formation, _, _, units = FORMATION_KJ[rng.choice(sorted(FORMATION_KJ))]
            return [atoms_enthalpy(formula), formation, units]
        if name == "free_valence":
            valence = rng.randint(0, 6)
            return [valence, rng.randint(0, valence)]
        if name == "stage":
            return [rng.choice((rng.randint(100, 6000), rng.randint(HOT_K - 300, HOT_K + 300))),
                    round(rng.choice((rng.random(), rng.uniform(0.85, 0.95), rng.uniform(0.45, 0.55))), 3)]
        if name == "water_state":
            return [rng.choice((rng.randint(150, 500), rng.randint(150, 500), rng.randint(1, 6000)))]
        if name == "condensed":
            below = rng.choice((*CONDENSE_K.values(), rng.randint(50, 2000)))
            return [below, rng.choice((rng.randint(max(1, below - 200), below + 200), rng.randint(50, 2000)))]
        if name == "water_share":
            molecules = rng.choice((0, rng.randint(1, 20000), rng.randint(1, 20000)))
            return [rng.randint(0, molecules), molecules]
        if name == "water_atoms":
            atoms = rng.randrange(18000, 22001, 1000)
            return [rng.randint(0, atoms // 3), atoms]
        if name in ("reaction_heat", "move_allowed") and rng.random() < 0.3:  # close calls
            made = rng.randint(100, 1500)
            return [made, max(0, made + rng.randint(-20, 20))]
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


LESSONS = ChemLessons(SIM, RULES)


def lesson_gate() -> LessonGate:
    """The gate of the lessons ``chem predict <rule> <input> <value> ...`` (topic ``predict_chem``)."""
    return LESSONS


def run(seed: int, rules: int = 6, **params) -> Rollout:
    """:meth:`Chem.run`: the rollout of a seed under round 6 (the default) or ``rules=7``."""
    return simulation().run(seed, rules, **params)


def conserved(r: Rollout) -> Verdict:
    """:meth:`Chem.conserved` for a rollout of either rule set."""
    return simulation().conserved(r)


def owns(prompt: str) -> bool:
    """True for a question about a rollout or a table of either rule set, or a lesson of :data:`LESSONS`."""
    return simulation().owns(prompt) or LESSONS.owns(prompt)


def check(prompt: str, answer: str) -> Verdict:
    """Judge a question about a rollout or table of either rule set (replayed or looked up) or a
    lesson (recomputed from its inputs)."""
    if LESSONS.owns(prompt):
        return LESSONS.check(prompt, answer)
    return simulation().check(prompt, answer)
