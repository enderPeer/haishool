"""Level 13 of the ladder: the corrected chain, from the first second to a society, on one clock.

Round 6's ``world`` chained five toys in an order without times and with hand-offs a review found
wrong or arbitrary (``haishool/cosmos/world.py``, frozen). This level chains the eleven levels
that exist now, the first five under their ``rules=7``, and repairs the seams:

    nucleo        level 1, rules 7, **our** universe for every seed (the what-if universes stay
                  level 1's own): the hydrogen and helium the first minutes leave
    stars         level 7: earlier stars enrich the gas from the first stars until the cloud forms
                  (``formation_time``, drawn from the seed); replaces "3 % per generation"
    cloud         hand-off: the cloud's mass, spin and cooling
    collapse      level 2, rules 7: the cloud falls together; a star and a disc
    star          hand-off: the star's mass, light and **lifetime**, the zone, the disc's solids and gas
    planets       level 3, rules 7, with that star, those solids and that gas
    surface_k     hand-off: the k-th rocky planet in the temperate zone, innermost first: its light,
                  its kind of surface, its temperature and whether its water is liquid
    chemistry_k   level 4, rules 7: the surface of a planet with an ocean cools and is sparked
    replicators_k level 5, rules 7: the planet's vessel
    cells_k, bodies_k, senses_k, signals_k, society_k      levels 8 to 12

A world rollout's steps are these **eras**, in this order, planet after planet. An era is only
there if it happened: ``planets`` needs a star, solids and time; every era of a planet needs the
state below it (see "The ladder") and time on the star's clock.

**One clock.** Every era prints ``age``, the whole years since the Big Bang at which it starts,
and every simulated era ``duration``, the years its run stands for; the next era starts at
``age + duration``. The physical eras take their own units, and each is of the real order of
magnitude (all from memory): level 1 ends after 3000 s (printed as ``seconds``; that is 0 years;
the real light elements are made in the first 3 to 20 minutes, which those 3000 s cover); the
stars make metals from :data:`FIRST_STARS` = 250 million years on (level 7's time 0; the real
first stars: about 100 to 200 million years; 250 is the nearest point of level 7's grid of
0.25 Gyr); the collapse is level 2's 6 free-fall times of 428000 years, 2.6 million years (its
own conversion for a core of 1.5 solar masses in 0.1 pc; the chain gives every cloud that
density; a real core becomes a star with a disc in some 0.1 to 1 million years, so the toy is
slow by a few times); the planets are level 3's 50 million years (real: gas giants form within
the 1 to 10 million years the disc gas lasts, rocky planets in 10 to 100 million). The star's clock starts when
the collapse ends: ``lifetime`` = 10^10 years * mass / luminosity, to 3 digits (the sun 10^10; from
memory of the textbook scaling; it gives 4.1 * 10^9 years at 1.35 solar masses, real stars some
3 * 10^9, 1.3 * 10^8 at 5, real about 10^8, and 10^12 for the smallest red dwarfs, which in truth
burn even longer), and
``star_end = age + lifetime``. **Nothing runs past
star_end**: an era, or one more try of an era, that would end after it does not happen, and the
world's ``outcome`` is ``out_of_time``.

The years of the later eras are **toy calibration, not knowledge**: nobody knows how long a
planet waits for replicators, cells or language. :data:`SPAN` gives each level's run the years
the same step took on the earth, the only example there is, rounded so that a step of the level
is a round number of years (all from memory; Ga is 10^9 years ago): chemistry 100 million
(oceans by 4.4 Ga), replicators 400 million (80 steps of 5 million; life by 3.5 Ga for certain,
by 3.7 to 4.0 Ga on disputed evidence), cells 2160 million (60 steps of 36 million; cells with a
nucleus by about 1.7 Ga), bodies 1200 million (animals at 0.6 Ga), senses 120 million (eyes and
nerves by 0.52 Ga), signals 480 million (from the first eyes to speech, which arose some time in
the last few hundred thousand years; nobody knows when), society 20000 (level 12's own clock in
years; farming 12000 and writing 5000 years ago). Together with the collapse and the planets they
add up to 4.51 * 10^9 years, the earth's age within a percent; that is how they were chosen. They
are **toy rates**: one planet's dates, made the pace of every planet; how long replicators, cells,
bodies, senses or language take elsewhere, or whether they come at all, no one knows.

**Tries.** A run of a biological level is 300 generations or so in one vessel; a planet has many
vessels and much time. So each of the levels 5 to 12 is tried again while it has not reached
what the next level needs: at most :data:`MAX_TRIES` = 6 times (a toy cap on the cost of a
world, nothing more), each try a fresh run of the level (seed ``(seed * 10 + k) * 1000 + level
* 10 + try``, so no two levels, planets or worlds share a random stream; its free parameters
drawn anew, the hand-off the same) and each costing the level's span on the clock:
``duration = tries * span``. The era reports the try that got furthest (``kept``: the
first that reached what the next level needs, else the first that reached the level's lower
rung, else the last). A small star gives every try; a star like the sun 4 tries at cells; a
star above about 1.55 solar masses dies before the first cells are through, and one above
about 1.3 before the earth's own schedule of one try a level.

**The ladder** is defined once, :data:`LINKS` and :data:`LADDER`. The links, in order: ``metals``
(the cloud has some), ``star`` (the clump shines), ``planets``, ``temperate`` (a rocky planet in
the zone), ``water`` (one of them has a surface with liquid water), ``organics`` (level 4 left
precursors), ``replicators`` (level 5 ends with some), ``cells`` and ``complex_cells`` (level 8
ends ``cells`` or ``complex_cells``; ``complex_cells``), ``bodies`` (level 9 ends
``multicellular`` or ``animals_like``), ``senses`` and ``groups`` (level 10 ends ``sensing``,
``seeing`` or ``brained``; and its agents gain from staying near kin that call, ``group yes``),
``signals`` and ``language`` (level 11 ends ``calls``, ``lexicon`` or ``language``;
``language``), ``society`` and ``states`` (level 12 ends ``tribes``, ``villages``, ``chiefdoms``
or ``states``; ``states``). The next level runs only if the level below reached its upper link
(for level 9: ``animals_like``), so the links a world holds are always a prefix of the list. The
world's ``rung`` names the first missing link, ``no_metals no_star no_planets no_temperate
no_water no_organics sterile``, or else the last link held, ``replicators cells complex_cells
bodies senses groups signals language society states``. Of several planets the one that got
furthest counts (the inner one of equals). ``outcome`` is the rung, or ``out_of_time`` when the
star's death is what stopped the climb; ``by_now`` is the rung from the eras that are over
13.8 * 10^9 years after the Big Bang.

That bodies need complex cells, senses animals, signals groups and a society a language is the
chain's own rule (level 9 could run with simple cells, level 12 with a lexicon); it makes the
ladder a ladder. On the earth the complex forms came in this order (animals after cells with a
nucleus, language after animals that live in groups), but not every simpler one did: bacteria
made simple many-celled filaments before cells with a nucleus existed, and animals signal to
mates, rivals and predators without living in groups. It is no law.

**The hand-offs.** Each is a function of this module, takes the numbers the eras print (3
significant digits, counts and years exact) and is a lesson of :data:`LESSONS` (``q world7
predict star_mass cloud_mass 0 point 1 5 star_share 0 point 5 5 1. a 0 point 0 8 2 7.``), so a
world's hand-offs can be recomputed from its lines alone. Products of printed numbers are worked
out in exact decimals and a 5 rounds up. What each is, where it comes from, what is toy:

* ``helium_share``: level 7 starts from the helium level 1 prints; the traces count as hydrogen
  (:func:`haishool.evo.stars.handoff_in`).
* ``formation_year``, ``enrichment_years``: the cloud forms at ``formation_time`` (0.25 to 12 Gyr
  in steps of 0.25, drawn); level 7 runs from the first stars until then with the ``efficiency``
  drawn from 0.15, 0.2, 0.3, 0.45, 0.6 per Gyr (the cloud's neighbourhood; 0.3 is level 7's own
  default and gives the sun's 0.0135 at 9.25 Gyr). A cloud of 0.25 Gyr is metal-free. Metals,
  helium and the oxygen to iron ratio are level 7's (closed box with delayed iron); round 6's
  generations are gone.
* ``cooling``: 0.1 below the critical metallicity :data:`Z_CRIT` = 10^-5 (after Bromm and Loeb
  2003: about 10^-3.5 of the sun's; the toy rounds), else the seed's own draw, 0.2 to 0.6
  (turbulence differs from cloud to cloud; decay in about a free-fall time after Mac Low 1999).
  Level 2's ``cooling`` is a drag that takes random motion out of the cloud: one toy number for
  radiative cooling and the decay of turbulence together, not a cooling rate.
  Round 6's "0.1 + 20 Z" made metal-rich clouds form heavier stars for no reason.
* ``cloud_mass``: drawn from 0.1, 0.15, 0.2, 0.3, 0.5, 0.7, 1, 1.5, 2, 3, 5, 8 solar masses: a
  flat sample in the logarithm, **not the real mass function** (by the Kroupa 2001 law between
  0.08 and 100 solar masses 76 % of all stars are below 0.5, 18 % from 0.5 to 1.5, 5 % from
  1.5 to 8, 0.6 % above: :func:`kroupa_share`, the constants ``stars_*``). Metal-free gas cools
  only to about 200 K instead of 10 K, its Jeans mass is (200 / 10)^1.5 = 90 times larger (after
  Abel, Bryan and Norman 2002; Bromm, Coppi and Larson 2002): 90 times the drawn mass.
* ``star_mass`` = cloud_mass * star_share (level 2's largest bound clump; 0.3 to 0.74 there, inside
  the real 25 to 75 % of a core that end in the star, after Matzner and McKee 2000);
  ``fusion`` yes from 0.08 solar masses.
* ``luminosity``: level 3's piecewise law (0.23 M^2.3 below 0.43, M^4 to 2, 1.4 M^3.5 to 55,
  32000 M above; textbook fit, from memory there), 0 without fusion. ``lifetime`` as above.
  ``frost_line`` = 2.7 au * sqrt(L), ``hz_inner``, ``hz_outer``: level 3's flux limits.
* ``disc_solids`` = 26640 earth masses * disc_share * cloud_mass * metallicity, ``disc_gas`` the
  same with 1 - metallicity: 8 % (:data:`DISC_RETAINED`, toy) of what level 2 leaves in its disc
  stays as the disc of level 3, at 333000 earth masses a solar mass. That puts 2 to 10 % of the
  star's mass into the disc (real: about 1 %, up to 10 % when young, after Williams and Cieza
  2011); 8 % is chosen so that level 3 grows gas giants around some stars of the sun's make-up
  and around more of the metal-rich (the observed trend, after Fischer and Valenti 2005; the
  measured shares are in the table below). Round 6 gave every star the same 10000 * Z.
* ``flux`` = luminosity / orbit^2; ``zone`` (level 3's ``climate``); ``surface``: ``thin`` below
  0.3 earth masses (no air: Mars), ``enveloped`` when the planet kept disc gas or weighs more
  than 10 (level 3's bounds), else ``terran``; ``t_eq`` = 254.6 K * flux^(1/4) (albedo 0.3);
  ``greenhouse``: level 3's thermostat for a terran planet (33 K at the earth's flux, more
  further out), 5 K for a thin one (Mars; from memory), 0 for an enveloped one, where the
  temperature is that of the cloud tops; ``temperature`` = t_eq + greenhouse, a half up;
  ``water``: ice, liquid (273 to 373 K) or vapor by level 4's rule, ``none`` under an envelope.
  Round 6 put the earth at 311 K and froze 85 of 190 planets "in the zone". Here the zone, the
  temperature and level 3's own count of habitable planets are one rule: a terran planet in the
  zone is at 273 to 294 K, so liquid, and level 3's ``habitable`` equals the number of terran
  surfaces in every system (51 of 51 on seeds 1 to 60). The rung ``no_water`` therefore means
  only that every temperate rocky planet is thin (frozen) or enveloped; it is mostly the rung of
  metal-poor clouds, whose discs make planets of Mars's size.
* ``monomers`` = 100 * precursors: **the size of the vessel, not chemistry** (toy; nobody knows
  how much reduced carbon reached any pond). ``inflow`` = 20 monomers a step per spark of level
  4 (toy: the energy that made the precursors keeps making them; a vessel without sparks is
  closed). ``hydrolysis`` from the temperature (:func:`haishool.cosmos.life.hydrolysis_at`).
  Whether the planet's air was reducing (one vessel in five) and how many sparks it gets are
  level 4's own draws: nobody knows.
* ``fidelity`` = 1 - mu and ``nutrient`` = 12500 * inflow (level 8's hand-off);
  ``gene_length``: the length level 8 draws, halved until twice it fits under the error
  threshold ``l_max`` of the handed fidelity (toy: genes a compartment could not copy were not
  there to be enclosed; without this rule two worlds in five lose their genes at once for a
  reason that is a throw of dice, not a planet).
* ``efficiency`` from the cells' energy (level 9's hand-off); ``light`` = 200 * flux, the
  producers' capacity (200 is level 9's default; toy). ``brightness`` = flux, at most 1, the
  light level 10's eyes work with (toy). ``body_size``, ``predators`` (level 10's hand-off),
  ``neurons``, ``band`` (level 11's): the bodies, hunters, brains and groups of the level below.
* ``duration``, ``era_end``, ``in_time``: the clock.

What does **not** depend on anything below, and is drawn by each level from the try's seed: the
free parameters the era's ``drawn`` record lists (mutation rates, the meanings a group talks
about, the land of a society, ...). And two seams that stay open: the surface chemistry does
not depend on the cloud's make-up (level 4 cools its own fixed mixes; the temperature decides
ice, liquid or vapor and nothing else), and language here is learned and passed on, not
inherited: what evolved is the ears, the voice, the neurons and the group (level 10).

Lines (seed 85 verbatim, cut where marked ``...``; numbers as digits; one line each in the data).
Its star is a red dwarf of 0.0827 solar masses that shines for 10^12 years; its one temperate
planet needs three tries for complex cells and is a world of states 12.6 * 10^9 years after the
Big Bang:

    world7 seed 8 5 params. formation_time 3 point 2 5. efficiency 0 point 6. cloud_mass 0 point 1 5.
        spin 0 point 1 1. cooling 0 point 5 8. t_gas 2 point 5.
    world7 seed 8 5 timeline. eras nucleo stars cloud collapse star planets surface_1 chemistry_1 replicators_1
        cells_1 bodies_1 senses_1 signals_1 society_1. rung states. outcome states.
    world7 seed 8 5 era cloud. age 3 2 5 0 0 0 0 0 0 0. cloud_mass 0 point 1 5. spin 0 point 1 1. cooling 0 point 5 8.
    world7 seed 8 5 era star. age 3 2 5 2 5 6 8 0 0 0. fusion yes. star_mass 0 point 0 8 2 7.
        luminosity 7 point 4 5 e minus 4. lifetime 1 1 1 0 0 0 0 0 0 0 0 0 0. star_end 1 1 1 3 2 5 2 5 6 8 0 0 0.
        frost_line 0 point 0 7 3 7. ...
    world7 seed 8 5 era surface_1. age 3 3 0 2 5 6 8 0 0 0. orbit 0 point 0 3 6 5. mass 0 point 5 3 9. gas 0.
        flux 0 point 5 5 9. surface terran. t_eq 2 2 0. greenhouse 6 2 point 7. temperature 2 8 3. water liquid.
    world7 seed 8 5 era cells_1. age 3 8 0 2 5 6 8 0 0 0. duration 6 4 8 0 0 0 0 0 0 0. tries 3. kept 3.
        replicators 1 9 0 4. fidelity 0 point 9 9 9. nutrient 3 2 5 0 0 0 0. l_max 1 7 3 0. gene_length 2 0. cells 4 0 0.
        genome_length 1 2 8. complex 1. energy_per_cell 1 4 6. first_chromosome 3 5. result complex_cells.
    world7 seed 8 5 era cells_1 drawn. gene_types 6. gene_drawn 2 0. selfish 0 point 1. threshold 7 2.
        innovation 0 point 0 0 5.
    world7 seed 8 5 era senses_1. age 1 1 4 8 2 5 6 8 0 0 0. duration 1 2 0 0 0 0 0 0 0. tries 1. kept 1. body_size 2 5 6.
        body_types 7. predators 0 point 3. ... ears 1 point 9 9. neurons 6 3 point 7. talkers 0 point 9 4 5. ...
    world7 seed 8 5 era society_1. age 1 2 5 6 2 5 6 8 0 0 0. duration 1 2 0 0 0 0. tries 6. kept 6. ... technologies 1 0.
        technology_list fire stone_tools clothing boats pottery farming metal writing mathematics printing. ...
    q world7 seed 8 5 era cells_1 result. a complex_cells.
    q world7 seed 8 5 final rung. a states.
    q world7 seed 8 5 final by_now. a states.
    q world7 seed 8 5 final age_at_end. a 1 2 5 6 2 6 8 8 0 0 0.
    q world7 seed 8 5 why states. a metals star planets temperate water organics replicators cells complex_cells
        bodies senses groups signals language society states.
    q world7 seed 8 5 timeline. a nucleo stars cloud collapse star planets surface_1 chemistry_1 replicators_1
        cells_1 bodies_1 senses_1 signals_1 society_1.
    q world7 predict lifetime star_mass 0 point 0 8 2 7 luminosity 7 point 4 5 e minus 4. a 1 1 1 0 0 0 0 0 0 0 0 0 0.
    q world7 predict in_time age 3 8 0 2 5 6 8 0 0 0 span 2 1 6 0 0 0 0 0 0 0 star_end 1 1 1 3 2 5 2 5 6 8 0 0 0. a yes.

and, for a world that reached ``signals``, the language of its furthest planet, taken over from
level 11 (:func:`haishool.evo.signals.language_lines`) under the world's seed:

    world7 seed 8 5 lexicon. fire na. sky_danger le pe. predator ma. young bu la. small me pa pa. far lu pe.
        near ma su. big me ma.
    world7 seed 8 5 grammar. order property_first. compositional 1.
    q world7 seed 8 5 word predator. a ma.
    q world7 seed 8 5 say predator small. a me pa pa ma.
    q world7 seed 8 5 meaning me pa pa ma. a predator small.
    q world7 seed 8 5 say young near. a ma su bu la.
    q world7 seed 8 5 final sentence. a me pa pa ma.

The tables are seedless: ``q world7 constant max_tries. a 6.``, ``q world7 span cells. a 2 1 6 0
0 0 0 0 0 0.``, ``q world7 ladder after cells. a complex_cells.``.

**Gate** (:func:`conserved`). The chain is walked again from the seed and the parameters with
the stored level rollouts in place of fresh runs: every stored level must be the run the chain
asks for (seed and every parameter handed over or drawn), must pass its own level's
``conserved``, and no level may be missing or left over; the eras rebuilt from them, with every
hand-off recomputed from the printed numbers, must be the stored eras. On top: the mass ledger
(hydrogen + helium + metals = 1 within 1e-9 before and after level 7), the clock (each era
starts where the one before ends, no era ends after the star, a missing era or try is one that
would have), the ladder (the links held are a prefix; the rung never falls from era to era of a
planet) and the summary, which must follow from the eras. :func:`check` replays a seed's world
and compares: words, counts and years exactly, a measured number when it rounds to the three
digits printed; hand-off questions go through :data:`LESSONS` (4 digits, counts and words exact).

**Plausibility**, measured on this code over seeds 1 to 300 (:data:`CENSUS` holds the counts;
``tests/test_evo_world7.py`` holds the targets on seeds 1 to 40 and pins four whole worlds):

    every world     the gate holds; every line is dense, at most 128     300 of 300; 85 to 322 lines a world, 148
                    tokens and passes ``check``; every hand-off           on average, the longest 121 tokens
                    lesson gives what the next era prints
    one universe    the era nucleo is the same in every world             hydrogen 0.753, helium 0.247
    metals          0 only before the first metals; the sun's 0.0135      4 worlds metal-free (formation 0.25);
                    for efficiency 0.3 at 9.25 Gyr                        else 0.000162 to 0.0286
    stars           0.08 to 100 solar masses and brown dwarfs             0.0398 to 82.3; 41 do not shine, 109
                                                                          below 0.5, 74 to 1.5, 74 to 8, 2 above
    metal-free      one massive star, no solids, no planets               4 of 4: 3.73 to 82.3 solar masses
    planets         gas giants more often around metal-rich stars         0 of 130 systems below 0.008, 11 of 84
                                                                          from 0.008 to 0.016, 17 of 45 above;
                                                                          9 of 50 stars of 0.7 to 1.4 solar masses
    surfaces        terran planets in the zone have liquid water,         161 terran at 274 to 294 K, 23 thin at
                    thin ones are frozen                                  208 to 266 K, 1 enveloped
    the star's      no cells around a star above 1.55 solar masses,       heaviest star with a cells era 1.52,
    clock           no bodies above 1.37                                  with a bodies era 1.36; 69 worlds end
                                                                          out_of_time, 43 of them before a
                                                                          first try at cells
    the ladder      thinner with every rung                               see CENSUS: 143 worlds get a vessel,
                                                                          138 end with replicators, 50 with
                                                                          complex cells, 49 with bodies, 29
                                                                          with groups, 27 with signals, 5 with a
                                                                          language, 2 with a society beyond bands
    tries           more tries, more rungs; the cap is felt               cells: 50 of 111 vessels reach complex
                                                                          cells, 44 use all 6 tries; senses: 20
                                                                          of 48 use all 6; signals: 23 of 28;
                                                                          society: 5 of 5
    cost            a world in seconds (not the 2 s of one level: a      alone, seeds 1 to 40: 0.95 to 5.9 s,
                    world runs up to 11 levels, levels 5 to 12 up to      median 1.5 s, mean 2.0 s, 16 of 40
                    6 times each)                                         over 2 s; the collapse alone is 1.2 s
                                                                          of every world, a try of level 11
                                                                          0.4 s. Twelve at a time on a shared
                                                                          machine, seeds 1 to 300: 1.1 to
                                                                          11.6 s, median 2.6 s

What the census also shows, plainly: the upper rungs are rare (a language in 5 worlds of 300, a
society beyond bands in 2). Where a climb stops for good it is mostly the levels' own rules
meeting, not the clock: a vessel handed a fidelity of 0.95 reached complex cells in 0 of 69
trial runs; a world without predators cannot give groups (level 10's kin gain is 0 without
attacks); bodies of 64 to 512 cells carry at most 16 to 128 neurons (a quarter of the cells),
and level 11 gives a brain 1 word per 8 neurons, so 19 of 28 groups stop at calls; of the 5
groups with a language 3 stay bands (seed 29: 15 % talkers, success 0.10, bands; seed 85: 95 %
talkers, success 0.88, states). More worlds, not other rules, are the way to more of them.
Every rung occurs (seeds 1 to 300; in seeds 1 to 200 all but ``no_organics``, ``complex_cells``
and ``society``, which seeds 205, 219 and 207 reach).

The worlds that climb highest circle red dwarfs, because only the clock of a long-lived star
leaves room for six tries of a level (seed 85: 0.0827 solar masses). That is the toy's arithmetic,
not a finding: the chain knows nothing of the flares of red dwarfs or of planets that turn one
face to their star, which may make such worlds worse homes than the sun's.

Forced regimes, seeds 1 to 20 each (every gate holds): a cloud before the first metals
(``formation_time`` 0.25) makes one star of 5.2 to 379 solar masses and no planets, 20 of 20; a
cloud of 0.1 solar masses makes brown dwarfs, 20 of 20 ``no_star``; a cloud of 8 makes stars of
3 to 5.4 solar masses whose climb ends by ``sterile`` at most, 8 worlds ``out_of_time``; a cloud of
1.5 solar masses at 9.25 Gyr with efficiency 0.3 has the sun's 0.0136 and gas giants around 6
of 20 stars, at 12 Gyr with 0.6 (0.0286) around 18 of 20, at 1 Gyr with 0.15 (0.000494) around
none, and 16 of those 20 worlds end ``no_water``: their temperate planets are thin.

Reproducibility. The chain draws six numbers from ``random.Random(seed)`` and computes with
exact decimals, + - * / and sqrt, except where it calls a level's function (level 3's luminosity
law uses a power; a printed third digit could in principle differ between C libraries at a
rounding edge). Everything else is each level's own. The numbers were written with Python 3.11
and numpy 2.4; the four pinned worlds of the test file are the canary.
"""

from __future__ import annotations

import math
import random
import re
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, field
from decimal import ROUND_HALF_UP, Context, Decimal

from haishool.cosmos import Rollout, chem, gravity, life, nucleo, planets
from haishool.evo import Input, LessonGate, Rule, bodies, cells, senses, sig, signals, society, stars
from haishool.truth import Line, Verdict, is_finite, num, parse_num

SIM = "world7"

#: years since the Big Bang today (13.8 * 10^9, from memory)
NOW = 13_800_000_000
GYR = 1_000_000_000
#: the grid of level 7 (0.25 Gyr) in years
QUARTER = 250_000_000
#: years after the Big Bang at which level 7's time 0 is put: the first stars
FIRST_STARS = QUARTER
#: years of one free-fall time of level 2 (its T_FF_MYR = 0.428 for 1.5 solar masses in 0.1 pc)
T_FF_YEARS = int(round(gravity.T_FF_MYR * 1_000_000))
#: tries a biological level gets at most (a cap on the cost of a world; toy)
MAX_TRIES = 6
#: a training line is at most this long (tokens, a "." counted as one)
MAX_TOKENS = 128

#: years the whole run of a level stands for (see the module docstring: toy calibration on the earth)
SPAN: dict[str, int] = {
    "collapse": 6 * T_FF_YEARS,
    "planets": 50_000_000,
    "chemistry": 100_000_000,
    "replicators": 400_000_000,
    "cells": 2_160_000_000,
    "bodies": 1_200_000_000,
    "senses": 120_000_000,
    "signals": 480_000_000,
    "society": 20_000,
}


def kroupa_share(low: float, high: float) -> float:
    """Share of all stars between 0.08 and 100 solar masses that weigh ``low`` to ``high``, by the
    broken power law of Kroupa (2001, from memory): dN/dm ~ m^-1.3 below 0.5, m^-2.3 above.

    >>> [round(kroupa_share(a, b), 3) for a, b in ((0.08, 0.5), (0.5, 1.5), (1.5, 8), (8, 100))]
    [0.761, 0.182, 0.051, 0.006]
    """
    def count(a: float, b: float) -> float:
        a, b = max(a, 0.08), min(b, 100.0)
        total = 0.0
        if a < min(b, 0.5):
            total += (a ** -0.3 - min(b, 0.5) ** -0.3) / 0.3
        if b > max(a, 0.5):
            total += 0.5 * (max(a, 0.5) ** -1.3 - b ** -1.3) / 1.3
        return total
    return count(low, high) / count(0.08, 100.0)


#: every constant of the hand-offs, with where it comes from; the code reads ``value`` only
CONSTANTS: dict[str, dict[str, float | int | str]] = {
    "now": {"value": NOW, "notes": "years since the Big Bang today, 13.8e9 (from memory)"},
    "first_stars": {"value": FIRST_STARS, "notes": "toy: years at which level 7's time 0 is put; the real first "
                                                   "stars: about 1e8 to 2e8 years (from memory)"},
    "free_fall_years": {"value": T_FF_YEARS, "notes": "level 2's free-fall time for 1.5 solar masses in 0.1 pc; toy: "
                                                      "every cloud of the chain has that density"},
    "max_tries": {"value": MAX_TRIES, "notes": "toy: tries of a biological level at most; a cap on the cost of a world"},
    "z_crit": {"value": 1e-5, "notes": "critical metallicity below which gas cools badly and fragments into heavy "
                                       "clumps; toy rounding of about 10^-3.5 solar (after Bromm and Loeb 2003)"},
    "cooling_primordial": {"value": 0.1, "notes": "toy: level 2's cooling for gas below z_crit"},
    "jeans_factor": {"value": 90, "notes": "(200 K / 10 K)^1.5 rounded: Jeans mass of metal-free gas over that of "
                                           "cold gas (after Abel, Bryan and Norman 2002; from memory)"},
    "min_star": {"value": 0.08, "notes": "solar masses from which hydrogen burns (from memory)"},
    "sun_lifetime": {"value": 10 * GYR, "notes": "main-sequence lifetime of the sun in years, about 1e10 (from "
                                                 "memory); lifetime = this * mass / luminosity"},
    "disc_retained": {"value": 0.08, "notes": "toy: share of level 2's disc that is the disc of level 3; chosen so "
                                              "that some stars of the sun's make-up get a gas giant"},
    "earths_per_sun": {"value": 333000, "notes": "earth masses in a solar mass, about 333000 (from memory)"},
    "thin_mass": {"value": planets.HABITABLE_MASS[0], "notes": "earth masses below which a planet keeps no air: level "
                                                               "3's toy bound between Mars and Venus"},
    "envelope_mass": {"value": planets.HABITABLE_MASS[1], "notes": "earth masses above which a planet is taken to keep "
                                                                   "a thick envelope: level 3's toy bound"},
    "thin_greenhouse": {"value": 5.0, "notes": "K a thin air adds: Mars, about 5 K (from memory)"},
    "chem_start_k": {"value": 4600, "notes": "K at which a surface starts in level 4: above its 4536 K, where every "
                                             "bond is broken"},
    "chem_steps": {"value": 30, "notes": "toy: cooling steps of a surface, the middle of level 4's range"},
    "vessel_per_molecule": {"value": 100, "notes": "toy: monomers of level 5 per precursor molecule of level 4: the "
                                                   "size of the vessel, not a yield of chemistry"},
    "inflow_per_spark": {"value": 20, "notes": "toy: monomers per step of level 5 for each spark of level 4"},
    "gene_min": {"value": 20, "notes": "bases: the shortest gene of level 8's list"},
    "gene_margin": {"value": 2, "notes": "toy: the first genes are at most l_max over this"},
    "light_earth": {"value": 200, "notes": "toy: level 9's light (producers' capacity) under the earth's flux; its "
                                           "default"},
    "stars_below_half": {"value": round(kroupa_share(0.08, 0.5), 3),
                         "notes": "share of real stars below 0.5 solar masses, from the formula (Kroupa 2001 between "
                                  "0.08 and 100); not what this chain samples"},
    "stars_half_to_one_and_half": {"value": round(kroupa_share(0.5, 1.5), 3), "notes": "0.5 to 1.5, from the formula"},
    "stars_one_and_half_to_eight": {"value": round(kroupa_share(1.5, 8.0), 3), "notes": "1.5 to 8, from the formula"},
    "stars_above_eight": {"value": round(kroupa_share(8.0, 100.0), 4), "notes": "above 8, from the formula"},
}

Z_CRIT = float(CONSTANTS["z_crit"]["value"])
COOLING_PRIMORDIAL = float(CONSTANTS["cooling_primordial"]["value"])
JEANS_FACTOR = int(CONSTANTS["jeans_factor"]["value"])
MIN_STAR = float(CONSTANTS["min_star"]["value"])
SUN_LIFETIME = int(CONSTANTS["sun_lifetime"]["value"])
DISC_RETAINED = float(CONSTANTS["disc_retained"]["value"])
EARTHS_PER_SUN = int(CONSTANTS["earths_per_sun"]["value"])
THIN_MASS = float(CONSTANTS["thin_mass"]["value"])
ENVELOPE_MASS = float(CONSTANTS["envelope_mass"]["value"])
THIN_GREENHOUSE = float(CONSTANTS["thin_greenhouse"]["value"])
CHEM_START_K = int(CONSTANTS["chem_start_k"]["value"])
CHEM_STEPS = int(CONSTANTS["chem_steps"]["value"])
VESSEL = int(CONSTANTS["vessel_per_molecule"]["value"])
INFLOW_PER_SPARK = int(CONSTANTS["inflow_per_spark"]["value"])
GENE_MIN = int(CONSTANTS["gene_min"]["value"])
GENE_MARGIN = int(CONSTANTS["gene_margin"]["value"])
LIGHT_EARTH = int(CONSTANTS["light_earth"]["value"])

#: what random_params draws from, in its order
FORMATION_QUARTERS = (1, 48)
EFFICIENCIES = (0.15, 0.2, 0.3, 0.45, 0.6)
CLOUD_MASSES = (0.1, 0.15, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0)
SPIN_RANGE = (0.02, 0.30)
COOLING_RANGE = (0.2, 0.6)
T_GAS = (1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0)
GENE_LENGTHS = (20, 40, 80, 160, 320, 640)
#: surfaces in a system at most (the planet number is one digit of the seeds below)
MAX_SURFACES = 9

PARAM_KEYS = ("formation_time", "efficiency", "cloud_mass", "spin", "cooling", "t_gas")
#: the sizes of the sub-simulations and the cap on the tries; ``random_params`` draws the rest
DEFAULTS: dict[str, int] = {"particles": gravity.N, "atoms": 20000, "max_tries": MAX_TRIES}

#: the links of the chain in order, and the rungs: the ladder is defined here and nowhere else
LINKS = ("metals", "star", "planets", "temperate", "water", "organics", "replicators", "cells", "complex_cells",
         "bodies", "senses", "groups", "signals", "language", "society", "states")
#: links before a planet's own climb
LOWER = 6
#: the rung of a world that holds ``i`` links: the first missing link by its "no" name, then the last link held
LADDER = ("no_metals", "no_star", "no_planets", "no_temperate", "no_water", "no_organics", "sterile",
          *LINKS[LOWER:])
OUT_OF_TIME = "out_of_time"
OUTCOMES = (*LADDER, OUT_OF_TIME)

FIXED_ERAS = ("nucleo", "stars", "cloud", "collapse", "star")
PLANET_ERAS = ("surface", "chemistry", "replicators", "cells", "bodies", "senses", "signals", "society")
#: era -> the metrics of its record line, in order
ERAS: dict[str, tuple[str, ...]] = {
    "nucleo": ("age", "seconds", "hydrogen", "helium", "traces"),
    "stars": ("age", "duration", "efficiency", "gas_fraction", "hydrogen", "helium", "metallicity", "oxygen", "carbon",
              "iron"),
    "cloud": ("age", "cloud_mass", "spin", "cooling"),
    "collapse": ("age", "duration", "stage", "clumps", "star_share", "companions", "disc_share", "disc_radius",
                 "star_time"),
    "star": ("age", "fusion", "star_mass", "luminosity", "lifetime", "star_end", "frost_line", "hz_inner", "hz_outer",
             "disc_solids", "disc_gas"),
    "planets": ("age", "duration", "t_gas", "planets", "rocky", "super_earths", "icy", "ice_giants", "gas_giants",
                "largest_mass", "temperate"),
    "surface": ("age", "orbit", "mass", "gas", "flux", "surface", "t_eq", "greenhouse", "temperature", "water"),
    "chemistry": ("age", "duration", "mix", "sparks", "h2o", "organic", "precursors"),
    "replicators": ("age", "duration", "tries", "kept", "monomers", "inflow", "hydrolysis", "mu",
                    "first_replicator_step", "peak_replicators", "replicators", "lineages", "max_length", "result"),
    "cells": ("age", "duration", "tries", "kept", "replicators", "fidelity", "nutrient", "l_max", "gene_length",
              "cells", "genome_length", "complex", "energy_per_cell", "first_chromosome", "result"),
    "bodies": ("age", "duration", "tries", "kept", "light", "transfer", "species", "max_size", "cell_types",
               "consumer_size", "consumer_types", "trophic_levels", "air", "predator_share", "result"),
    "senses": ("age", "duration", "tries", "kept", "body_size", "body_types", "predators", "air", "brightness",
               "agents", "eyes", "ears", "neurons", "talkers", "predator_pressure", "kin_gain", "group", "result"),
    "signals": ("age", "duration", "tries", "kept", "ears_voice", "brain", "pressure", "band", "split", "capacity",
                "success", "words", "meanings", "compositional", "order", "word_length", "result"),
    "society": ("age", "duration", "tries", "kept", "understood", "lexicon", "grammar", "people", "groups",
                "cooperation", "technologies", "technology_list", "farming", "writing", "hierarchy", "first_farming",
                "first_writing", "result"),
}
SUMMARY_KEYS = ("hydrogen", "helium", "metallicity", "formation_time", "star_mass", "star_lifetime", "planets",
                "temperate", "oceans", "planet", "rung", "outcome", "by_now", "age_at_end", "eras", "technologies",
                "words", "sentence", "meaning")

KEYS: dict[str, str] = {
    "formation_time": "Gyr after the Big Bang at which the cloud forms, 0.25 to 12 in steps of 0.25 (parameter)",
    "efficiency": "level 7's star formation efficiency per Gyr where the cloud forms (parameter)",
    "cloud_mass": "mass of the cloud in solar masses: the draw (parameter), 90 times it without metals (era cloud)",
    "spin": "rotational over potential energy of the cloud, 0.02 to 0.3 (parameter; level 2's spin)",
    "cooling": "level 2's cooling: the draw 0.2 to 0.6 (parameter), 0.1 below the critical metallicity (era cloud)",
    "t_gas": "Myr at which the disc gas is gone (parameter; level 3's t_gas)",
    "age": "whole years since the Big Bang at which the era starts",
    "duration": "whole years the era's run stands for: its span, times its tries for a level that is tried again",
    "seconds": "seconds of level 1's run, 3000: less than a year, so the era nucleo has no duration in years",
    "tries": "runs of the level made for this era, 1 to max_tries, each a fresh seed and each costing the span",
    "kept": "the try the era reports: the first that reached what the next level needs, else the first that "
            "reached the level's lower rung, else the last",
    "result": "the level's own word for how the kept try ended (its outcome)",
    "hydrogen": "mass fraction of hydrogen: after the first minutes (era nucleo), of the cloud (era stars, final)",
    "helium": "mass fraction of helium: after the first minutes (era nucleo), of the cloud (era stars, final)",
    "traces": "mass fraction of deuterium, helium-3 and lithium-7 after the first minutes",
    "gas_fraction": "share of level 7's box that is still gas when the cloud forms",
    "metallicity": "mass fraction of the cloud in everything heavier than helium (level 7)",
    "oxygen": "mass fraction of oxygen in the cloud",
    "carbon": "mass fraction of carbon in the cloud",
    "iron": "mass fraction of iron in the cloud",
    "stage": "level 2's stage word at its end: cloud, contracting, disc, fragmenting, star or multiple",
    "clumps": "bound clumps at the end of level 2 (count)",
    "star_share": "share of the cloud's mass in the largest bound clump at the end of level 2: the star",
    "companions": "other bound clumps (count)",
    "disc_share": "share of the cloud's mass in the disc around the star at the end of level 2",
    "disc_radius": "radius of that disc in level 2's units (0.1 pc)",
    "star_time": "free-fall times after which the largest clump first holds a quarter of the cloud, or never",
    "fusion": "yes when the star reaches 0.08 solar masses and shines, else no",
    "star_mass": "mass of the star in solar masses: cloud_mass * star_share, to 3 digits, a 5 rounding up",
    "luminosity": "light of the star in solar luminosities: level 3's law, 0 without fusion",
    "lifetime": "years the star shines: 10^10 * star_mass / luminosity, to 3 digits (0 without fusion)",
    "star_end": "years since the Big Bang at which the star dies: age + lifetime; nothing runs past it",
    "frost_line": "au beyond which ice is solid: 2.7 * sqrt(luminosity)",
    "hz_inner": "inner edge of the temperate zone in au (level 3: flux 1.107)",
    "hz_outer": "outer edge of the temperate zone in au (level 3: flux 0.356)",
    "disc_solids": "solids of the disc in earth masses: 26640 * disc_share * cloud_mass * metallicity",
    "disc_gas": "gas of the disc in earth masses: 26640 * disc_share * cloud_mass * (1 - metallicity)",
    "planets": "planets at the end of level 3 (count; final: 0 when it did not run)",
    "rocky": "rocky planets (count)",
    "super_earths": "super-earths (count)",
    "icy": "small icy planets (count)",
    "ice_giants": "ice giants (count)",
    "gas_giants": "gas giants (count)",
    "largest_mass": "mass of the heaviest planet in earth masses",
    "temperate": "rocky planets whose flux is temperate (count): each has a surface era",
    "orbit": "orbit of the planet in au",
    "mass": "mass of the planet in earth masses",
    "gas": "disc gas the planet kept, in earth masses",
    "flux": "starlight at the planet in units of what the earth receives: luminosity / orbit^2",
    "surface": "thin (below 0.3 earth masses), enveloped (kept gas, or above 10 earth masses) or terran",
    "t_eq": "K of the planet without greenhouse at albedo 0.3: 254.6 * flux^(1/4)",
    "greenhouse": "K the air adds: level 3's thermostat (terran), 5 (thin), 0 (enveloped: cloud tops)",
    "temperature": "whole K: t_eq + greenhouse, a half up. It decides ice, liquid or vapor and nothing else",
    "water": "ice, liquid (273 to 373 K), vapor, or none under an envelope",
    "mix": "level 4's element mix of the surface: ocean, or reducing for one vessel in five (level 4's own draw)",
    "sparks": "sparks after the cooling (level 4's own draw, 0 to 20)",
    "h2o": "water molecules at the end of level 4 (count)",
    "organic": "molecules with a carbon-hydrogen bond at the end of level 4 (count)",
    "precursors": "of those, the ones with a C-C, C-N or C-O bond (count): what the vessel is filled from",
    "monomers": "free monomers level 5 starts with: 100 * precursors (the size of the vessel; toy)",
    "inflow": "monomers per step entering level 5: 20 * sparks (toy)",
    "hydrolysis": "level 5's chance per bond and step that water breaks it, from the temperature",
    "mu": "level 5's chance that a copied letter is wrong (drawn by the kept try)",
    "first_replicator_step": "step of level 5 at which the first replicator appeared, or never",
    "peak_replicators": "most replicators at any step of level 5 (count)",
    "replicators": "replicators at the end of level 5 (count): what level 8 encloses",
    "lineages": "lineages of replicators at the end of level 5 (count)",
    "max_length": "longest replicator that could keep itself in level 5 (letters), 0 when none can",
    "fidelity": "level 8's chance that a base is copied right: 1 - mu",
    "nutrient": "bases per generation entering level 8: 12500 * inflow",
    "l_max": "level 8's error threshold for a protocell with the handed fidelity (bases)",
    "gene_length": "bases of a gene: level 8's draw, halved until twice it is at most l_max (not below 20)",
    "cells": "cells at the end of level 8 (count)",
    "genome_length": "mean bases of a cell's distinct genes at the end of level 8",
    "complex": "share of the cells with an engulfed partner",
    "energy_per_cell": "energy of a cell in level 8's units, 10 without a working partner",
    "first_chromosome": "generation of level 8 at which half the cells had linked genes, or never",
    "light": "level 9's light, the producers' capacity in biomass units: 200 * flux (toy)",
    "transfer": "level 9's transfer efficiency, from the cells' energy: 0.05 at 10 units, 0.2 at 160",
    "species": "lineages at the end of level 9 (count)",
    "max_size": "cells of the largest common body (count)",
    "cell_types": "most cell types among common bodies (count)",
    "consumer_size": "cells of the largest common grazer or predator (count; 0 without any)",
    "consumer_types": "most cell types among common grazers and predators (count)",
    "trophic_levels": "established levels of the food web, 0 to 4",
    "air": "oxygen in the air in today's levels: at the end of level 9 (era bodies); the same, at most 1, as "
           "level 10 takes it (era senses)",
    "predator_share": "share of all biomass in predators",
    "body_size": "cells of the bodies level 10 gives sensors to: consumer_size, or max_size without consumers",
    "body_types": "their kinds of cell: consumer_types, or cell_types without consumers",
    "predators": "level 10's predator density: 3 * predator_share, at most 0.3; 0 below 3 trophic levels",
    "brightness": "level 10's light, 0 dark to 1: the flux, at most 1 (toy)",
    "agents": "agents at the end of level 10 (count)",
    "eyes": "mean eye level 0 to 5",
    "ears": "mean ear level 0 to 5",
    "neurons": "mean neurons per agent at the end of level 10",
    "talkers": "share of the agents with ears and a voice",
    "predator_pressure": "chance to be attacked in a generation at the end of level 10",
    "kin_gain": "share of the death risk that warning calls of kin could remove",
    "group": "yes when kin_gain is at least 0.01: the agents gain from staying near kin",
    "ears_voice": "level 11's share of talkers: talkers",
    "brain": "level 11's neurons per agent: neurons, rounded",
    "pressure": "level 11's predator pressure: predator_pressure",
    "band": "agents of level 11's first group: at most 60 of the agents",
    "split": "1 when the agents form two bands that seldom meet (120 or more of them), else 0",
    "capacity": "words a brain holds in level 11: brain // 8",
    "success": "expected accuracy when one agent tells another a meaning, at the end of level 11",
    "words": "distinct words in use at the end of level 11 (count; final: of the furthest planet, 0 without)",
    "meanings": "meanings with a settled utterance (count)",
    "compositional": "share of the taught pairs the language says by its rule",
    "order": "thing_first, property_first or none",
    "word_length": "mean syllables of a word",
    "understood": "level 12's success of communication: success",
    "lexicon": "level 12's words: words",
    "grammar": "level 12's compositional share: compositional",
    "people": "people at the end of level 12 (count)",
    "groups": "groups at the end of level 12 (count)",
    "cooperation": "share of cooperators at the end of level 12",
    "technologies": "technologies at the end of level 12 (count; final: of the furthest planet, 0 without)",
    "technology_list": "their names in order, or none",
    "farming": "yes or no",
    "writing": "yes or no",
    "hierarchy": "levels of command in the largest group (count)",
    "first_farming": "year of level 12 at which farming began, or never",
    "first_writing": "year of level 12 at which writing began, or never",
    "star_lifetime": "the star's lifetime in years (0 without fusion)",
    "oceans": "planets with a terran surface and liquid water (count)",
    "planet": "number of the planet that got furthest (the inner one of equals), 0 when none has a chemistry era",
    "rung": "the first missing link or the last link held: " + " ".join(LADDER),
    "outcome": "the rung, or out_of_time when the star's death stopped the climb",
    "by_now": "the rung from the eras that are over 13.8e9 years after the Big Bang",
    "age_at_end": "years since the Big Bang at which the last era of the furthest planet (else of the world) ends",
    "eras": "number of eras; in the timeline line: their names",
    "sentence": "an utterance of the furthest planet's language (the first pair it can say, else its first word), "
                "or none",
    "meaning": "what that utterance means, or none",
    "why": "the links of the chain that held, of: " + " ".join(LINKS) + " (nothing when none did)",
    "constant": "a constant of the hand-offs, by name (see CONSTANTS)",
    "span": "years a level's run stands for (see SPAN): toy calibration on the earth's dates",
    "ladder": "the rung after a rung (none after the last)",
}


# ---------------------------------------------------------------------------------------------
# numbers as the lines print them


#: exact decimal arithmetic for the hand-offs, whatever the caller's decimal context is
_EXACT = Context(prec=60, rounding=ROUND_HALF_UP)


def _dec(x: float | int) -> Decimal:
    """A number as the decimal it prints as (``repr`` is the shortest text that gives the float back)."""
    return Decimal(x) if isinstance(x, int) else Decimal(repr(float(x)))


def _half_up(d: Decimal, digits: int = 3) -> Decimal:
    """``d`` to ``digits`` significant digits, a 5 rounding up: the school rule, on exact decimals.

    >>> [float(_half_up(Decimal(t))) for t in ("1.0095", "0.5865", "0.99951", "0", "1175000000000")]
    [1.01, 0.587, 1.0, 0.0, 1180000000000.0]
    """
    if d == 0:
        return Decimal(0)
    return d.quantize(Decimal((0, (1,), d.adjusted() - digits + 1)), rounding=ROUND_HALF_UP, context=_EXACT)


def _product(*factors: float | int) -> Decimal:
    out = Decimal(1)
    for f in factors:
        out = _EXACT.multiply(out, _dec(f))
    return out


def dense_value(value: float | int | str) -> str:
    """A metric as it stands in a line: a word, an exact whole number, or a number to 3 digits.

    >>> dense_value(0.01478), dense_value(9002568000), dense_value(288), dense_value("terran")
    ('0 point 0 1 4 8', '9 0 0 2 5 6 8 0 0 0', '2 8 8', 'terran')
    """
    if isinstance(value, str):
        return value
    if isinstance(value, bool):
        raise TypeError("a metric is a word or a number, not a bool")
    if isinstance(value, int):
        return num(value)
    return num(sig(float(value)))


def table_value(value: float | int | str) -> str:
    """A parameter or a constant as it was put in (up to 6 digits)."""
    if isinstance(value, (str, int)):
        return dense_value(value)
    return num(float(value), sig=6)


def tokens(text: str) -> int:
    """The length of a line as the trainer counts it."""
    return len(text.replace(".", " . ").split())


def _shown(value: float | int | str) -> float | int | str:
    """A level's number as an era holds it: floats to 3 significant digits, the rest as it is."""
    if isinstance(value, bool):
        raise TypeError("a metric is a word or a number, not a bool")
    return sig(float(value)) if isinstance(value, float) else value


# ---------------------------------------------------------------------------------------------
# the hand-offs: plain functions of the numbers the eras print. Each is a lesson (RULES).


def helium_share(hydrogen: float, helium: float) -> float:
    """Helium mass fraction of the gas level 7 starts from, given what level 1 prints: the traces
    count as hydrogen (level 7's own hand-off).

    >>> helium_share(0.753, 0.247)
    0.247
    """
    return stars.handoff_in({"hydrogen": hydrogen, "helium": helium})["helium"]


def first_gas(hydrogen: float, helium: float) -> dict[str, float]:
    """The two parameters level 7 starts from: :func:`helium_share`, and the rest as hydrogen."""
    return {"hydrogen": stars.handoff_in({"hydrogen": hydrogen, "helium": helium})["hydrogen"],
            "helium": helium_share(hydrogen, helium)}


def formation_year(formation_time: float) -> int:
    """Years since the Big Bang at which the cloud forms (``formation_time`` in Gyr, a multiple of 0.25).

    >>> formation_year(9.25)
    9250000000
    """
    return int(round(formation_time * 4)) * QUARTER


def enrichment_years(formation_time: float) -> int:
    """Years earlier stars had to enrich the gas: from the first stars to the cloud."""
    return formation_year(formation_time) - FIRST_STARS


def cooling_of(metallicity: float, drawn: float) -> float:
    """Level 2's cooling: 0.1 below the critical metallicity, else the cloud's own draw."""
    return COOLING_PRIMORDIAL if metallicity < Z_CRIT else drawn


def cloud_mass_of(drawn: float, metallicity: float) -> float:
    """Solar masses of the cloud: the draw, 90 times it below the critical metallicity.

    >>> cloud_mass_of(0.7, 0.0135), cloud_mass_of(0.7, 0.0)
    (0.7, 63.0)
    """
    return float(_product(JEANS_FACTOR, drawn)) if metallicity < Z_CRIT else drawn


def star_mass_of(cloud_mass: float, star_share: float) -> float:
    """``cloud_mass * star_share`` to 3 digits, in exact decimals, a 5 rounding up.

    >>> star_mass_of(1.5, 0.59), star_mass_of(1.5, 0.453), star_mass_of(63.0, 0.414)
    (0.885, 0.68, 26.1)
    """
    return float(_half_up(_product(cloud_mass, star_share)))


def fusion_of(star_mass: float) -> str:
    return "yes" if star_mass >= MIN_STAR else "no"


def luminosity_of(star_mass: float) -> float:
    """Solar luminosities, to 3 digits: level 3's law, 0 below 0.08 solar masses.

    >>> luminosity_of(1.0), luminosity_of(0.5), luminosity_of(0.05)
    (1.0, 0.0625, 0.0)
    """
    return sig(planets.luminosity7(star_mass)) if star_mass >= MIN_STAR else 0.0


def lifetime_of(star_mass: float, luminosity: float) -> int:
    """Whole years on the main sequence: 10^10 * mass / luminosity, to 3 digits (0 without light).

    >>> lifetime_of(1.0, 1.0), lifetime_of(1.5, 5.06), lifetime_of(0.05, 0.0)
    (10000000000, 2960000000, 0)
    """
    if luminosity <= 0:
        return 0
    return int(_half_up(_EXACT.divide(_product(SUN_LIFETIME, star_mass), _dec(luminosity))))


def frost_line_of(luminosity: float) -> float:
    return sig(planets.FROST_AU * math.sqrt(luminosity))


def hz_inner_of(luminosity: float) -> float:
    return sig(planets.hz_inner(luminosity))


def hz_outer_of(luminosity: float) -> float:
    return sig(planets.hz_outer(luminosity))


def disc_solids_of(disc_share: float, cloud_mass: float, metallicity: float) -> float:
    """Earth masses of solids in level 3's disc, to 3 digits.

    >>> disc_solids_of(0.3, 1.8, 0.0135), disc_solids_of(0.3, 1.8, 0.0)
    (194.0, 0.0)
    """
    return float(_half_up(_product(EARTHS_PER_SUN, DISC_RETAINED, disc_share, cloud_mass, metallicity)))


def disc_gas_of(disc_share: float, cloud_mass: float, metallicity: float) -> float:
    """Earth masses of gas in level 3's disc, to 3 digits."""
    rest = _EXACT.subtract(Decimal(1), _dec(metallicity))
    return float(_half_up(_EXACT.multiply(_product(EARTHS_PER_SUN, DISC_RETAINED, disc_share, cloud_mass), rest)))


def flux_of(luminosity: float, orbit: float) -> float:
    return sig(planets.flux(luminosity, orbit))


def zone_of(flux: float) -> str:
    """Level 3's ``climate``: runaway, temperate or frozen."""
    return planets.climate(flux)


def surface_of(mass: float, gas: float) -> str:
    """``thin`` below 0.3 earth masses, ``enveloped`` with disc gas or above 10, else ``terran``.

    >>> surface_of(0.107, 0.0), surface_of(1.0, 0.0), surface_of(4.0, 0.2), surface_of(12.0, 0.0)
    ('thin', 'terran', 'enveloped', 'enveloped')
    """
    if mass < THIN_MASS:
        return "thin"
    return "enveloped" if gas > 0 or mass > ENVELOPE_MASS else "terran"


def t_eq_of(flux: float) -> float:
    return sig(planets.equilibrium_temperature(flux))


def greenhouse_of(flux: float, mass: float, gas: float) -> float:
    """K the air adds, by the kind of surface (see the module docstring).

    >>> greenhouse_of(1.0, 1.0, 0.0), greenhouse_of(0.431, 0.107, 0.0), greenhouse_of(0.5, 12.0, 0.0)
    (33.0, 5.0, 0.0)
    """
    kind = surface_of(mass, gas)
    if kind == "thin":
        return THIN_GREENHOUSE
    return sig(planets.greenhouse(flux)) if kind == "terran" else 0.0


def temperature_of(flux: float, greenhouse: float) -> int:
    """Whole K: equilibrium temperature plus greenhouse, a half rounds up.

    >>> temperature_of(1.0, 33.0), temperature_of(0.431, 5.0)
    (288, 211)
    """
    return math.floor(planets.equilibrium_temperature(flux) + greenhouse + 0.5)


def water_of(temperature: int, mass: float, gas: float) -> str:
    """``ice``, ``liquid`` or ``vapor`` by level 4's rule; ``none`` under an envelope."""
    return "none" if surface_of(mass, gas) == "enveloped" else chem.water_state(temperature)


def monomers_of(precursors: int) -> int:
    return VESSEL * precursors


def inflow_of(sparks: int) -> int:
    return INFLOW_PER_SPARK * sparks


def hydrolysis_of(temperature: int) -> float:
    return life.hydrolysis_at(temperature)


def fidelity_of(mu: float) -> float:
    return cells.handoff_in({"mu": mu})["fidelity"]


def nutrient_of(inflow: int) -> int:
    return cells.handoff_in({"inflow": inflow})["inflow"]


def gene_length_of(drawn: int, l_max: float) -> int:
    """The drawn length, halved until twice it is at most ``l_max`` (never below 20 bases).

    >>> gene_length_of(640, 896.0), gene_length_of(80, 896.0), gene_length_of(320, 35.8)
    (320, 80, 20)
    """
    length = int(drawn)
    while length > GENE_MIN and GENE_MARGIN * length > l_max:
        length //= 2
    return length


def efficiency_of(energy_per_cell: float) -> float:
    return bodies.handoff_in({"outcome": "complex_cells", "energy_per_cell": energy_per_cell})["efficiency"]


def light_of(flux: float) -> float:
    """Level 9's light: 200 * flux, a whole number, a half up.

    >>> light_of(1.0), light_of(0.356)
    (200.0, 71.0)
    """
    return float(_product(LIGHT_EARTH, flux).quantize(Decimal(1), rounding=ROUND_HALF_UP, context=_EXACT))


def brightness_of(flux: float) -> float:
    return min(1.0, flux)


def _senses_handed(era: dict) -> dict:
    """Level 10's own hand-off from a bodies era (``handoff_out`` of level 9, ``handoff_in`` of level 10)."""
    summary = {key: era[key] for key in ("consumer_size", "max_size", "consumer_types", "cell_types", "trophic_levels",
                                         "predator_share", "species")}
    return senses.handoff_in(bodies.handoff_out({**summary, "oxygen": era["air"]}))


def body_size_of(consumer_size: int, max_size: int) -> int:
    return _senses_handed({"consumer_size": consumer_size, "max_size": max_size, "consumer_types": 1, "cell_types": 1,
                           "trophic_levels": 0, "predator_share": 0.0, "air": 1.0, "species": 1})["body_size"]


def predators_of(predator_share: float, trophic_levels: int) -> float:
    return _senses_handed({"consumer_size": 1, "max_size": 1, "consumer_types": 1, "cell_types": 1,
                           "trophic_levels": trophic_levels, "predator_share": predator_share, "air": 1.0,
                           "species": 1})["predators"]


def neurons_of(neurons: float) -> int:
    return signals.handoff_in({"neurons": neurons})["neurons"]


def band_of(agents: int) -> int:
    return signals.handoff_in({"population": agents, "group": "yes"})["population"]


def duration_of(tries: int, span: int) -> int:
    return tries * span


def era_end(age: int, duration: int) -> int:
    return age + duration


def in_time(age: int, span: int, star_end: int) -> str:
    """``yes`` when a run of ``span`` years that starts at ``age`` is over before the star dies."""
    return "yes" if age + span <= star_end else "no"


def planet_seed(seed: int, k: int) -> int:
    """The seed of planet ``k`` of a world: its level 4 runs with it."""
    return seed * 10 + k


def try_seed(seed: int, k: int, level: int, j: int) -> int:
    """The seed of try ``j`` of level number ``level`` (5, 8 to 12) on planet ``k``: one stream per
    world, planet, level and try.

    >>> try_seed(29, 1, 8, 2)
    291082
    """
    return planet_seed(seed, k) * 1000 + level * 10 + j


# ---------------------------------------------------------------------------------------------
# the levels of a planet's climb


@dataclass(frozen=True)
class Level:
    """One of the levels 5 to 12 as the chain uses it."""
    name: str
    #: the level's number on the ladder (it is part of the seed of its tries)
    number: int
    module: object
    #: keyword arguments of the level's run besides its parameters
    extra: dict
    #: the eras of the planet so far (by base name) -> the parameters handed over
    handed: Callable[[dict], dict]
    #: (era key, handed parameter) for the era's record
    shown: tuple[tuple[str, str], ...]
    #: (era key, "summary" | "params" | "derived", key there) for the era's record
    results: tuple[tuple[str, str, str], ...]
    #: era -> the level reached its lower rung; era -> it reached what the next level needs
    lower: Callable[[dict], bool]
    upper: Callable[[dict], bool]
    #: the links of the ladder this level decides: the lower rung and, if it has a name, the upper
    links: tuple[str, ...]

    def draws(self, seed: int) -> dict:
        """The level's own random parameters for a try."""
        if self.name == "replicators":
            return life.random_params(random.Random(seed), rules=7)
        return self.module.random_params(random.Random(seed))

    def params(self, seed: int, handed: dict) -> dict:
        """What a try runs with: the level's draws, the hand-off on top, and the chain's rule for
        the first genes."""
        p = {**self.draws(seed), **handed}
        if self.name == "cells":
            p["gene_length"] = gene_length_of(p["gene_length"], cells.derived(p)["l_max"])
        return p

    def fields(self, r: Rollout) -> dict:
        source = {"summary": r.summary, "params": r.params, "derived": getattr(r, "derived", {})}
        return {key: _shown(source[where][name]) for key, where, name in self.results}


def _life_handed(eras: dict) -> dict:
    return {"monomers": monomers_of(eras["chemistry"]["precursors"]), "inflow": inflow_of(eras["chemistry"]["sparks"]),
            "hydrolysis": hydrolysis_of(eras["surface"]["temperature"])}


def _cells_handed(eras: dict) -> dict:
    below = eras["replicators"]
    return {"replicators": cells.handoff_in({"replicators": below["replicators"]})["replicators"],
            "fidelity": fidelity_of(below["mu"]), "inflow": nutrient_of(below["inflow"])}


def _bodies_handed(eras: dict) -> dict:
    return {"complex_cells": "yes", "efficiency": efficiency_of(eras["cells"]["energy_per_cell"]),
            "light": light_of(eras["surface"]["flux"])}


def _senses_given(eras: dict) -> dict:
    below = eras["bodies"]
    handed = _senses_handed(below)
    return {"body_size": body_size_of(below["consumer_size"], below["max_size"]), "cell_types": handed["cell_types"],
            "predators": predators_of(below["predator_share"], below["trophic_levels"]), "oxygen": handed["oxygen"],
            "light": brightness_of(eras["surface"]["flux"])}


def _signals_handed(eras: dict) -> dict:
    below = eras["senses"]
    handed = signals.handoff_in({**{key: below[key] for key in ("talkers", "neurons", "predator_pressure", "group")},
                                 "population": below["agents"]})
    return {**handed, "neurons": neurons_of(below["neurons"]), "population": band_of(below["agents"])}


def _society_handed(eras: dict) -> dict:
    below = eras["signals"]
    return society.handoff_in({key: below[key] for key in ("success", "words", "compositional")})


def _senses_reached(e: dict) -> bool:
    return e["result"] in ("sensing", "seeing", "brained")


LEVELS: tuple[Level, ...] = (
    Level("replicators", 5, life, {"rules": 7}, _life_handed,
          (("monomers", "monomers"), ("inflow", "inflow"), ("hydrolysis", "hydrolysis")),
          (("mu", "params", "mu"), ("first_replicator_step", "summary", "first_replicator_step"),
           ("peak_replicators", "summary", "peak_replicators"), ("replicators", "summary", "replicators"),
           ("lineages", "summary", "lineages"), ("max_length", "summary", "max_length"),
           ("result", "summary", "outcome")),
          lambda e: e["replicators"] > 0, lambda e: e["replicators"] > 0,
          ("replicators",)),
    Level("cells", 8, cells, {}, _cells_handed,
          (("replicators", "replicators"), ("fidelity", "fidelity"), ("nutrient", "inflow")),
          (("l_max", "derived", "l_max"), ("gene_length", "params", "gene_length"), ("cells", "summary", "cells"),
           ("genome_length", "summary", "genome_length"), ("complex", "summary", "complex"),
           ("energy_per_cell", "summary", "energy_per_cell"), ("first_chromosome", "summary", "first_chromosome"),
           ("result", "summary", "outcome")),
          lambda e: e["result"] in ("cells", "complex_cells"), lambda e: e["result"] == "complex_cells",
          ("cells", "complex_cells")),
    Level("bodies", 9, bodies, {}, _bodies_handed,
          (("light", "light"), ("transfer", "efficiency")),
          (("species", "summary", "species"), ("max_size", "summary", "max_size"),
           ("cell_types", "summary", "cell_types"), ("consumer_size", "summary", "consumer_size"),
           ("consumer_types", "summary", "consumer_types"), ("trophic_levels", "summary", "trophic_levels"),
           ("air", "summary", "oxygen"), ("predator_share", "summary", "predator_share"),
           ("result", "summary", "outcome")),
          lambda e: e["result"] in ("multicellular", "animals_like"), lambda e: e["result"] == "animals_like",
          ("bodies",)),
    Level("senses", 10, senses, {}, _senses_given,
          (("body_size", "body_size"), ("body_types", "cell_types"), ("predators", "predators"), ("air", "oxygen"),
           ("brightness", "light")),
          (("agents", "summary", "population"), ("eyes", "summary", "eyes"), ("ears", "summary", "ears"),
           ("neurons", "summary", "neurons"), ("talkers", "summary", "talkers"),
           ("predator_pressure", "summary", "predator_pressure"), ("kin_gain", "summary", "kin_gain"),
           ("group", "summary", "group"), ("result", "summary", "outcome")),
          _senses_reached, lambda e: _senses_reached(e) and e["group"] == "yes",
          ("senses", "groups")),
    Level("signals", 11, signals, {}, _signals_handed,
          (("ears_voice", "ears_voice"), ("brain", "neurons"), ("pressure", "predator_pressure"),
           ("band", "population"), ("split", "split")),
          (("capacity", "params", "capacity"), ("success", "summary", "success"), ("words", "summary", "words"),
           ("meanings", "summary", "meanings"), ("compositional", "summary", "compositional"),
           ("order", "summary", "order"), ("word_length", "summary", "word_length"),
           ("result", "summary", "outcome")),
          lambda e: e["result"] in ("calls", "lexicon", "language"), lambda e: e["result"] == "language",
          ("signals", "language")),
    Level("society", 12, society, {}, _society_handed,
          (("understood", "success"), ("lexicon", "words"), ("grammar", "compositional")),
          (("people", "summary", "population"), ("groups", "summary", "groups"),
           ("cooperation", "summary", "cooperation"), ("technologies", "summary", "technologies"),
           ("technology_list", "summary", "technology_list"), ("farming", "summary", "farming"),
           ("writing", "summary", "writing"), ("hierarchy", "summary", "hierarchy"),
           ("first_farming", "summary", "first_farming"), ("first_writing", "summary", "first_writing"),
           ("result", "summary", "outcome")),
          lambda e: e["result"] in ("tribes", "villages", "chiefdoms", "states"), lambda e: e["result"] == "states",
          ("society", "states")),
)
LEVEL_OF = {level.name: level for level in LEVELS}
#: a level's free parameter that the chain renames in the ``drawn`` record (the era shows the one used)
DRAWN_RENAMED = {"cells": {"gene_length": "gene_drawn"}}

#: the level behind an era, and its own gate
_MODULES = {"nucleo": nucleo, "stars": stars, "collapse": gravity, "planets": planets, "chemistry": chem,
            **{level.name: level.module for level in LEVELS}}


def _base(era: str) -> str:
    """``cells_2`` -> ``cells``; a fixed era is its own base."""
    head, _, tail = era.rpartition("_")
    return head if head in PLANET_ERAS and tail.isdigit() else era


def _number(era: str) -> int:
    """The planet number of an era name, 0 for a fixed era."""
    head, _, tail = era.rpartition("_")
    return int(tail) if head in PLANET_ERAS and tail.isdigit() else 0


def kept_try(level: Level, tried: list[dict]) -> int:
    """Which try an era reports (from 1): the first that reached what the next level needs, else the
    first that reached the level's lower rung, else the last."""
    for test in (level.upper, level.lower):
        for j, fields in enumerate(tried, start=1):
            if test(fields):
                return j
    return len(tried)


# ---------------------------------------------------------------------------------------------
# the rollout


@dataclass
class WorldRollout(Rollout):
    """A :class:`Rollout` whose steps are the eras, with the rollout of every level that ran."""
    #: era name -> the level's rollout (for an era with tries: the kept one)
    levels: dict[str, Rollout] = field(default_factory=dict)
    #: era name -> every try of the level, in order
    attempts: dict[str, list[Rollout]] = field(default_factory=dict)
    #: the furthest planet's level-11 run, cut down to its language (``None`` below the rung signals)
    language: signals.SignalsRollout | None = None

    def value(self, text: float | int | str) -> str:
        return dense_value(text)

    def era(self, name: str) -> dict | None:
        return next((s for s in self.steps if s["era"] == name), None)

    def planet(self, k: int) -> list[dict]:
        """The eras of planet ``k``, in order."""
        return [s for s in self.steps if _number(s["era"]) == k]


def random_params(rng: random.Random, rules: int = 7) -> dict[str, float]:
    """Six draws, always in this order: ``formation_time`` (0.25 to 12 Gyr in steps of 0.25),
    ``efficiency``, ``cloud_mass``, ``spin`` (0.02 to 0.3), ``cooling`` (0.2 to 0.6; always drawn, so
    the stream does not shift for metal-free clouds) and ``t_gas``.

    >>> random_params(random.Random(3))
    {'formation_time': 4.0, 'efficiency': 0.6, 'cloud_mass': 2.0, 'spin': 0.06, 'cooling': 0.57, 't_gas': 6.0}
    """
    if rules != 7:
        raise ValueError(f"world7 is a round-7 level: rules must be 7, not {rules!r}")
    return {"formation_time": 0.25 * rng.randint(*FORMATION_QUARTERS), "efficiency": rng.choice(EFFICIENCIES),
            "cloud_mass": rng.choice(CLOUD_MASSES), "spin": round(rng.uniform(*SPIN_RANGE), 2),
            "cooling": round(rng.uniform(*COOLING_RANGE), 2), "t_gas": rng.choice(T_GAS)}


def _is_seed(seed: object) -> bool:
    return isinstance(seed, int) and not isinstance(seed, bool) and 0 <= seed <= 10 ** 9


def _params(seed: int, given: dict) -> dict[str, float | int]:
    if not _is_seed(seed):
        raise ValueError(f"a world seed is a whole number from 0 to 10^9, not {seed!r}")
    unknown = set(given) - set(PARAM_KEYS) - set(DEFAULTS) - {"rules"}
    if unknown:
        raise TypeError(f"unknown parameter(s) {sorted(unknown)}; known: {sorted((*PARAM_KEYS, *DEFAULTS))}")
    given = dict(given)
    if given.pop("rules", 7) != 7:
        raise ValueError("world7 is a round-7 level: rules must be 7")
    p = {**random_params(random.Random(seed)), **DEFAULTS, **given}
    for key in DEFAULTS:
        if isinstance(p[key], bool) or int(p[key]) != p[key]:
            raise ValueError(f"{key} must be a whole number, not {p[key]!r}")
        p[key] = int(p[key])
    for key in PARAM_KEYS:
        if isinstance(p[key], (bool, str)) or not is_finite(p[key]):
            raise ValueError(f"{key} must be a number, not {p[key]!r}")
        p[key] = float(p[key])
    quarters = p["formation_time"] * 4
    if quarters != int(quarters) or not 1 <= quarters <= 55:
        raise ValueError(f"formation_time must be a multiple of 0.25 from 0.25 to 13.75 Gyr, not {p['formation_time']}")
    if not (0 < p["efficiency"] <= 4 and p["cloud_mass"] > 0 and 0 <= p["spin"] <= 1 / 3 and p["cooling"] > 0
            and p["t_gas"] >= 0 and p["particles"] >= 32 and p["atoms"] >= 1000 and 1 <= p["max_tries"] <= 9):
        raise ValueError(f"bad parameters: {p}")
    return p


class _Broken(Exception):
    """A stored world that is not what the chain makes of its seed (the gate's finding)."""


class _Live:
    """The chain's source of level rollouts when a world is run: it runs them."""

    def __init__(self) -> None:
        self.levels: dict[str, Rollout] = {}
        self.attempts: dict[str, list[Rollout]] = {}

    def get(self, name: str, j: int, seed: int, kwargs: dict) -> Rollout:
        r = _MODULES[_base(name)].run(seed, **kwargs)
        if j:
            self.attempts.setdefault(name, []).append(r)
        else:
            self.levels[name] = r
        return r

    def keep(self, name: str, kept: int) -> None:
        self.levels[name] = self.attempts[name][kept - 1]


class _Stored:
    """The chain's source of level rollouts in the gate: the ones a world holds, each checked to be
    the run the chain asks for and to pass its own level's gate."""

    def __init__(self, world: WorldRollout) -> None:
        self.world = world
        self.used: set[tuple[str, int]] = set()

    def get(self, name: str, j: int, seed: int, kwargs: dict) -> Rollout:
        if j:
            tries = self.world.attempts.get(name, [])
            if j > len(tries):
                raise _Broken(f"{name}: try {j} is missing")
            r = tries[j - 1]
        else:
            r = self.world.levels.get(name)
            if r is None:
                raise _Broken(f"level {name} is missing")
        module = _MODULES[_base(name)]
        if r.sim != module.SIM or r.seed != seed:
            raise _Broken(f"{name}: not the run of seed {seed} of level {module.SIM}")
        for key, value in kwargs.items():
            got = r.params.get(key)
            if got != value or isinstance(got, str) != isinstance(value, str):
                raise _Broken(f"{name}: the level ran with {key} {got!r}, the chain hands it {value!r}")
        verdict = module.conserved(r)
        if not verdict.ok:
            raise _Broken(f"level {name}: {verdict.reason}")
        self.used.add((name, j))
        return r

    def keep(self, name: str, kept: int) -> None:
        if self.world.levels.get(name) is not self.world.attempts[name][kept - 1]:
            raise _Broken(f"{name}: the level kept is not try {kept}")
        self.used.add((name, 0))

    def leftover(self) -> list[str]:
        held = {(name, 0) for name in self.world.levels}
        held |= {(name, j) for name, tries in self.world.attempts.items() for j in range(1, len(tries) + 1)}
        return sorted(f"{name} try {j}" if j else name for name, j in held - self.used)


def nucleo_era(level: Rollout) -> dict:
    s = level.summary
    return {"era": "nucleo", "age": 0, "seconds": int(s["end_time"]), "hydrogen": sig(s["hydrogen"]),
            "helium": sig(s["helium"]), "traces": sig(s["traces"])}


def stars_era(level: Rollout, formation_time: float) -> dict:
    s = level.summary
    return {"era": "stars", "age": FIRST_STARS, "duration": enrichment_years(formation_time),
            "efficiency": level.params["efficiency"],
            **{key: sig(s[key]) for key in ("gas_fraction", "hydrogen", "helium", "metallicity", "oxygen", "carbon",
                                            "iron")}}


def cloud_era(enriched: dict, p: dict) -> dict:
    z = enriched["metallicity"]
    return {"era": "cloud", "age": formation_year(p["formation_time"]), "cloud_mass": cloud_mass_of(p["cloud_mass"], z),
            "spin": p["spin"], "cooling": cooling_of(z, p["cooling"])}


def collapse_era(level: Rollout, cloud: dict) -> dict:
    s = level.summary
    return {"era": "collapse", "age": cloud["age"], "duration": SPAN["collapse"], "stage": s["stage"],
            "clumps": s["clumps"], "star_share": _shown(s["star"]), "companions": s["companions"],
            "disc_share": _shown(s["disc_mass"]), "disc_radius": _shown(s["disc_radius"]),
            "star_time": _shown(s["star_time"])}


def star_era(enriched: dict, cloud: dict, collapse: dict) -> dict:
    """The ``star`` era from the numbers the eras before it print."""
    age = era_end(collapse["age"], collapse["duration"])
    mass = star_mass_of(cloud["cloud_mass"], collapse["star_share"])
    light = luminosity_of(mass)
    shines = fusion_of(mass) == "yes"
    lifetime = lifetime_of(mass, light)
    z = enriched["metallicity"]
    return {"era": "star", "age": age, "fusion": fusion_of(mass), "star_mass": mass, "luminosity": light,
            "lifetime": lifetime, "star_end": era_end(age, lifetime),
            "frost_line": frost_line_of(light) if shines else 0.0, "hz_inner": hz_inner_of(light) if shines else 0.0,
            "hz_outer": hz_outer_of(light) if shines else 0.0,
            "disc_solids": disc_solids_of(collapse["disc_share"], cloud["cloud_mass"], z),
            "disc_gas": disc_gas_of(collapse["disc_share"], cloud["cloud_mass"], z)}


def surface_eras(level: Rollout, star: dict, age: int) -> list[dict]:
    """One era per rocky planet of a level-3 run whose flux is temperate, innermost first."""
    rocky = sorted((b for b in level.system if b["type"] == "rocky"), key=lambda b: (b["orbit"], b["id"]))
    out: list[dict] = []
    for body in rocky:
        orbit, mass, gas = sig(body["orbit"]), sig(body["mass"]), sig(body["gas"])
        flux = flux_of(star["luminosity"], orbit)
        if zone_of(flux) != "temperate" or len(out) == MAX_SURFACES:
            continue
        warming = greenhouse_of(flux, mass, gas)
        kelvin = temperature_of(flux, warming)
        out.append({"era": f"surface_{len(out) + 1}", "age": age, "orbit": orbit, "mass": mass, "gas": gas,
                    "flux": flux, "surface": surface_of(mass, gas), "t_eq": t_eq_of(flux), "greenhouse": warming,
                    "temperature": kelvin, "water": water_of(kelvin, mass, gas)})
    return out


def planets_era(level: Rollout, star: dict, t_gas: float, temperate: int) -> dict:
    s = level.summary
    return {"era": "planets", "age": star["age"], "duration": SPAN["planets"], "t_gas": t_gas,
            **{key: s[key] for key in ("planets", "rocky", "super_earths", "icy", "ice_giants", "gas_giants")},
            "largest_mass": sig(s["largest_mass"]), "temperate": temperate}


def ocean(surface: dict) -> bool:
    """True for a surface era with liquid water on a terran planet: the planet gets a chemistry era."""
    return surface["surface"] == "terran" and surface["water"] == "liquid"


def chem_params(seed: int, k: int, temperature: int, atoms: int) -> dict:
    """What level 4 runs with on planet ``k``: its own draws of the mix (reducing or not) and the
    sparks, the planet's temperature, the chain's sizes."""
    own = chem.random_params(random.Random(planet_seed(seed, k)), rules=7)
    return {"mix": "reducing" if own["mix"] == "reducing" else "ocean", "t_start": CHEM_START_K, "t_end": temperature,
            "steps": CHEM_STEPS, "atoms": atoms, "sparks": own["sparks"], "rules": 7}


def chemistry_era(level: Rollout, k: int, age: int) -> dict:
    last = level.steps[-1]
    return {"era": f"chemistry_{k}", "age": age, "duration": SPAN["chemistry"], "mix": level.params["mix"],
            "sparks": level.params["sparks"], "h2o": last["h2o"], "organic": last["organic"],
            "precursors": last["precursors"]}


def _climb(seed: int, k: int, surface: dict, star: dict, p: dict, source: _Live | _Stored) -> list[dict]:
    """The eras of planet ``k`` after its surface: chemistry, then the levels 5 to 12, each as often
    as the ladder and the clock allow."""
    out: list[dict] = []
    age, end = surface["age"], star["star_end"]
    if not ocean(surface) or in_time(age, SPAN["chemistry"], end) != "yes":
        return out
    name = f"chemistry_{k}"
    out.append(chemistry_era(source.get(name, 0, planet_seed(seed, k),
                                        chem_params(seed, k, surface["temperature"], p["atoms"])), k, age))
    eras = {"surface": surface, "chemistry": out[-1]}
    if out[-1]["precursors"] <= 0:
        return out
    age = era_end(age, SPAN["chemistry"])
    for level in LEVELS:
        name, span = f"{level.name}_{k}", SPAN[level.name]
        handed = level.handed(eras)
        tried: list[dict] = []
        for j in range(1, p["max_tries"] + 1):
            if in_time(era_end(age, duration_of(j - 1, span)), span, end) != "yes":
                break
            ts = try_seed(seed, k, level.number, j)
            tried.append(level.fields(source.get(name, j, ts, {**level.params(ts, handed), **level.extra})))
            if level.upper(tried[-1]):
                break
        if not tried:
            break
        kept = kept_try(level, tried)
        source.keep(name, kept)
        era = {"era": name, "age": age, "duration": duration_of(len(tried), span), "tries": len(tried), "kept": kept,
               **{key: handed[param] for key, param in level.shown}, **tried[kept - 1]}
        out.append(era)
        eras[level.name] = era
        if not level.upper(era):
            break
        age = era_end(age, era["duration"])
    return out


def _chain(seed: int, p: dict, source: _Live | _Stored) -> list[dict]:
    """The eras of a world, with the level rollouts taken from ``source``."""
    steps = [nucleo_era(source.get("nucleo", 0, seed, {"rules": 7}))]
    gas = first_gas(steps[0]["hydrogen"], steps[0]["helium"])
    span = enrichment_years(p["formation_time"])
    steps.append(stars_era(source.get("stars", 0, seed, {"efficiency": p["efficiency"], "t_end": span / GYR, **gas}),
                           p["formation_time"]))
    enriched = steps[-1]
    cloud = cloud_era(enriched, p)
    steps.append(cloud)
    collapse = collapse_era(source.get("collapse", 0, seed, {"spin": cloud["spin"], "cooling": cloud["cooling"],
                                                             "n": p["particles"], "rules": 7}), cloud)
    steps.append(collapse)
    star = star_era(enriched, cloud, collapse)
    steps.append(star)
    if star["fusion"] == "yes" and star["disc_solids"] > 0 \
            and in_time(star["age"], SPAN["planets"], star["star_end"]) == "yes":
        system = source.get("planets", 0, seed, {"star_mass": star["star_mass"], "disc_mass": star["disc_solids"],
                                                 "t_gas": p["t_gas"], "gas_mass": star["disc_gas"], "rules": 7})
        surfaces = surface_eras(system, star, era_end(star["age"], SPAN["planets"]))
        steps.append(planets_era(system, star, p["t_gas"], len(surfaces)))
        for k, surface in enumerate(surfaces, start=1):
            steps.append(surface)
            steps += _climb(seed, k, surface, star, p, source)
    return steps


def _slim_language(level: signals.SignalsRollout) -> signals.SignalsRollout:
    """A level-11 rollout cut down to what its language lines need."""
    last = signals.STEPS
    return signals.SignalsRollout(level.sim, level.seed, dict(level.params), [], dict(level.summary),
                                  world=level.world, states=[], languages={last: level.languages[last]})


def run(seed: int, **params: float | int) -> WorldRollout:
    """One world: the eras of ``seed``. Parameters not given are the seed's own (:func:`random_params`)
    or the default sizes; ``run(seed)`` is what :func:`check` replays. ``run(seed, max_tries=1)`` is a
    faster world in which every level gets one try."""
    p = _params(seed, params)
    source = _Live()
    steps = _chain(seed, p, source)
    talk = _eras(steps).get(f"signals_{furthest(steps)[0]}")
    language = _slim_language(source.levels[talk["era"]]) if talk and LEVEL_OF["signals"].lower(talk) else None
    r = WorldRollout(SIM, seed, p, steps, summarise(steps, p, language), source.levels, source.attempts, language)
    _remember(r)
    return r


# ---------------------------------------------------------------------------------------------
# the ladder: what the eras say


def _eras(steps: list[dict]) -> dict[str, dict]:
    return {s["era"]: s for s in steps}


def planet_links(steps: list[dict], k: int) -> list[str]:
    """The links of planet ``k``'s own climb that hold (``replicators`` onward), in order: a prefix."""
    era, held = _eras(steps), []
    for level in LEVELS:
        e = era.get(f"{level.name}_{k}")
        if e is None or not level.lower(e):
            break
        held.append(level.links[0])
        if not level.upper(e):
            break
        held += level.links[1:]
    return held


def furthest(steps: list[dict]) -> tuple[int, list[str]]:
    """The planet whose climb got furthest (the inner one of equals) and the links of its climb;
    planet 0 when no planet has a chemistry era."""
    best, links, score = 0, [], -1
    for s in steps:
        if _base(s["era"]) == "chemistry":
            k = _number(s["era"])
            mine = planet_links(steps, k) if s["precursors"] > 0 else []
            mark = len(mine) + (s["precursors"] > 0)
            if mark > score:
                best, links, score = k, mine, mark
    return best, links


def links_held(steps: list[dict]) -> list[str]:
    """The links of the chain that hold, in the order of :data:`LINKS`: always a prefix of it."""
    era = _eras(steps)
    surfaces = [s for s in steps if _base(s["era"]) == "surface"]
    vats = [s for s in steps if _base(s["era"]) == "chemistry"]
    held = {
        "metals": "stars" in era and era["stars"]["metallicity"] > 0,
        "star": "star" in era and era["star"]["fusion"] == "yes",
        "planets": "planets" in era and era["planets"]["planets"] > 0,
        "temperate": bool(surfaces),
        "water": any(ocean(s) for s in surfaces),
        "organics": any(s["precursors"] > 0 for s in vats),
    }
    out = []
    for link in LINKS[:LOWER]:
        if not held[link]:
            return out
        out.append(link)
    return out + furthest(steps)[1]


def rung_of(steps: list[dict]) -> str:
    """The first missing link, or the last link held."""
    return LADDER[len(links_held(steps))]


def stopped_by_clock(steps: list[dict], max_tries: int = MAX_TRIES) -> bool:
    """True when the next thing the chain would have run did not fit into the star's life (which
    is the only reason for a due era, or a due try, to be missing)."""
    era = _eras(steps)
    held = links_held(steps)
    if len(held) < 2:
        return False
    if "planets" not in era:
        return era["star"]["disc_solids"] > 0
    if "water" not in held:
        return False
    k = furthest(steps)[0]
    if k == 0:
        return True  # an ocean, and no time for its chemistry
    if era[f"chemistry_{k}"]["precursors"] <= 0:
        return False
    for level in LEVELS:
        e = era.get(f"{level.name}_{k}")
        if e is None:
            return True
        if not level.upper(e):
            return e["tries"] < max_tries
    return False


def _ends(step: dict) -> int:
    return step["age"] + step.get("duration", 0)


def age_at_end(steps: list[dict]) -> int:
    k = furthest(steps)[0]
    return _ends([s for s in steps if _number(s["era"]) == k][-1])


def sentence_of(language: signals.SignalsRollout | None) -> tuple[str, str]:
    """An utterance of a language and its meaning: the first pair it can say, else its first word."""
    if language is None:
        return "none", "none"
    tongue, world = language.languages[signals.STEPS], language.world
    for thing in tongue["things"]:
        for prop in world["properties"]:
            said = signals.utterances_for(tongue, thing, prop)
            if said:
                return signals.text_of(said[0]), f"{thing} {prop}"
    for meaning in world["meanings"]:
        if meaning in tongue["lexicon"]:
            return signals.text_of(tongue["lexicon"][meaning]), meaning
    return "none", "none"


def summarise(steps: list[dict], p: dict, language: signals.SignalsRollout | None) -> dict[str, float | int | str]:
    era = _eras(steps)
    surfaces = [s for s in steps if _base(s["era"]) == "surface"]
    k = furthest(steps)[0]
    rung = rung_of(steps)
    talk, city = era.get(f"signals_{k}"), era.get(f"society_{k}")
    sentence, meaning = sentence_of(language)
    return {"hydrogen": era["stars"]["hydrogen"], "helium": era["stars"]["helium"],
            "metallicity": era["stars"]["metallicity"], "formation_time": p["formation_time"],
            "star_mass": era["star"]["star_mass"], "star_lifetime": era["star"]["lifetime"],
            "planets": era["planets"]["planets"] if "planets" in era else 0, "temperate": len(surfaces),
            "oceans": sum(1 for s in surfaces if ocean(s)), "planet": k, "rung": rung,
            "outcome": OUT_OF_TIME if stopped_by_clock(steps, p["max_tries"]) else rung,
            "by_now": rung_of([s for s in steps if _ends(s) <= NOW]), "age_at_end": age_at_end(steps),
            "eras": len(steps), "technologies": city["technologies"] if city else 0,
            "words": talk["words"] if talk else 0, "sentence": sentence, "meaning": meaning}


# ---------------------------------------------------------------------------------------------
# replay: the world a seed names


_SLIM: OrderedDict[int, WorldRollout] = OrderedDict()
_SLIM_MAX = 512


def _remember(r: WorldRollout) -> None:
    """Keep the eras, the summary and the language of a seed's own world for :func:`check` (the
    level rollouts are not kept: a world holds some megabytes of them)."""
    if not canonical(r):
        return
    _SLIM[r.seed] = WorldRollout(SIM, r.seed, dict(r.params), [dict(s) for s in r.steps], dict(r.summary), {}, {},
                                 r.language)
    _SLIM.move_to_end(r.seed)
    while len(_SLIM) > _SLIM_MAX:
        _SLIM.popitem(last=False)


def _replay(seed: int) -> WorldRollout:
    """The gate's own copy of a seed's world, without the level rollouts; never handed out."""
    if seed not in _SLIM:
        run(seed)
    _SLIM.move_to_end(seed)
    return _SLIM[seed]


def rollout(seed: int) -> WorldRollout:
    """The world a seed names in the lines and in :func:`check`, with every level rollout."""
    return run(seed)


def canonical(r: Rollout) -> bool:
    """True when ``r`` is the world its seed names, the only one :func:`check` can replay."""
    return r.sim == SIM and _is_seed(r.seed) and r.params == {**random_params(random.Random(r.seed)), **DEFAULTS}


# ---------------------------------------------------------------------------------------------
# the gate


def _timeline_holds(r: WorldRollout) -> str:
    names = [s["era"] for s in r.steps]
    if len(set(names)) != len(names):
        return "an era stands twice"
    fixed = [n for n in names if _number(n) == 0]
    if tuple(fixed[:len(FIXED_ERAS)]) != FIXED_ERAS or fixed[len(FIXED_ERAS):] not in ([], ["planets"]) \
            or names[:len(fixed)] != fixed:
        return f"the eras are out of order: {' '.join(names)}"
    rest, k = names[len(fixed):], 0
    while rest:
        k += 1
        mine = [n for n in rest if _number(n) == k]
        if not mine or rest[:len(mine)] != mine or mine != [f"{b}_{k}" for b in PLANET_ERAS[:len(mine)]]:
            return f"the eras of planet {k} are out of order: {' '.join(names)}"
        rest = rest[len(mine):]
    for s in r.steps:
        if set(s) != {"era", *ERAS[_base(s["era"])]}:
            return f"era {s['era']} does not hold its metrics"
    return ""


def _mass_holds(r: WorldRollout) -> str:
    first, enriched = r.levels["nucleo"], r.levels["stars"]
    fresh = nucleo.run(r.seed, rules=7)
    if first.summary != fresh.summary or first.params != fresh.params:
        return "the nucleo level is not our universe under rules 7"
    start = math.fsum((enriched.params["hydrogen"], enriched.params["helium"]))
    end = math.fsum((enriched.summary["hydrogen"], enriched.summary["helium"], enriched.summary["metallicity"]))
    if abs(start - 1.0) > 1e-9 or abs(end - 1.0) > 1e-9:
        return f"the mass fractions sum to {start!r} before and {end!r} after the stars, not 1"
    if min(enriched.summary[k] for k in ("hydrogen", "helium", "metallicity")) < 0:
        return "a negative mass fraction"
    if enriched.summary["hydrogen"] > enriched.params["hydrogen"] + 1e-12:
        return "stars made hydrogen"
    return ""


def _clock_holds(r: WorldRollout) -> str:
    era, p = _eras(r.steps), r.params
    star = era["star"]
    for s in r.steps:
        if not isinstance(s["age"], int) or isinstance(s["age"], bool) or s["age"] < 0 or s.get("duration", 0) < 0:
            return f"era {s['era']}: age and duration must be whole years, not negative"
    if era["nucleo"]["age"] != 0 or era["stars"]["age"] != FIRST_STARS \
            or _ends(era["stars"]) != era["cloud"]["age"] or era["cloud"]["age"] != formation_year(p["formation_time"]) \
            or era["collapse"]["age"] != era["cloud"]["age"] or _ends(era["collapse"]) != star["age"]:
        return "the clock of the fixed eras does not add up"
    if star["star_end"] != star["age"] + star["lifetime"]:
        return "star_end is not the star's age plus its lifetime"
    if "planets" in era and era["planets"]["age"] != star["age"]:
        return "the planets do not start with the star"
    k = 0
    while f"surface_{k + 1}" in era:
        k += 1
        mine = r.planet(k)
        if mine[0]["age"] != _ends(era["planets"]):
            return f"surface_{k} does not start when the planets are done"
        for before, after in zip(mine, mine[1:]):
            if after["age"] != _ends(before):
                return f"{after['era']} does not start where {before['era']} ends"
            if _ends(after) <= after["age"]:
                return f"{after['era']} takes no time"
        for s in mine[1:]:
            if "tries" in s and (not 1 <= s["kept"] <= s["tries"] <= p["max_tries"]
                                 or s["duration"] != s["tries"] * SPAN[_base(s["era"])]):
                return f"{s['era']}: tries, kept and duration do not fit"
    for s in r.steps:
        if s["era"] not in FIXED_ERAS and _ends(s) > star["star_end"]:
            return f"era {s['era']} ends after the star"
    return ""


def _ladder_holds(r: WorldRollout) -> str:
    held = links_held(r.steps)
    if tuple(held) != LINKS[:len(held)]:
        return "the links held are not a prefix of the ladder"
    fixed = [s for s in r.steps if _number(s["era"]) == 0]
    k = 0
    while r.era(f"surface_{k + 1}") is not None:
        k += 1
        mine, seen = r.planet(k), -1
        for i in range(1, len(mine) + 1):
            now = LADDER.index(rung_of(fixed + mine[:i]))
            if now < seen:
                return f"planet {k}: the rung falls"
            seen = now
    return ""


def _summary_follows(r: WorldRollout) -> str:
    s = r.summary
    if s != summarise(r.steps, r.params, r.language):
        return "the summary does not follow from the eras"
    if s["rung"] not in LADDER or s["outcome"] not in OUTCOMES or s["by_now"] not in LADDER:
        return "a rung that is not on the ladder"
    if LADDER.index(s["by_now"]) > LADDER.index(s["rung"]):
        return "more by now than in the end"
    if not 0 <= s["oceans"] <= s["temperate"] <= (r.era("planets") or {"rocky": 0})["rocky"] <= s["planets"]:
        return "the counts of the summary do not nest"
    if s["metallicity"] == 0 and s["planets"]:
        return "planets without metals"
    talk = r.era(f"signals_{s['planet']}")
    speaks = talk is not None and LEVEL_OF["signals"].lower(talk)
    if speaks != (r.language is not None):
        return "a language exactly where the furthest planet reached signals"
    if speaks:
        level, last = r.levels[talk["era"]], signals.STEPS
        if r.language.seed != level.seed or r.language.languages[last] != level.languages[last] \
                or r.language.world != level.world:
            return "the language is not the furthest planet's"
    return ""


def conserved(r: Rollout) -> Verdict:
    """The chain's gate (see the module docstring for the list)."""
    if not isinstance(r, WorldRollout) or r.sim != SIM or not r.steps:
        return Verdict(False, None, "not a world7 rollout")
    try:
        reason = _timeline_holds(r)
        if reason:
            return Verdict(False, None, reason)
        p = _params(r.seed, dict(r.params))
        if p != r.params:
            return Verdict(False, None, "the parameters are not a world's")
        source = _Stored(r)
        try:
            steps = _chain(r.seed, p, source)
        except _Broken as broken:
            return Verdict(False, None, str(broken))
        if source.leftover():
            return Verdict(False, None, f"levels the chain did not ask for: {', '.join(source.leftover())}")
        for mine, rebuilt in zip(r.steps, steps):
            if mine != rebuilt:
                wrong = sorted(key for key in mine if mine.get(key) != rebuilt.get(key))
                return Verdict(False, None, f"era {mine['era']} does not follow from the hand-offs: {' '.join(wrong)}")
        if len(steps) != len(r.steps):
            return Verdict(False, None, "the eras are not the ones the chain gives")
        for holds in (_mass_holds, _clock_holds, _ladder_holds, _summary_follows):
            reason = holds(r)
            if reason:
                return Verdict(False, None, reason)
    except (KeyError, IndexError, TypeError, AttributeError, ValueError) as e:
        return Verdict(False, None, f"malformed world7 rollout: {type(e).__name__}: {e}")
    return Verdict(True, None, "every level's gate, every hand-off, the clock and the ladder hold")


# ---------------------------------------------------------------------------------------------
# lessons: the hand-offs with their inputs


def _quarters(formation_time: float) -> bool:
    return formation_time * 4 == int(formation_time * 4)


_STAR = Input("star_mass", 0.0, 1000.0, places=3)
_LIGHT = Input("luminosity", 0.0, 1e8, places=4)
_Z = Input("metallicity", 0.0, 0.05, places=5)
_FLUX = Input("flux", 0.01, 100.0, places=3)
_MASS = Input("mass", 0.1, 20.0, places=2)
_GAS = Input("gas", 0.0, 10.0, places=2)
_AGE = Input("age", 0, 10 ** 13, True)

RULES: dict[str, Rule] = {
    "helium_share": Rule((Input("hydrogen", 0.5, 0.9), Input("helium", 0.1, 0.5)), helium_share,
                         "helium mass fraction of the gas the first stars form from. helium over hydrogen plus "
                         "helium to 4 decimals. the traces count as hydrogen"),
    "formation_year": Rule((Input("formation_time", 0.25, 13.75, places=2),), formation_year,
                           "years since the big bang at which the cloud forms. formation_time in gyr times 1 e 9",
                           _quarters),
    "enrichment_years": Rule((Input("formation_time", 0.25, 13.75, places=2),), enrichment_years,
                             "years earlier stars enrich the gas. formation_year minus 2 5 0 0 0 0 0 0 0. the first "
                             "stars are put at 2 5 0 million years. toy", _quarters),
    "cooling": Rule((_Z, Input("drawn", 0.2, 0.6, places=2)), cooling_of,
                    "cooling of the cloud in level 2. 0 point 1 when metallicity is below 1 e minus 5. otherwise "
                    "the value drawn for the cloud"),
    "cloud_mass": Rule((Input("drawn", 0.1, 8.0, places=2), _Z), cloud_mass_of,
                       "solar masses of the cloud. the drawn mass. 9 0 times it when metallicity is below 1 e "
                       "minus 5. metal free gas stays warm and breaks into heavy clumps"),
    "star_mass": Rule((Input("cloud_mass", 0.1, 720.0, places=2), Input("star_share", 0.0, 1.0)), star_mass_of,
                      "solar masses of the star. cloud_mass times star_share to 3 digits. a 5 rounds up"),
    "fusion": Rule((_STAR,), fusion_of, "yes when the star has 0 point 0 8 solar masses or more and shines. else no"),
    "luminosity": Rule((_STAR,), luminosity_of,
                       "solar luminosities to 3 digits. 0 below 0 point 0 8 solar masses. 0 point 2 3 times mass to "
                       "the 2 point 3 below 0 point 4 3. mass to the 4 below 2. 1 point 4 times mass to the 3 point 5 "
                       "below 5 5. 3 2 0 0 0 times mass above"),
    "lifetime": Rule((Input("star_mass", 0.08, 1000.0, places=3), Input("luminosity", 1e-4, 1e8, places=4)),
                     lifetime_of,
                     "whole years the star shines. 1 e 1 0 times star_mass over luminosity to 3 digits. a 5 rounds up"),
    "frost_line": Rule((_LIGHT,), frost_line_of,
                       "au beyond which ice is solid. 2 point 7 times square root of luminosity. 3 digits"),
    "hz_inner": Rule((_LIGHT,), hz_inner_of,
                     "inner edge of the temperate zone in au. square root of luminosity over 1 point 1 0 7. 3 digits"),
    "hz_outer": Rule((_LIGHT,), hz_outer_of,
                     "outer edge of the temperate zone in au. square root of luminosity over 0 point 3 5 6. 3 digits"),
    "disc_solids": Rule((Input("disc_share", 0.0, 1.0), Input("cloud_mass", 0.1, 720.0, places=2), _Z),
                        disc_solids_of,
                        "earth masses of solids in the disc of level 3. 2 6 6 4 0 times disc_share times cloud_mass "
                        "times metallicity to 3 digits. 2 6 6 4 0 is 8 percent of 3 3 3 0 0 0 earth masses. toy share"),
    "disc_gas": Rule((Input("disc_share", 0.0, 1.0), Input("cloud_mass", 0.1, 720.0, places=2), _Z), disc_gas_of,
                     "earth masses of gas in the disc of level 3. 2 6 6 4 0 times disc_share times cloud_mass times "
                     "one minus metallicity to 3 digits"),
    "flux": Rule((Input("luminosity", 1e-4, 1e8, places=4), Input("orbit", 0.01, 3000.0, places=3)), flux_of,
                 "starlight at a planet in units of what the earth receives. luminosity over orbit squared. 3 digits"),
    "zone": Rule((_FLUX,), zone_of,
                 "runaway when flux is above 1 point 1 0 7. frozen below 0 point 3 5 6. else temperate"),
    "surface": Rule((_MASS, _GAS), surface_of,
                    "thin below 0 point 3 earth masses. else enveloped when the planet kept gas or weighs more than "
                    "1 0. else terran"),
    "t_eq": Rule((_FLUX,), t_eq_of,
                 "kelvin without greenhouse at albedo 0 point 3. 2 5 4 point 6 times fourth root of flux. 3 digits"),
    "greenhouse": Rule((_FLUX, _MASS, _GAS), greenhouse_of,
                       "kelvin the air adds. thin 5. enveloped 0. terran 5 0 0 above flux 1 point 1 0 7. 3 3 from "
                       "flux 1 on. 3 3 plus 6 7 point 3 times one minus flux down to 0 point 3 5 6. 0 below"),
    "temperature": Rule((_FLUX, Input("greenhouse", 0.0, 500.0, places=1)), temperature_of,
                        "whole kelvin. 2 5 4 point 6 times fourth root of flux plus greenhouse. a half rounds up"),
    "water": Rule((Input("temperature", 50, 1500, True), _MASS, _GAS), water_of,
                  "none under an envelope. else ice below 2 7 3 kelvin. liquid to 3 7 3. vapor above"),
    "monomers": Rule((Input("precursors", 0, 1000, True),), monomers_of,
                     "monomers level 5 starts with. 1 0 0 times precursors. the size of the vessel. toy"),
    "inflow": Rule((Input("sparks", 0, 200, True),), inflow_of,
                   "monomers per step entering level 5. 2 0 times sparks. toy"),
    "hydrolysis": Rule((Input("temperature", 200, 400, True),), hydrolysis_of,
                       "chance per bond and step that water breaks it. 0 point 0 0 2 below 2 6 0 kelvin. then 0 point "
                       "0 0 5. 0 point 0 1 from 2 7 5. 0 point 0 2 from 2 9 0. 0 point 0 5 from 3 0 5. 0 point 1 from "
                       "3 2 0. toy"),
    "fidelity": Rule((Input("mu", 0.0, 0.5),), fidelity_of,
                     "chance that a base is copied right. one minus mu. at least 0 point 5"),
    "nutrient": Rule((Input("inflow", 0, 4000, True),), nutrient_of,
                     "bases per generation entering level 8. 1 2 5 0 0 times inflow"),
    "gene_length": Rule((Input("drawn", 20, 640, True), Input("l_max", 1.0, 3000.0, places=1)), gene_length_of,
                        "bases of the first genes. the drawn length halved until twice it is at most l_max. never "
                        "below 2 0. toy", lambda drawn, l_max: drawn in GENE_LENGTHS),
    "efficiency": Rule((Input("energy_per_cell", 1.0, 200.0, places=1),), efficiency_of,
                       "transfer efficiency of level 9. 0 point 0 5 at energy 1 0. 0 point 2 at 1 6 0. between by the "
                       "logarithm. 2 decimals. toy"),
    "light": Rule((Input("flux", 0.3, 1.2),), light_of,
                  "light of level 9. 2 0 0 times flux. whole. a half rounds up. toy"),
    "brightness": Rule((Input("flux", 0.0, 2.0),), brightness_of, "light of level 1 0. the flux. at most 1. toy"),
    "body_size": Rule((Input("consumer_size", 0, 2 ** 20, True), Input("max_size", 1, 2 ** 20, True)), body_size_of,
                      "cells of the bodies that get sensors. consumer_size. max_size when there are no consumers"),
    "predators": Rule((Input("predator_share", 0.0, 1.0), Input("trophic_levels", 0, 4, True)), predators_of,
                      "predator density of level 1 0. 0 below 3 trophic levels. else 3 times predator_share. at "
                      "most 0 point 3"),
    "neurons": Rule((Input("neurons", 0.0, 1024.0, places=1),), neurons_of,
                    "neurons level 1 1 counts with. the mean neurons rounded to a whole number"),
    "band": Rule((Input("agents", 0, 200, True),), band_of,
                 "agents of the first group in level 1 1. the agents of level 1 0. at most 6 0"),
    "duration": Rule((Input("tries", 1, 9, True), Input("span", 1, 3 * GYR, True)), duration_of,
                     "years an era takes. tries times the span of its level"),
    "era_end": Rule((_AGE, Input("duration", 0, 10 ** 13, True)), era_end,
                    "years since the big bang at which an era ends and the next begins. age plus duration. also the "
                    "death of a star. its age plus its lifetime"),
    "in_time": Rule((_AGE, Input("span", 1, 3 * GYR, True), Input("star_end", 0, 2 * 10 ** 13, True)), in_time,
                    "yes when age plus span is at most star_end. a run that would end after the star dies does not "
                    "happen"),
}


def _star(rng: random.Random) -> float:
    return star_mass_of(cloud_mass_of(rng.choice(CLOUD_MASSES), 0.0 if rng.random() < 0.05 else 0.01),
                        round(rng.uniform(0.25, 0.75), 3))


def _shining_star(rng: random.Random) -> float:
    while True:
        mass = _star(rng)
        if mass >= MIN_STAR:
            return mass


def _shining(rng: random.Random) -> float:
    return luminosity_of(_shining_star(rng))


def _age(rng: random.Random) -> int:
    return rng.randint(1, 60) * QUARTER + rng.choice((0, SPAN["collapse"], SPAN["collapse"] + SPAN["planets"]))


def _span(rng: random.Random) -> int:
    return SPAN[rng.choice(tuple(SPAN))]


def _metals(rng: random.Random) -> float:
    return 0.0 if rng.random() < 0.15 else sig(rng.uniform(0.0001, 0.03))


def _zone_flux(rng: random.Random) -> float:
    return sig(rng.uniform(0.2, 1.4))


def _planet(rng: random.Random) -> tuple[float, float]:
    """A rocky planet's mass and kept gas: about a fifth thin, a third enveloped, half terran."""
    mass = sig(10 ** rng.uniform(-1.0, 1.2))
    return mass, (sig(rng.uniform(0.01, 0.4) * mass) if rng.random() < 0.3 else 0.0)


def _in_time(rng: random.Random) -> list:
    """A clock question with the star's death near the end of the run as often as far from it:
    yes and no about evenly (the edge, age + span = star_end, is a yes)."""
    age, span = _age(rng), _span(rng)
    factor = rng.choice((0.3, 0.6, 0.9, 0.99, 1.01, 1.5, 4.0, 40.0))
    return [age, span, age + (span if rng.random() < 0.1 else int(span * factor))]


def _lifetime(rng: random.Random) -> list:
    mass = _shining_star(rng)
    return [mass, luminosity_of(mass)]


def _lit(rng: random.Random) -> list:
    light = _shining(rng)
    return [light, sig(math.sqrt(light) * rng.uniform(0.5, 2.5))]


def _warmed(rng: random.Random) -> list:
    flux = _zone_flux(rng)
    return [flux, greenhouse_of(flux, *_planet(rng))]


#: how :meth:`World7Lessons.generate` draws the inputs of a rule (the others: evenly in their ranges)
DRAWS: dict[str, Callable[[random.Random], list]] = {
    "formation_year": lambda rng: [0.25 * rng.randint(1, 55)],
    "enrichment_years": lambda rng: [0.25 * rng.randint(1, 55)],
    "cooling": lambda rng: [_metals(rng), round(rng.uniform(*COOLING_RANGE), 2)],
    "cloud_mass": lambda rng: [rng.choice(CLOUD_MASSES), _metals(rng)],
    "star_mass": lambda rng: [cloud_mass_of(rng.choice(CLOUD_MASSES), _metals(rng)), round(rng.uniform(0.2, 0.8), 3)],
    "fusion": lambda rng: [sig(rng.uniform(0.02, 0.14)) if rng.random() < 0.8 else _star(rng)],
    "luminosity": lambda rng: [_star(rng)],
    "lifetime": _lifetime,
    "frost_line": lambda rng: [_shining(rng)],
    "hz_inner": lambda rng: [_shining(rng)],
    "hz_outer": lambda rng: [_shining(rng)],
    "disc_solids": lambda rng: [round(rng.uniform(0.1, 0.5), 3), rng.choice(CLOUD_MASSES), _metals(rng)],
    "disc_gas": lambda rng: [round(rng.uniform(0.1, 0.5), 3), rng.choice(CLOUD_MASSES), _metals(rng)],
    "flux": _lit,
    "zone": lambda rng: [sig(rng.uniform(*rng.choice(((0.05, 0.355), (0.356, 1.107), (1.108, 3.0)))))],
    "surface": lambda rng: list(_planet(rng)),
    "t_eq": lambda rng: [_zone_flux(rng)],
    "greenhouse": lambda rng: [_zone_flux(rng), *_planet(rng)],
    "temperature": _warmed,
    "water": lambda rng: [rng.randint(150, 450), *_planet(rng)],
    "monomers": lambda rng: [rng.randint(0, 200)],
    "inflow": lambda rng: [rng.randint(0, 20)],
    "hydrolysis": lambda rng: [rng.randint(240, 340)],
    "fidelity": lambda rng: [rng.choice((0.001, 0.002, 0.005, 0.01, 0.02, 0.05, round(rng.uniform(0.0, 0.1), 3)))],
    "nutrient": lambda rng: [INFLOW_PER_SPARK * rng.randint(0, 20)],
    "gene_length": lambda rng: [rng.choice(GENE_LENGTHS),
                                sig(rng.choice((38, 95, 190, 380, 950, 1900)) * rng.uniform(0.9, 1.1))],
    "light": lambda rng: [sig(rng.uniform(0.356, 1.107))],
    "brightness": lambda rng: [sig(rng.uniform(0.2, 1.3))],
    "body_size": lambda rng: [0 if rng.random() < 0.3 else 2 ** rng.randint(1, 10), 2 ** rng.randint(1, 10)],
    "predators": lambda rng: [sig(rng.uniform(0.0, 0.15)), rng.choice((2, 3, 3, 4, 4))],
    "neurons": lambda rng: [sig(rng.uniform(0.0, 260.0))],
    "band": lambda rng: [rng.randint(0, 90)],
    "efficiency": lambda rng: [sig(10 * 16 ** rng.uniform(-0.05, 1.0))],
    "duration": lambda rng: [rng.randint(1, MAX_TRIES), SPAN[rng.choice([lv.name for lv in LEVELS])]],
    "era_end": lambda rng: [_age(rng), rng.randint(1, MAX_TRIES) * _span(rng) if rng.random() < 0.7
                            else lifetime_of(*_lifetime(rng))],
    "in_time": _in_time,
}


class World7Lessons(LessonGate):
    """The gate of the hand-off lessons. It judges like every :class:`LessonGate`; only the inputs of
    the generated lessons are drawn where the worlds have them (:data:`DRAWS`) instead of evenly."""

    def generate(self, rng: random.Random, n: int) -> list[Line]:
        names = sorted(self.rules)
        out: list[Line] = []
        tries = 0
        while len(out) < n:
            tries += 1
            if tries > 50 * n + 1000:
                raise RuntimeError("world7: the lesson rules reject almost every drawn input")
            name = rng.choice(names)
            if name in DRAWS:
                values = DRAWS[name](rng)
            else:
                values = [rng.randint(int(s.low), int(s.high)) if s.integer
                          else round(rng.uniform(s.low, s.high), s.places) for s in self.rules[name].inputs]
            ln = self.line(name, values)
            if ln is not None:
                out.append(ln)
        return out


LESSONS = World7Lessons(SIM, RULES)


def lesson_gate() -> LessonGate:
    return LESSONS


def handoffs(r: Rollout) -> list[tuple[str, tuple, float | int | str]]:
    """Every hand-off of one world as ``(rule, inputs, value the next era prints)``: the inputs are
    numbers its eras and records print, so each can be checked against :data:`RULES`."""
    era, p = _eras(r.steps), r.params
    first, enriched, cloud, collapse, star = (era[n] for n in FIXED_ERAS)
    z = enriched["metallicity"]
    out: list[tuple[str, tuple, float | int | str]] = [
        ("helium_share", (first["hydrogen"], first["helium"]), first["helium"]),
        ("formation_year", (p["formation_time"],), cloud["age"]),
        ("enrichment_years", (p["formation_time"],), enriched["duration"]),
        ("era_end", (enriched["age"], enriched["duration"]), cloud["age"]),
        ("cooling", (z, p["cooling"]), cloud["cooling"]),
        ("cloud_mass", (p["cloud_mass"], z), cloud["cloud_mass"]),
        ("era_end", (collapse["age"], collapse["duration"]), star["age"]),
        ("star_mass", (cloud["cloud_mass"], collapse["star_share"]), star["star_mass"]),
        ("fusion", (star["star_mass"],), star["fusion"]),
        ("luminosity", (star["star_mass"],), star["luminosity"]),
        ("disc_solids", (collapse["disc_share"], cloud["cloud_mass"], z), star["disc_solids"]),
        ("disc_gas", (collapse["disc_share"], cloud["cloud_mass"], z), star["disc_gas"]),
    ]
    if star["fusion"] == "yes":
        out += [("lifetime", (star["star_mass"], star["luminosity"]), star["lifetime"]),
                ("era_end", (star["age"], star["lifetime"]), star["star_end"]),
                ("frost_line", (star["luminosity"],), star["frost_line"]),
                ("hz_inner", (star["luminosity"],), star["hz_inner"]),
                ("hz_outer", (star["luminosity"],), star["hz_outer"])]
        if star["disc_solids"] > 0:
            out.append(("in_time", (star["age"], SPAN["planets"], star["star_end"]),
                        "yes" if "planets" in era else "no"))
    for surface in (s for s in r.steps if _base(s["era"]) == "surface"):
        k = _number(surface["era"])
        body = (surface["mass"], surface["gas"])
        out += [("era_end", (star["age"], SPAN["planets"]), surface["age"]),
                ("flux", (star["luminosity"], surface["orbit"]), surface["flux"]),
                ("zone", (surface["flux"],), "temperate"),
                ("surface", body, surface["surface"]),
                ("t_eq", (surface["flux"],), surface["t_eq"]),
                ("greenhouse", (surface["flux"], *body), surface["greenhouse"]),
                ("temperature", (surface["flux"], surface["greenhouse"]), surface["temperature"]),
                ("water", (surface["temperature"], *body), surface["water"])]
        if not ocean(surface):
            continue
        vat = era.get(f"chemistry_{k}")
        out.append(("in_time", (surface["age"], SPAN["chemistry"], star["star_end"]), "yes" if vat else "no"))
        if vat is None or vat["precursors"] <= 0:
            continue
        age, eras = _ends(vat), {"surface": surface, "chemistry": vat}
        for level in LEVELS:
            e, span = era.get(f"{level.name}_{k}"), SPAN[level.name]
            if e is None:
                out.append(("in_time", (age, span, star["star_end"]), "no"))
                break
            out += [("in_time", (age, span, star["star_end"]), "yes"),
                    ("duration", (e["tries"], span), e["duration"])]
            out += _level_handoffs(level, e, eras, r.seed)
            eras[level.name] = e
            if not level.upper(e):
                if e["tries"] < p["max_tries"]:
                    out.append(("in_time", (_ends(e), span, star["star_end"]), "no"))
                break
            age = _ends(e)
            if level is not LEVELS[-1]:
                out.append(("era_end", (e["age"], e["duration"]), age))
    return out


def _level_handoffs(level: Level, e: dict, eras: dict, seed: int) -> list[tuple[str, tuple, float | int | str]]:
    surface, name = eras["surface"], level.name
    if name == "replicators":
        vat = eras["chemistry"]
        return [("monomers", (vat["precursors"],), e["monomers"]), ("inflow", (vat["sparks"],), e["inflow"]),
                ("hydrolysis", (surface["temperature"],), e["hydrolysis"])]
    below = eras[LEVELS[LEVELS.index(level) - 1].name]
    if name == "cells":
        drawn = level.draws(try_seed(seed, _number(e["era"]), level.number, e["kept"]))["gene_length"]
        return [("fidelity", (below["mu"],), e["fidelity"]), ("nutrient", (below["inflow"],), e["nutrient"]),
                ("gene_length", (drawn, e["l_max"]), e["gene_length"])]
    if name == "bodies":
        return [("efficiency", (below["energy_per_cell"],), e["transfer"]), ("light", (surface["flux"],), e["light"])]
    if name == "senses":
        return [("body_size", (below["consumer_size"], below["max_size"]), e["body_size"]),
                ("predators", (below["predator_share"], below["trophic_levels"]), e["predators"]),
                ("brightness", (surface["flux"],), e["brightness"])]
    if name == "signals":
        return [("neurons", (below["neurons"],), e["brain"]), ("band", (below["agents"],), e["band"])]
    return []


def handoff_lines(r: Rollout) -> list[Line]:
    """The hand-offs of one world as lessons, asked with the numbers its eras print."""
    out, seen = [], set()
    for name, inputs, _ in handoffs(r):
        ln = LESSONS.line(name, list(inputs))
        if ln is None:
            raise ValueError(f"world {r.seed}: hand-off {name} {inputs} is outside the lesson's ranges")
        if ln.text not in seen:
            seen.add(ln.text)
            out.append(ln)
    return out


# ---------------------------------------------------------------------------------------------
# prompts and their judgement

_N = r"(0|[1-9](?: \d)*)"
_ERA_Q = re.compile(rf"^world7 seed {_N} era ([a-z]+)(?:_([1-9]))? ([a-z0-9_]+)$")
_SEED_Q = re.compile(rf"^world7 seed {_N} (final|param|why) ([a-z0-9_]+)$")
_TIMELINE_Q = re.compile(rf"^world7 seed {_N} timeline$")
_TALK_Q = re.compile(rf"^world7 seed {_N} (word|meaning|say|order)((?: [a-z_]+)*)$")
_TABLE_Q = re.compile(r"^world7 (constant|span) ([a-z0-9_]+)$")
_LADDER_Q = re.compile(r"^world7 ladder after ([a-z_]+)$")


def parse(prompt: str) -> tuple | None:
    """A world7 prompt as ``(kind, ...)``; ``None`` when it is not one. Kinds: ``("era", seed, era,
    key)``, ``("final" | "param" | "why", seed, key)``, ``("timeline", seed)``, ``("word" | "meaning"
    | "say" | "order", seed, words)``, ``("constant" | "span", name)``, ``("ladder", rung)``. Lessons
    (``world7 predict ...``) are :data:`LESSONS`'s."""
    m = _ERA_Q.match(prompt)
    if m:
        base, number, key = m[2], m[3], m[4]
        if base not in ERAS or (number is not None) != (base in PLANET_ERAS) or key not in ERAS[base]:
            return None
        seed = int(parse_num(m[1]))
        return ("era", seed, base + (f"_{number}" if number else ""), key) if _is_seed(seed) else None
    m = _SEED_Q.match(prompt)
    if m:
        known = {"final": SUMMARY_KEYS, "param": PARAM_KEYS, "why": LADDER}[m[2]]
        seed = int(parse_num(m[1]))
        return (m[2], seed, m[3]) if m[3] in known and _is_seed(seed) else None
    m = _TIMELINE_Q.match(prompt)
    if m:
        seed = int(parse_num(m[1]))
        return ("timeline", seed) if _is_seed(seed) else None
    m = _TALK_Q.match(prompt)
    if m:
        seed, words = int(parse_num(m[1])), tuple(m[3].split())
        if not _is_seed(seed):
            return None
        if m[2] == "word" and len(words) == 1 and words[0] in signals.VOCABULARY + signals.PROPERTIES:
            return "word", seed, words
        if m[2] == "say" and len(words) == 2 and words[0] in signals.VOCABULARY and words[1] in signals.PROPERTIES:
            return "say", seed, words
        if m[2] == "meaning" and 1 <= len(words) <= 2 * signals.MAX_SYLLABLES \
                and all(len(w) == 2 and w[0] in signals.CONSONANTS and w[1] in signals.VOWELS for w in words):
            return "meaning", seed, words
        if m[2] == "order" and not words:
            return "order", seed, words
        return None
    m = _TABLE_Q.match(prompt)
    if m:
        return (m[1], m[2]) if m[2] in (CONSTANTS if m[1] == "constant" else SPAN) else None
    m = _LADDER_Q.match(prompt)
    if m:
        return ("ladder", m[1]) if m[1] in LADDER else None
    return None


def owns(prompt: str) -> bool:
    return parse(prompt) is not None or LESSONS.owns(prompt)


def _said(q: tuple) -> list[str] | None:
    """Every answer the language of a world accepts for a question about it, the one the lines give
    first; ``None`` when the world has no language."""
    language = _replay(q[1]).language
    if language is None:
        return None
    tongue, words = language.languages[signals.STEPS], q[2]
    if q[0] == "order":
        return [tongue["order"]]
    if q[0] == "say":
        return [signals.text_of(u) for u in signals.utterances_for(tongue, *words)] or ["none"]
    if q[0] == "word":
        return [signals.text_of(tongue["lexicon"].get(words[0]) or tongue["properties"].get(words[0]))]
    return signals.meanings_of(tongue, tuple(words)) or ["none"]


def _truth(q: tuple) -> float | int | str | None:
    kind = q[0]
    if kind == "constant":
        return CONSTANTS[q[1]]["value"]
    if kind == "span":
        return SPAN[q[1]]
    if kind == "ladder":
        i = LADDER.index(q[1])
        return LADDER[i + 1] if i + 1 < len(LADDER) else "none"
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
    if kind == "why":
        return (" ".join(links_held(r.steps)) or "nothing") if r.summary["rung"] == q[2] else None
    said = _said(q)
    return None if said is None else said[0]


def truth(prompt: str) -> float | int | str | None:
    """The chain's own value for a prompt; ``None`` when the prompt is not a world prompt or cannot
    be answered (an era the seed does not have, a ``why`` of another rung, a world without language)."""
    q = parse(prompt)
    return None if q is None else _truth(q)


def _judge(value: float | int | str, answer: str, exact: bool = False) -> Verdict:
    """Words and whole numbers exactly; a measured number when the answer rounds to the three digits
    printed; ``exact`` numbers (parameters, constants) exactly."""
    expected, answer = (table_value if exact else dense_value)(value), answer.strip()
    if isinstance(value, str):
        ok = answer == value
        return Verdict(ok, expected, "exact word" if ok else "wrong word")
    got = parse_num(answer)
    if not is_finite(got):
        return Verdict(False, expected, "not a number")
    if exact or isinstance(value, int):
        ok = got == value
        return Verdict(ok, expected, "exact" if ok else "not the exact value")
    ok = num(sig(float(got))) == expected
    return Verdict(ok, expected, "the three digits printed" if ok else "not the three digits printed")


def check(prompt: str, answer: str) -> Verdict:
    """Replay the world a prompt names (or look up the table, or recompute the lesson) and compare."""
    if LESSONS.owns(prompt):
        return LESSONS.check(prompt, answer)
    q = parse(prompt)
    if q is None:
        return Verdict(False, None, "not my question")
    if q[0] in ("word", "meaning", "say", "order"):
        said = _said(q)
        if said is None:
            return Verdict(False, None, f"world7 seed {q[1]} has no language")
        ok = answer.strip() in said
        return Verdict(ok, said[0], "what the settled language gives" if ok else "not in the language")
    value = _truth(q)
    if value is None:
        if q[0] == "era":
            return Verdict(False, None, f"world7 seed {q[1]} has no era {q[2]}")
        return Verdict(False, None, f"world7 seed {q[1]} did not end at {q[2]}")
    return _judge(value, answer, exact=q[0] in ("param", "constant", "span") or q[::2] == ("final", "formation_time"))


# ---------------------------------------------------------------------------------------------
# lines


def params_line(r: Rollout) -> Line:
    """``world7 seed 3 params. formation_time 4. efficiency 0 point 6. cloud_mass 2. ...``"""
    fields = " ".join(f"{k} {table_value(r.params[k])}." for k in PARAM_KEYS)
    return Line(f"{SIM} seed {num(r.seed)} params. {fields}", topic=SIM, kind="record")


def param_lines(r: Rollout) -> list[Line]:
    return [Line(f"{SIM} seed {num(r.seed)} param {k}", table_value(r.params[k]), SIM, "fact") for k in PARAM_KEYS]


def timeline_line(r: Rollout) -> Line:
    """``world7 seed 4 timeline. eras nucleo stars cloud collapse star. rung no_star. outcome no_star.``"""
    names = " ".join(s["era"] for s in r.steps)
    return Line(f"{SIM} seed {num(r.seed)} timeline. eras {names}. rung {r.summary['rung']}. "
                f"outcome {r.summary['outcome']}.", topic=SIM, kind="record")


def era_line(r: Rollout, t: int) -> Line:
    """``world7 seed 3 era cloud. age 4 0 0 0 0 0 0 0 0 0. cloud_mass 2. spin 0 point 0 6. cooling 0 point 5 7.``"""
    step = r.steps[t]
    fields = " ".join(f"{k} {dense_value(step[k])}." for k in ERAS[_base(step["era"])])
    return Line(f"{SIM} seed {num(r.seed)} era {step['era']}. {fields}", topic=SIM, kind="record")


def drawn_line(r: Rollout, t: int) -> Line | None:
    """The free parameters the kept try of a level drew from its seed (``None`` for other eras):
    ``world7 seed 3 era cells_1 drawn. gene_types 4. gene_drawn 8 0. selfish 0 point 0 2. ...``"""
    step = r.steps[t]
    level = LEVEL_OF.get(_base(step["era"]))
    if level is None:
        return None
    draws = level.draws(try_seed(r.seed, _number(step["era"]), level.number, step["kept"]))
    handed = {param for _, param in level.shown} | {"complex_cells"}
    renamed = DRAWN_RENAMED.get(level.name, {})
    fields = " ".join(f"{renamed.get(k, k)} {table_value(v)}." for k, v in draws.items() if k not in handed)
    return Line(f"{SIM} seed {num(r.seed)} era {step['era']} drawn. {fields}", topic=SIM, kind="record")


def era_queries(r: Rollout, t: int) -> list[Line]:
    step = r.steps[t]
    return [Line(f"{SIM} seed {num(r.seed)} era {step['era']} {k}", dense_value(step[k]), SIM, "fact")
            for k in ERAS[_base(step["era"])]]


def final_lines(r: Rollout) -> list[Line]:
    return [Line(f"{SIM} seed {num(r.seed)} final {k}", table_value(v) if k == "formation_time" else dense_value(v),
                 SIM, "calc") for k, v in r.summary.items()]


def why_line(r: Rollout) -> Line:
    """``q world7 seed 4 why no_star. a metals.``: the links of the chain that held."""
    return Line(f"{SIM} seed {num(r.seed)} why {r.summary['rung']}", " ".join(links_held(r.steps)) or "nothing",
                SIM, "calc")


def language_lines(r: Rollout) -> list[Line]:
    """The language of the furthest planet as level 11 writes it, under the world's seed: the
    lexicon, the grammar and the irregular phrases as records, and the questions ``word``,
    ``meaning``, ``say`` and ``order``. Empty for a world below the rung ``signals``."""
    language = getattr(r, "language", None)
    if language is None:
        return []
    theirs, mine = f"{signals.SIM} seed {num(language.seed)}", f"{SIM} seed {num(r.seed)}"
    out = []
    for ln in signals.language_lines(language):
        if not ln.prompt.startswith(theirs + " "):
            raise ValueError(f"an unexpected language line: {ln.prompt!r}")
        out.append(Line(mine + ln.prompt[len(theirs):], ln.answer, SIM, ln.kind, dict(ln.meta)))
    return out


def records() -> list[Line]:
    """The seedless records: the ladder, the spans, and one record per hand-off rule."""
    spans = " ".join(f"{name} {num(years)}." for name, years in SPAN.items())
    return [Line(f"{SIM} ladder. links {' '.join(LINKS)}. rungs {' '.join(LADDER)}.", topic=SIM, kind="record"),
            Line(f"{SIM} spans. {spans}", topic=SIM, kind="record")] + LESSONS.records()


def table_lines() -> list[Line]:
    """Every question the tables answer (fixed; no seed involved)."""
    out = [Line(f"{SIM} constant {name}", table_value(row["value"]), SIM, "fact") for name, row in CONSTANTS.items()]
    out += [Line(f"{SIM} span {name}", num(years), SIM, "fact") for name, years in SPAN.items()]
    return out + [Line(f"{SIM} ladder after {rung}", LADDER[i + 1] if i + 1 < len(LADDER) else "none", SIM, "fact")
                  for i, rung in enumerate(LADDER)]


def lines(r: Rollout) -> list[Line]:
    """Every line of one world, which must be the world its seed names (:func:`rollout`): the
    parameter and timeline records, a record per era (and one of the draws of a level's kept try),
    then the questions: every metric of every era, the summary, the parameters, the timeline, the
    ``why``, the language if there is one, and the hand-offs as they happened."""
    if not canonical(r):
        raise ValueError(f"lines only for the world a seed names (rollout({r.seed})): the gate replays that one")
    _remember(r)
    out = [params_line(r), timeline_line(r)]
    for t in range(len(r.steps)):
        out.append(era_line(r, t))
        drawn = drawn_line(r, t)
        if drawn is not None:
            out.append(drawn)
    for t in range(len(r.steps)):
        out += era_queries(r, t)
    out += final_lines(r) + param_lines(r)
    out.append(Line(f"{SIM} seed {num(r.seed)} timeline", " ".join(s["era"] for s in r.steps), SIM, "fact"))
    out.append(why_line(r))
    out += language_lines(r) + handoff_lines(r)
    for ln in out:
        if tokens(ln.text) > MAX_TOKENS:
            raise ValueError(f"a line of {tokens(ln.text)} tokens, more than {MAX_TOKENS}: {ln.text[:60]} ...")
    return out


#: practice hand-off lessons :func:`generate` adds per world
PRACTICE_PER_SEED = 40


def generate(rng: random.Random, n: int) -> list[Line]:
    """``n`` lines: the table questions, then world after world on seeds drawn from ``rng`` (each
    with :data:`PRACTICE_PER_SEED` practice lessons) until there are ``n``. A world costs seconds;
    to build training data from whole worlds, call ``lines(rollout(seed))`` per seed instead."""
    if n <= 0:
        return []
    out = table_lines()
    for seed in rng.sample(range(1, 9999), min(9998, n // 100 + 1)):
        if len(out) >= n:
            break
        out += lines(run(seed)) + LESSONS.generate(rng, PRACTICE_PER_SEED)
    return out[:n]


class World7:
    """The chain as a :class:`haishool.cosmos.Simulation` and a :class:`haishool.truth.Gate`."""

    sim = SIM
    topic = SIM
    KEYS = KEYS

    def run(self, seed: int, **params: float | int) -> WorldRollout:
        return run(seed, **params)

    def rollout(self, seed: int) -> WorldRollout:
        return rollout(seed)

    def random_params(self, rng: random.Random, rules: int = 7) -> dict[str, float]:
        return random_params(rng, rules)

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


_SIM = World7()


def simulation() -> World7:
    return _SIM


#: what seeds 1 to 300 give on this code: worlds per rung, per outcome, and per rung by now
CENSUS: dict[str, dict[str, int]] = {
    "rung": {"no_metals": 4, "no_star": 41, "no_planets": 1, "no_temperate": 72, "no_water": 24, "no_organics": 2,
             "sterile": 18, "replicators": 34, "cells": 54, "complex_cells": 1, "bodies": 1, "senses": 19,
             "groups": 2, "signals": 22, "language": 3, "society": 1, "states": 1},
    "outcome": {"no_metals": 4, "no_star": 41, "no_planets": 1, "no_temperate": 72, "no_water": 24, "sterile": 4,
                "replicators": 2, "cells": 37, "senses": 19, "groups": 1, "signals": 21, "language": 3,
                "society": 1, "states": 1, "out_of_time": 69},
    "by_now": {"no_metals": 4, "no_star": 41, "no_planets": 1, "no_temperate": 72, "no_water": 24, "no_organics": 2,
               "sterile": 18, "replicators": 92, "cells": 15, "complex_cells": 12, "bodies": 3, "senses": 7,
               "groups": 1, "signals": 7, "states": 1},
}
