"""Level 3 of the toy universe: a protoplanetary disc grows planets.

This text describes round 6, the default. Round 7 adds corrected rules as an option
(``rules=7``); they are described in the section "Rules 7" at the end.

Not an N-body code. A star of mass ``m_star`` (0.5-1.5 solar) is ringed by a disc of solids
whose surface density falls as r^-1.5 from 0.3 to 30 au and adds up to ``disc_mass`` earth
masses (10-300). The star's luminosity follows from its mass (L = M^3.5) and puts the frost line
at r_frost = 2.7 au * sqrt(L): beyond it ice triples the solid surface density. Gas (100 times
the rock mass, no ice; 30 % of it within a planet's reach, the rest falls onto the star or is
blown away) sits in the disc until ``t_gas`` Myr (1-10), fading linearly.

**Accretion.** The disc is cut into 24 logarithmic zones. Each zone holds one embryo that sweeps
up the zone's solids (toy law, integrated exactly):

    dM/dt = eps * Sigma_rem * A_zone / (TAU_SWEEP * P)      S_rem(t) = S_0 * exp(-eps t / (TAU_SWEEP P))

proportional to the remaining surface density Sigma_rem (times the zone area A_zone), inversely
to the orbital period P = sqrt(a^3 / m_star) years. TAU_SWEEP = 0.2 Myr is the e-folding time of
a zone's solids at 1 au around 1 solar mass; ``eps`` is a seeded efficiency per zone (0.6-1.4),
the first stochastic element.

**Collisions**, the second. Two neighbouring embryos are unstable when their orbits can come
within K_HILL = 2 sqrt(3) mutual Hill radii of each other (the two-planet Hill stability limit):

    (a2 - a1) - ecc * (a1 + a2) < K_HILL * R_hill      R_hill = ((m1 + m2) / (3 m_star))^(1/3) * (a1 + a2) / 2

with ecc = E_GAS = 0.05 while gas damps the orbits and ecc = E_LATE = 0.2 once the gas is gone.
Both are toy values, as 24 zones are coarser than real embryo spacing: damped, only heavy
neighbours meet; stirred, the orbits of neighbouring zones cross, as in the late stage of giant
impacts. An unstable pair merges with a seeded chance per Myr (e-folding TAU_MERGE_GAS = 1 Myr,
TAU_MERGE_LATE = 3 Myr); the merged body sits at the mass-weighted orbit and feeds on both
zones. With probability P_EJECT = 0.3 a merger throws a fraction (0.02-0.2) of the combined
solids out as debris, which is counted and never comes back.

**Gas.** An embryo beyond the frost line whose solid mass passes ``core_threshold`` (10 earth
masses) while gas is present runs away, capturing gas at dM_gas/dt = M / TAU_RUNAWAY (0.3 Myr),
never more than its zone still holds within reach (0.3 * 100 * rock, fading linearly to zero at
``t_gas``). A core of 5-10 earth masses beyond the frost line captures a little gas,
dM_gas/dt = M_core / TAU_ENVELOPE (10 Myr).

**Types** at any moment: ``gas`` when the envelope outweighs the core; ``ice`` when a body
beyond the frost line has 5 or more earth masses of solids (a core that only grew heavy after
the gas had gone stays an ice giant); ``rocky`` for every other body, including small icy bodies
far out. A planet is a body of at least 0.1 earth masses. **Stages**: ``dust`` (no planet yet),
``embryos``, ``giants_forming`` (gas is there and a core beyond the frost line has reached the
threshold), ``clearing`` (gas gone, some neighbours still unstable), ``done`` (gas gone, every
pair stable). A finished system turns to ``clearing`` again when its planets, still growing,
come into each other's reach.

Output: 30 steps at fixed times up to 50 Myr (:data:`TIMES`), each a dict of the metrics in
:data:`STATE_KEYS` plus a ledger (:data:`LEDGER_KEYS`) that :meth:`Planets.conserved` checks.
Dense lines (:func:`lines`), measured numbers to 3 significant digits, the parameters that were
put in as given:

    planets seed 3 params. m_star 0 point 7 4. disc_mass 1 6 7 point 8. t_gas 4 point 3 3. core_threshold 1 0. ...
    planets seed 3 step 1 4. time 4. planets 2 2. rocky 1 5. ice 5. gas 2. largest_mass 4 0 point 8. ...
    q planets seed 3 step 1 4 gas. a 2.
    q planets seed 3 step 1 4 next stage. a clearing.       (what happens next)
    q planets seed 3 final habitable. a 1.                   (how it ends)

Gates (:meth:`Planets.conserved`): solids in bodies + solids left in zones + debris = initial
solids, to 1e-9 earth masses; the debris is the sum of what the logged mergers threw out;
captured gas never exceeds the gas there was; planets = rocky + ice + gas; a body's mass never
drops and no body vanishes except in a logged merger, and a merger loses no more than its logged
debris; gas is only captured beyond the frost line; time strictly increasing.
:meth:`Planets.check` replays the seed (parameters from :func:`random_params` with
``random.Random(seed)``) and compares with a 5 % tolerance for measured numbers, exactly for
counts and words. A prompt has one spelling (no leading zeros), and an answer must be a finite
number.

Plausibility targets (what the systems must look like; ``tests/test_cosmos_planets.py`` holds
them), with the values measured over the canonical seeds 1 to 2000:

    every seed        the gates pass; the solids add up                 worst error 2.3e-13 earth masses
                      starts as ``dust``, ends ``clearing`` or ``done``  done 1998, clearing 2
                      2 to 10 planets at the end, from up to 24         3 .. 9, median 6; 4 .. 8 in 99.8 %
                      no gas or ice giant inside the frost line         0 of 5919 at the end; on the way 1
                                                                        (seed 1874, carried in by a merger)
    most seeds        a rocky planet at the end                         1999 of 2000 runs
                      rocky planets sit inside the frost line           93 % of them (the rest: small icy bodies)
                      gas giants sit a few au out                       1.05 .. 26 frost lines, median 2.4
                      a rocky planet in the habitable zone              0 in 292 runs, 1 in 1653, 2 in 55
    the rules         no gas giant from a light disc                    none below 124 earth masses of solids
                      the heavier the disc, the more gas giants         runs with one: 0 % below 100, 10 % at
                                                                        100-150, 50 % at 150-200, 76 % above 200
                      the longer the gas lasts, the more gas giants     discs above 200: 0.27 giants each when
                                                                        t_gas < 3, 1.29 at 3-6, 2.07 above 6
                      gas giants weigh about as much as real ones       31 .. 1460 earth masses, median 258
                                                                        (Saturn 95, Jupiter 318)

What is real and what is toy. Real: Kepler's third law for the periods; the Hill radius and the
2 sqrt(3) stability limit of two planets; a frost line that scales with sqrt(L), and L = M^3.5
for stars like the sun; the r^-1.5 disc with more solids beyond the frost line; a core of about
10 earth masses that runs away with gas while the gas lasts; the book-keeping of mass (the
constants 2.7 au, 3.5, the factor 3 for ice and the 10 earth masses are textbook values written
down from memory). Toy, tuned to give plausible systems and not derived: the sweep law and its time TAU_SWEEP (real growth
goes through runaway and oligarchic stages and pebbles); one embryo per zone and only 24 zones;
the two eccentricities and the two merger times (a real late stage is chaotic N-body dynamics,
here it is a seeded coin per pair and substep); debris as a single seeded fraction; the gas
budget and capture times; no migration, no resonances, no photo-evaporation, no moons. So the
inner ``rocky`` planets come out heavy (median 2.9 earth masses, up to 31, which no real disc
would leave without an atmosphere), and a star like the sun grows a gas giant only from about
200 earth masses of solids (t_gas 5 Myr), more than its own disc is usually reckoned to have
held. The numbers of a run tell the toy's rules; they are no forecast of a real system.

Reproducibility. Random numbers are the raw PCG64 stream of the seed, which numpy keeps fixed
across versions, turned into fractions here (:class:`_Rng`) and not by ``Generator``, whose
methods may change; sums that enter the state are exactly rounded (``math.fsum``); every float
array is float64; nothing is sorted. The elementary functions (exp, cbrt, power) may differ in
the last digit between machines and numpy builds, so two machines need not agree bit for bit.
The lines agree all the same as long as no decision sits on a knife's edge: a change of 1e-12
in the star and disc mass, ten thousand times such a difference, moves not one line in seeds
1 to 2000. The one place where two times did meet exactly, a mid-step time against ``t_gas``,
has an explicit rule (:data:`T_EPS`).

Rules 7
=======

Everything above describes round 6, which stays as it is: the version-5 models were trained on
its rollouts. A review of its rules found one that contradicts the next level, one seeded coin
with no model behind it, and several numbers that are off from the published ones. The
corrections are an option, never a change of the default:

    run(seed, rules=7, **random_params(random.Random(seed), rules=7))     one corrected run
    rollout(seed, rules=7)                 the same, cached: the run a seed stands for
    simulation(rules=7)                    the level with rules 7 as the default of its methods
    lines(rollout)                         the lines; the rollout says which rules it is of

A rules-7 rollout carries ``params["rules"] == 7`` (a round-6 rollout has no such key), names its
parameters as the other levels do (``star_mass``, ``frost_line``, ``hz_inner``, ``hz_outer``) and
keeps the finished planets with their climate in ``rollout.system``. Its run is plain Python on
floats; numpy only supplies the raw PCG64 bits, in round 6's order (the 24 zone efficiencies,
then one draw per unstable pair and substep; a meeting itself draws nothing any more).

What changed, the published value each rule follows, and what stays toy:

1. **Habitable zone by flux** (:func:`flux`, :func:`climate`, :func:`habitable`). Round 6 put the
   zone at 0.95-1.7 au * sqrt(L), and level 6 then gave a planet 278.3 K * L^(1/4) / sqrt(a) + 33 K:
   two climates, by which 42 % of the "habitable" planets were frozen. Now one function decides:
   flux = L / a^2 in units of the earth's, habitable between the maximum-greenhouse limit 0.356
   and the runaway-greenhouse limit 1.107 (after Kopparapu et al. 2013, 2014, the values for the
   sun and one earth mass, from memory), that is 0.950 to 1.676 au * sqrt(L). The temperature that
   goes with it (:func:`surface_temperature`, for the chain) is the equilibrium temperature for a
   Bond albedo of 0.3, 254.6 K * flux^(1/4) (the earth's effective temperature is 255 K), plus a
   greenhouse of 33 K at the earth's flux. Toy: the two limits are the sun's for every star (the
   published ones are lower for a cooler star); the greenhouse between the outer edge and the
   earth's flux is a straight line, 33 + 67.3 * (1 - flux), standing in for the carbonate-silicate
   thermostat, with 67.3 set so that the outer edge is at 273 K; above 1.107 it is a flat 500 K
   (about Venus: 737 K at the surface against some 230 K effective; a small planet that lost its
   air, as Mercury did, would stay near its equilibrium temperature instead); below 0.356 it is 0
   (that no greenhouse of CO2 holds 273 K there is what the limit means; a real frozen planet is
   brighter and colder still, or keeps a few kelvin of greenhouse as Mars does). A gas giant has
   no surface: its temperature is the equilibrium temperature alone (112 K at Jupiter's orbit;
   Jupiter radiates at about 124 K, part of it its own heat). Result: the earth 288 K, the zone
   273 to 294 K, just inside the inner edge 761 K, the orbit of Venus 799 K, the orbit of Mars
   278 K (Mars itself, at about 210 K, is too light to hold the air the rule assumes), just beyond
   the outer edge 197 K. So water is liquid (273 to 373 K) exactly where the flux is temperate; the
   gate holds every planet with a surface to it. (Until the review of this section the straight
   line ran on beyond the outer edge and gave a frozen planet 100 to 273 K, 147 K at Neptune's
   flux where the equilibrium temperature is 46 K, and a hot Jupiter got the 500 K of a runaway
   greenhouse; both were wrong physics and are gone.)
2. **Planets throw each other out** (:func:`safronov`, :func:`scatters`). Round 6 merged every
   unstable pair and threw out a seeded 2-20 % of the solids in 30 % of the mergers, so solid
   bodies of 100-240 earth masses piled up (Uranus and Neptune weigh 14.5 and 17.1 in all). Real: a
   body whose escape speed at the surface exceeds its orbital speed scatters what it meets out
   of the system instead of swallowing it. The number used is (v_escape / v_orbit)^2 =
   2 (m / M) (a / R), for a body of the earth's density 0.141 * m^(2/3) * a / M: 0.141 for the earth,
   34 for Jupiter (21 with its real radius; gas giants are larger, far above 1 either way). The
   code calls it the Safronov number, loosely: Safronov's own number compares the escape speed
   with the random speed of the bodies, and (m / M) (a / R), half of this one, is the squared
   "Safronov number" of Ford and Rasio 2008; that scattering turns from accretion to ejection
   where the escape speed passes the orbital speed is the rule of thumb of, for one, Wyatt et al.
   2017 (both from memory). When bodies meet, the heaviest decides: from 1 on it ejects the
   others with their gas and the unswept solids of their zones and stays where it is; below 1
   they merge without loss. Toy: the threshold of exactly 1 (a real body just above it ejects
   slowly and still swallows much of what it meets), the earth's density for every body, that a
   scattered body always leaves (none is merely moved outward), and that the cleared zones leave
   at once. The cost is open: two thirds of the solids leave the system, because the outer zones
   hold most of them and a body there reaches 1 at 0.6 earth masses (at 10 au around the sun).
3. **Own prompts, shared names.** A rules-7 story says ``rules 7`` after its seed, so no prompt
   has two answers; counts are named for what they count (``gas_giants``, ``ice_giants``,
   ``ejected`` next to the fraction ``debris``); the lessons have the word ``planets7``.
4. **Parameters** (``random_params(rng, rules=7)``, the same draws in the same order). Gas discs do
   not last uniformly 1-10 Myr: the share of young stars with a disc falls roughly as
   exp(-t / 2.5 Myr) (after Mamajek 2009; Haisch, Lada and Lada 2001: half gone by 3 Myr, nearly
   all by 6). Now ``t_gas`` = 1 Myr + an exponential of mean 2 Myr, at most 10
   (:func:`gas_lifetime`): 37 % of the discs are left at 3 Myr and 8 % at 6 (of the seeds 1 to
   2000: 36 % and 8 %; exp(-t / 2.5 Myr) itself gives 30 % and 9 %). And the solids scale with
   the star, ``disc_mass`` = ``star_mass`` * uniform(10, 300) earth masses (observed dust masses
   rise at least linearly with the star's mass). Toy: the shift of 1 Myr, the cap at 10, the
   uniform range.
5. **Luminosity and grid** (:func:`luminosity7`). M^3.5 only fits near one solar mass. Now the
   piecewise textbook law: 0.23 M^2.3 below 0.43 solar masses, M^4 to 2, 1.4 M^3.5 to 55, 32000 M
   above (after Duric 2004 and Salaris and Cassisi 2005, from memory): 0.0625 instead of 0.088 at
   0.5 solar masses, 5.06 instead of 4.13 at 1.5. The frost line stays 2.7 au * sqrt(L). The zone
   edges scale with it, sqrt(L) * 0.3 * 100^(i / 24) au, so frost line and habitable zone fall on
   the same zones for every star (the embryos of zones 6 to 8, at the zones' geometric middles, start
   in the habitable zone; zone 11 is the first icy one) and no star is shut out: round 6 could give no star below 0.39 solar masses a
   habitable planet, because its grid began at 0.3 au. Toy: the outer edge scaling with sqrt(L)
   (a dust disc's inner rim does), and the main-sequence luminosity although the young star was
   brighter.
6. **Migration** (:func:`migrated_orbit`). Round 6 had no giant inside the frost line because
   nothing moved. Real: a planet that has opened a gap drifts inward with the disc gas (type II
   migration, after Lin and Papaloizou 1986) until the gas is gone; about 0.5-1 % of sun-like
   stars have a hot Jupiter (transit surveys find about half a percent, radial-velocity ones about one) (51 Peg b, the first planet found around such a star, sits at
   0.05 au). Now a body of type ``gas`` drifts while gas is present, da/dt = -(a / tau) * (1 - t /
   t_gas) with tau = TAU_MIG7 * a^1.5 / sqrt(M), integrated exactly per substep, and stops at
   A_PARK7. A giant that reaches its inner neighbour's orbit meets it at once (rule 2), and a
   body that has left its zone stops sweeping it. The frost-line gate on gas capture is gone (the
   core threshold alone decides; the review found it changed nothing, inner bodies get heavy
   only after the gas has left). Toy: TAU_MIG7 = 1 Myr at 1 au (about a million orbits), A_PARK7 =
   0.05 au for every star (the grid of a star below 0.4 solar masses begins inside it, and a giant
   there parks among the inner planets; ``random_params`` draws 0.5 to 1.5 solar masses), no
   type I migration of the small bodies, no resonances.
7. **Types** (:func:`planet_type`). Round 6's ``rocky`` was the rest class and held bodies of up
   to 58 earth masses and every small body beyond the frost line. Now: ``gas`` when the envelope
   weighs at least as much as the solids; beyond the frost line ``ice_giant`` from 5 earth masses
   of solids, else ``icy``; inside it ``super_earth`` from 10 earth masses of solids (the toy's own
   core threshold: a giant's core, had gas been left), else ``rocky``. Habitable is a rocky planet
   of 0.3 to 10 earth masses with a temperate flux that holds no disc gas: a core of 5 earth masses
   or more that met the gas keeps a hydrogen envelope (``rocky`` only says it is lighter than the
   core), and a thousandth of an earth mass of hydrogen is a sky of about a thousand bars (the
   earth's air is 0.86 millionths of its mass for 1 bar), no earth-like one (in the seeds 1 to 12000 no rocky planet in the zone holds any; with a core threshold of
   1 earth mass in a gas-poor disc many do).
   Toy: the 0.3 (between Mars, 0.107, which lost its air, and Venus, 0.815, which kept it) and the
   10; planets larger than about 1.6 earth radii, 5 to 6 earth masses of rock, mostly carry gas
   (after Rogers 2015) and the published zone is for 0.1-5 earth masses, but with 5 as the bound
   nearly half of the runs would have no habitable planet (955 of 2000), because the inner planets
   stay heavy (below). A planet is still a body of 0.1 earth masses or more.
8. **Gas budget** (``gas_mass``, optional). Round 6 ties the gas to the rock: 100 times a zone's
   rock, of which a planet can reach 30 %. With ``gas_mass`` earth masses of disc gas given, a zone
   holds its share of that gas by the same r^-1.5 profile (:func:`zone_gas`), so metals change the
   solids and not the gas. Without it the round-6 rule holds. GAS_REACH stays 0.3 in both (toy).
9. **Merged orbit** (:func:`merged_orbit`). A mass-weighted mean of two orbits is no conserved
   quantity; two bodies on circular orbits keep their orbital angular momentum m * sqrt(a). The
   merged body now sits at ((m1 sqrt(a1) + m2 sqrt(a2)) / (m1 + m2))^2, and the gate checks it
   (never outside the two orbits: two giants parked at 0.05 au merge at 0.05 au exactly).

Left as they were, and why: the frost line, the r^-1.5 disc with ice tripling the solids, one
embryo per zone (the zone's solids stand in for the isolation mass, :func:`zone_solids`), the
sweep law, the core threshold with its two capture times, Hill stability with 2 sqrt(3) and the
two eccentricities, the merger times and the output times; the review found them sound or
honestly labelled. EARTH_PER_SUN also stays round 6's 332954 (from rounded kilogram values; the
measured ratio is 332946, 2.5e-5 less), so that both rule sets share one Hill radius. Not taken
from the review: a second word in ``Rollout.sim`` (the prompt says ``rules 7`` instead, as the
other corrected levels do); PLANET_MIN = 0.05 (it would change the counts of light discs for
the sake of Mercury); GAS_REACH = 0.17 with ``gas_mass`` (measured on the seeds 1 to 2000 with
``gas_mass`` = 10000 earth masses per solar mass of star: the giants weigh 21.5 .. 399 earth
masses, median 84.5, about as with the round-6 budget, so nothing needs lowering).

Lines of a rules-7 rollout (:func:`lines`; only ``rollout(seed, rules=7)`` has lines, because the
gate replays the seed's own parameters): the kinds of round 6, plus one record and six questions
per planet of the finished system (planet 1 is the innermost):

    planets seed 3 rules 7 params. star_mass 0 point 7 4. disc_mass 1 2 4 point 2. t_gas 1 point 9 2. core_threshold 1 0. luminosity 0 point 3. frost_line 1 point 4 8. hz_inner 0 point 5 2. hz_outer 0 point 9 1 8.
    planets seed 3 rules 7 step 1 4. time 4. planets 1 3. rocky 7. super_earths 0. icy 2. ice_giants 4. gas_giants 0. ejected 7. largest_mass 7 point 0 2. innermost_au 0 point 1 8 1. outermost_au 1 4 point 9. debris 0 point 5. gas_present no. stage clearing.
    q planets seed 3 rules 7 step 1 4 ejected. a 7.
    q planets seed 3 rules 7 step 1 4 next stage. a clearing.
    planets seed 3 rules 7 planet 2. orbit 0 point 4 9 9. mass 4 point 3 8. type rocky. flux 1 point 2 1. temperature 7 6 7. habitable no.
    q planets seed 3 rules 7 planet 2 temperature. a 7 6 7.
    q planets seed 3 rules 7 final habitable. a 0.

and lessons that carry their inputs (:data:`LESSONS`, 24 rules, topic ``predict_planets7``); each
is computed by the very function the run calls:

    q planets7 predict flux luminosity 4 point 7 8 6 orbit 1 point 1 4 8. a 3 point 6 3 2.
    q planets7 predict surface_temperature flux 0 point 4 6 1. a 2 7 9.
    q planets7 predict scatters mass 9 point 5 7 3 orbit 0 point 1 0 7 star_mass 1 point 2 8. a no.
    planets7 predict flux. starlight a planet receives in units of what the earth receives. luminosity over orbit squared. luminosity in solar units. orbit in au.

Gates under rules 7 (:func:`_conserved7`): solids in bodies + solids left in zones + debris =
initial solids, to 1e-9 earth masses; the debris, the ejected gas and the count of ejected bodies
are the sums of the logged events; gas in bodies = gas captured - gas ejected; captured gas never
exceeds what the disc held; planets = rocky + super_earths + icy + ice_giants + gas_giants, the
counts are the types of the bodies, and every type follows from the body's solids, gas and
orbit; every logged meeting is an ejection by a body whose Safronov number is 1 or more (the
debris is exactly what the ejected bodies and their zones held, the survivor keeps mass and
orbit) or a merger below 1 that loses nothing and keeps sum(m sqrt(a)) to 1e-9; outside an event
no body loses mass or vanishes, and only a body with gas moves, inward, while the disc has gas,
never inside 0.05 au; the bodies stay in order of orbit; the final system is the planets of the
last step with flux, temperature and the habitable word recomputed; every habitable planet is a
rocky one without disc gas and has liquid water (273 to 373 K), and a planet with a surface has
liquid water exactly when its flux is temperate. :func:`check` judges ``planets seed <s> rules 7
...`` by the seed's rules-7 rollout: words, counts and whole kelvin exactly, a parameter that was
put in as given, measured numbers within 5 %.

Measured over the canonical seeds 1 to 2000 under rules 7 (``tests/test_cosmos_planets_rules7.py``
holds the targets on seeds 1 to 40), with round 6 in brackets:

    every seed        the gates pass; the solids add up                 worst error 1.1e-13 earth masses
                      ends ``done``                                     2000 of 2000; 2 to 9 planets, median 6
    solids per body   ice giants                                        5.0 .. 32.9, median 10.7 (5 .. 201, 26.9)
                      cores of gas giants                               10.0 .. 31.3, median 13.2 (10 .. 243, 61)
                      heaviest solid body of all                        32.9 earth masses (243)
    gas giants        runs with one                                     10.9 % (35.75 %); none from a disc below
                                                                        119 earth masses or with gas for under 1.3 Myr
                      what they weigh                                   22 .. 455 earth masses, median 82
                                                                        (Saturn 95, Jupiter 318)
                      inside the frost line                             3.0 % of runs (0); at 0.1 au or closer 0.85 %
    habitable         no planet, one, two                               471, 1499, 30 runs (292, 1653, 55)
                      their water is liquid                             1559 of 1559 planets, 273 .. 294 K (round 6:
                                                                        745 of 1763 frozen by level 6's rule)
                      liquid water on a surface outside the zone        none: 761 K or more inside, 197 K or less beyond
                      what they weigh                                   0.30 .. 9.99 earth masses, median 3.3
    debris            share of the solids that left                     0.16 .. 0.86, median 0.68 (median 0.04)
    small stars       a habitable planet at 0.1, 0.2, 0.3, 0.39 solar   160, 173, 173, 170 of 200 runs each, with 140
                      masses (round 6: none at 0.2 and 0.39)            earth masses of solids per solar mass, gas 3 Myr
    speed             slowest run                                       about 20 ms (most take 5 ms)

What is still wrong, stated plainly. The inner planets stay heavy (rocky median 2.6 earth masses,
32 % of the habitable ones above 5): a disc heavy enough to build a giant's core holds about six
times the solar system's inner rock, and no rule here removes it. 17 of the 5782 rocky planets
keep some disc gas (up to 2.4 earth masses on 9.5 of rock) and are still called ``rocky``, though
never habitable. Every rocky planet closer in than the zone gets Venus' greenhouse, the airless
ones too. Two thirds of the solids are ejected, which is high and untuned. The orbit of a body
that scatters others does not change. No pair of planets ends closer than 1.56 in orbit ratio
(Venus and the earth: 1.38). The stage word goes back from ``done`` to ``clearing`` in 7 of 2000
runs, as it could in round 6. The round-7 level ``haishool.evo.stars`` still uses M^3.5 for its
luminosity; the two agree only near one solar mass. The numbers of a run tell the toy's rules;
they are no forecast of a real system.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

import numpy as np

from haishool.cosmos import Rollout, close, query_lines, seeds, state_line, summary_lines
from haishool.evo import Input, LessonGate, Rule
from haishool.truth import Line, Verdict, is_finite, num, parse_num

SIM = "planets"

#: earth masses in one solar mass as round 6 has it: the rounded kilogram values 1.98847e30 kg /
#: 5.9722e24 kg (from memory) give 332954; the measured ratio is 332946 (IAU 2015 nominal GM values),
#: so this is 2.5e-5 high. Round 6 is frozen with it (only the cube root enters there, via the Hill
#: radius); rules 7 keeps it so that the two share one Hill radius.
EARTH_PER_SUN = 1.98847e30 / 5.9722e24

N_ZONES = 24
R_IN, R_OUT = 0.3, 30.0  # au
ICE_FACTOR = 3.0
FROST_AU = 2.7  # frost line of a 1 L_sun star
GAS_TO_ROCK = 100.0
GAS_REACH = 0.3  # share of a zone's gas a planet can capture
TAU_SWEEP = 0.2  # Myr, at 1 au around 1 solar mass
EPS_RANGE = (0.6, 1.4)
K_HILL = 2.0 * math.sqrt(3.0)  # mutual Hill radii; two-planet stability limit (Gladman 1993, from memory)
E_GAS = 0.05  # eccentricity while gas damps the orbits (toy value, about the disc's aspect ratio)
E_LATE = 0.2  # eccentricity of stirred orbits once the gas is gone (toy value)
TAU_MERGE_GAS, TAU_MERGE_LATE = 1.0, 3.0  # Myr
P_EJECT = 0.3
EJECT_FRACTION = (0.02, 0.2)
ICE_CORE = 5.0  # earth masses
TAU_RUNAWAY, TAU_ENVELOPE = 0.3, 10.0  # Myr
PLANET_MIN = 0.1  # earth masses
HABITABLE = (0.95, 1.7)  # au, scaled by sqrt(L)
SUBSTEPS = 10  # integration substeps per output interval
T_EPS = 1e-9  # Myr; two times closer than this count as the same moment (a mid-step time against t_gas)
_TWO_M53 = 1.0 / 9007199254740992.0
_DIGITS = frozenset("0123456789")

#: output times in Myr
TIMES = [0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.7, 1, 1.5, 2, 2.5, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20,
         25, 30, 35, 40, 45, 50]
STAGES = ("dust", "embryos", "giants_forming", "clearing", "done")

PARAM_KEYS = ("m_star", "disc_mass", "t_gas", "core_threshold", "r_frost", "luminosity")
#: the parameters that are put in; they are written as given, the derived ones to 3 digits
INPUT_KEYS = frozenset({"m_star", "disc_mass", "t_gas", "core_threshold"})
STATE_KEYS = ("time", "planets", "rocky", "ice", "gas", "largest_mass", "innermost_au", "outermost_au",
              "debris", "gas_present", "stage")
LEDGER_KEYS = ("solids_in_bodies", "zone_solids", "debris_mass", "gas_captured", "gas_left")
SUMMARY_KEYS = ("planets", "rocky", "ice", "gas", "largest_mass", "habitable", "total_planet_mass")
COUNT_KEYS = frozenset({"planets", "rocky", "ice", "gas", "habitable"})
WORD_KEYS = frozenset({"gas_present", "stage"})

KEYS: dict[str, str] = {
    "m_star": "mass of the star, solar masses (parameter)",
    "disc_mass": "solids in the disc at the start, earth masses (parameter)",
    "t_gas": "time at which the disc gas is gone, Myr (parameter)",
    "core_threshold": "core mass from which a body beyond the frost line runs away with gas, earth masses",
    "r_frost": "frost line, au: 2.7 au * sqrt(L / L_sun)",
    "luminosity": "luminosity of the star, solar luminosities: (m_star)^3.5",
    "time": "time since the disc formed, Myr",
    "planets": "bodies of at least 0.1 earth masses (count)",
    "rocky": "planets that are neither ice nor gas (count)",
    "ice": "planets beyond the frost line with at least 5 earth masses of solids, envelope lighter than the core (count)",
    "gas": "planets whose gas envelope outweighs the core (count)",
    "largest_mass": "mass of the heaviest body, solids and gas, earth masses",
    "innermost_au": "orbit of the innermost planet, au (0 when there is none)",
    "outermost_au": "orbit of the outermost planet, au (0 when there is none)",
    "debris": "fraction of the initial solids thrown out by collisions",
    "gas_present": "yes while the disc still holds gas, else no",
    "stage": "dust, embryos, giants_forming, clearing or done",
    "solids_in_bodies": "ledger: solids inside all bodies, earth masses",
    "zone_solids": "ledger: solids not yet swept up, earth masses",
    "debris_mass": "ledger: solids thrown out, earth masses",
    "gas_captured": "ledger: gas captured by all bodies so far, earth masses",
    "gas_left": "ledger: gas still in the disc within reach of the planets, earth masses",
    "habitable": "final rocky planets between 0.95 and 1.7 au scaled by sqrt(L) (count)",
    "total_planet_mass": "final mass of all planets, solids and gas, earth masses",
}


@dataclass
class PlanetRollout(Rollout):
    """A :class:`Rollout` that also keeps every body at every step and the merger log."""
    #: per step, per body: ``{"id", "a", "solid", "gas", "type"}`` (embryos below 0.1 included)
    bodies: list[list[dict]] = field(default_factory=list)
    #: ``{"t", "ids", "kept", "mass_before", "debris"}`` per merger (rules 7 adds ``ejected``,
    #: ``gas_ejected``, ``theta``, ``members``, ``orbit`` and ``mass_after``, see ``_Disc7.merge``)
    events: list[dict] = field(default_factory=list)
    #: rules 7 only: the planets at the end, innermost first, each ``{"id", "orbit", "mass", "solid",
    #: "gas", "type", "flux", "temperature", "habitable"}`` (what the next level starts from)
    system: list[dict] = field(default_factory=list)

    def value(self, text: float | int | str) -> str:
        return dense_value(text)


def dense_value(value: float | int | str) -> str:
    """A metric as it stands in a line: a word, an exact count, or a number to 3 significant digits.

    The number is rounded first: ``num(1052.53, sig=3)`` alone would write every decimal of a
    number from 1000 on.

    >>> dense_value(1052.53), dense_value(0.03234), dense_value(7), dense_value("done")
    ('1 0 5 0', '0 point 0 3 2 3', '7', 'done')
    """
    if isinstance(value, str):
        return value
    if isinstance(value, int):
        return num(value)
    return num(float(f"{value:.3g}"), sig=3)


def param_value(key: str, value: float) -> str:
    """A parameter as it stands in the ``params`` record: an input as given (``disc_mass 253.5``
    is ``2 5 3 point 5``, to at most 5 digits), a derived one like any measured number.

    >>> param_value("disc_mass", 253.5), param_value("m_star", 0.74), param_value("r_frost", 1.5936)
    ('2 5 3 point 5', '0 point 7 4', '1 point 5 9')
    """
    return num(float(value), sig=5) if key in INPUT_KEYS else dense_value(value)


def _rule_set(rules: object) -> int:
    """6 (round 6, frozen) or 7 (the corrected rules); anything else is no rule set of this level."""
    if isinstance(rules, bool) or rules not in (6, 7):
        raise ValueError(f"planets knows the rules 6 (round 6, frozen) and 7 (corrected), not {rules!r}")
    return int(rules)


def _carried_rules(r: Rollout) -> int:
    """The rule set a rollout was run under: 7 when its parameters say so, else round 6."""
    return 7 if isinstance(r.params, dict) and r.params.get("rules") == 7 else 6


def random_params(rng: random.Random, rules: int = 6) -> dict[str, float]:
    """Star mass uniform 0.5-1.5, solids uniform 10-300 earth masses, gas lifetime 1-10 Myr.
    With ``rules=7`` (:func:`_random_params7`): ``star_mass`` drawn as before, solids that scale
    with the star, a gas lifetime of 1 Myr plus an exponential of mean 2 Myr."""
    if _rule_set(rules) == 7:
        return _random_params7(rng)
    return {"m_star": round(rng.uniform(0.5, 1.5), 2),
            "disc_mass": round(rng.uniform(10.0, 300.0), 1),
            "t_gas": round(rng.uniform(1.0, 10.0), 2),
            "core_threshold": 10.0}


def frost_line(m_star: float, rules: int = 6) -> float:
    """Frost line in au: 2.7 au * sqrt(L), with L = M^3.5 (round 6) or :func:`luminosity7` (``rules=7``)."""
    if _rule_set(rules) == 7:
        return frost_line7(m_star)
    return FROST_AU * math.sqrt(m_star ** 3.5)


class _Rng:
    """Uniform numbers taken straight from the PCG64 bit stream.

    NumPy keeps the bits of a seeded ``PCG64`` the same in every version; it does not promise
    that ``Generator.random`` and ``Generator.uniform`` go on turning them into the same numbers.
    The conversion (the top 53 bits as a fraction) is therefore done here. It gives exactly what
    ``numpy.random.default_rng(seed)`` gives in numpy 2.4, which the tests hold.
    """

    def __init__(self, seed: int) -> None:
        self._bits = np.random.PCG64(seed)

    def random(self) -> float:
        return (int(self._bits.random_raw()) >> 11) * _TWO_M53

    def uniform(self, lo: float, hi: float, size: int | None = None) -> float | np.ndarray:
        if size is None:
            return lo + (hi - lo) * self.random()
        raw = self._bits.random_raw(size)
        return lo + (hi - lo) * ((raw >> np.uint64(11)).astype(np.float64) * _TWO_M53)


def _total(x: np.ndarray) -> float:
    """An exactly rounded sum: the same whatever order or vector width numpy would add in."""
    return math.fsum(x.tolist())


def _period(a: np.ndarray, m_star: float) -> np.ndarray:
    """Orbital period in years (Kepler's third law in au, solar masses, years)."""
    return np.sqrt(a ** 3 / m_star)


def _unstable(a: np.ndarray, m: np.ndarray, m_star: float, ecc: float) -> np.ndarray:
    """For each neighbouring pair: can the two orbits come within K_HILL mutual Hill radii?"""
    pair_mass = (m[:-1] + m[1:]) / (3.0 * m_star * EARTH_PER_SUN)
    r_hill = np.cbrt(pair_mass) * (a[:-1] + a[1:]) / 2.0
    return (a[1:] - a[:-1]) - ecc * (a[:-1] + a[1:]) < K_HILL * r_hill


class _Disc:
    """The mutable state: one row per zone/body, merged zones become one row."""

    def __init__(self, rng: _Rng, m_star: float, disc_mass: float, t_gas: float) -> None:
        self.m_star, self.t_gas = m_star, t_gas
        self.r_frost = frost_line(m_star)
        edges = R_IN * (R_OUT / R_IN) ** (np.arange(N_ZONES + 1) / N_ZONES)
        self.r_in, self.r_out = edges[:-1].copy(), edges[1:].copy()
        self.a = np.sqrt(self.r_in * self.r_out)
        # mass of a zone with Sigma = S0 r^-1.5: 4 pi S0 (sqrt(r_out) - sqrt(r_in)), times the ice factor
        ice = np.where(self.a > self.r_frost, ICE_FACTOR, 1.0)
        weight = ice * (np.sqrt(self.r_out) - np.sqrt(self.r_in))
        self.s_rem = disc_mass * weight / _total(weight)
        self.gas0 = GAS_REACH * GAS_TO_ROCK * self.s_rem / ice
        self.m_solid = np.zeros(N_ZONES)
        self.m_gas = np.zeros(N_ZONES)
        self.gas_cap = np.zeros(N_ZONES)
        self.eps = rng.uniform(*EPS_RANGE, size=N_ZONES)
        self.ids = np.arange(N_ZONES, dtype=np.int64)
        self.debris = 0.0
        self.events: list[dict] = []
        self.step = 0  # index of the output step the current substeps lead to

    @property
    def mass(self) -> np.ndarray:
        return self.m_solid + self.m_gas

    def gas_at(self, t: float) -> bool:
        """Is there still gas at time ``t``? A time within T_EPS of ``t_gas`` counts as ``t_gas``:
        the gas is gone. (A mid-step time such as 3.0 + 3 * 0.1 + 0.05 is 3.3499999999999996.)"""
        return t < self.t_gas - T_EPS

    def gas_available(self, t: float) -> np.ndarray:
        share = 1.0 - t / self.t_gas if self.gas_at(t) else 0.0
        return np.maximum(0.0, self.gas0 * share - self.gas_cap)

    def runaway(self, t: float, core_threshold: float) -> np.ndarray:
        return self.gas_at(t) & (self.a > self.r_frost) & (self.m_solid >= core_threshold)

    def advance(self, rng: _Rng, t: float, dt: float, core_threshold: float) -> None:
        # accretion: exact exponential update of the zone's remaining solids
        rate = self.eps / (TAU_SWEEP * _period(self.a, self.m_star))
        swept = self.s_rem * -np.expm1(-rate * dt)
        self.s_rem = self.s_rem - swept
        self.m_solid = self.m_solid + swept
        # gas capture, capped by what the zone still holds at mid-step
        gas_here = self.gas_at(t + dt / 2)
        if gas_here:
            beyond = self.a > self.r_frost
            fast = beyond & (self.m_solid >= core_threshold)
            slow = beyond & ~fast & (self.m_solid >= ICE_CORE)
            want = np.where(fast, self.mass * math.expm1(dt / TAU_RUNAWAY),
                            np.where(slow, self.m_solid * dt / TAU_ENVELOPE, 0.0))
            got = np.minimum(want, self.gas_available(t + dt / 2))
            self.gas_cap = self.gas_cap + got
            self.m_gas = self.m_gas + got
        self.collide(rng, t + dt, dt, gas_here)

    def collide(self, rng: _Rng, t: float, dt: float, gas_here: bool) -> None:
        if len(self.a) < 2:
            return
        ecc, tau = (E_GAS, TAU_MERGE_GAS) if gas_here else (E_LATE, TAU_MERGE_LATE)
        p_merge = -math.expm1(-dt / tau)
        joins = [bool(u and rng.random() < p_merge) for u in _unstable(self.a, self.mass, self.m_star, ecc)]
        if not any(joins):
            return
        groups: list[list[int]] = [[0]]
        for i, join in enumerate(joins):
            if join:
                groups[-1].append(i + 1)
            else:
                groups.append([i + 1])
        for g in groups:
            if len(g) > 1:
                self.merge(rng, t, g)
        keep = [g[0] for g in groups]
        for name in ("r_in", "r_out", "a", "s_rem", "gas0", "m_solid", "m_gas", "gas_cap", "eps", "ids"):
            setattr(self, name, getattr(self, name)[keep])

    def merge(self, rng: _Rng, t: float, g: list[int]) -> None:
        """Fold the zones in ``g`` into the first one; the heaviest body's id survives (of two
        equally heavy bodies the inner one: ``argmax`` takes the first)."""
        i = g[0]
        m = self.mass[g]
        solid = _total(self.m_solid[g])
        total = _total(m)
        debris = 0.0
        if rng.random() < P_EJECT:
            debris = solid * rng.uniform(*EJECT_FRACTION)
        kept = int(self.ids[g][int(np.argmax(m))])
        self.events.append({"t": t, "step": self.step, "ids": [int(x) for x in self.ids[g]], "kept": kept,
                            "mass_before": total, "debris": debris})
        self.a[i] = _total(self.a[g] * m) / total if total > 0 else math.sqrt(self.r_in[g[0]] * self.r_out[g[-1]])
        self.r_out[i] = self.r_out[g[-1]]
        self.ids[i] = kept
        self.eps[i] = _total(self.eps[g]) / len(g)
        for name in ("s_rem", "gas0", "m_gas", "gas_cap"):
            arr = getattr(self, name)
            arr[i] = _total(arr[g])
        self.m_solid[i] = solid - debris
        self.debris += debris

    def types(self) -> list[str]:
        out = []
        for a, s, g in zip(self.a, self.m_solid, self.m_gas):
            if g >= s and g > 0:
                out.append("gas")
            elif a > self.r_frost and s >= ICE_CORE:
                out.append("ice")
            else:
                out.append("rocky")
        return out

    def stage(self, t: float, n_planets: int, core_threshold: float) -> str:
        if n_planets == 0:
            return "dust"
        if not self.gas_at(t):
            unstable = len(self.a) > 1 and bool(_unstable(self.a, self.mass, self.m_star, E_LATE).any())
            return "clearing" if unstable else "done"
        return "giants_forming" if bool(self.runaway(t, core_threshold).any()) else "embryos"

    def snapshot(self, t: float, disc_mass: float, core_threshold: float) -> tuple[dict, list[dict]]:
        mass = self.mass
        types = self.types()
        big = mass >= PLANET_MIN
        counts = {k: sum(1 for ty, b in zip(types, big) if b and ty == k) for k in ("rocky", "ice", "gas")}
        n = int(big.sum())
        step = {
            "time": float(t), "planets": n, **counts,
            "largest_mass": float(mass.max()) if len(mass) else 0.0,
            "innermost_au": float(self.a[big].min()) if n else 0.0,
            "outermost_au": float(self.a[big].max()) if n else 0.0,
            "debris": float(self.debris / disc_mass),
            "gas_present": "yes" if self.gas_at(t) else "no",
            "stage": self.stage(t, n, core_threshold),
            "solids_in_bodies": _total(self.m_solid), "zone_solids": _total(self.s_rem),
            "debris_mass": float(self.debris), "gas_captured": _total(self.gas_cap),
            "gas_left": _total(self.gas_available(t)),
        }
        bodies = [{"id": int(i), "a": float(a), "solid": float(s), "gas": float(g), "type": ty}
                  for i, a, s, g, ty in zip(self.ids, self.a, self.m_solid, self.m_gas, types)]
        return step, bodies


class Planets:
    """The simulation and its gate. Rollouts replayed by :meth:`check` are cached per seed.

    ``Planets()`` is round 6. ``Planets(rules=7)`` runs, draws and writes under the corrected
    rules unless a call names another rule set; ``conserved``, ``check`` and ``owns`` judge both
    kinds whatever the object was made with (a rollout and a prompt say which rules they are of)."""

    sim = SIM
    topic = SIM
    KEYS = KEYS
    rules = 6

    def __init__(self, rules: int = 6) -> None:
        self._cache: dict[int, PlanetRollout] = {}
        self._cache7: dict[int, PlanetRollout] = {}
        if _rule_set(rules) == 7:
            self.rules = 7
            self.KEYS = KEYS7

    def _rules(self, rules: int | None) -> int:
        return self.rules if rules is None else _rule_set(rules)

    def random_params(self, rng: random.Random, rules: int | None = None) -> dict[str, float]:
        return random_params(rng, self._rules(rules))

    def run(self, seed: int, m_star: float = 1.0, disc_mass: float = 100.0, t_gas: float = 3.0,
            core_threshold: float = 10.0, rules: int | None = None, star_mass: float | None = None,
            gas_mass: float | None = None) -> PlanetRollout:
        """Deterministic for a given seed and parameters (the PCG64 stream of ``seed``).

        ``t_gas = 0`` is a disc without gas. Raises ``ValueError`` for a star or disc without
        mass, a negative gas lifetime or a core threshold that is not positive.

        ``rules=7`` runs the corrected rules (:func:`_run7`, section "Rules 7" of the module
        docstring) and the rollout says so in ``params["rules"]``. ``star_mass`` is the rules-7
        name of ``m_star`` (either is taken under both rule sets); ``gas_mass`` (earth masses of
        disc gas, rules 7 only) replaces the gas budget that is tied to the solids."""
        rules = self._rules(rules)
        if star_mass is not None:
            if m_star != 1.0 and float(m_star) != float(star_mass):
                raise ValueError(f"m_star and star_mass name the same mass: {m_star!r}, {star_mass!r}")
            m_star = star_mass
        if rules == 7:
            return _run7(seed, m_star, disc_mass, t_gas, core_threshold, gas_mass)
        if gas_mass is not None:
            raise ValueError("gas_mass is a parameter of rules 7; round 6 ties the gas to the solids")
        m_star, disc_mass, t_gas, core_threshold = float(m_star), float(disc_mass), float(t_gas), float(core_threshold)
        if not (math.isfinite(m_star) and m_star > 0 and math.isfinite(disc_mass) and disc_mass > 0):
            raise ValueError(f"m_star and disc_mass must be positive: {m_star!r}, {disc_mass!r}")
        if not (math.isfinite(t_gas) and t_gas >= 0 and core_threshold > 0):
            raise ValueError(f"t_gas must not be negative, core_threshold must be positive: {t_gas!r}, {core_threshold!r}")
        rng = _Rng(seed)
        disc = _Disc(rng, m_star, disc_mass, t_gas)
        params = {"m_star": m_star, "disc_mass": disc_mass, "t_gas": t_gas, "core_threshold": core_threshold,
                  "r_frost": disc.r_frost, "luminosity": m_star ** 3.5, "gas_initial": _total(disc.gas0)}
        r = PlanetRollout(SIM, int(seed), params, [])
        step, bodies = disc.snapshot(TIMES[0], disc_mass, core_threshold)
        r.steps.append(step)
        r.bodies.append(bodies)
        for k, (t0, t1) in enumerate(zip(TIMES, TIMES[1:]), start=1):
            dt = (t1 - t0) / SUBSTEPS
            disc.step = k
            for i in range(SUBSTEPS):
                disc.advance(rng, t0 + i * dt, dt, core_threshold)
            step, bodies = disc.snapshot(t1, disc_mass, core_threshold)
            r.steps.append(step)
            r.bodies.append(bodies)
        r.events = disc.events
        last = r.steps[-1]
        lo, hi = (h * math.sqrt(params["luminosity"]) for h in HABITABLE)
        planets = [b for b in r.bodies[-1] if b["solid"] + b["gas"] >= PLANET_MIN]
        r.summary = {
            **{k: last[k] for k in ("planets", "rocky", "ice", "gas", "largest_mass")},
            "habitable": sum(1 for b in planets if b["type"] == "rocky" and lo <= b["a"] <= hi),
            "total_planet_mass": math.fsum(b["solid"] + b["gas"] for b in planets),
        }
        return r

    def rollout(self, seed: int, rules: int | None = None) -> PlanetRollout:
        """The canonical run of a seed: parameters from ``random_params(random.Random(seed))``
        (with ``rules=7``: from ``random_params(random.Random(seed), rules=7)``, corrected rules)."""
        if self._rules(rules) == 7:
            if seed not in self._cache7:
                if len(self._cache7) >= 512:
                    self._cache7.clear()
                self._cache7[seed] = _run7(seed, **_random_params7(random.Random(seed)))
            return self._cache7[seed]
        if seed not in self._cache:
            if len(self._cache) >= 512:
                self._cache.clear()
            self._cache[seed] = self.run(seed, **random_params(random.Random(seed)))
        return self._cache[seed]

    def conserved(self, r: Rollout) -> Verdict:
        if _carried_rules(r) == 7:
            return _conserved7(r)
        initial = float(r.params["disc_mass"])
        gas_initial = float(r.params["gas_initial"])
        events = getattr(r, "events", [])
        prev = None
        for t, s in enumerate(r.steps):
            total = s["solids_in_bodies"] + s["zone_solids"] + s["debris_mass"]
            if abs(total - initial) > 1e-9:
                return Verdict(False, num(initial, sig=6), f"step {t}: solids {total!r} != initial {initial!r}")
            if not close(s["debris"] * initial, s["debris_mass"], rel=1e-9):
                return Verdict(False, None, f"step {t}: debris fraction does not match the debris mass")
            if abs(math.fsum(e["debris"] for e in events if e["step"] <= t) - s["debris_mass"]) > 1e-9:
                return Verdict(False, None, f"step {t}: debris does not match the merger log")
            if s["gas_captured"] + s["gas_left"] > gas_initial + 1e-9 or s["gas_left"] < 0:
                return Verdict(False, None, f"step {t}: more gas captured than the disc held")
            if s["planets"] != s["rocky"] + s["ice"] + s["gas"]:
                return Verdict(False, num(s["rocky"] + s["ice"] + s["gas"]), f"step {t}: planets != rocky + ice + gas")
            if s["stage"] not in STAGES or s["gas_present"] not in ("yes", "no"):
                return Verdict(False, None, f"step {t}: unknown stage or gas_present")
            if prev is not None:
                if not s["time"] > prev["time"]:
                    return Verdict(False, None, f"step {t}: time does not increase")
                if s["gas_captured"] < prev["gas_captured"] - 1e-9:
                    return Verdict(False, None, f"step {t}: captured gas went down")
                if s["debris_mass"] < prev["debris_mass"] or s["zone_solids"] > prev["zone_solids"] + 1e-9:
                    return Verdict(False, None, f"step {t}: debris or swept solids went backwards")
            prev = s
        return self._bodies_keep_mass(r)

    @staticmethod
    def _bodies_keep_mass(r: Rollout) -> Verdict:
        """A body's mass never drops, and no body vanishes, except in a logged merger; a merger
        loses no more than its logged debris; and gas is only ever captured beyond the frost line."""
        bodies = getattr(r, "bodies", None) or []
        events = getattr(r, "events", [])
        r_frost = float(r.params["r_frost"])
        for n, e in enumerate(events):
            if not 1 <= e["step"] < len(bodies) or e["kept"] not in e["ids"]:
                return Verdict(False, None, f"merger {n}: bad step or survivor")
            if any(e["kept"] in f["ids"] for f in events[n + 1:] if f["step"] == e["step"]):
                continue  # merged again before the next output step
            now = next((b for b in bodies[e["step"]] if b["id"] == e["kept"]), None)
            if now is None or now["solid"] + now["gas"] < e["mass_before"] - e["debris"] - 1e-9:
                return Verdict(False, None, f"merger {n} at step {e['step']}: mass was lost beyond the logged debris")
        for t in range(1, len(bodies)):
            merged = {i for e in events if e["step"] == t for i in e["ids"]}
            before = {b["id"]: b for b in bodies[t - 1]}
            for b in bodies[t]:
                if b["id"] in merged:
                    continue
                was = before[b["id"]]
                if b["solid"] + b["gas"] < was["solid"] + was["gas"] - 1e-9:
                    return Verdict(False, None, f"step {t}: body {b['id']} lost mass without a merger")
                if b["gas"] > was["gas"] + 1e-9 and b["a"] <= r_frost:
                    return Verdict(False, None, f"step {t}: body {b['id']} captured gas inside the frost line")
            if not set(before) - {b["id"] for b in bodies[t]} <= merged:
                return Verdict(False, None, f"step {t}: a body vanished without a merger")
        return Verdict(True, None, "solids, gas, counts, masses and time are consistent")

    def parse(self, prompt: str) -> tuple[int, str, int, str] | None:
        """``(seed, where, step, key)`` with ``where`` in ``step``, ``next``, ``final``, ``params``."""
        w = prompt.split()
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
        if not isinstance(step, int) or not 0 <= step < len(TIMES) or num(step).split() != digits:
            return None
        if len(rest) == 2 and rest[0] == "next" and rest[1] in STATE_KEYS + LEDGER_KEYS and step + 1 < len(TIMES):
            return seed, "next", step, rest[1]
        if len(rest) == 1 and rest[0] in STATE_KEYS + LEDGER_KEYS:
            return seed, "step", step, rest[0]
        return None

    def owns(self, prompt: str) -> bool:
        """True for a round-6 story question (``planets seed 7 step 3 gas``) and for a rules-7 one
        (``planets seed 7 rules 7 step 3 gas_giants``); the lessons have their own gate, :data:`LESSONS`."""
        return self.parse(prompt) is not None or parse7(prompt) is not None

    def truth(self, prompt: str) -> float | int | str | None:
        """The simulation's own value for a prompt, or ``None`` when it is not this topic's."""
        q = self.parse(prompt)
        if q is None:
            q7 = parse7(prompt)
            return None if q7 is None else _truth7(self.rollout(q7[0], rules=7), q7)
        seed, where, step, key = q
        r = self.rollout(seed)
        if where == "final":
            return r.summary[key]
        if where == "params":
            return r.params[key]
        return r.steps[step + (where == "next")][key]

    def check(self, prompt: str, answer: str) -> Verdict:
        if self.parse(prompt) is None:
            q7 = parse7(prompt)
            if q7 is None:
                return Verdict(False, None, "not my question")
            return _check7(self.rollout(q7[0], rules=7), q7, answer)
        value = self.truth(prompt)
        if value is None:
            return Verdict(False, None, "not my question")
        words = prompt.split()
        key = words[-1]
        expected = param_value(key, value) if words[-2] == "params" else dense_value(value)
        if key in WORD_KEYS:
            ok = answer.strip() == value
            return Verdict(ok, expected, "exact word" if ok else "wrong word")
        got = parse_num(answer)
        if not is_finite(got):  # "1 e 9 9 9" parses to infinity, which is close to nothing
            return Verdict(False, expected, "not a number")
        if key in COUNT_KEYS:
            ok = got == value
            return Verdict(ok, expected, "exact count" if ok else "wrong count")
        ok = close(float(got), float(value), rel=0.05)
        return Verdict(ok, expected, "within 5 percent" if ok else "more than 5 percent off")

    def records(self) -> list[Line]:
        return []

    def generate(self, rng: random.Random, n: int, rules: int | None = None) -> list[Line]:
        """All lines of ``n`` canonical rollouts with seeds drawn from ``rng``."""
        rules = self._rules(rules)
        out: list[Line] = []
        for seed in seeds(rng, n):
            out += lines(self.rollout(seed, rules))
        return out

    def lines(self, r: Rollout, every: int = 1, rules: int | None = None) -> list[Line]:
        return lines(r, every, rules)

    def lesson_gate(self) -> LessonGate:
        return LESSONS


def _digits_end(words: list[str], start: int) -> int:
    """Index after the run of single-digit tokens that begins at ``start``."""
    i = start
    while i < len(words) and words[i] in _DIGITS:
        i += 1
    return i


def params_line(r: Rollout) -> Line:
    """``planets seed 3 params. m_star 0 point 7 4. disc_mass 1 6 7 point 8. t_gas 4 point 3 3. ...``"""
    fields = " ".join(f"{k} {param_value(k, r.params[k])}." for k in PARAM_KEYS)
    return Line(f"{r.sim} seed {num(r.seed)} params. {fields}", topic=r.sim, kind="record")


def lines(r: Rollout, every: int = 1, rules: int | None = None) -> list[Line]:
    """The parameter record, a state record and questions for every ``every``-th step (now and
    next), and the final questions. The ledger is left out: it is the gate's, not the model's.

    The rule set is the one the rollout carries (``params["rules"]`` is 7, or absent for round 6);
    ``rules`` may name it and must then agree. A rules-7 rollout gives the lines of :func:`_lines7`
    (``planets seed 3 rules 7 ...``), never a round-6 prompt."""
    if rules is not None and _rule_set(rules) != _carried_rules(r):
        raise ValueError(f"the rollout was run under rules {_carried_rules(r)}, not {rules!r}")
    if _carried_rules(r) == 7:
        return _lines7(r, every)
    keys = list(STATE_KEYS)
    out = [params_line(r)]
    for t in range(0, len(r.steps), every):
        out.append(state_line(r, t, keys))
        out += query_lines(r, t, keys)
    return out + summary_lines(r)


# ---------------------------------------------------------------------------------------------
# Rules 7: the corrected level (section "Rules 7" of the module docstring). Nothing above this
# line computes differently than in round 6: run(), random_params(), rollout(), conserved(),
# check(), owns(), lines() and generate() hand over to the functions below when the rule set is 7.
# The rules-7 run is plain Python on floats (numpy only supplies the raw PCG64 bits), so every
# rule below is the very function a lesson asks about.
# ---------------------------------------------------------------------------------------------

#: the word of the rules-7 lessons (``planets7 predict flux ...``). A rules-7 *story* keeps the
#: word ``planets`` and says ``rules 7`` after its seed. (``planets predict solids_remaining`` is
#: the round-6 lesson of ``haishool.cosmos.predict``; the two never share a prompt.)
SIM7 = "planets7"

#: starlight in units of what the earth receives above which a rocky planet's ocean boils away
#: (runaway greenhouse) and below which no amount of CO2 keeps it liquid (maximum greenhouse):
#: after Kopparapu et al. 2013, 2014, values for the sun and one earth mass, from memory
S_RUNAWAY, S_MAX_GREENHOUSE = 1.107, 0.356
#: K at the earth's flux for a Bond albedo of 0.3: 278.3 K * 0.7^(1/4) (the earth's effective temperature is 255 K)
T_EQ_EARTH = 254.6
GREENHOUSE_EARTH = 33.0  # K the earth's air adds (288 K surface minus 255 K)
GREENHOUSE_SLOPE = 67.3  # K per unit of missing flux: toy thermostat, set so that the outer edge is at 273 K
GREENHOUSE_RUNAWAY = 500.0  # K: toy, about Venus (737 K at the surface, about 230 K effective)
GREENHOUSE_FROZEN = 0.0  # K: beyond the outer edge CO2 freezes out; the toy leaves the bare equilibrium temperature
WATER_K = (273, 373)  # whole kelvin between which water is liquid (level 6's hand-off uses the same)
HABITABLE_MASS = (0.3, 10.0)  # earth masses: toy lower bound between Mars (lost its air) and Venus (kept it)
SUPER_EARTH = 10.0  # earth masses of solids from which an inner body is a ``super_earth`` (the default core threshold)
AU_KM = 1.495978707e8
R_EARTH_AU = 6371.0 / AU_KM  # the earth's mean radius in au
TAU_MIG7 = 1.0  # Myr: drift time of a gas giant at 1 au around the sun at t = 0 (toy, about a million orbits)
A_PARK7 = 0.05  # au: the disc's inner cavity, where a migrating giant stops (toy)
T_GAS_SHIFT7, T_GAS_MEAN7, T_GAS_MAX7 = 1.0, 2.0, 10.0  # Myr: gas lifetime = shift + exponential(mean), capped
DISC_PER_STAR7 = (10.0, 300.0)  # earth masses of solids per solar mass of star
TYPES7 = ("rocky", "super_earth", "icy", "ice_giant", "gas")
#: type of a body -> the count of planets of that type in a step
TYPE_COUNTS7 = {"rocky": "rocky", "super_earth": "super_earths", "icy": "icy", "ice_giant": "ice_giants",
                "gas": "gas_giants"}
CLIMATES7 = ("runaway", "temperate", "frozen")

PARAM_KEYS7 = ("star_mass", "disc_mass", "t_gas", "core_threshold", "luminosity", "frost_line", "hz_inner", "hz_outer")
#: the parameters that are put in; they are written as given and judged exactly
INPUT_KEYS7 = frozenset({"star_mass", "disc_mass", "t_gas", "core_threshold"})
STATE_KEYS7 = ("time", "planets", "rocky", "super_earths", "icy", "ice_giants", "gas_giants", "ejected",
               "largest_mass", "innermost_au", "outermost_au", "debris", "gas_present", "stage")
LEDGER_KEYS7 = ("solids_in_bodies", "zone_solids", "debris_mass", "gas_captured", "gas_in_bodies", "gas_ejected",
                "gas_left")
SUMMARY_KEYS7 = ("planets", "rocky", "super_earths", "icy", "ice_giants", "gas_giants", "ejected", "largest_mass",
                 "habitable", "inner_giants", "total_planet_mass")
#: per planet of the finished system (``planets seed 3 rules 7 planet 2 temperature``), innermost is planet 1
PLANET_KEYS7 = ("orbit", "mass", "type", "flux", "temperature", "habitable")

KEYS7: dict[str, str] = {
    "star_mass": "mass of the star, solar masses (parameter)",
    "disc_mass": "solids in the disc at the start, earth masses (parameter)",
    "t_gas": "time at which the disc gas is gone, Myr (parameter)",
    "core_threshold": "core mass from which a body runs away with gas, earth masses (parameter)",
    "luminosity": "luminosity of the star, solar luminosities: the piecewise main-sequence law of luminosity7()",
    "frost_line": "frost line, au: 2.7 au * sqrt(luminosity)",
    "hz_inner": "inner edge of the habitable zone, au: sqrt(luminosity / 1.107)",
    "hz_outer": "outer edge of the habitable zone, au: sqrt(luminosity / 0.356)",
    "time": "time since the disc formed, Myr",
    "planets": "bodies of at least 0.1 earth masses (count): rocky + super_earths + icy + ice_giants + gas_giants",
    "rocky": "planets inside the frost line with less than 10 earth masses of solids, envelope lighter than the core (count)",
    "super_earths": "planets inside the frost line with at least 10 earth masses of solids, envelope lighter than the core (count)",
    "icy": "planets beyond the frost line with less than 5 earth masses of solids (count)",
    "ice_giants": "planets beyond the frost line with at least 5 earth masses of solids, envelope lighter than the core (count)",
    "gas_giants": "planets whose gas envelope weighs at least as much as the core (count)",
    "ejected": "bodies thrown out of the system so far (count)",
    "largest_mass": "mass of the heaviest body, solids and gas, earth masses",
    "innermost_au": "orbit of the innermost planet, au (0 when there is none)",
    "outermost_au": "orbit of the outermost planet, au (0 when there is none)",
    "debris": "fraction of the initial solids thrown out of the system",
    "gas_present": "yes while the disc still holds gas, else no",
    "stage": "dust, embryos, giants_forming, clearing or done",
    "solids_in_bodies": "ledger: solids inside all bodies, earth masses",
    "zone_solids": "ledger: solids not swept up (and still in the system), earth masses",
    "debris_mass": "ledger: solids thrown out, earth masses",
    "gas_captured": "ledger: gas captured by all bodies so far, ejected bodies included, earth masses",
    "gas_in_bodies": "ledger: gas inside the bodies that are still there, earth masses",
    "gas_ejected": "ledger: gas that left with ejected bodies, earth masses",
    "gas_left": "ledger: gas still in the disc within reach of the planets, earth masses",
    "habitable": "final rocky planets of 0.3 to 10 earth masses without disc gas whose flux is 0.356 to 1.107 (count); "
                 "per planet: yes or no",
    "inner_giants": "final gas giants at or inside the frost line (count)",
    "total_planet_mass": "final mass of all planets, solids and gas, earth masses",
    "orbit": "per planet: orbit, au",
    "mass": "per planet: solids and gas, earth masses",
    "type": "per planet: rocky, super_earth, icy, ice_giant or gas",
    "flux": "per planet: starlight in units of what the earth receives, luminosity / orbit^2",
    "temperature": "per planet: whole K the toy climate rule gives for its flux: at the surface, or for a gas giant "
                   "the equilibrium temperature of its cloud tops",
}


# ---------------------------------------------------------------------------------------------
# the rules: every function here is called by the rules-7 run and answers a lesson (LESSONS)
# ---------------------------------------------------------------------------------------------

def luminosity7(star_mass: float) -> float:
    """Solar luminosities of a main-sequence star of ``star_mass`` solar masses: the piecewise
    textbook law, ``0.23 M^2.3`` below 0.43, ``M^4`` to 2, ``1.4 M^3.5`` to 55, ``32000 M`` above."""
    m = float(star_mass)
    if m < 0.43:
        return 0.23 * m ** 2.3
    if m < 2.0:
        return (m * m) * (m * m)
    if m < 55.0:
        return 1.4 * m ** 3.5
    return 32000.0 * m


def frost_line7(star_mass: float) -> float:
    """Frost line in au: ``2.7 * sqrt(luminosity7(star_mass))``."""
    return FROST_AU * math.sqrt(luminosity7(star_mass))


def flux(luminosity: float, orbit: float) -> float:
    """Starlight at ``orbit`` au in units of what the earth receives: ``luminosity / orbit^2``."""
    return luminosity / (orbit * orbit)


def climate(flux: float) -> str:
    """``runaway`` above 1.107 (the ocean boils away), ``frozen`` below 0.356 (no greenhouse is
    enough), else ``temperate``: the habitable zone."""
    if flux > S_RUNAWAY:
        return "runaway"
    return "frozen" if flux < S_MAX_GREENHOUSE else "temperate"


def hz_inner(luminosity: float) -> float:
    """Inner edge of the habitable zone in au: where the flux is 1.107."""
    return math.sqrt(luminosity / S_RUNAWAY)


def hz_outer(luminosity: float) -> float:
    """Outer edge of the habitable zone in au: where the flux is 0.356."""
    return math.sqrt(luminosity / S_MAX_GREENHOUSE)


def habitable(luminosity: float, orbit: float, mass: float) -> str:
    """``yes`` when a rocky planet of ``mass`` earth masses at ``orbit`` au can keep liquid water:
    a temperate flux and 0.3 to 10 earth masses. (The run asks this of a ``rocky`` planet that
    holds no disc gas; every other planet is not habitable.)"""
    in_zone = climate(flux(luminosity, orbit)) == "temperate"
    return "yes" if in_zone and HABITABLE_MASS[0] <= mass <= HABITABLE_MASS[1] else "no"


def equilibrium_temperature(flux: float) -> float:
    """K of a planet without greenhouse that reflects 30 % of the light: ``254.6 * flux^(1/4)``."""
    return T_EQ_EARTH * math.sqrt(math.sqrt(flux))


def greenhouse(flux: float) -> float:
    """K the air adds: 500 in a runaway (flux above 1.107), 33 from the earth's flux up to there,
    below it ``33 + 67.3 * (1 - flux)`` down to the outer edge 0.356 (the carbonate-silicate
    thermostat as a straight line), and nothing beyond the edge: there no greenhouse of CO2 holds
    273 K, the planet is frozen."""
    if flux > S_RUNAWAY:
        return GREENHOUSE_RUNAWAY
    if flux >= 1.0:
        return GREENHOUSE_EARTH
    if flux >= S_MAX_GREENHOUSE:
        return GREENHOUSE_EARTH + GREENHOUSE_SLOPE * (1.0 - flux)
    return GREENHOUSE_FROZEN


def surface_temperature(flux: float) -> int:
    """Whole K at the surface: equilibrium temperature plus greenhouse, a half rounds up. Water is
    liquid (273 to 373 K) exactly where the flux is temperate: 273 to 294 K in the zone, 761 K or
    more inside it, 197 K or less beyond it."""
    return math.floor(equilibrium_temperature(flux) + greenhouse(flux) + 0.5)


def _temperature7(flux: float, type_: str) -> int:
    """Whole K of a planet in the finished system: :func:`surface_temperature` for a body with a
    surface; for a gas giant, which has none, the equilibrium temperature of its cloud tops (a
    half rounds up)."""
    if type_ == "gas":
        return math.floor(equilibrium_temperature(flux) + 0.5)
    return surface_temperature(flux)


def period(orbit: float, star_mass: float) -> float:
    """Orbital period in years (Kepler's third law in au, solar masses, years)."""
    return math.sqrt(orbit ** 3 / star_mass)


def solids_remaining(remaining: float, efficiency: float, orbit: float, star_mass: float, dt: float) -> float:
    """Solids a zone still holds after ``dt`` Myr of sweeping: the round-6 sweep law,
    ``remaining * exp(-efficiency * dt / (TAU_SWEEP * period))``."""
    return remaining * math.exp(-efficiency * dt / (TAU_SWEEP * period(orbit, star_mass)))


def gas_share(time: float, t_gas: float) -> float:
    """Share of the disc gas still there at ``time`` Myr: ``1 - time / t_gas``, zero from ``t_gas``
    on (a time within T_EPS of ``t_gas`` counts as ``t_gas``)."""
    return 1.0 - time / t_gas if time < t_gas - T_EPS else 0.0


def gas_wanted(solid: float, gas: float, dt: float, core_threshold: float = 10.0) -> float:
    """Earth masses of gas a body tries to capture in ``dt`` Myr while gas is present: a core of
    ``core_threshold`` or more runs away (``(solid + gas) * (exp(dt / 0.3) - 1)``), a core of 5 or
    more takes ``solid * dt / 10``, a lighter one nothing."""
    if solid >= core_threshold:
        return (solid + gas) * math.expm1(dt / TAU_RUNAWAY)
    if solid >= ICE_CORE:
        return solid * dt / TAU_ENVELOPE
    return 0.0


def hill_radius(mass_1: float, mass_2: float, orbit_1: float, orbit_2: float, star_mass: float) -> float:
    """Mutual Hill radius of two planets in au (masses in earth masses, the star in solar masses)."""
    return math.cbrt((mass_1 + mass_2) / (3.0 * star_mass * EARTH_PER_SUN)) * (orbit_1 + orbit_2) / 2.0


def closest_approach(orbit_1: float, orbit_2: float, eccentricity: float) -> float:
    """How close two orbits of that eccentricity come, in au (negative when they cross)."""
    return float(abs(orbit_2 - orbit_1) - eccentricity * (orbit_1 + orbit_2))


def unstable(closest: float, hill_radius: float) -> str:
    """``yes`` when two neighbours can come within 2 sqrt(3) mutual Hill radii of each other."""
    return "yes" if closest < K_HILL * hill_radius else "no"


def safronov(mass: float, orbit: float, star_mass: float) -> float:
    """Square of the escape speed at the surface over the orbital speed, for a body of the
    earth's density: ``2 (m / M) (a / R)`` with ``R = R_earth * m^(1/3)``, that is
    ``0.14105 * mass^(2/3) * orbit / star_mass``. From 1 on a body throws what it meets out of
    the system."""
    if mass <= 0:
        return 0.0
    return 2.0 * mass / (star_mass * EARTH_PER_SUN) * orbit / (R_EARTH_AU * math.cbrt(mass))


def scatters(mass: float, orbit: float, star_mass: float) -> str:
    """``yes`` when the heavier of two meeting bodies ejects the other (Safronov number of 1 or
    more), ``no`` when the two merge."""
    return "yes" if safronov(mass, orbit, star_mass) >= 1.0 else "no"


def merged_orbit(mass_1: float, orbit_1: float, mass_2: float, orbit_2: float) -> float:
    """Orbit in au of two merged bodies on circular orbits: orbital angular momentum
    ``m * sqrt(a)`` is kept, ``((m1 sqrt(a1) + m2 sqrt(a2)) / (m1 + m2))^2``. The result lies
    between the two orbits; where rounding would put it a last digit outside, it is the nearer
    orbit (two bodies parked at 0.05 au merge at 0.05 au, not at 0.04999999999999998)."""
    root = (mass_1 * math.sqrt(orbit_1) + mass_2 * math.sqrt(orbit_2)) / (mass_1 + mass_2)
    return min(max(root * root, min(orbit_1, orbit_2)), max(orbit_1, orbit_2))


def _drift_clock(time: float, t_gas: float) -> float:
    """Integral of the gas share from 0 to ``time``: ``t - t^2 / (2 t_gas)``, with t at most ``t_gas``."""
    t = min(time, t_gas)
    return t - t * t / (2.0 * t_gas)


def migrated_orbit(orbit: float, star_mass: float, time: float, dt: float, t_gas: float) -> float:
    """Orbit in au of a gas giant after ``dt`` Myr of drifting inward with the disc gas from
    ``time`` on: ``da/dt = -(a / tau) * (1 - t / t_gas)`` with ``tau = TAU_MIG7 * a^1.5 /
    sqrt(star_mass)``, integrated exactly; never inside A_PARK7 (a body already there stays)."""
    if orbit <= A_PARK7 or t_gas <= 0:
        return float(orbit)
    clock = _drift_clock(time + dt, t_gas) - _drift_clock(time, t_gas)
    if clock <= 0:
        return float(orbit)
    x = orbit ** 1.5 - 1.5 * math.sqrt(star_mass) / TAU_MIG7 * clock
    return A_PARK7 if x <= A_PARK7 ** 1.5 else min(float(orbit), x ** (2.0 / 3.0))


def planet_type(solid: float, gas: float, orbit: float, frost_line: float) -> str:
    """``gas`` when the envelope weighs at least as much as the solids; else beyond the frost line
    ``ice_giant`` from 5 earth masses of solids and ``icy`` below, inside it ``super_earth`` from
    10 earth masses of solids and ``rocky`` below."""
    if gas >= solid and gas > 0:
        return "gas"
    if orbit > frost_line:
        return "ice_giant" if solid >= ICE_CORE else "icy"
    return "super_earth" if solid >= SUPER_EARTH else "rocky"


def _zone_edges(scale: float = 1.0) -> list[float]:
    """The 25 edges of the 24 logarithmic zones in au, for a star with ``sqrt(L) = scale``."""
    return [scale * R_IN * (R_OUT / R_IN) ** (i / N_ZONES) for i in range(N_ZONES + 1)]


def _zone_shares() -> tuple[tuple[float, ...], tuple[float, ...], int]:
    """Share of the disc's solids and of its gas in each zone, and the first zone beyond the
    frost line. Sigma ~ r^-1.5 gives a zone ``sqrt(r_out) - sqrt(r_in)``; ice triples the solids.
    The grid scales with sqrt(L) like the frost line, so the shares are the same for every star."""
    e = _zone_edges()
    gas = [math.sqrt(e[i + 1]) - math.sqrt(e[i]) for i in range(N_ZONES)]
    icy = [math.sqrt(e[i] * e[i + 1]) > FROST_AU for i in range(N_ZONES)]
    solid = [(ICE_FACTOR if cold else 1.0) * w for cold, w in zip(icy, gas)]
    s, g = math.fsum(solid), math.fsum(gas)
    return tuple(w / s for w in solid), tuple(w / g for w in gas), icy.index(True)


ZONE_SOLIDS7, ZONE_GAS7, FIRST_ICY_ZONE7 = _zone_shares()


def zone_solids(disc_mass: float, zone: int) -> float:
    """Earth masses of solids in zone ``zone`` (0 is the innermost) before anything is swept up.
    One embryo grows per zone, so this stands in for the isolation mass."""
    return float(disc_mass) * ZONE_SOLIDS7[zone]


def zone_gas(disc_mass: float, zone: int, gas_mass: float | None = None) -> float:
    """Earth masses of gas within a planet's reach in a zone at the start: GAS_REACH of the zone's
    gas. Without ``gas_mass`` the gas is 100 times the zone's rock (round 6); with it the disc's
    gas is ``gas_mass``, spread like the surface density."""
    if gas_mass is None:
        rock = zone_solids(disc_mass, zone) / (ICE_FACTOR if zone >= FIRST_ICY_ZONE7 else 1.0)
        return GAS_REACH * GAS_TO_ROCK * rock
    return GAS_REACH * float(gas_mass) * ZONE_GAS7[zone]


def gas_lifetime(draw: float) -> float:
    """Myr at which the disc gas is gone, for a uniform ``draw`` in [0, 1): 1 Myr plus an
    exponential of mean 2 Myr, at most 10, to 2 decimals."""
    return round(min(T_GAS_MAX7, T_GAS_SHIFT7 - T_GAS_MEAN7 * math.log(1.0 - draw)), 2)


RULES7: dict[str, Rule] = {
    "luminosity": Rule((Input("star_mass", 0.1, 3, places=2),), luminosity7,
                       "luminosity of a main sequence star in solar units by its mass in solar masses. below "
                       "0 point 4 3 it is 0 point 2 3 times mass to the power 2 point 3. from 0 point 4 3 to 2 it is "
                       "mass to the power 4. from 2 to 5 5 it is 1 point 4 times mass to the power 3 point 5. "
                       "above 5 5 it is 3 2 0 0 0 times mass"),
    "frost_line": Rule((Input("star_mass", 0.1, 3, places=2),), frost_line7,
                       "frost line in au. 2 point 7 times the square root of the luminosity. beyond it water is ice"),
    "flux": Rule((Input("luminosity", 0.01, 6), Input("orbit", 0.05, 6)), flux,
                 "starlight a planet receives in units of what the earth receives. luminosity over orbit squared. "
                 "luminosity in solar units. orbit in au"),
    "climate": Rule((Input("flux", 0.05, 1.6),), climate,
                    "runaway when flux above 1 point 1 0 7. frozen when flux below 0 point 3 5 6. else temperate. "
                    "the habitable zone is the temperate range"),
    "hz_inner": Rule((Input("luminosity", 0.01, 6),), hz_inner,
                     "inner edge of the habitable zone in au. square root of luminosity over 1 point 1 0 7"),
    "hz_outer": Rule((Input("luminosity", 0.01, 6),), hz_outer,
                     "outer edge of the habitable zone in au. square root of luminosity over 0 point 3 5 6"),
    "habitable": Rule((Input("luminosity", 0.05, 2), Input("orbit", 0.2, 2.5), Input("mass", 0.05, 12)), habitable,
                      "yes when a rocky planet without disc gas can keep liquid water. flux is luminosity over "
                      "orbit squared. "
                      "flux from 0 point 3 5 6 to 1 point 1 0 7 and mass from 0 point 3 to 1 0 earth masses. else no"),
    "equilibrium_temperature": Rule((Input("flux", 0.05, 5),), equilibrium_temperature,
                                    "kelvin of a planet without greenhouse that reflects 3 0 percent of the light. "
                                    "2 5 4 point 6 times flux to the power 0 point 2 5"),
    "greenhouse": Rule((Input("flux", 0.1, 2),), greenhouse,
                       "kelvin the air adds to the equilibrium temperature. 5 0 0 when flux above 1 point 1 0 7. "
                       "3 3 when flux from 1 to 1 point 1 0 7. from 0 point 3 5 6 to 1 it is 3 3 plus 6 7 point 3 "
                       "times one minus flux. below 0 point 3 5 6 it is 0. the planet is frozen"),
    "surface_temperature": Rule((Input("flux", 0.1, 2),), surface_temperature,
                                "surface temperature in whole kelvin. equilibrium temperature 2 5 4 point 6 times flux "
                                "to the power 0 point 2 5 plus greenhouse. a half rounds up. water is liquid from "
                                "2 7 3 to 3 7 3"),
    "period": Rule((Input("orbit", 0.05, 40), Input("star_mass", 0.1, 3, places=2)), period,
                   "orbital period in years. square root of orbit cubed over star_mass. orbit in au. star_mass in "
                   "solar masses"),
    "solids_remaining": Rule((Input("remaining", 0, 100), Input("efficiency", 0.6, 1.4), Input("orbit", 0.05, 30),
                              Input("star_mass", 0.5, 1.5, places=2), Input("dt", 0, 2)), solids_remaining,
                             "solids left in a zone after dt myr of sweeping. remaining times exp of minus efficiency "
                             "times dt over tau_sweep times period. tau_sweep 0 point 2. period is the square root of "
                             "orbit cubed over star_mass"),
    "gas_share": Rule((Input("time", 0, 12, places=2), Input("t_gas", 1, 10, places=2)), gas_share,
                      "share of the disc gas still there. one minus time over t_gas. zero from t_gas on. times in myr"),
    "gas_wanted": Rule((Input("solid", 0, 30), Input("gas", 0, 100), Input("dt", 0.005, 0.5)), gas_wanted,
                       "earth masses of gas a body tries to capture in dt myr while gas is present. solid at least "
                       "1 0 runs away and wants solid plus gas times exp of dt over 0 point 3 minus one. solid from "
                       "5 to 1 0 wants solid times dt over 1 0. below 5 nothing"),
    "hill_radius": Rule((Input("mass_1", 0.01, 300), Input("mass_2", 0.01, 300), Input("orbit_1", 0.05, 30),
                         Input("orbit_2", 0.05, 30), Input("star_mass", 0.5, 1.5, places=2)), hill_radius,
                        "mutual hill radius of two planets in au. cube root of mass_1 plus mass_2 over 3 times the "
                        "star mass in earth masses. times the mean of the two orbits. one solar mass is counted "
                        "as 3 3 2 9 5 4 earth masses"),
    "closest_approach": Rule((Input("orbit_1", 0.05, 30), Input("orbit_2", 0.05, 30),
                              Input("eccentricity", 0, 0.3, places=2)), closest_approach,
                             "how close two orbits come in au. the difference of the orbits minus eccentricity times "
                             "their sum. eccentricity 0 point 0 5 while gas damps the orbits and 0 point 2 after. "
                             "negative when the orbits cross"),
    "unstable": Rule((Input("closest", -1, 3), Input("hill_radius", 0.001, 0.5, places=4)), unstable,
                     "yes when two neighbours can meet. closest approach below 2 times the square root of 3 times "
                     "the mutual hill radius. that is 3 point 4 6 4 hill radii"),
    "safronov": Rule((Input("mass", 0.01, 400), Input("orbit", 0.05, 40), Input("star_mass", 0.5, 1.5, places=2)),
                     safronov,
                     "square of escape speed at the surface over orbital speed for a body of the density of the "
                     "earth. 0 point 1 4 1 0 5 times mass to the power 2 over 3 times orbit over star_mass. mass in "
                     "earth masses. orbit in au. star_mass in solar masses"),
    "scatters": Rule((Input("mass", 0.05, 10), Input("orbit", 0.05, 10), Input("star_mass", 0.5, 1.5, places=2)),
                     scatters,
                     "yes when the heavier of two meeting bodies throws the other out of the system. its safronov "
                     "number 0 point 1 4 1 0 5 times mass to the power 2 over 3 times orbit over star_mass is at "
                     "least 1. below 1 the two merge"),
    "merged_orbit": Rule((Input("mass_1", 0.01, 100), Input("orbit_1", 0.05, 30), Input("mass_2", 0.01, 100),
                          Input("orbit_2", 0.05, 30)), merged_orbit,
                         "orbit in au of two merged bodies. angular momentum mass times square root of orbit is "
                         "kept. the mass weighted mean of the square roots of the orbits. squared. never "
                         "outside the two orbits"),
    "migrated_orbit": Rule((Input("orbit", 0.05, 30), Input("star_mass", 0.5, 1.5, places=2),
                            Input("time", 0, 10, places=2), Input("dt", 0.005, 0.5), Input("t_gas", 1, 10, places=2)),
                           migrated_orbit,
                           "orbit in au of a gas giant after dt myr of drift with the disc gas. orbit to the power "
                           "1 point 5 falls by 1 point 5 times the square root of star_mass times g of time plus dt "
                           "minus g of time. g of t is t minus t squared over 2 t_gas with t at most t_gas. never "
                           "inside 0 point 0 5 au"),
    "planet_type": Rule((Input("solid", 0.1, 15), Input("gas", 0, 6), Input("orbit", 0.1, 6),
                         Input("frost_line", 0.5, 5)), planet_type,
                        "gas when the gas weighs at least as much as the solids. else beyond the frost line "
                        "ice_giant from 5 earth masses of solids and icy below. else inside it super_earth from "
                        "1 0 earth masses of solids and rocky below"),
    "zone_solids": Rule((Input("disc_mass", 10, 450, places=1), Input("zone", 0, N_ZONES - 1, integer=True)),
                        zone_solids,
                        "earth masses of solids in one of 2 4 zones before sweeping. the zones are equal steps in the "
                        "logarithm of the orbit from 0 point 3 to 3 0 au times the square root of the luminosity. "
                        "surface density falls as orbit to the power minus 1 point 5. ice triples it from zone "
                        "1 1 on. zone 0 is the innermost. this mass stands in for the isolation mass"),
    "gas_lifetime": Rule((Input("draw", 0, 0.999),), gas_lifetime,
                         "time in myr at which the disc gas is gone for a uniform draw between 0 and 1. 1 minus "
                         "2 times the natural log of one minus draw. at most 1 0. rounded to 2 decimals"),
}
#: the gate of the rules-7 lessons: ``planets7 predict <rule> <input> <value> ...``, topic ``predict_planets7``
LESSONS = LessonGate(SIM7, RULES7)
RULES = RULES7


def lesson_gate() -> LessonGate:
    return LESSONS


# ---------------------------------------------------------------------------------------------
# the rules-7 run
# ---------------------------------------------------------------------------------------------

def _random_params7(rng: random.Random) -> dict[str, float]:
    """In round 6's draw order: ``star_mass`` uniform 0.5-1.5, ``disc_mass`` = star mass times a
    uniform 10-300 earth masses, ``t_gas`` from :func:`gas_lifetime`."""
    star_mass = round(rng.uniform(0.5, 1.5), 2)
    return {"star_mass": star_mass,
            "disc_mass": round(star_mass * rng.uniform(*DISC_PER_STAR7), 1),
            "t_gas": gas_lifetime(rng.random()),
            "core_threshold": 10.0}


class _Body7:
    """One row of the rules-7 disc: a body and the zone (or merged zones) it feeds on."""
    __slots__ = ("id", "a", "r_in", "r_out", "s_rem", "gas0", "gas_cap", "solid", "gas", "eps")

    def __init__(self, id_: int, a: float, r_in: float, r_out: float, s_rem: float, gas0: float, eps: float) -> None:
        self.id, self.a, self.r_in, self.r_out = id_, a, r_in, r_out
        self.s_rem, self.gas0, self.eps = s_rem, gas0, eps
        self.gas_cap = self.solid = self.gas = 0.0

    @property
    def mass(self) -> float:
        return self.solid + self.gas


class _Disc7:
    """The mutable state under rules 7: a list of bodies in order of orbit."""

    def __init__(self, rng: _Rng, star_mass: float, disc_mass: float, t_gas: float,
                 gas_mass: float | None = None) -> None:
        self.star_mass, self.t_gas = star_mass, t_gas
        self.luminosity = luminosity7(star_mass)
        self.r_frost = frost_line7(star_mass)
        edges = _zone_edges(math.sqrt(self.luminosity))
        eps = rng.uniform(*EPS_RANGE, size=N_ZONES).tolist()
        self.rows = [_Body7(i, math.sqrt(edges[i] * edges[i + 1]), edges[i], edges[i + 1], zone_solids(disc_mass, i),
                            zone_gas(disc_mass, i, gas_mass), eps[i]) for i in range(N_ZONES)]
        self.debris = 0.0
        self.gas_ejected = 0.0
        self.n_ejected = 0
        self.events: list[dict] = []
        self.step = 0  # index of the output step the current substeps lead to

    def gas_at(self, t: float) -> bool:
        return t < self.t_gas - T_EPS

    def available(self, b: _Body7, t: float) -> float:
        return max(0.0, b.gas0 * gas_share(t, self.t_gas) - b.gas_cap)

    def advance(self, rng: _Rng, t: float, dt: float, core_threshold: float) -> None:
        mid = t + dt / 2
        gas_here = self.gas_at(mid)
        for b in self.rows:
            if b.a >= b.r_in:  # a body that has migrated out of its zone leaves the rest there as a belt
                left = solids_remaining(b.s_rem, b.eps, b.a, self.star_mass, dt)
                b.solid += b.s_rem - left
                b.s_rem = left
            if gas_here:
                got = min(gas_wanted(b.solid, b.gas, dt, core_threshold), self.available(b, mid))
                b.gas_cap += got
                b.gas += got
        if gas_here:
            self.migrate(t, dt)
        self.collide(rng, t + dt, dt, gas_here)

    def migrate(self, t: float, dt: float) -> None:
        """Gas giants drift inward with the disc gas; one that reaches its inner neighbour's orbit
        meets it at once."""
        rows, moved = self.rows, False
        for b in rows:
            if b.gas >= b.solid and b.gas > 0:
                b.a = migrated_orbit(b.a, self.star_mass, t, dt, self.t_gas)
                moved = True
        if not moved:
            return
        i = 1
        while i < len(rows):
            if rows[i].a <= rows[i - 1].a:
                rows[i - 1:i + 1] = [self.merge(t + dt, [rows[i - 1], rows[i]])]
                i = max(1, i - 1)
            else:
                i += 1

    def pair_unstable(self, x: _Body7, y: _Body7, ecc: float) -> bool:
        return unstable(closest_approach(x.a, y.a, ecc),
                        hill_radius(x.mass, y.mass, x.a, y.a, self.star_mass)) == "yes"

    def collide(self, rng: _Rng, t: float, dt: float, gas_here: bool) -> None:
        rows = self.rows
        if len(rows) < 2:
            return
        ecc, tau = (E_GAS, TAU_MERGE_GAS) if gas_here else (E_LATE, TAU_MERGE_LATE)
        p_merge = -math.expm1(-dt / tau)
        joins = [bool(self.pair_unstable(x, y, ecc) and rng.random() < p_merge) for x, y in zip(rows, rows[1:])]
        if not any(joins):
            return
        groups: list[list[_Body7]] = [[rows[0]]]
        for b, join in zip(rows[1:], joins):
            if join:
                groups[-1].append(b)
            else:
                groups.append([b])
        self.rows = [self.merge(t, g) if len(g) > 1 else g[0] for g in groups]

    def merge(self, t: float, g: list[_Body7]) -> _Body7:
        """The bodies in ``g`` meet. The heaviest (of equally heavy ones the first) decides: with
        a Safronov number of 1 or more it throws the others out of the system, with their gas
        and the unswept solids of their zones, and stays where it is; below 1 all merge into one
        body at the orbit that keeps their angular momentum. No random number is drawn."""
        masses = [b.mass for b in g]
        k = masses.index(max(masses))
        big = g[k]
        total = math.fsum(masses)
        theta = safronov(masses[k], big.a, self.star_mass)
        event = {"t": t, "step": self.step, "ids": sorted(b.id for b in g), "kept": big.id, "mass_before": total,
                 "theta": theta,
                 "members": [{"id": b.id, "a": b.a, "solid": b.solid, "gas": b.gas, "zone": b.s_rem} for b in g]}
        if scatters(masses[k], big.a, self.star_mass) == "yes":
            others = [b for b in g if b is not big]
            debris = math.fsum([b.solid for b in others] + [b.s_rem for b in others])
            gas_out = math.fsum(b.gas for b in others)
            out = big
            out.gas0 = math.fsum(b.gas0 for b in g)
            out.gas_cap = math.fsum(b.gas_cap for b in g)
            self.n_ejected += len(others)
            event.update(debris=debris, gas_ejected=gas_out, ejected=sorted(b.id for b in others))
        else:
            orbit, mass = g[0].a, masses[0]
            for b, m in zip(g[1:], masses[1:]):
                if mass + m > 0:
                    orbit = merged_orbit(mass, orbit, m, b.a)
                mass += m
            r_in, r_out = min(b.r_in for b in g), max(b.r_out for b in g)
            if total <= 0:
                orbit = math.sqrt(r_in * r_out)
            out = _Body7(big.id, orbit, r_in, r_out, math.fsum(b.s_rem for b in g), math.fsum(b.gas0 for b in g),
                         math.fsum(b.eps for b in g) / len(g))
            out.gas_cap = math.fsum(b.gas_cap for b in g)
            out.solid = math.fsum(b.solid for b in g)
            out.gas = math.fsum(b.gas for b in g)
            debris = gas_out = 0.0
            event.update(debris=0.0, gas_ejected=0.0, ejected=[])
        event.update(orbit=out.a, mass_after=out.mass)
        self.events.append(event)
        self.debris += debris
        self.gas_ejected += gas_out
        return out

    def stage(self, t: float, n_planets: int, core_threshold: float) -> str:
        if n_planets == 0:
            return "dust"
        rows = self.rows
        if not self.gas_at(t):
            return "clearing" if any(self.pair_unstable(x, y, E_LATE) for x, y in zip(rows, rows[1:])) else "done"
        return "giants_forming" if any(b.solid >= core_threshold for b in rows) else "embryos"

    def snapshot(self, t: float, disc_mass: float, core_threshold: float) -> tuple[dict, list[dict]]:
        rows = self.rows
        types = [planet_type(b.solid, b.gas, b.a, self.r_frost) for b in rows]
        orbits = [b.a for b in rows if b.mass >= PLANET_MIN]
        counts = dict.fromkeys(TYPE_COUNTS7.values(), 0)
        for b, ty in zip(rows, types):
            if b.mass >= PLANET_MIN:
                counts[TYPE_COUNTS7[ty]] += 1
        step = {
            "time": float(t), "planets": len(orbits), **counts, "ejected": self.n_ejected,
            "largest_mass": max((b.mass for b in rows), default=0.0),
            "innermost_au": min(orbits, default=0.0), "outermost_au": max(orbits, default=0.0),
            "debris": self.debris / disc_mass,
            "gas_present": "yes" if self.gas_at(t) else "no",
            "stage": self.stage(t, len(orbits), core_threshold),
            "solids_in_bodies": math.fsum(b.solid for b in rows), "zone_solids": math.fsum(b.s_rem for b in rows),
            "debris_mass": self.debris, "gas_captured": math.fsum(b.gas_cap for b in rows),
            "gas_in_bodies": math.fsum(b.gas for b in rows), "gas_ejected": self.gas_ejected,
            "gas_left": math.fsum(self.available(b, t) for b in rows),
        }
        bodies = [{"id": b.id, "a": b.a, "solid": b.solid, "gas": b.gas, "type": ty} for b, ty in zip(rows, types)]
        return step, bodies


def _system7(bodies: list[dict], luminosity: float) -> list[dict]:
    """The planets among ``bodies`` (in order of orbit) with the climate their star gives them."""
    out = []
    for b in bodies:
        mass = b["solid"] + b["gas"]
        if mass < PLANET_MIN:
            continue
        light = flux(luminosity, b["a"])
        bare_rock = b["type"] == "rocky" and b["gas"] == 0  # disc gas left on a planet is no earth-like sky
        out.append({"id": b["id"], "orbit": b["a"], "mass": mass, "solid": b["solid"], "gas": b["gas"],
                    "type": b["type"], "flux": light, "temperature": _temperature7(light, b["type"]),
                    "habitable": habitable(luminosity, b["a"], mass) if bare_rock else "no"})
    return out


def _run7(seed: int, star_mass: float = 1.0, disc_mass: float = 100.0, t_gas: float = 3.0,
          core_threshold: float = 10.0, gas_mass: float | None = None) -> PlanetRollout:
    """One rules-7 rollout on round 6's output times (:data:`TIMES`) and substeps. Every formula
    is one of the lesson functions above; the random numbers are round 6's (the zone
    efficiencies, then one draw per unstable pair and substep)."""
    star_mass, disc_mass, t_gas, core_threshold = float(star_mass), float(disc_mass), float(t_gas), float(core_threshold)
    if not (math.isfinite(star_mass) and star_mass > 0 and math.isfinite(disc_mass) and disc_mass > 0):
        raise ValueError(f"star_mass and disc_mass must be positive: {star_mass!r}, {disc_mass!r}")
    if not (math.isfinite(t_gas) and t_gas >= 0 and core_threshold > 0):
        raise ValueError(f"t_gas must not be negative, core_threshold must be positive: {t_gas!r}, {core_threshold!r}")
    if gas_mass is not None:
        gas_mass = float(gas_mass)
        if not (math.isfinite(gas_mass) and gas_mass >= 0):
            raise ValueError(f"gas_mass must not be negative: {gas_mass!r}")
    rng = _Rng(seed)
    disc = _Disc7(rng, star_mass, disc_mass, t_gas, gas_mass)
    params: dict[str, float | int] = {
        "star_mass": star_mass, "disc_mass": disc_mass, "t_gas": t_gas, "core_threshold": core_threshold,
        "luminosity": disc.luminosity, "frost_line": disc.r_frost, "hz_inner": hz_inner(disc.luminosity),
        "hz_outer": hz_outer(disc.luminosity), "gas_initial": math.fsum(b.gas0 for b in disc.rows)}
    if gas_mass is not None:
        params["gas_mass"] = gas_mass
    params["rules"] = 7
    r = PlanetRollout(SIM, int(seed), params, [])
    step, bodies = disc.snapshot(TIMES[0], disc_mass, core_threshold)
    r.steps.append(step)
    r.bodies.append(bodies)
    for k, (t0, t1) in enumerate(zip(TIMES, TIMES[1:]), start=1):
        dt = (t1 - t0) / SUBSTEPS
        disc.step = k
        for i in range(SUBSTEPS):
            disc.advance(rng, t0 + i * dt, dt, core_threshold)
        step, bodies = disc.snapshot(t1, disc_mass, core_threshold)
        r.steps.append(step)
        r.bodies.append(bodies)
    r.events = disc.events
    r.system = _system7(r.bodies[-1], disc.luminosity)
    last = r.steps[-1]
    r.summary = {
        **{k: last[k] for k in SUMMARY_KEYS7[:8]},
        "habitable": sum(1 for p in r.system if p["habitable"] == "yes"),
        "inner_giants": sum(1 for p in r.system if p["type"] == "gas" and p["orbit"] <= disc.r_frost),
        "total_planet_mass": math.fsum(p["mass"] for p in r.system),
    }
    return r


def _conserved7(r: Rollout) -> Verdict:
    """The gate of a rules-7 rollout (the list is in the module docstring, section "Rules 7")."""
    p = r.params
    if any(k not in p for k in (*PARAM_KEYS7, "gas_initial")):
        return Verdict(False, None, "a rules-7 rollout names star_mass, frost_line and the zone in its parameters")
    initial, gas_initial, lum = float(p["disc_mass"]), float(p["gas_initial"]), float(p["luminosity"])
    events = getattr(r, "events", [])
    bodies = getattr(r, "bodies", None) or []
    if len(bodies) != len(r.steps):
        return Verdict(False, None, "a rules-7 rollout keeps its bodies at every step")
    prev = None
    for t, s in enumerate(r.steps):
        past = [e for e in events if e["step"] <= t]
        total = s["solids_in_bodies"] + s["zone_solids"] + s["debris_mass"]
        if abs(total - initial) > 1e-9:
            return Verdict(False, num(initial, sig=6), f"step {t}: solids {total!r} != initial {initial!r}")
        if not close(s["debris"] * initial, s["debris_mass"], rel=1e-9):
            return Verdict(False, None, f"step {t}: debris fraction does not match the debris mass")
        if abs(math.fsum(e["debris"] for e in past) - s["debris_mass"]) > 1e-9:
            return Verdict(False, None, f"step {t}: debris does not match the event log")
        if abs(math.fsum(e["gas_ejected"] for e in past) - s["gas_ejected"]) > 1e-9:
            return Verdict(False, None, f"step {t}: ejected gas does not match the event log")
        if s["ejected"] != sum(len(e["ejected"]) for e in past):
            return Verdict(False, None, f"step {t}: ejected bodies do not match the event log")
        if abs(s["gas_in_bodies"] - (s["gas_captured"] - s["gas_ejected"])) > 1e-9:
            return Verdict(False, None, f"step {t}: gas in bodies != gas captured - gas ejected")
        if s["gas_captured"] + s["gas_left"] > gas_initial + 1e-9 or s["gas_left"] < 0:
            return Verdict(False, None, f"step {t}: more gas captured than the disc held")
        counted = sum(s[k] for k in TYPE_COUNTS7.values())
        if s["planets"] != counted:
            return Verdict(False, num(counted), f"step {t}: planets != rocky + super_earths + icy + ice_giants + gas_giants")
        if s["stage"] not in STAGES or s["gas_present"] not in ("yes", "no"):
            return Verdict(False, None, f"step {t}: unknown stage or gas_present")
        here = bodies[t]
        if abs(math.fsum(b["solid"] for b in here) - s["solids_in_bodies"]) > 1e-9 \
                or abs(math.fsum(b["gas"] for b in here) - s["gas_in_bodies"]) > 1e-9:
            return Verdict(False, None, f"step {t}: the bodies do not hold the solids and gas of the ledger")
        if any(y["a"] <= x["a"] for x, y in zip(here, here[1:])):
            return Verdict(False, None, f"step {t}: the bodies are not in order of orbit")
        if any(b["type"] != planet_type(b["solid"], b["gas"], b["a"], float(p["frost_line"])) for b in here):
            return Verdict(False, None, f"step {t}: a body's type does not follow from its solids, gas and orbit")
        planets = [b for b in here if b["solid"] + b["gas"] >= PLANET_MIN]
        if any(s[key] != sum(1 for b in planets if b["type"] == ty) for ty, key in TYPE_COUNTS7.items()):
            return Verdict(False, None, f"step {t}: the counts are not the types of the bodies")
        if prev is not None:
            if not s["time"] > prev["time"]:
                return Verdict(False, None, f"step {t}: time does not increase")
            if s["gas_captured"] < prev["gas_captured"] - 1e-9:
                return Verdict(False, None, f"step {t}: captured gas went down")
            if s["debris_mass"] < prev["debris_mass"] or s["zone_solids"] > prev["zone_solids"] + 1e-9:
                return Verdict(False, None, f"step {t}: debris or swept solids went backwards")
        prev = s
    verdict = _events7(r, events, bodies)
    if not verdict.ok:
        return verdict
    verdict = _bodies7(r, events, bodies)
    if not verdict.ok:
        return verdict
    system = getattr(r, "system", [])
    if system != _system7(bodies[-1], lum):
        return Verdict(False, None, "the final system is not the planets of the last step with their climate")
    for n, planet in enumerate(system, start=1):
        liquid = WATER_K[0] <= planet["temperature"] <= WATER_K[1]
        if planet["habitable"] == "yes" and not liquid:
            return Verdict(False, None, f"planet {n}: habitable, but its water is not liquid")
        if planet["habitable"] == "yes" and (planet["type"] != "rocky" or planet["gas"] != 0):
            return Verdict(False, None, f"planet {n}: habitable, but no rocky planet without disc gas")
        if planet["type"] != "gas" and liquid != (climate(planet["flux"]) == "temperate"):
            return Verdict(False, None, f"planet {n}: liquid water and a temperate flux do not go together")
    last = r.steps[-1]
    if any(r.summary.get(k) != last[k] for k in SUMMARY_KEYS7[:8]) \
            or r.summary.get("habitable") != sum(1 for x in system if x["habitable"] == "yes") \
            or r.summary.get("inner_giants") != sum(1 for x in system
                                                    if x["type"] == "gas" and x["orbit"] <= float(p["frost_line"])):
        return Verdict(False, None, "the summary does not show how the run ended")
    return Verdict(True, None, "solids, gas, angular momentum, counts, climate and time are consistent")


def _events7(r: Rollout, events: list[dict], bodies: list[list[dict]]) -> Verdict:
    """Every logged meeting is an ejection by a body with a Safronov number of 1 or more (the
    others leave with their solids, zone solids and gas; the survivor keeps mass and orbit) or a
    merger below 1 that loses no mass and keeps the orbital angular momentum."""
    star_mass = float(r.params["star_mass"])
    for n, e in enumerate(events):
        members = {m["id"]: m for m in e["members"]}
        if not 1 <= e["step"] < len(bodies) or e["kept"] not in members or sorted(members) != e["ids"]:
            return Verdict(False, None, f"event {n}: bad step, survivor or members")
        kept = members[e["kept"]]
        kept_mass = kept["solid"] + kept["gas"]
        masses = [m["solid"] + m["gas"] for m in e["members"]]
        if kept_mass < max(masses) or abs(math.fsum(masses) - e["mass_before"]) > 1e-9:
            return Verdict(False, None, f"event {n}: the survivor is not the heaviest, or the masses do not add up")
        theta = safronov(kept_mass, kept["a"], star_mass)
        if not close(theta, e["theta"], rel=1e-9) or (theta >= 1.0) != bool(e["ejected"]):
            return Verdict(False, None, f"event {n}: ejection or merger does not follow from the safronov number")
        if e["ejected"]:
            others = [m for m in e["members"] if m["id"] != e["kept"]]
            if sorted(m["id"] for m in others) != e["ejected"] \
                    or abs(math.fsum([m["solid"] for m in others] + [m["zone"] for m in others]) - e["debris"]) > 1e-9 \
                    or abs(math.fsum(m["gas"] for m in others) - e["gas_ejected"]) > 1e-9:
                return Verdict(False, None, f"event {n}: the debris is not what the ejected bodies and their zones held")
            if e["orbit"] != kept["a"] or abs(e["mass_after"] - kept_mass) > 1e-9:
                return Verdict(False, None, f"event {n}: the survivor of an ejection changed mass or orbit")
        else:
            if e["debris"] != 0 or e["gas_ejected"] != 0 or abs(e["mass_after"] - e["mass_before"]) > 1e-9:
                return Verdict(False, None, f"event {n}: a merger lost mass")
            before = math.fsum(m * math.sqrt(x["a"]) for m, x in zip(masses, e["members"]))
            after = e["mass_after"] * math.sqrt(e["orbit"])
            if abs(before - after) > 1e-9 * max(1.0, before):
                return Verdict(False, None, f"event {n}: a merger did not keep the orbital angular momentum")
        if any(e["kept"] in f["ids"] for f in events[n + 1:] if f["step"] == e["step"]):
            continue  # met another body before the next output step
        now = next((b for b in bodies[e["step"]] if b["id"] == e["kept"]), None)
        if now is None or now["solid"] + now["gas"] < e["mass_after"] - 1e-9:
            return Verdict(False, None, f"event {n} at step {e['step']}: the survivor lost mass")
    return Verdict(True)


def _bodies7(r: Rollout, events: list[dict], bodies: list[list[dict]]) -> Verdict:
    """Outside a logged event no body loses mass or vanishes, and only a body that holds gas moves,
    inward, while the disc has gas, and never inside A_PARK7."""
    for t in range(1, len(bodies)):
        met = {i for e in events if e["step"] == t for i in e["ids"]}
        before = {b["id"]: b for b in bodies[t - 1]}
        for b in bodies[t]:
            if b["id"] in met:
                continue
            was = before.get(b["id"])
            if was is None:
                return Verdict(False, None, f"step {t}: body {b['id']} appeared from nowhere")
            if b["solid"] + b["gas"] < was["solid"] + was["gas"] - 1e-9:
                return Verdict(False, None, f"step {t}: body {b['id']} lost mass without an event")
            if b["a"] != was["a"] and not (r.steps[t - 1]["gas_present"] == "yes" and b["gas"] > 0
                                           and A_PARK7 <= b["a"] < was["a"]):
                return Verdict(False, None, f"step {t}: body {b['id']} moved without gas to carry it")
        if not set(before) - {b["id"] for b in bodies[t]} <= met:
            return Verdict(False, None, f"step {t}: a body vanished without an event")
    return Verdict(True)


# ---------------------------------------------------------------------------------------------
# the rules-7 lines and their gate
# ---------------------------------------------------------------------------------------------

def param_value7(key: str, value: float) -> str:
    """A rules-7 parameter as it stands in a line: an input as given, a derived one to 3 digits."""
    return num(float(value), sig=5) if key in INPUT_KEYS7 else dense_value(value)


def parse7(prompt: str) -> tuple[int, str, int, str] | None:
    """``planets seed 3 rules 7 step 1 4 next stage`` -> ``(3, "next", 14, "stage")``; ``where`` is
    ``step``, ``next``, ``final``, ``params`` or ``planet`` (then the number is the planet's, from
    1). ``None`` for anything else, a round-6 prompt included."""
    w = prompt.split()
    if len(w) < 7 or w[0] != SIM or w[1] != "seed" or " ".join(w) != prompt:
        return None
    i = _digits_end(w, 2)
    seed = parse_num(w[2:i])
    if not isinstance(seed, int) or num(seed).split() != w[2:i] or w[i:i + 2] != ["rules", "7"]:  # no leading zeros
        return None
    where, rest = w[i + 2], w[i + 3:]
    if where in ("final", "params"):
        keys = SUMMARY_KEYS7 if where == "final" else PARAM_KEYS7
        return (seed, where, -1, rest[0]) if len(rest) == 1 and rest[0] in keys else None
    if where not in ("step", "planet"):
        return None
    j = _digits_end(rest, 0)
    digits, rest = rest[:j], rest[j:]
    n = parse_num(digits)
    if not isinstance(n, int) or num(n).split() != digits:
        return None
    if where == "planet":
        return (seed, "planet", n, rest[0]) if 1 <= n <= N_ZONES and len(rest) == 1 and rest[0] in PLANET_KEYS7 else None
    if not 0 <= n < len(TIMES):
        return None
    if len(rest) == 2 and rest[0] == "next" and rest[1] in STATE_KEYS7 + LEDGER_KEYS7 and n + 1 < len(TIMES):
        return seed, "next", n, rest[1]
    if len(rest) == 1 and rest[0] in STATE_KEYS7 + LEDGER_KEYS7:
        return seed, "step", n, rest[0]
    return None


def _truth7(r: PlanetRollout, q: tuple[int, str, int, str]) -> float | int | str | None:
    """The value a parsed rules-7 prompt asks for in the seed's canonical rollout ``r``; ``None``
    for a planet the system does not have."""
    _, where, n, key = q
    if where == "final":
        return r.summary[key]
    if where == "params":
        return r.params[key]
    if where == "planet":
        return r.system[n - 1][key] if n <= len(r.system) else None
    return r.steps[n + (where == "next")][key]


def _check7(r: PlanetRollout, q: tuple[int, str, int, str], answer: str) -> Verdict:
    """Compare with the seed's rules-7 rollout: a word, a count and a whole kelvin exactly, a
    parameter that was put in exactly as given, every measured number within 5 percent."""
    value = _truth7(r, q)
    if value is None:
        return Verdict(False, None, f"the system of seed {q[0]} has no planet {q[2]}")
    where, key = q[1], q[3]
    expected = param_value7(key, value) if where == "params" else dense_value(value)
    if isinstance(value, str):
        ok = answer.strip() == value
        return Verdict(ok, expected, "exact word" if ok else "wrong word")
    got = parse_num(answer)
    if not is_finite(got):  # "1 e 9 9 9" parses to infinity, which is close to nothing
        return Verdict(False, expected, "not a number")
    if isinstance(value, int):
        ok = got == value
        what = "kelvin" if key == "temperature" else "count"
        return Verdict(ok, expected, f"exact {what}" if ok else f"wrong {what}")
    if where == "params" and key in INPUT_KEYS7:
        ok = float(got) == float(value)
        return Verdict(ok, expected, "the parameter as given" if ok else "wrong parameter")
    ok = close(float(got), float(value), rel=0.05)
    return Verdict(ok, expected, "within 5 percent" if ok else "more than 5 percent off")


def _canonical7(r: Rollout) -> bool:
    """Was ``r`` run with the parameters its seed stands for under rules 7?"""
    if not isinstance(r.seed, int) or isinstance(r.seed, bool) or r.seed < 0 or "gas_mass" in r.params:
        return False
    return all(r.params.get(k) == v for k, v in _random_params7(random.Random(r.seed)).items())


def params_line7(r: Rollout) -> Line:
    """``planets seed 3 rules 7 params. star_mass 0 point 7 4. disc_mass 1 2 4 point 2. t_gas 1 point 9 2. ...``"""
    fields = " ".join(f"{k} {param_value7(k, r.params[k])}." for k in PARAM_KEYS7)
    return Line(f"{SIM} seed {num(r.seed)} rules 7 params. {fields}", topic=SIM, kind="record", meta={"rules": 7})


def _lines7(r: Rollout, every: int = 1) -> list[Line]:
    """The lines of ``rollout(seed, rules=7)``: the kinds of round 6 with ``rules 7`` after the
    seed and the rules-7 keys, and a record and questions for every planet of the finished
    system. Only the seed's canonical rollout has lines: :func:`check` replays the seed's own
    parameters, and one prompt must have one answer."""
    if not _canonical7(r):
        raise ValueError(f"lines() wants rollout({r.seed!r}, rules=7): check() replays the seed's random parameters")
    head = f"{SIM} seed {num(r.seed)} rules 7"
    value = dense_value

    def line(prompt: str, answer: str = "", kind: str = "record") -> Line:
        return Line(prompt, answer, SIM, kind, {"rules": 7})

    out = [params_line7(r)]
    for t in range(0, len(r.steps), every):
        step = r.steps[t]
        out.append(line(f"{head} step {num(t)}. " + " ".join(f"{k} {value(step[k])}." for k in STATE_KEYS7)))
        out += [line(f"{head} step {num(t)} {k}", value(step[k]), "fact") for k in STATE_KEYS7]
        if t + 1 < len(r.steps):
            out += [line(f"{head} step {num(t)} next {k}", value(r.steps[t + 1][k]), "calc") for k in STATE_KEYS7]
    for n, planet in enumerate(getattr(r, "system", []), start=1):
        out.append(line(f"{head} planet {num(n)}. " + " ".join(f"{k} {value(planet[k])}." for k in PLANET_KEYS7)))
        out += [line(f"{head} planet {num(n)} {k}", value(planet[k]), "calc") for k in PLANET_KEYS7]
    out += [line(f"{head} final {k}", value(r.summary[k]), "calc") for k in SUMMARY_KEYS7]
    return out


_SIM = Planets()
_SIM7 = Planets(rules=7)


def simulation(rules: int = 6) -> Planets:
    """The level as an object: round 6 by default, the corrected rules with ``rules=7``."""
    return _SIM7 if _rule_set(rules) == 7 else _SIM


def run(seed: int, rules: int = 6, **params: float) -> PlanetRollout:
    """``simulation().run`` as a function: round 6 by default, the corrected rules with ``rules=7``."""
    return _SIM.run(seed, rules=_rule_set(rules), **params)


def rollout(seed: int, rules: int = 6) -> PlanetRollout:
    """The run a seed stands for in the lines and in :func:`check` (cached; do not change it)."""
    return _SIM.rollout(seed, _rule_set(rules))


def conserved(r: Rollout) -> Verdict:
    return _SIM.conserved(r)


def check(prompt: str, answer: str) -> Verdict:
    """Judge an answer to a question about a rollout: ``planets seed 3 step 1 4 gas`` (round 6) or
    ``planets seed 3 rules 7 step 1 4 gas_giants``. Lessons (``planets7 predict ...``) are judged
    by :data:`LESSONS`."""
    return _SIM.check(prompt, answer)


def owns(prompt: str) -> bool:
    return _SIM.owns(prompt)
