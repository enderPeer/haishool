"""Level 2 of the toy universe: a rotating gas cloud collapses into clumps and a flattened disc.

(Round 6 called the two-fold patterns it saw spiral arms; they were mostly pairs of clumps, and
rules 7 name them for what they are.)

What follows describes the level as round 6 wrote it, which is still what every function does
by default. A review in round 7 found several of its rules wrong or arbitrary; the corrected
level is the same code called with ``rules=7`` and is described in the last section, "Rules 7".

A cloud of N equal particles (256) starts as a near-Gaussian ball (sigma 0.5 R, cut at the
cloud radius R) in solid-body rotation about z, plus random motions. Gravity is softened
(Plummer, ``eps`` = 0.05 R) and integrated with a kick-drift-kick leapfrog at a fixed step.
Units: G = 1, total mass M = 1, cloud radius R = 1, so the time unit is sqrt(R^3 / GM) = 1 and
the free-fall time of a uniform sphere is t_ff = pi / sqrt(8) = 1.11. A run lasts 4 t_ff and
writes 41 states, one every 0.1 t_ff.

Parameters (``random_params`` draws them from the seed):

* ``spin`` (0.05 to 0.4): the rotational kinetic energy as a fraction of |W|, the potential
  energy; 0.5 would be a cloud held up by rotation alone. (It is not Peebles' lambda, which
  cannot exceed about 0.25 for a bound cloud of this shape.) The random motions are chosen so
  that the cloud starts in virial balance with every axis held up: the tensor virial theorem
  for a sphere asks for a kinetic energy of |W|/6 along z and |W|/3 in the plane, of which
  rotation takes its share. Without cooling the cloud therefore stays a slowly sloshing
  spheroid. Above spin 1/3 rotation alone exceeds what the plane can hold, so such a cloud
  spreads and turns oblate even without cooling.
* ``cooling`` (0, or 0.1 to 0.6): the strength of the dissipation. After every leapfrog step
  the vertical velocity and the cylindrical-radial velocity of each particle are divided by
  1 + 4 cooling dt / t_ff, so cooling 0.25 is one e-folding of the non-circular motion per
  free-fall time. The azimuthal velocity is untouched, so every particle keeps its Lz; the
  total momentum is then reset to zero (a uniform shift, which can only lower the energy and
  leaves Lz unchanged because the centre of mass sits at the origin). The kinetic energy taken
  out is booked as ``radiated``. With cooling 0 the run is purely conservative.

What is real and what is toy. Real: Newton's gravity between the particles (softened; the
force is the exact gradient of the softened potential), a time-reversible second-order
integrator, the virial balance of the start, and the conservation laws of the gate. The
outcome is the textbook one for the right reason: cooling removes random motion but not
angular momentum, so the cloud can shrink along its axis but not across it, and a cold disc is
unstable against its own gravity and breaks up. Toy: 256 particles stand in for a gas, and
there is no gas: no pressure, no shocks, no radiation, no star. The "cooling" is a drag
towards circular motion about the z axis, which it is told in advance; it is not gas physics.
Nothing holds a clump up but the softening, so clumps shrink to the softening length and the
energy of a cooled run (3 to 14 times the initial binding energy at the end) measures ``eps``,
not nature. Strongly cooled discs end thinner (flattening 0.01 to 0.1) than any real disc,
and thinner than the softening. Because nothing resists the first collapse, the cloud
fragments while it flattens: a smooth disc that only later grows arms, as in a real gas cloud,
hardly appears (stage ``disc`` in 18 of the 1640 states of seeds 1 to 40). ``spiral`` is a
two-fold-pattern detector that cannot tell arms from a bar or a pair of clumps. The units are
not tied to any real cloud. The motion is chaotic: the numbers of one seed are exact for the
replay but mean nothing beyond the statistics below.

What happens: with cooling the cloud loses its support, the low-angular-momentum half of the
mass sinks into a central condensation (the largest clump, the later star) and the rest
settles into a thin disc that breaks into clumps; fast rotators also form pairs and two-armed
patterns. Most cooled runs go cloud, (collapsing,) fragmenting, clumped. An uncooled cloud
that sloshes inwards may read ``collapsing`` for a few states and returns to ``cloud``.

Plausibility targets, with the values measured on this code (``tests/test_cosmos_gravity.py``
holds each of them on a part of the seeds; fixed parameters on seeds 101 to 112, the 40 seeds
are 1 to 40 with ``random_params``):

    every run (148)   the gate holds                                  148 of 148
                      energy + radiated within 2 %                    cooled 0.7 % at worst (0.35 %
                                                                      before the rounding to 3
                                                                      digits), uncooled 0.002 %
                      Lz within 1 %                                   1e-15 before rounding
    spin 0.35,        a disc: smallest flattening below 0.2           0.007 to 0.053
    cooling 0.4, 0.6  a two-fold pattern shows                        24 of 24 runs
    (24 runs)         ends in at least 3 clumps                       3 to 8, the largest holds
                                                                      0.52 to 0.66 of the mass
    spin 0.05, 0.1,   one clump with over 0.7 of the mass             0.73 to 0.85
    cooling 0.6       collapse (radius halved) within 1.5 t_ff        1.1 to 1.3
    (24 runs)         ends tighter than every fast rotator            radius 0.02 to 0.08
                                                                      against 0.16 to 0.56
                      a two-fold pattern is rare                      5 runs, 1 or 2 states each
    cooling 0,        no disc: flattening never below 0.5             0.53 to 0.97 (above 0.7
    spin 0.05 to 0.4                                                  up to spin 0.2)
    (60 runs)         no collapse, no spiral, no clump over 0.1       none, none, 0.051 at most
    seeds 1 to 40     the 33 cooled runs flatten below 0.5            0.014 to 0.43
                      and end fragmenting or clumped                  33 (30 clumped)
                      the 7 uncooled runs stay a cloud above 0.6      0.62 to 0.93
                      a rerun gives the same states                   40 of 40

The step is dt = t_ff / 250 = 0.0044 (25 steps between states, 1000 per run): the densest
clumps that form (mass ~0.7 inside the softening length, orbital period ~0.14) get ~30 steps
per orbit. A run takes 1 to 3 s on one CPU core, 1.4 to 1.7 s on average over seeds 1 to 40
on a machine busy with other work (O(N^2) forces by numpy broadcasting; the fixed-order sums
cost about a fifth of that).

Reproducibility. ``check`` replays a run, and the motion is chaotic, so a last-digit
difference between the machine that wrote the lines and the one that judges would grow into a
different run. The run therefore uses only operations that IEEE 754 defines to the last bit
(+ - * / sqrt on float64, element by element) in an order this file fixes: every sum is
``_total`` (pairs of neighbours, level by level) instead of ``ndarray.sum``, whose order is
numpy's own business; the start is drawn from Python's ``random.Random(seed).random()``, the
one stream documented to stay the same across versions, with normal numbers made by adding
twelve uniform ones (no logarithm, no numpy generator); there is no BLAS, no exp, no
trigonometric function, no power, no k-d tree and no Python ``sum`` of floats (3.12 changed
its rounding). ``tests/test_cosmos_gravity.py`` replays a small run in pure Python floats and
gets the same bits, and also with numpy's AVX2 code paths switched off. It cannot run another
numpy version: ``test_known_rollout`` is the canary for that, and if it fails on the training
machine the lines must be rebuilt there.

Metrics per state (``KEYS``), rounded to 3 significant digits; these lines are from
``rollout(5)`` (spin 0.27, cooling 0.5):

    gravity seed 5 step 1 2. time 1 point 2. radius 0 point 3 6 5. flattening 0 point 5 7 6.
        clumps 8. largest_clump 0 point 2 6 6. spiral no. energy minus 0 point 7 2 8.
        angular_momentum 0 point 3 3 3. stage fragmenting.               (one line)
    q gravity seed 5 step 1 2 clumps. a 8.
    q gravity seed 5 step 1 2 next spiral. a yes.            (the state 0.1 t_ff later)
    q gravity seed 5 final clumps. a 5.                      (how the run ends)
    q gravity seed 5 final collapse_time. a 1 point 3.       (``a never.`` without collapse)

``clumps`` are friends-of-friends groups of at least 5 particles with linking length
max(0.1 radius, eps) (the floor because gravity makes nothing smaller than the softening).
``spiral`` is yes when, for the mass within 2 half-mass radii, the m=2 Fourier mode of the
azimuths exceeds 0.2 (Poisson noise is ~0.07 for 200 particles) and also exceeds the m=1 mode,
while the flattening is below 0.5: a two-fold pattern in a flattened system. The m=1 condition
keeps a single off-centre clump from counting in most cases. ``stage`` is the first that
applies: clumped (the largest clump holds half the mass), fragmenting (2+ clumps, the largest
at least 10 %), disc (flattening below 0.5), collapsing (half-mass radius below 0.75 of its
initial value), cloud.

Gate (``conserved``): mass is constant exactly; the total momentum stays zero; Lz is conserved
within 1 %; energy + radiated is conserved within 2 % of the larger of the initial and the
current |energy|; radiated never falls and is zero without cooling; with cooling the energy
never rises by more than 0.5 % of its value between states. ``check`` replays the seed
(``rollout``: the parameters come from ``random_params(random.Random(seed))``) and compares
with 5 % tolerance (clumps: a whole number within 1; words exactly). ``lines`` refuses a run
with other parameters than its seed names, because ``check`` could not replay it.

Rules 7
-------

    r = run(seed, rules=7, **random_params(random.Random(seed), rules=7))    # or rollout(seed, rules=7)
    conserved(r); lines(r); check("gravity seed 5 rules 7 final star_time", "1 point 3")
    simulation(rules=7).run(seed, spin=0.2, cooling=0.3)                     # the level bound to rules 7

Without ``rules`` (or with ``rules=6``) every function gives what it gave in round 6, bit for
bit; ``tests/test_round6_frozen.py`` and ``tests/test_cosmos_gravity.py`` hold that. A rules-7
run has ``"rules": 7`` in its params and its lines say ``rules 7`` after the seed, so the two
runs a seed names cannot be mixed up. It writes 61 states (6 t_ff):

    gravity seed 5 rules 7 params. spin 0 point 1 9. cooling 0 point 5. cooling_time 0 point 2.
        bar_unstable yes. particles 2 5 6. softening 0 point 0 5. t_ff 1 point 1 1.
        t_ff_myr 0 point 4 2 8. angular_momentum_z 0 point 2 7 5.
        centrifugal_radius 0 point 0 7 5 6. radius_start 0 point 6 4 2.          (one line)
    gravity seed 5 rules 7 step 1 2. time 1 point 2. radius 0 point 2 5 5. flattening 0 point 6 6 2.
        clumps 3. largest_clump 0 point 0 7 4 2. disc_mass 0 point 6 8 8. disc_radius 0 point 2 7 3.
        pattern none. energy minus 0 point 9 0 9. radiated 0 point 5 5 5. support 0 point 6 3 1.
        rotation 0 point 1 1 6. random 0 point 1 9 9. jeans_number 6 point 0 8. toomre_q none.
        stage fragmenting.                                                       (one line)
    q gravity seed 5 rules 7 params spin. a 0 point 1 9.
    q gravity seed 5 rules 7 step 1 2 clumps. a 3.
    q gravity seed 5 rules 7 step 1 2 next stage. a star.
    q gravity seed 5 rules 7 final star_time. a 1 point 3.
    q gravity predict toomre_q sigma 0 point 7 9 3 kappa 1 7 point 0 8 surface_density 9 point 6 1. a 0 point 4 1 9 5.

The last line is a lesson (``LESSONS``, topic ``predict_gravity``): it carries its inputs, and
its answer is computed by the function the simulation itself calls. ``KEYS7``, ``SUMMARY_KEYS7``
and ``PARAM_KEYS7`` say what every word means.

Each correction, the rule of the real world it follows, and what stays toy:

1. The causes are written down. Round 6 printed outcomes but never ``spin`` or ``cooling``.
   The ``params`` record gives them once per seed, with the softening, the particle number and
   the time unit. Every state also prints what holds the cloud up: ``support`` = 2 K / |W|
   (the virial theorem: 1 is balance), ``rotation`` = T_rot / |W| and ``random`` = (K - T_rot) /
   |W| (the beta and alpha that models of rotating collapse are started with, after Boss and
   Bodenheimer 1979), and ``radiated``, so that energy + radiated = constant can be read off.
   ``angular_momentum`` was one number repeated 41 times; it now stands once in the params
   (``angular_momentum_z``) and once as a ``final`` question, and ``rotation`` takes its place.
   Toy: T_rot is that of a rigid body with the same Lz, and it is measured about the z axis.

2. The cooling. Round 6 divided the velocity towards and along a z axis it was told in
   advance, then reset the momentum. Rules 7 use an inelastic drag between neighbours: every
   pair of particles closer than ``H7`` = 0.15 R that approach each other loses the fraction
   ``drag_factor`` = 10 * cooling * dt / t_ff of its closing speed, as equal and opposite
   pushes along the line between them (never more than a full stop per particle). This is the
   inelastic-collision picture of particles that meet (after Brahic 1977, for the bodies of
   planetary rings), which the sticky-particle models of gas clouds in galaxies took up, and
   has the form of the
   artificial viscosity of particle hydrodynamics, which acts on approaching pairs only
   (after Monaghan and Gingold 1983). Taken from them: encounters that take kinetic energy
   and conserve momentum and angular momentum pair by pair, with no axis and no reset
   (measured on seeds 1 to 100: every component of the angular momentum to 1e-15, the
   momentum to 5e-16, energy + radiated within 0.4 %, 0.7 % after the rounding to 3 digits).
   The more neighbours a particle has, the faster it cools, as radiating gas does with its
   density. Invented: the reach 0.15 R, the rate 10, the cap. ``cooling_time`` = 1 / (10
   cooling) is the e-folding time of the closing speed of one pair of neighbours (the time
   in which it falls to 1 / e = 0.37 of what it was; the pair is never quite stopped): 1 t_ff
   at cooling 0.1, 0.17 t_ff at 0.6. A particle with c approaching neighbours is braked c
   times as fast, up to the cap: from c = 1 / drag_factor on (42 neighbours at cooling 0.6,
   250 at 0.1) each pair loses 1 / c, and an inflow towards a common centre then loses half
   its speed in a single step whatever the cooling. So inside a forming clump the drag is as
   strong as the time step allows, which is a property of the toy and its step, not of gas.
   The real rule this range is laid across: a cloud can collapse freely and break
   up when it cools faster than it falls (after Rees and Ostriker 1977); real molecular gas
   cools far faster than that. The number is a toy rate, not a measured cooling time, and
   there is no temperature. The review proposed reach 0.25 and rate 60; measured on seeds 1
   to 40 that drag worked as a viscosity that held the cloud up: the first star came at 2.3
   to 3.5 t_ff, later with more cooling (r = +0.5), and neither the star's mass nor the
   final flattening depended on cooling (|r| below 0.1). With 0.15 and 10 the star comes at
   1.1 to 2.1 t_ff (in one run of 84 at 3.6) and the trends below have the right sign. Still toy: the drag
   also brakes a converging flow, so it is a viscosity as well as a cooling; and a star is a
   softened point, so the energy at the end (4 to 17 times the initial binding energy)
   measures ``eps``, not nature.

3. Clumps. Round 6 counted friends-of-friends groups of 5 particles with a linking length
   that changed with the radius; the count flickered, and unbound groups were counted even in
   uncooled clouds. Rules 7: linking length ``eps``, at least 14 particles, and bound (kinetic
   energy about the group's own velocity plus the softened potential energy of its members
   below zero). With the cloud taken as 1.5 solar masses, 14 particles are 0.082 solar masses,
   just above the smallest mass that can burn hydrogen (about 0.075 to 0.08 solar masses), so
   every counted clump would be a star by its mass, the smallest ones only just, and none a
   planet. The gate compares the count
   exactly. Measured: no clump in any state of an uncooled run; the count changes 6.1 times
   in the 60 steps of a cooled run, by 2 or more 0.3 times. Toy: 14 particles do not resolve
   a fragment (that takes of the order of 100, after Bate and Burkert 1997); two clumps that
   touch are one group for a state, so the count can drop and rise again (in 5 of 84 cooled
   runs the largest clump falls back below a quarter of the mass at some point); the count
   is a property of one chaotic run, not a law.

4. ``pattern`` instead of ``spiral``. The round-6 detector answered yes for any two-fold
   mass distribution, mostly for a pair of clumps; trailing arms were not there to be seen,
   and 256 particles cannot resolve them (simulations of discs that show arms use tens of
   thousands of particles and more). Rules 7 name what
   is there: ``pair`` (a second bound clump with a tenth of the mass), else for the flat mass
   outside the clumps ``lopsided`` (one-fold mode) or ``bar`` (two-fold mode) when the Fourier
   amplitude A passes Rayleigh's test A^2 n > 6.9 (for a round distribution of n points the
   chance of that is e^-6.9, 1 in 1000, for each of the two modes; this replaces the fixed
   0.2), else ``none``. The word arms is
   never used. Measured in 5124 cooled states: none 4912, pair 124, lopsided 45, bar 43.

5. Stability numbers and ``stage``. The stage was a ladder of hand-picked thresholds on the
   noisy count. Now it is, on bound clumps: ``multiple`` (two or more, the largest a star),
   ``star`` (one clump with a quarter of the mass), ``fragmenting`` (any bound clump),
   ``disc`` (flattening below 0.5), ``contracting`` (radius below 0.75 of the first), ``cloud``;
   a pure function (:func:`stage7`) of numbers the lines print. Two published criteria are
   printed with it. ``jeans_number``: how many Jeans masses, pi^2.5 / 6 * sigma^3 / sqrt(G^3
   rho) (Jeans 1902), the mass outside the clumps holds. ``toomre_q``: sigma kappa / (3.36 G
   Sigma) (Toomre 1964), stable above 1, printed only where there is something flat
   (flattening below 0.5), else ``none``; kappa is taken as sqrt 2 times the rotation rate,
   the value for a flat rotation curve, which is an assumption (around a point mass it is 1
   times the rotation rate, so where the star dominates Q is printed up to 1.4 times too
   high), and 3.36 is the constant for a disc of stars (a gas disc has pi). ``bar_unstable``
   in the params is yes above T_rot / |W| = 0.14 (after Ostriker and Peebles 1973, who found
   it for rotating systems of stars in equilibrium; asking it of the cloud's start, before
   any collapse, is the toy's use of it). Measured: of 45 cooled runs at or below 0.14, 2
   end with more than one clump, of 39 above it 18. Honest limits: by this definition (the
   mass of a sphere one Jeans length across) a uniform sphere in virial balance already
   holds 1.9 Jeans masses, so a number near 2 means balance, not collapse; and the Jeans
   number counts random motion only, so a fast rotator starts with a high one (2.1 to 8.6
   at the start, rising with spin) and is stable all the same; it does not tell cooled from
   uncooled runs. Q was below 1 in 21 of the 939 states that have one. ``disc``, a flat
   cloud without any clump, did not occur in seeds 1 to 100: the centre falls together first,
   as in a real cloud, and the disc is what surrounds the star afterwards (``flattening``,
   ``disc_mass``, ``disc_radius``).

6. The centre, the disc and ``star_time``. Once clumps exist, the centre of mass is empty
   space and the half-mass ``radius`` is the distance of the largest clump from it.
   ``radius`` is kept as it was (the stage and the continuity need it). The centre is now the
   centre of mass of the largest bound clump once it holds a quarter of the mass (a star).
   ``disc_mass`` is the mass outside every bound clump within one cloud radius of the centre,
   ``disc_radius`` its median distance from the axis, and once a star exists ``flattening``
   is that of this mass about the star. ``collapse_time`` (radius halved, or never) is
   replaced by ``star_time``: the time of the first state with a star, or ``never``, which
   means not within the 6 t_ff of the run. Times are compared exactly (they are multiples
   of 0.1). Measured: a number for all 84 cooled runs of seeds 1 to 100 (1.1 to 2.1, one run
   3.6), never for all 16 uncooled ones.

7. Units. t_ff = pi / sqrt(8) is right for a uniform sphere of the cloud's mass and radius
   (:func:`free_fall_time` of the density 3 / (4 pi)), but the cloud is a Gaussian ball whose
   denser inner half falls in about 0.75 of it. For one clock in the chain the units get a
   label: 1.5 solar masses (the mass the round-6 ``world`` gives the cloud; ``world7`` draws
   0.1 to 8 and keeps this clock) within 0.1 parsec, the
   size of a dense core (its mean density, about 5000 hydrogen molecules per cm^3 with 2.8
   hydrogen masses per molecule for the helium, is at the
   thin end of what is observed, 10^4 to 10^5), has t_ff = 0.428 million years
   (``T_FF_MYR``, from the same formula in cgs units).
   Then ``eps`` is about 1000 au and one particle about 6 Jupiter masses, and the disc radii
   of the toy, 0.2 to 0.6 R, are 4000 to 13000 au, a hundred times the tens to hundreds of au
   of real discs around young stars. The spin is not the main reason. Observed cores have
   rotational over gravitational energy of the order of 0.02 (after Goodman and others 1993),
   which is the low end of the toy's range, and there the centrifugal radius is about 0.01 R,
   some 200 au, the size of a real disc; yet ``disc_radius`` is still 0.25 to 0.3 R (seeds 1
   to 100, spin 0.03 and 0.04). ``disc_radius`` is the median distance of all the loose mass
   within one cloud radius, so it measures the envelope that has not fallen in as much as a
   disc held up by rotation, and nothing smaller than the softening (1000 au) can be resolved.
   The higher spins, up to 0.3 (several times the largest that Goodman and others found),
   widen it to 0.6 R, and nothing here (no magnetic braking) carries angular momentum away.
   The scaling is a label for the clock, not a claim about a real cloud.

8. The start. ``random_params(rng, rules=7)`` draws spin from 0.02 to 0.3 instead of 0.05
   to 0.4 (same draws, same order): above 1/3 the round-6 start was out of virial balance
   (2 K / |W| up to 1.13), and ``evolve`` now refuses such a spin. Measured ``support`` at the
   start: 0.89 to 1.10. :func:`centrifugal_radius` of the cloud's mean specific angular
   momentum is printed in the params (0.01 to 0.13 R; the softening is 0.05 R).

9. Cost. The force uses six preallocated buffers with the same operations in the same order
   as round 6 (a run without cooling gives the same bits under both rules). The drag takes
   the cap of a pair from its two particles (the same numbers, since rounding is monotone),
   and ``measure7`` computes the squared distances once for the clump finder and both
   potential energies. A run takes 0.7 to 1.9 s for its 1500 steps (mean 1.1 s, seeds 1 to
   100, on a machine busy with other work); an uncooled run 0.7 to 0.8 s, and the slowest are
   slow rotators with strong cooling, whose star of some 150 particles inside the softening
   length gives over 10000 pairs to damp at every step. These are times on a performance core:
   on a CPU with efficiency cores, a process the system has moved to one of them takes about
   twice as long (seed 1 on an i9-13900K: 1.44 to 1.59 s held to the performance cores, 3.3 to
   3.75 s held to the efficiency cores).

10. Softening and resolution: unchanged (N = 256, eps = 0.05), but said. eps is a quarter of
   the mean distance between particles at the start, so encounters between single particles
   matter: by the rule of thumb the relaxation time is 0.1 N / ln N = 4.6 crossing times,
   about the length of a run, so an uncooled cloud is not a collisionless one. It also
   spreads: the mass within one cloud radius falls to 0.82 to 0.89 within the first t_ff and
   stands at 0.73 to 0.9 after 6 t_ff. That is not evaporation (0.4 to 2.7 % of the
   particles start unbound): the start is in virial balance as a whole but not orbit by
   orbit, since every particle gets the same spread of speeds wherever it is, and the outer
   ones swing out beyond the radius the cloud was cut at. A clump is a softened point; its
   size is eps. ``support`` falls below 1 once a clump is smaller than the softening,
   because there the potential is deeper than the forces it exerts; that is not a collapse.

11. Lessons. ``LESSONS`` (``lesson_gate()``) holds 18 rules that carry their inputs:
   free_fall_time, sphere_density, softened_force, softened_potential, spin_rate,
   plane_speed, virial_ratio, rotation_share, random_share, jeans_mass, jeans_number,
   toomre_q, bar_unstable, centrifugal_radius, cooling_time, drag_factor, stage, pattern.
   Each is the function of that name (``stage7`` and ``pattern7`` for the last two) which
   ``initial_state``, ``evolve``, ``measure7`` or ``run`` calls (``softened_force`` and
   ``softened_potential`` call the simulation's force and potential routines for two
   bodies), so a lesson is true of the simulation.

Where the code departs from the review's proposal: the constants of the drag (item 2);
``params <key>`` instead of ``param <key>`` in the questions (the word of the record);
``disc_mass`` leaves out every bound clump, not only the largest, so that a companion is not
counted as disc; ``jeans_number`` is taken of the mass outside the clumps and ``toomre_q``
of the ring around the disc radius, because the half-mass radius is no size once a star
exists; ``toomre_q`` is the word none where nothing is flat (a Q of 0.3 for a round cloud
that never collapses would mislead); there is no ``disc_stable`` lesson, because the
simulation calls no such function; T_FF stays the round-6 constant, which
:func:`free_fall_time` reproduces; the round-6 force routine is left as it is.

What happens under rules 7 (seeds 1 to 100; 84 cooled, 16 uncooled): an uncooled cloud stays
a cloud (no bound clump in 976 states, flattening never below 0.61). A cooled cloud loses
its support, its centre falls together into a star within 1.1 to 2.1 t_ff (one run of 84:
3.6), and the rest settles around it, the flatter the stronger the cooling. Trends over the cooled runs
(correlation coefficients; trends, not numbers to predict): more spin gives a lighter star
(-0.91; 0.55 to 0.74 of the mass at spin up to 0.1, 0.31 to 0.48 from 0.2), a wider disc
(+0.68), a later star (+0.50) and more companions (+0.59; 28 of 29 slow rotators end as one
star, 13 of 19 fast ones as several); more cooling gives a flatter system (smallest
flattening -0.59; median 0.38 for cooling from 0.4, 0.56 up to 0.2) and a slightly earlier
star (-0.21). The numbers of one seed are exact for the replay and mean nothing beyond that.

The rules-7 gate is the same as above. ``check`` compares words, whole numbers, times and the
params exactly and every other number within 5 %. For ``world7`` the summary gives ``star``
(mass fraction of the largest bound clump), ``companions`` (the other bound clumps: born by
the fragmentation of the cloud, so companion stars or brown dwarfs; the planets of the chain
come from level 3. Their mass is the cloud's times their share: at least 14 particles, 0.082
solar masses for a cloud of 1.5, less for a lighter cloud of ``world7``), ``disc_mass``, ``disc_radius``, ``star_time`` and ``patterns``; the clock is
time * T_FF_MYR.
"""

from __future__ import annotations

import copy
import math
import random
import re
from collections.abc import Iterator
from functools import lru_cache

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from haishool.cosmos import Rollout, close, query_lines, state_line, summary_lines
from haishool.evo import Input, LessonGate, Rule, sig
from haishool.truth import Line, Verdict, num, parse_num

SIM = "gravity"
#: free-fall time of a uniform sphere with G = M = R = 1
T_FF = math.pi / math.sqrt(8.0)
#: states written per run (one every OUT_DT free-fall times) and leapfrog steps between them
N_OUT, OUT_DT, SUB_STEPS = 41, 0.1, 25
#: particles, total mass and softening length of the run a seed names
N, MASS, EPS = 256, 1.0, 0.05
#: the leapfrog step
DT = OUT_DT * T_FF / SUB_STEPS
#: damping rate of the non-circular motion per unit cooling, in 1 / t_ff
COOL_RATE = 4.0
#: a clump has at least this many particles; linking length as a fraction of the half-mass radius
MIN_MEMBERS, LINK_FRACTION = 5, 0.1
#: m=2 amplitude that counts as a spiral, and the flattening below which a system is a disc
SPIRAL_M2, DISC_FLATTENING = 0.2, 0.5
#: stage thresholds: half the mass in one clump; the largest clump of a fragmenting system; contraction
CLUMPED_FRACTION, FRAGMENT_FRACTION, COLLAPSING_RADIUS = 0.5, 0.1, 0.75
#: gate tolerances: energy + radiated and Lz drift, energy rise between states with cooling, total momentum
ENERGY_TOL, LZ_TOL, RISE_TOL, MOMENTUM_TOL = 0.02, 0.01, 0.005, 1e-6

# rules 7 (see the docstring section of that name); everything above stays as round 6 wrote it
#: states of a rules-7 run: 6 free-fall times, so that the disc around the star has time to settle
N_OUT7 = 61
#: the pairwise drag: reach (in cloud radii) and rate per unit cooling per free-fall time; toy numbers
H7, COOL_RATE7 = 0.15, 10.0
#: a clump is a bound friends-of-friends group (linking length EPS) of at least this many particles
MIN_MEMBERS7 = 14
#: mass fraction of the largest bound clump from which it is called a star; of the second for a pair
STAR_FRACTION, PAIR_FRACTION = 0.25, 0.1
#: T_rot / |W| above which a rotating body turns into a bar (after Ostriker and Peebles 1973)
BAR_LIMIT = 0.14
#: Rayleigh's test for a Fourier amplitude A of n points: A * A * n above this has a chance of e^-6.9 = 0.001
RAYLEIGH, PATTERN_MIN = 6.9, 40
#: the largest spin that can start in virial balance, and the initial cloud radius
SPIN_MAX7, CLOUD_RADIUS = 1.0 / 3.0, 1.0
#: fewest particles in the annulus for a Toomre Q
ANNULUS_MIN = 8
#: no training line may be longer (a state line has 16 metrics; ``lines7`` refuses to write a longer one)
MAX_TOKENS7 = 128
#: the label that ties the units to a real dense core: the mass ``world`` gives the cloud, a radius,
#: and cgs constants (G, solar mass in g, parsec in cm, year in s)
CLOUD_MASS_MSUN, CLOUD_RADIUS_PC = 1.5, 0.1
G_CGS, MSUN_G, PC_CM, YEAR_S = 6.674e-8, 1.989e33, 3.0857e18, 3.156e7

KEYS: dict[str, str] = {
    "time": "time since the start in free-fall times (t_ff = pi / sqrt(8) with G = M = R = 1)",
    "radius": "half-mass radius about the centre of mass, in units of the initial cloud radius",
    "flattening": "rms vertical extent over rms in-plane extent of the mass within 2 half-mass radii "
                  "(1 sphere, small disc)",
    "clumps": "friends-of-friends groups of at least 5 particles, linking length max(0.1 radius, eps)",
    "largest_clump": "mass fraction of the biggest clump (0 when there is none)",
    "spiral": "yes when the m=2 mode within 2 half-mass radii exceeds 0.2 and the m=1 mode "
              "while flattening is below 0.5",
    "energy": "total energy, kinetic plus softened potential (G = M = R = 1)",
    "angular_momentum": "z component of the total angular momentum about the centre of mass",
    "stage": "one of cloud, collapsing, disc, fragmenting, clumped",
    "mass": "total mass; a gate diagnostic (constant), not written to the training lines",
    "momentum": "magnitude of the total momentum; a gate diagnostic (zero up to rounding), "
                "not written to the training lines",
    "radiated": "kinetic energy the cooling has taken out so far; a gate diagnostic (energy + radiated is "
                "constant), not written to the training lines",
}
#: the metrics that go into the training lines, in this order
LINE_KEYS = ["time", "radius", "flattening", "clumps", "largest_clump", "spiral", "energy", "angular_momentum", "stage"]
#: the summary of a run, asked as ``gravity seed 7 final <key>``
SUMMARY_KEYS: dict[str, str] = {
    "clumps": "clumps in the last state",
    "largest_clump": "largest_clump in the last state",
    "flattening": "flattening in the last state",
    "min_flattening": "the smallest flattening of the run",
    "spiral_ever": "yes when any state had spiral yes",
    "collapse_time": "first time (in t_ff) the half-mass radius is below half its initial value, or never "
                     "(no cooling, or a fast rotator that ends as a wide pair of clumps)",
}
STAGES = ("cloud", "collapsing", "disc", "fragmenting", "clumped")

_PROMPT = re.compile(r"^gravity seed ((?:[0-9] )*[0-9]) (?:step ((?:[0-9] )*[0-9]) (next )?|final )([a-z_]+)$")

#: rules 7: the metrics of a state
KEYS7: dict[str, str] = {
    "time": "time since the start in free-fall times of a uniform sphere with the cloud's mass and radius "
            "(t_ff = pi / sqrt(8) with G = M = R = 1); the denser inner half falls in about 0.75 of it",
    "radius": "half-mass radius about the centre of mass, in units of the initial cloud radius; once clumps "
              "exist it is the distance of the largest from the centre of mass rather than a size",
    "flattening": "rms height over rms in-plane extent / sqrt 2 (1 sphere, small disc): of the mass within 2 "
                  "half-mass radii of the centre of mass, and once a star exists of the disc about the star",
    "clumps": "bound clumps: friends-of-friends groups (linking length eps) of at least 14 particles whose "
              "internal kinetic plus potential energy is negative",
    "largest_clump": "mass fraction of the largest bound clump (0 when there is none); a star from 0.25",
    "disc_mass": "mass fraction outside every bound clump and within one cloud radius of the centre (the "
                 "star's centre of mass once a star exists, else the cloud's)",
    "disc_radius": "median distance of that mass from the axis through the centre (0 when there is none)",
    "pattern": "none, pair (a second bound clump with a tenth of the mass), lopsided or bar (a significant "
               "one-fold or two-fold mode of the flat mass outside the clumps); never arms",
    "energy": "total energy, kinetic plus softened potential (G = M = R = 1)",
    "radiated": "kinetic energy the pairwise drag has taken out so far; energy + radiated is constant",
    "support": "virial ratio 2 K / |W|: 1 in balance, below 1 gravity wins; once a clump is smaller than the "
               "softening it stays below 1, because there the potential is deeper than the forces it exerts",
    "rotation": "share of the binding held up by rotation, T_rot / |W| with T_rot = Lz * Lz / (2 I)",
    "random": "share of the binding held up by random motion, (K - T_rot) / |W|",
    "jeans_number": "Jeans masses in the mass outside the clumps (the whole cloud before the first clump): that "
                    "mass over pi^2.5 / 6 * sigma^3 / sqrt(density), with its random speed sigma within 2 of its "
                    "half-mass radii about the centre and the mean density inside that radius; rotation and the "
                    "pull of a star are not counted, so it overstates how easily a rotating body breaks up",
    "toomre_q": "Toomre's Q of the mass outside the clumps between 0.5 and 2 disc radii, taking the epicyclic "
                "rate as sqrt 2 times the rotation rate (a flat rotation curve); stable above 1; the word none "
                "when flattening is 0.5 or more (nothing flat to ask) or fewer than 8 particles are there",
    "stage": "one of cloud, contracting, disc, fragmenting, star, multiple",
    "angular_momentum": "z component of the total angular momentum about the centre of mass; constant, so it "
                        "is written once (params angular_momentum_z) and checked by the gate in every state",
    "mass": "total mass; a gate diagnostic (constant), not written to the training lines",
    "momentum": "magnitude of the total momentum; a gate diagnostic (zero up to rounding), not written to the "
                "training lines",
}
#: rules 7: the metrics of the training lines, in this order
LINE_KEYS7 = ["time", "radius", "flattening", "clumps", "largest_clump", "disc_mass", "disc_radius", "pattern",
              "energy", "radiated", "support", "rotation", "random", "jeans_number", "toomre_q", "stage"]
#: rules 7: the summary of a run, asked as ``gravity seed 7 rules 7 final <key>``
SUMMARY_KEYS7: dict[str, str] = {
    "stage": "stage of the last state",
    "clumps": "bound clumps in the last state",
    "star": "mass fraction of the largest bound clump in the last state",
    "companions": "bound clumps besides the largest in the last state (born by the fragmentation "
                  "of the cloud: companion stars or brown dwarfs, not planets)",
    "disc_mass": "disc_mass in the last state",
    "disc_radius": "disc_radius in the last state",
    "flattening": "flattening in the last state",
    "min_flattening": "the smallest flattening of the run",
    "star_time": "time (in t_ff) of the first state whose largest bound clump holds a quarter of the mass, "
                 "or never (not within the run of 6 t_ff)",
    "patterns": "the words among pair, lopsided, bar that some state showed, in this order, or none",
    "angular_momentum_z": "angular_momentum in the last state (the same as at the start)",
}
#: rules 7: the givens of a run, written once as the ``params`` record
PARAM_KEYS7: dict[str, str] = {
    "spin": "rotational energy over |W| at the start (0.02 to 0.3)",
    "cooling": "strength of the pairwise drag (0, or 0.1 to 0.6)",
    "cooling_time": "1 / (10 cooling): free-fall times in which the closing speed of an approaching pair of "
                    "neighbours falls to 1 / e of it; a toy rate, not a measured cooling time; never without cooling",
    "bar_unstable": "yes when spin is above 0.14 (the limit for a rotating system in equilibrium, asked of the start)",
    "particles": "number of particles",
    "softening": "softening length in cloud radii",
    "t_ff": "the free-fall time in code units",
    "t_ff_myr": "the same in million years for 1.5 solar masses within 0.1 parsec (a label, not a fit)",
    "angular_momentum_z": "angular_momentum of the first state",
    "centrifugal_radius": "radius of the circular orbit that carries the cloud's mean specific angular "
                          "momentum about the whole mass",
    "radius_start": "radius of the first state",
}
#: the params that differ from seed to seed and are asked as ``gravity seed 7 rules 7 params <key>``
PARAM_QUESTIONS7 = ("spin", "cooling", "cooling_time", "bar_unstable", "centrifugal_radius")
STAGES7 = ("cloud", "contracting", "disc", "fragmenting", "star", "multiple")
PATTERNS7 = ("none", "pair", "lopsided", "bar")
#: rules-7 metrics the gate compares exactly (whole numbers and multiples of 0.1)
EXACT7 = frozenset({"clumps", "companions", "time", "star_time"})

_PROMPT7 = re.compile(r"^gravity seed ((?:[0-9] )*[0-9]) rules 7 "
                      r"(?:step ((?:[0-9] )*[0-9]) (next )?|(final )|params )([a-z_]+)$")


def _sig3(x: float) -> float:
    """Round to 3 significant digits, so ``num`` prints at most 3 of them."""
    return float(f"{x:.3g}")


def _total(a: np.ndarray) -> np.ndarray:
    """The sum over the last axis in a fixed order: neighbours are added in pairs, level by
    level, with element-wise additions only. ``a.sum()`` would leave the order (and so the last
    digit) to numpy's reduction loops, which may change between versions and CPUs."""
    if a.shape[-1] == 0:
        return np.zeros(a.shape[:-1])
    while a.shape[-1] > 1:
        n = a.shape[-1]
        half = a[..., 0:n - 1:2] + a[..., 1:n:2]
        a = np.concatenate([half, a[..., n - 1:]], -1) if n % 2 else half
    return a[..., 0]


def _square(a: np.ndarray) -> np.ndarray:
    """x^2 + y^2 + z^2 of the rows of an (N, 3) array, added in this order."""
    return a[:, 0] * a[:, 0] + a[:, 1] * a[:, 1] + a[:, 2] * a[:, 2]


def _weighted(m: np.ndarray, a: np.ndarray) -> np.ndarray:
    """sum_i m_i a_i for the rows of an (N, 3) array: the three components."""
    return _total((m[:, None] * a).T)


def _kinetic(v: np.ndarray, m: np.ndarray) -> float:
    return 0.5 * float(_total(m * _square(v)))


def _normal(rnd: random.Random, count: int) -> np.ndarray:
    """``count`` numbers that are normal to a good approximation: twelve uniform numbers added
    up, minus 6 (mean 0, variance 1, nothing beyond 6). Python's ``random()`` is the one stream
    whose reproducibility across versions is documented, and additions are exact to the last
    bit everywhere, which the logarithms of a true normal generator are not."""
    u = np.array([rnd.random() for _ in range(12 * count)]).reshape(count, 12)
    g = u[:, 0]
    for j in range(1, 12):
        g = g + u[:, j]
    return g - 6.0


def _pairs(x: np.ndarray, eps2: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Coordinate differences x_i - x_j and softened squared distances, all (N, N)."""
    dx = x[:, 0][:, None] - x[:, 0][None, :]
    dy = x[:, 1][:, None] - x[:, 1][None, :]
    dz = x[:, 2][:, None] - x[:, 2][None, :]
    return dx, dy, dz, dx * dx + dy * dy + dz * dz + eps2


def _accel(x: np.ndarray, m: np.ndarray, eps2: float) -> np.ndarray:
    """Softened accelerations, O(N^2): a_i = -sum_j m_j (x_i - x_j) / (r_ij^2 + eps^2)^1.5."""
    dx, dy, dz, r2 = _pairs(x, eps2)
    w = m / (r2 * np.sqrt(r2))
    return -np.stack([_total(w * dx), _total(w * dy), _total(w * dz)], 1)


def _potential(x: np.ndarray, m: np.ndarray, eps2: float) -> float:
    r2 = _pairs(x, eps2)[3]
    inv = 1.0 / np.sqrt(r2)
    np.fill_diagonal(inv, 0.0)
    return -0.5 * float(_total(m * _total(inv * m)))


# ---------------------------------------------------------------------------------------------
# the rules as named functions: the simulation calls them, and each is a lesson (LESSONS)
# ---------------------------------------------------------------------------------------------

def free_fall_time(density: float) -> float:
    """Time for a uniform, pressure-free sphere of this density to fall together, with G = 1:
    sqrt(3 pi / (32 density)). Pass G times a density in cgs units to get seconds."""
    return math.sqrt(3.0 * math.pi / (32.0 * density))


def sphere_density(mass: float, radius: float) -> float:
    """Mean density of a sphere: mass / (4/3 pi radius^3)."""
    return mass / (4.0 / 3.0 * math.pi * radius * radius * radius)


def softened_force(mass_1: float, mass_2: float, distance: float, softening: float) -> float:
    """The pull between two bodies, m1 m2 d / (d^2 + s^2)^1.5, by the simulation's own force
    routine applied to the two of them (so it is the force every run uses, to the last bit)."""
    x = np.array([[0.0, 0.0, 0.0], [float(distance), 0.0, 0.0]])
    m = np.array([float(mass_1), float(mass_2)])
    return float(mass_1) * float(_accel(x, m, softening * softening)[0, 0])


def softened_potential(mass_1: float, mass_2: float, distance: float, softening: float) -> float:
    """The potential energy of two bodies, -m1 m2 / sqrt(d^2 + s^2), by the simulation's own
    potential routine; :func:`softened_force` is its gradient."""
    x = np.array([[0.0, 0.0, 0.0], [float(distance), 0.0, 0.0]])
    m = np.array([float(mass_1), float(mass_2)])
    return _potential(x, m, softening * softening)


def spin_rate(spin: float, potential: float, inertia: float) -> float:
    """Angular velocity of the solid-body start: the rotational energy inertia omega^2 / 2 is
    ``spin`` times the potential energy |W|."""
    return math.sqrt(2.0 * spin * potential / inertia)


def plane_speed(spin: float, potential: float, mass: float) -> float:
    """Random speed along x and along y at the start: the plane holds |W|/3 of kinetic energy
    (tensor virial theorem for a sphere), of which rotation takes ``spin`` |W|."""
    return math.sqrt(max(0.0, 1.0 / 3.0 - spin) * potential / mass)


def virial_ratio(kinetic: float, potential: float) -> float:
    """2 K / |W|: 1 in virial balance, below 1 the system must contract, above 1 it expands."""
    return 2.0 * kinetic / potential


def rotation_share(lz: float, inertia: float, potential: float) -> float:
    """beta = T_rot / |W| with T_rot = Lz^2 / (2 I): the share of the binding that rotation holds up."""
    return 0.5 * lz * lz / (inertia * potential)


def random_share(kinetic: float, lz: float, inertia: float, potential: float) -> float:
    """alpha = (K - T_rot) / |W|: the share held up by random (thermal) motion."""
    return (kinetic - 0.5 * lz * lz / inertia) / potential


def jeans_mass(sigma: float, density: float) -> float:
    """The smallest mass that gravity can pull together against random motions of speed
    ``sigma`` at this density, G = 1: pi^2.5 / 6 * sigma^3 / sqrt(density) (Jeans 1902)."""
    return math.pi * math.pi * math.sqrt(math.pi) / 6.0 * sigma * sigma * sigma / math.sqrt(density)


def jeans_number(mass: float, sigma: float, density: float) -> float:
    """How many Jeans masses a body holds: mass / jeans_mass."""
    return mass / jeans_mass(sigma, density)


def toomre_q(sigma: float, kappa: float, surface_density: float) -> float:
    """Toomre's Q for a disc of stars, G = 1: sigma kappa / (3.36 surface density); a thin
    disc is stable against its own gravity above 1 (Toomre 1964)."""
    return sigma * kappa / (3.36 * surface_density)


def bar_unstable(spin: float) -> str:
    """yes when the rotational share T_rot / |W| is above 0.14: a rotating system in equilibrium
    is then unstable to a bar (after Ostriker and Peebles 1973)."""
    return "yes" if spin > BAR_LIMIT else "no"


def centrifugal_radius(j: float, mass: float) -> float:
    """Radius of the circular orbit with specific angular momentum ``j`` about ``mass``, G = 1."""
    return j * j / mass


def cooling_time(cooling: float) -> float:
    """Rules 7: free-fall times in which the pairwise drag takes the closing speed of an
    approaching pair of neighbours down to 1 / e of it (the e-folding time), 1 / (COOL_RATE7 *
    cooling). A toy rate, not a measured cooling time."""
    return 1.0 / (COOL_RATE7 * cooling)


def drag_factor(cooling: float, dt_tff: float) -> float:
    """Rules 7: the fraction of its closing speed an approaching pair loses in one step of
    ``dt_tff`` free-fall times: COOL_RATE7 * cooling * dt_tff. (In a crowd :func:`_drag7` caps
    it at 1 over the number of approaching pairs of the busier particle of the pair.)"""
    return COOL_RATE7 * cooling * dt_tff


def stage7(radius: float, flattening: float, clumps: int, largest: float, radius0: float) -> str:
    """Rules 7, the first that applies: multiple (two or more bound clumps, the largest a
    star), star (a bound clump with a quarter of the mass), fragmenting (any bound clump), disc
    (flattening below 0.5), contracting (radius below 0.75 of its start), cloud."""
    if clumps >= 2 and largest >= STAR_FRACTION:
        return "multiple"
    if clumps >= 1 and largest >= STAR_FRACTION:
        return "star"
    if clumps >= 1:
        return "fragmenting"
    if flattening < DISC_FLATTENING:
        return "disc"
    if radius < COLLAPSING_RADIUS * radius0:
        return "contracting"
    return "cloud"


def pattern7(second: float, particles: int, flattening: float, m1: float, m2: float) -> str:
    """Rules 7: the shape of the mass in the plane. ``pair`` when the second bound clump holds
    a tenth of the mass. Else the ``particles`` outside every bound clump are asked: fewer than
    40 or not flat (flattening 0.5 or more) is ``none``; an amplitude A of the one-fold (m1) or
    two-fold (m2) Fourier mode counts when A * A * particles > 6.9 (Rayleigh's test: chance
    e^-6.9, 1 in 1000, for a round distribution); ``lopsided`` when m1 counts and is at least m2,
    ``bar`` when m2 counts and exceeds m1, else ``none``."""
    if second >= PAIR_FRACTION:
        return "pair"
    if particles < PATTERN_MIN or flattening >= DISC_FLATTENING:
        return "none"
    if m1 * m1 * particles > RAYLEIGH and m1 >= m2:
        return "lopsided"
    if m2 * m2 * particles > RAYLEIGH and m2 > m1:
        return "bar"
    return "none"


#: rules 7: the free-fall time in million years when the cloud is CLOUD_MASS_MSUN within CLOUD_RADIUS_PC
T_FF_MYR = sig(free_fall_time(G_CGS * sphere_density(CLOUD_MASS_MSUN * MSUN_G, CLOUD_RADIUS_PC * PC_CM))
               / (1e6 * YEAR_S))


def initial_state(seed: int, n: int, mass: float, spin: float, eps: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Positions, velocities and masses of the cloud: a near-Gaussian ball in solid-body
    rotation with random motions that hold up every axis (see the module docstring). Centre of
    mass and total momentum are zero."""
    rnd = random.Random(seed)
    pos = 0.5 * _normal(rnd, 12 * n).reshape(4 * n, 3)
    pos = pos[_square(pos) < 1.0][:n]
    if len(pos) < n:  # a draw lands inside with probability 0.74, so 4 n draws never run short
        raise ValueError("too few particles inside the cloud radius")
    m = np.full(n, mass / n)
    pos -= _weighted(m, pos) / mass
    w = abs(_potential(pos, m, eps * eps))
    plane = float(_total(m * (pos[:, 0] * pos[:, 0] + pos[:, 1] * pos[:, 1])))
    omega = spin_rate(spin, w, plane)
    in_plane = plane_speed(spin, w, mass)
    sigma = np.array([in_plane, in_plane, math.sqrt(w / (3.0 * mass))])
    vel = omega * np.stack([-pos[:, 1], pos[:, 0], np.zeros(n)], 1) + _normal(rnd, 3 * n).reshape(n, 3) * sigma
    vel -= _weighted(m, vel) / mass
    return pos, vel, m


def evolve(seed: int, spin: float = 0.2, cooling: float = 0.3, n: int = N, mass: float = MASS,
           eps: float = EPS, ledger: dict | None = None,
           rules: int = 6) -> Iterator[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """The N_OUT states of a run (N_OUT7 under ``rules=7``) as (positions, velocities, masses).
    The arrays are advanced in place between states, so copy what you keep. ``ledger["radiated"]``,
    when a dict is passed, is the kinetic energy the cooling has taken out up to the state just
    yielded."""
    ledger = {} if ledger is None else ledger
    ledger["radiated"] = 0.0
    _rules(rules)
    if spin < 0 or cooling < 0 or n < 2 * MIN_MEMBERS or mass <= 0 or eps <= 0:
        raise ValueError(f"bad parameters: spin {spin}, cooling {cooling}, n {n}, mass {mass}, eps {eps}")
    if rules >= 7:
        yield from _evolve7(seed, spin, cooling, n, mass, eps, ledger)
        return
    x, v, m = initial_state(seed, n, mass, spin, eps)
    eps2 = eps * eps
    damp = 1.0 / (1.0 + COOL_RATE * cooling * DT / T_FF)
    a = _accel(x, m, eps2)
    for k in range(N_OUT):
        yield x, v, m
        if k == N_OUT - 1:
            break
        for _ in range(SUB_STEPS):
            v += 0.5 * DT * a
            x += DT * v
            a = _accel(x, m, eps2)
            v += 0.5 * DT * a
            if cooling > 0:
                before = _kinetic(v, m)
                r2 = x[:, 0] * x[:, 0] + x[:, 1] * x[:, 1]
                r2[r2 == 0.0] = 1.0
                radial = (1.0 - damp) * (v[:, 0] * x[:, 0] + v[:, 1] * x[:, 1]) / r2
                v[:, 0] -= radial * x[:, 0]
                v[:, 1] -= radial * x[:, 1]
                v[:, 2] *= damp
                v -= _weighted(m, v) / mass
                ledger["radiated"] += before - _kinetic(v, m)


def _rules(rules: int) -> None:
    if rules not in (6, 7):
        raise ValueError(f"gravity knows rules 6 (round 6, the default) and rules 7, not {rules!r}")


def _accel7(x: np.ndarray, m: np.ndarray, eps2: float, buf: list[np.ndarray]) -> np.ndarray:
    """:func:`_accel` with the same operations in the same order, written into six (N, N)
    buffers instead of fresh arrays: the same bits, a third of the time. ``buf[0:4]`` are left
    holding x_i - x_j (three components) and the softened squared distances."""
    dx, dy, dz, r2, w, t = buf
    np.subtract(x[:, 0][:, None], x[:, 0][None, :], out=dx)
    np.subtract(x[:, 1][:, None], x[:, 1][None, :], out=dy)
    np.subtract(x[:, 2][:, None], x[:, 2][None, :], out=dz)
    np.multiply(dx, dx, out=r2)
    np.multiply(dy, dy, out=t)
    np.add(r2, t, out=r2)
    np.multiply(dz, dz, out=t)
    np.add(r2, t, out=r2)
    np.add(r2, eps2, out=r2)
    np.sqrt(r2, out=t)
    np.multiply(r2, t, out=t)
    np.divide(m, t, out=w)
    ax = _total(np.multiply(w, dx, out=t))  # _total copies before t is used again
    ay = _total(np.multiply(w, dy, out=t))
    az = _total(np.multiply(w, dz, out=t))
    return -np.stack([ax, ay, az], 1)


def _drag7(v: np.ndarray, m: np.ndarray, buf: list[np.ndarray], pick: np.ndarray, reach2: float,
           f0: float) -> float:
    """The rules-7 cooling, one step: every pair of particles closer than H7 that approach each
    other loses the fraction ``f0`` of its closing speed, as an equal and opposite push along
    the line between them (a particle in c such pairs is charged at most 1 / c per pair, so the
    pushes on it never add up to more than a full stop). Equal masses are assumed: the pushes
    then cancel in the total momentum and, lying along the line of centres, in the angular
    momentum. Returns the kinetic energy taken out. ``buf`` is what :func:`_accel7` left;
    ``pick`` is an (N, N) scratch array of bools."""
    dx, dy, dz, r2 = buf[0], buf[1], buf[2], buf[3]
    n = len(m)
    np.less(r2, reach2, out=pick)
    np.logical_and(pick, _upper(n), out=pick)
    flat = np.flatnonzero(pick)  # row by row: a fixed order of the pairs i < j
    i, j = np.divmod(flat, n)
    ex, ey, ez = dx.ravel().take(flat), dy.ravel().take(flat), dz.ravel().take(flat)
    dv = v.take(i, axis=0) - v.take(j, axis=0)
    closing = dv[:, 0] * ex + dv[:, 1] * ey + dv[:, 2] * ez
    keep = np.flatnonzero(closing < 0.0)  # approaching; two particles at one point have closing 0
    if len(keep) == 0:
        return 0.0
    i, j, closing = i.take(keep), j.take(keep), closing.take(keep)
    ex, ey, ez = ex.take(keep), ey.take(keep), ez.take(keep)
    count = np.bincount(i, minlength=n) + np.bincount(j, minlength=n)
    # per pair 0.5 f0 / max(1, f0 max(c_i, c_j)), taken per particle: rounding is monotone, so the
    # maximum of the two rounded products and the minimum of the two rounded quotients are the
    # very numbers the pair would get (the same bits, two passes over the pairs fewer)
    each = (0.5 * f0) / np.maximum(1.0, f0 * count)
    d2 = ex * ex + ey * ey + ez * ez
    if d2.min() <= 0.0:  # cannot happen for an approaching pair unless its distance underflows
        d2 = np.where(d2 > 0.0, d2, 1.0)
    s = np.minimum(each.take(i), each.take(j)) * closing / d2
    before = _kinetic(v, m)
    for c, e in enumerate((ex, ey, ez)):
        push = s * e
        # bincount adds its weights one after the other, in the order of the pairs
        v[:, c] += np.bincount(j, weights=push, minlength=n) - np.bincount(i, weights=push, minlength=n)
    return before - _kinetic(v, m)


@lru_cache(maxsize=4)
def _upper(n: int) -> np.ndarray:
    """True above the diagonal: each pair once, i < j."""
    return np.triu(np.ones((n, n), dtype=bool), 1)


def _evolve7(seed: int, spin: float, cooling: float, n: int, mass: float, eps: float,
             ledger: dict) -> Iterator[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """The N_OUT7 states of a rules-7 run: the same start and leapfrog as round 6, the buffered
    force, and the pairwise drag instead of the drag about the z axis. Nothing resets the
    momentum and nothing knows the axis."""
    if spin > SPIN_MAX7:
        raise ValueError(f"rules 7: spin {spin} is above 1/3, where no start is in virial balance")
    x, v, m = initial_state(seed, n, mass, spin, eps)
    eps2 = eps * eps
    f0 = drag_factor(cooling, DT / T_FF)
    buf = [np.empty((n, n)) for _ in range(6)]
    pick = np.empty((n, n), dtype=bool)
    reach2 = H7 * H7 + eps2  # the softened squared distance within which the drag acts
    a = _accel7(x, m, eps2, buf)
    for k in range(N_OUT7):
        yield x, v, m
        if k == N_OUT7 - 1:
            break
        for _ in range(SUB_STEPS):
            v += 0.5 * DT * a
            x += DT * v
            a = _accel7(x, m, eps2, buf)
            v += 0.5 * DT * a
            if cooling > 0:
                ledger["radiated"] += _drag7(v, m, buf, pick, reach2, f0)


def _square_distances(x: np.ndarray) -> np.ndarray:
    """dx^2 + dy^2 + dz^2 of all pairs, (N, N), added in the order of :func:`_pairs`: adding eps^2
    gives its softened squared distances to the bit (and adding 0, as ``_pairs(x, 0.0)`` does,
    changes nothing). Two arrays are allocated instead of ten."""
    d = np.subtract(x[:, 0][:, None], x[:, 0][None, :])
    d2 = d * d
    np.subtract(x[:, 1][:, None], x[:, 1][None, :], out=d)
    np.multiply(d, d, out=d)
    np.add(d2, d, out=d2)
    np.subtract(x[:, 2][:, None], x[:, 2][None, :], out=d)
    np.multiply(d, d, out=d)
    np.add(d2, d, out=d2)
    return d2


def _potential_d2(d2: np.ndarray, m: np.ndarray, eps2: float) -> float:
    """:func:`_potential` from the squared distances of :func:`_square_distances`: the same bits."""
    inv = d2 + eps2
    np.sqrt(inv, out=inv)
    np.divide(1.0, inv, out=inv)
    np.fill_diagonal(inv, 0.0)
    return -0.5 * float(_total(m * _total(inv * m)))


def bound_groups(x: np.ndarray, v: np.ndarray, m: np.ndarray, eps: float,
                 d2: np.ndarray | None = None) -> list[np.ndarray]:
    """Rules 7: the bound clumps of a state as arrays of particle indices, largest first (ties:
    the one with the lowest particle index first). A clump is a friends-of-friends group with
    linking length ``eps`` and at least MIN_MEMBERS7 particles whose kinetic energy about its
    own mean velocity plus the softened potential energy of its members is negative. ``d2``,
    when given, is :func:`_square_distances` of ``x`` (so a caller that has it need not pay twice)."""
    d2 = _square_distances(x) if d2 is None else d2
    _, label = connected_components(csr_matrix(d2 <= eps * eps), directed=False)
    out = []
    for g in np.flatnonzero(np.bincount(label) >= MIN_MEMBERS7):
        idx = np.flatnonzero(label == g)
        mg, vg = m[idx], v[idx]
        drift = _weighted(mg, vg) / float(_total(mg))
        if _kinetic(vg - drift, mg) + _potential_d2(d2[np.ix_(idx, idx)], mg, eps * eps) < 0.0:
            out.append(idx)
    out.sort(key=lambda idx: (-len(idx), int(idx[0])))
    return out


def _flatness(w: np.ndarray, z: np.ndarray) -> float:
    """rms height over rms in-plane extent / sqrt 2 of the rows of ``z`` (positions about a
    centre) with weights ``w``; 1 when there is nothing to measure."""
    if len(w) < 2:
        return 1.0
    plane = float(_total(w * (z[:, 0] * z[:, 0] + z[:, 1] * z[:, 1])))
    if plane <= 0.0:
        return 1.0
    w_all = float(_total(w))
    return math.sqrt(float(_total(w * z[:, 2] * z[:, 2])) / w_all) / math.sqrt(plane / (2.0 * w_all))


def _cylinder(z: np.ndarray, u: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Distance from the z axis, and the velocity components away from it and around it, for
    positions ``z`` and velocities ``u`` about a centre (a particle on the axis gets 0, 0)."""
    r2 = z[:, 0] * z[:, 0] + z[:, 1] * z[:, 1]
    safe = np.sqrt(np.where(r2 > 0.0, r2, 1.0))
    return (np.sqrt(r2), (z[:, 0] * u[:, 0] + z[:, 1] * u[:, 1]) / safe,
            (z[:, 0] * u[:, 1] - z[:, 1] * u[:, 0]) / safe)


def _spread2(w: np.ndarray, a: np.ndarray) -> float:
    """Weighted mean square of ``a`` about its weighted average."""
    w_all = float(_total(w))
    centre = float(_total(w * a)) / w_all
    return float(_total(w * (a - centre) * (a - centre))) / w_all


def measure7(x: np.ndarray, v: np.ndarray, m: np.ndarray, eps: float, radius0: float | None, time: float,
             radiated: float = 0.0) -> dict:
    """Rules 7: the metrics of one state at full precision (``run`` rounds them, and takes
    ``stage`` from the rounded numbers), plus what stands behind ``pattern`` (``m1``, ``m2``,
    ``pattern_particles``, ``pattern_flattening``, ``second_clump``) and behind the two stability
    numbers (``sigma``, ``density``, ``q_sigma``, ``q_kappa``, ``q_surface_density``)."""
    mass = float(_total(m))
    com = _weighted(m, x) / mass
    y = x - com
    r = np.sqrt(_square(y))
    order = np.argsort(r, kind="stable")
    radius = float(r[order][np.searchsorted(np.cumsum(m[order]), 0.5 * mass)])
    inside = r < 2.0 * radius

    d2 = _square_distances(x)  # once, for the clumps and the potential energy
    groups = bound_groups(x, v, m, eps, d2)
    shares = [float(_total(m[idx])) / mass for idx in groups]
    largest = shares[0] if shares else 0.0
    second = shares[1] if len(shares) > 1 else 0.0
    loose = np.ones(len(m), dtype=bool)
    for idx in groups:
        loose[idx] = False
    if largest >= STAR_FRACTION:  # the star is the centre, and it is at rest in its own frame
        ms = m[groups[0]]
        centre = _weighted(ms, x[groups[0]]) / float(_total(ms))
        drift = _weighted(ms, v[groups[0]]) / float(_total(ms))
    else:
        centre, drift = com, _weighted(m, v) / mass
    z, u = x - centre, v - drift

    # the disc: what is outside every bound clump and within the cloud radius of the centre
    disc = loose & (_square(z) < CLOUD_RADIUS * CLOUD_RADIUS)
    wd, zd, ud = m[disc], z[disc], u[disc]
    rd, out_d, around_d = _cylinder(zd, ud)
    disc_radius = float(np.sort(rd)[len(rd) // 2]) if len(rd) else 0.0
    flattening = _flatness(wd, zd) if largest >= STAR_FRACTION else _flatness(m[inside], y[inside])

    # pattern: the loose particles nearest the axis (the outer tenth dropped), if they lie flat
    wl, zl = m[loose], z[loose]
    rl = _cylinder(zl, zl)[0]
    near = np.argsort(rl, kind="stable")[:len(rl) - len(rl) // 10]
    wp, zp = wl[near], zl[near]
    m1 = m2 = 0.0
    if len(wp):
        w_all = float(_total(wp))
        xx, yy = zp[:, 0] * zp[:, 0], zp[:, 1] * zp[:, 1]
        safe = np.where(xx + yy > 0.0, xx + yy, 1.0)
        root = np.sqrt(safe)
        c1, s1 = float(_total(wp * zp[:, 0] / root)), float(_total(wp * zp[:, 1] / root))
        c2 = float(_total(wp * (xx - yy) / safe))
        s2 = float(_total(wp * 2.0 * zp[:, 0] * zp[:, 1] / safe))
        m1, m2 = math.sqrt(c1 * c1 + s1 * s1) / w_all, math.sqrt(c2 * c2 + s2 * s2) / w_all
    flat_p = _flatness(wp, zp)

    # the energies and what holds the cloud up
    kinetic = _kinetic(v, m)
    potential = _potential_d2(d2, m, eps * eps)
    lz = float(_total(m * (y[:, 0] * v[:, 1] - y[:, 1] * v[:, 0])))
    inertia = float(_total(m * (y[:, 0] * y[:, 0] + y[:, 1] * y[:, 1])))

    # Jeans: the mass outside the clumps (the whole cloud before the first clump), its random
    # speed within 2 of its half-mass radii and the mean density inside that radius
    jeans = sigma = density = half = 0.0
    if len(wl) >= ANNULUS_MIN:
        loose_mass = float(_total(wl))
        dl = np.sqrt(_square(zl))
        by_distance = np.argsort(dl, kind="stable")
        half = float(dl[by_distance][np.searchsorted(np.cumsum(wl[by_distance]), 0.5 * loose_mass)])
    if half > 0.0:
        core = dl < 2.0 * half
        wi = wl[core]
        ri, out_i, around_i = _cylinder(zl[core], u[loose][core])
        turning = float(_total(wi * ri * ri))
        omega = float(_total(wi * ri * around_i)) / turning if turning > 0.0 else 0.0
        shear = around_i - omega * ri
        sigma = math.sqrt((_spread2(wi, out_i) + float(_total(wi * shear * shear)) / float(_total(wi))
                           + _spread2(wi, u[loose][core][:, 2])) / 3.0)
        density = sphere_density(0.5 * loose_mass, half)
        jeans = jeans_number(loose_mass, sigma, density) if sigma > 0.0 else 0.0

    # Toomre: the disc between 0.5 and 2 disc radii, if what the state calls flattening is flat
    ring = (rd > 0.5 * disc_radius) & (rd < 2.0 * disc_radius)
    q: float | str = "none"
    q_sigma = q_kappa = q_surface = 0.0
    if flattening < DISC_FLATTENING and int(np.count_nonzero(ring)) >= ANNULUS_MIN:
        wr, rr = wd[ring], rd[ring]
        q_sigma = math.sqrt(_spread2(wr, out_d[ring]))
        q_kappa = math.sqrt(2.0) * abs(float(_total(wr * rr * around_d[ring])) / float(_total(wr * rr * rr)))
        q_surface = float(_total(wr)) / (math.pi * 3.75 * disc_radius * disc_radius)
        q = toomre_q(q_sigma, q_kappa, q_surface)

    p = _weighted(m, v)
    return {
        "time": time, "radius": radius, "flattening": flattening, "clumps": len(groups), "largest_clump": largest,
        "disc_mass": float(_total(wd)) / mass, "disc_radius": disc_radius,
        "pattern": pattern7(second, len(wp), flat_p, m1, m2),
        "energy": kinetic + potential, "radiated": radiated,
        "support": virial_ratio(kinetic, abs(potential)),
        "rotation": rotation_share(lz, inertia, abs(potential)),
        "random": random_share(kinetic, lz, inertia, abs(potential)),
        "jeans_number": jeans, "toomre_q": q,
        "stage": stage7(radius, flattening, len(groups), largest, radius if radius0 is None else radius0),
        "angular_momentum": lz, "mass": mass,
        "momentum": math.sqrt(float(p[0] * p[0] + p[1] * p[1] + p[2] * p[2])),
        "m1": m1, "m2": m2, "pattern_particles": len(wp), "pattern_flattening": flat_p, "second_clump": second,
        "sigma": sigma, "density": density, "q_sigma": q_sigma, "q_kappa": q_kappa, "q_surface_density": q_surface,
    }


def _fof(y: np.ndarray, m: np.ndarray, link: float) -> tuple[int, float]:
    """Number of friends-of-friends groups with at least MIN_MEMBERS particles and the mass
    fraction of the biggest one."""
    near = _pairs(y, 0.0)[3] <= link * link
    _, label = connected_components(near, directed=False)
    big = np.flatnonzero(np.bincount(label) >= MIN_MEMBERS)
    if len(big) == 0:
        return 0, 0.0
    masses = np.bincount(label, weights=m)  # one pass in particle order
    return int(len(big)), float(masses[big].max() / float(_total(m)))


def stage(radius: float, flattening: float, clumps: int, largest: float, radius0: float) -> str:
    if largest >= CLUMPED_FRACTION:
        return "clumped"
    if clumps >= 2 and largest >= FRAGMENT_FRACTION:
        return "fragmenting"
    if flattening < DISC_FLATTENING:
        return "disc"
    if radius < COLLAPSING_RADIUS * radius0:
        return "collapsing"
    return "cloud"


def measure(x: np.ndarray, v: np.ndarray, m: np.ndarray, eps: float, radius0: float | None, time: float,
            radiated: float = 0.0, rules: int = 6) -> dict:
    """The metrics of one state at full precision (``run`` rounds them), plus ``m1`` and ``m2``,
    the Fourier amplitudes behind ``spiral``. ``radius0`` is the first state's radius (``None``
    for the first state itself), ``radiated`` the ledger of :func:`evolve`. ``rules=7`` gives
    :func:`measure7`."""
    _rules(rules)
    if rules >= 7:
        return measure7(x, v, m, eps, radius0, time, radiated)
    mass = float(_total(m))
    y = x - _weighted(m, x) / mass
    r = np.sqrt(_square(y))
    order = np.argsort(r, kind="stable")
    radius = float(r[order][np.searchsorted(np.cumsum(m[order]), 0.5 * mass)])
    inside = r < 2.0 * radius
    w, yi = m[inside], y[inside]
    w_sum = float(_total(w))
    xx, yy = yi[:, 0] * yi[:, 0], yi[:, 1] * yi[:, 1]
    r2 = xx + yy
    flattening = (math.sqrt(float(_total(w * yi[:, 2] * yi[:, 2])) / w_sum)
                  / math.sqrt(float(_total(w * r2)) / (2.0 * w_sum)))
    # cos and sin of the azimuth and of twice the azimuth, without trigonometric functions
    safe = np.where(r2 > 0.0, r2, 1.0)
    root = np.sqrt(safe)
    c1, s1 = float(_total(w * yi[:, 0] / root)), float(_total(w * yi[:, 1] / root))
    c2 = float(_total(w * (xx - yy) / safe))
    s2 = float(_total(w * 2.0 * yi[:, 0] * yi[:, 1] / safe))
    m1, m2 = math.sqrt(c1 * c1 + s1 * s1) / w_sum, math.sqrt(c2 * c2 + s2 * s2) / w_sum
    clumps, largest = _fof(y, m, max(LINK_FRACTION * radius, eps))
    kinetic = _kinetic(v, m)
    p = _weighted(m, v)
    return {
        "time": time, "radius": radius, "flattening": flattening, "clumps": clumps, "largest_clump": largest,
        "spiral": "yes" if flattening < DISC_FLATTENING and m2 > SPIRAL_M2 and m2 > m1 else "no",
        "energy": kinetic + _potential(x, m, eps * eps),
        "angular_momentum": float(_total(m * (y[:, 0] * v[:, 1] - y[:, 1] * v[:, 0]))),
        "stage": stage(radius, flattening, clumps, largest, radius if radius0 is None else radius0),
        "mass": mass, "momentum": math.sqrt(float(p[0] * p[0] + p[1] * p[1] + p[2] * p[2])),
        "radiated": radiated, "m1": m1, "m2": m2,
    }


def random_params(rng: random.Random, rules: int = 6) -> dict[str, float]:
    """``spin`` uniform in [0.05, 0.4]; ``cooling`` 0 in one run of six (the conservative control),
    else uniform in [0.1, 0.6]; both to 2 decimals. Under ``rules=7`` the spin is uniform in
    [0.02, 0.3] (the same draws in the same order), so every start is in virial balance."""
    _rules(rules)
    spin = round(rng.uniform(0.02, 0.30), 2) if rules >= 7 else round(rng.uniform(0.05, 0.4), 2)
    cooling = 0.0 if rng.random() < 1 / 6 else round(rng.uniform(0.1, 0.6), 2)
    return {"spin": spin, "cooling": cooling}


class Gravity:
    """The level as a :class:`haishool.cosmos.Simulation`. ``Gravity()`` is round 6;
    ``Gravity(rules=7)`` (``simulation(rules=7)``) runs every seed under rules 7, which a single
    call can also ask for with ``run(seed, rules=7, ...)``. ``conserved``, ``check`` and ``owns``
    judge the runs and prompts of both rules whatever the instance was made with."""
    sim = SIM
    KEYS = KEYS
    KEYS7 = KEYS7

    def __init__(self, rules: int = 6) -> None:
        _rules(rules)
        self.rules = rules
        if rules >= 7:
            self.KEYS = KEYS7

    def random_params(self, rng: random.Random, rules: int = 6) -> dict[str, float]:
        _rules(rules)
        return random_params(rng, rules=max(rules, self.rules))

    def rollout(self, seed: int, rules: int = 6) -> Rollout:
        _rules(rules)
        return rollout(seed, rules=max(rules, self.rules))

    def run(self, seed: int, spin: float = 0.2, cooling: float = 0.3, n: int = N, mass: float = MASS,
            eps: float = EPS, rules: int = 6) -> Rollout:
        _rules(rules)
        if max(rules, self.rules) >= 7:
            return self._run7(seed, spin, cooling, n, mass, eps)
        steps: list[dict] = []
        radius0: float | None = None
        ledger: dict = {}
        for k, (x, v, m) in enumerate(evolve(seed, spin, cooling, n, mass, eps, ledger)):
            full = measure(x, v, m, eps, radius0, k * OUT_DT, ledger["radiated"])
            radius0 = full["radius"] if radius0 is None else radius0
            steps.append({key: _sig3(full[key]) if isinstance(full[key], float) else full[key] for key in KEYS})
        last = steps[-1]
        halved = [s["time"] for s in steps if s["radius"] < 0.5 * steps[0]["radius"]]
        summary = {
            "clumps": last["clumps"], "largest_clump": last["largest_clump"], "flattening": last["flattening"],
            "min_flattening": min(s["flattening"] for s in steps),
            "spiral_ever": "yes" if any(s["spiral"] == "yes" for s in steps) else "no",
            "collapse_time": halved[0] if halved else "never",
        }
        params = {"spin": spin, "cooling": cooling, "n": n, "mass": mass, "eps": eps, "dt": _sig3(DT),
                  "t_ff": _sig3(T_FF)}
        return Rollout(SIM, seed, params, steps, summary)

    def _run7(self, seed: int, spin: float, cooling: float, n: int, mass: float, eps: float) -> Rollout:
        """A rules-7 run: N_OUT7 states with the metrics KEYS7 (3 significant digits), the
        summary SUMMARY_KEYS7, and params that say ``rules 7`` and hold the givens of the
        ``params`` record. ``stage`` is :func:`stage7` of the rounded numbers of its state and
        the rounded first radius, so it can be recomputed from what the lines print; for the
        same reason ``toomre_q`` is ``none`` whenever the rounded flattening is 0.5 or more."""
        steps: list[dict] = []
        radius0: float | None = None
        ledger: dict = {}
        for k, (x, v, m) in enumerate(evolve(seed, spin, cooling, n, mass, eps, ledger, rules=7)):
            full = measure7(x, v, m, eps, radius0, k * OUT_DT, ledger["radiated"])
            radius0 = full["radius"] if radius0 is None else radius0
            step = {key: sig(full[key]) if isinstance(full[key], float) else full[key] for key in KEYS7}
            step["stage"] = stage7(step["radius"], step["flattening"], step["clumps"], step["largest_clump"],
                                   steps[0]["radius"] if steps else step["radius"])
            if step["flattening"] >= DISC_FLATTENING:  # the rule of toomre_q, on the number as printed
                step["toomre_q"] = "none"
            steps.append(step)
        first, last = steps[0], steps[-1]
        starred = [s["time"] for s in steps if s["largest_clump"] >= STAR_FRACTION]
        seen = {s["pattern"] for s in steps}
        summary = {
            "stage": last["stage"], "clumps": last["clumps"], "star": last["largest_clump"],
            "companions": max(0, last["clumps"] - 1),
            "disc_mass": last["disc_mass"], "disc_radius": last["disc_radius"], "flattening": last["flattening"],
            "min_flattening": min(s["flattening"] for s in steps),
            "star_time": starred[0] if starred else "never",
            "patterns": " ".join(word for word in PATTERNS7[1:] if word in seen) or "none",
            "angular_momentum_z": last["angular_momentum"],
        }
        params = {"spin": spin, "cooling": cooling, "n": n, "mass": mass, "eps": eps, "dt": _sig3(DT),
                  "t_ff": _sig3(T_FF), "rules": 7,
                  "cooling_time": sig(cooling_time(cooling)) if cooling > 0 else "never",
                  "bar_unstable": bar_unstable(spin), "t_ff_myr": T_FF_MYR,
                  "angular_momentum_z": first["angular_momentum"],
                  "centrifugal_radius": sig(centrifugal_radius(first["angular_momentum"] / mass, mass)),
                  "radius_start": first["radius"]}
        return Rollout(SIM, seed, params, steps, summary)

    def conserved(self, r: Rollout) -> Verdict:
        first = r.steps[0]
        e0, lz0 = first["energy"], first["angular_momentum"]
        for t, s in enumerate(r.steps):
            if s["mass"] != first["mass"]:
                return Verdict(False, None, f"mass changed at step {t}: {first['mass']} -> {s['mass']}")
            if s["momentum"] > MOMENTUM_TOL:
                return Verdict(False, None, f"momentum {s['momentum']} at step {t}")
            if not close(s["angular_momentum"], lz0, rel=LZ_TOL):
                return Verdict(False, None, f"angular momentum changed at step {t}: {lz0} -> {s['angular_momentum']}")
            if r.params["cooling"] == 0 and s["radiated"] != 0:
                return Verdict(False, None, f"energy radiated without cooling at step {t}: {s['radiated']}")
            before = r.steps[t - 1] if t else first
            if s["radiated"] < before["radiated"]:
                return Verdict(False, None, f"radiated energy fell at step {t}: "
                                            f"{before['radiated']} -> {s['radiated']}")
            if r.params["cooling"] > 0 and s["energy"] > before["energy"] + RISE_TOL * abs(before["energy"]):
                return Verdict(False, None, f"energy rose at step {t}: {before['energy']} -> {s['energy']}")
            if abs(s["energy"] + s["radiated"] - e0) > ENERGY_TOL * max(abs(e0), abs(s["energy"])):
                return Verdict(False, None, f"energy changed at step {t}: {e0} -> {s['energy']} "
                                            f"plus {s['radiated']} radiated")
        return Verdict(True)

    def owns(self, prompt: str) -> bool:
        return parse_prompt(prompt) is not None or parse_prompt7(prompt) is not None

    def check(self, prompt: str, answer: str) -> Verdict:
        parsed = parse_prompt(prompt)
        if parsed is None:
            return self._check7(prompt, answer)
        seed, step, key = parsed
        r = _replay(seed)
        if step is None:
            expected = r.summary[key]
        elif step >= len(r.steps):
            return Verdict(False, None, f"step {step} is beyond the run of {len(r.steps)} states")
        else:
            expected = r.steps[step][key]
        text = r.value(expected)
        if isinstance(expected, str):
            return Verdict(answer == expected, text)
        got = parse_num(answer)
        if got is None:
            return Verdict(False, text, "not a number")
        if key == "clumps":
            return Verdict(isinstance(got, int) and abs(got - expected) <= 1, text)
        return Verdict(close(got, expected, rel=0.05), text)

    def _check7(self, prompt: str, answer: str) -> Verdict:
        """The rules-7 story of a seed (``gravity seed 5 rules 7 ...``): words, whole numbers,
        times and the params exactly; every other number within 5 percent."""
        parsed = parse_prompt7(prompt)
        if parsed is None:
            return Verdict(False, None, "not my question")
        seed, step, key, kind = parsed
        r = _replay7(seed)
        if kind == "params":
            expected = _param7(r, key)
        elif kind == "final":
            expected = r.summary[key]
        elif step >= len(r.steps):
            return Verdict(False, None, f"step {step} is beyond the run of {len(r.steps)} states")
        else:
            expected = r.steps[step][key]
        text = r.value(expected)
        if isinstance(expected, str):
            return Verdict(answer == expected, text)
        got = parse_num(answer)
        if got is None:
            return Verdict(False, text, "not a number")
        if isinstance(expected, int):
            return Verdict(isinstance(got, int) and got == expected, text)
        if kind == "params" or key in EXACT7:
            return Verdict(got == expected, text)
        return Verdict(close(got, expected, rel=0.05), text)


def parse_prompt(prompt: str) -> tuple[int, int | None, str] | None:
    """``gravity seed 7 step 1 2 next clumps`` -> (7, 13, "clumps"); ``gravity seed 7 final clumps``
    -> (7, None, "clumps"); ``None`` when it is not a gravity prompt."""
    hit = _PROMPT.match(prompt)
    if not hit:
        return None
    seed_text, step_text, nxt, key = hit.groups()
    seed = parse_num(seed_text)
    if num(seed) != seed_text:  # no leading zeros
        return None
    if step_text is None:
        return (seed, None, key) if key in SUMMARY_KEYS else None
    step = parse_num(step_text)
    if num(step) != step_text or key not in KEYS:
        return None
    return seed, step + (1 if nxt else 0), key


def parse_prompt7(prompt: str) -> tuple[int, int | None, str, str] | None:
    """A prompt about the rules-7 run of a seed: ``gravity seed 7 rules 7 step 1 2 next clumps``
    -> (7, 13, "clumps", "step"); ``gravity seed 7 rules 7 final star_time`` -> (7, None,
    "star_time", "final"); ``gravity seed 7 rules 7 params spin`` -> (7, None, "spin", "params");
    ``None`` for anything else (a prompt without ``rules 7`` means round 6: :func:`parse_prompt`)."""
    hit = _PROMPT7.match(prompt)
    if not hit:
        return None
    seed_text, step_text, nxt, final, key = hit.groups()
    seed = parse_num(seed_text)
    if num(seed) != seed_text:  # no leading zeros
        return None
    if step_text is None:
        if final:
            return (seed, None, key, "final") if key in SUMMARY_KEYS7 else None
        return (seed, None, key, "params") if key in PARAM_KEYS7 else None
    step = parse_num(step_text)
    if num(step) != step_text or key not in KEYS7:
        return None
    return seed, step + (1 if nxt else 0), key, "step"


@lru_cache(maxsize=4096)
def _replay(seed: int) -> Rollout:
    return Gravity().run(seed, **random_params(random.Random(seed)))


@lru_cache(maxsize=4096)
def _replay7(seed: int) -> Rollout:
    return Gravity().run(seed, rules=7, **random_params(random.Random(seed), rules=7))


def rollout(seed: int, rules: int = 6) -> Rollout:
    """The run a prompt names: ``random_params`` from the seed, then ``run`` (both with the same
    ``rules``; a seed names one round-6 run and one rules-7 run). Cached (a run takes over a
    second and a feedback round asks about hundreds of seeds in any order; one rollout is about
    10 kB, 25 kB under rules 7); the caller gets a copy, so nothing it does reaches the gate."""
    _rules(rules)
    return copy.deepcopy(_replay7(seed) if rules >= 7 else _replay(seed))


def canonical(r: Rollout) -> bool:
    """True when ``r`` is the run its seed names (:func:`rollout`, with the rules its params
    record), the only one ``check`` can replay."""
    rules = r.params.get("rules", 6)
    if rules not in (6, 7):
        return False
    expected = {**random_params(random.Random(r.seed), rules=rules), "n": N, "mass": MASS, "eps": EPS}
    return r.sim == SIM and all(r.params.get(k) == v for k, v in expected.items())


def simulation(rules: int = 6) -> Gravity:
    """The level: round 6 by default, ``simulation(rules=7)`` for the corrected one."""
    return Gravity(rules)


def lines(r: Rollout, every: int = 1, rules: int | None = None) -> list[Line]:
    """Every state as a record line, the questions for every ``every``-th state (each shown
    metric now and ``next``) and the summary questions: 776 lines per run with ``every`` 1.
    A rules-7 rollout (its params say so) gives the rules-7 lines (:func:`lines7`); ``rules``
    need not be passed, and when it is it must be the rules the rollout was run with."""
    own = r.params.get("rules", 6)
    if rules is not None and rules != own:
        raise ValueError(f"the rollout was run with rules {own}, not {rules}")
    if not canonical(r):
        raise ValueError(f"lines only for the run a seed names (rollout({r.seed})): the gate replays that one, "
                         f"not parameters {r.params}")
    if own >= 7:
        return lines7(r, every)
    out: list[Line] = []
    for t in range(len(r.steps)):
        out.append(state_line(r, t, LINE_KEYS))
        if t % every == 0:
            out += query_lines(r, t, LINE_KEYS)
    return out + summary_lines(r)


def _param7(r: Rollout, key: str) -> float | int | str:
    """A field of the rules-7 ``params`` record."""
    return r.params[{"particles": "n", "softening": "eps"}.get(key, key)]


def lines7(r: Rollout, every: int = 1) -> list[Line]:
    """The lines of a rules-7 run. Every one says ``rules 7`` after the seed, so it cannot be
    taken for the round-6 run of the same seed: the ``params`` record and the questions for
    the params that differ between seeds, every state as a record, the questions for every
    ``every``-th state (now and ``next``), and the summary questions (2014 lines with ``every`` 1)."""
    if r.params.get("rules") != 7 or not canonical(r):
        raise ValueError(f"lines7 only for the rules-7 run a seed names (rollout({r.seed}, rules=7))")
    head = f"{SIM} seed {num(r.seed)} rules 7"
    out = [Line(f"{head} params. " + " ".join(f"{k} {r.value(_param7(r, k))}." for k in PARAM_KEYS7),
                topic=SIM, kind="record")]
    out += [Line(f"{head} params {k}", r.value(_param7(r, k)), SIM, "fact") for k in PARAM_QUESTIONS7]
    for t, step in enumerate(r.steps):
        out.append(Line(f"{head} step {num(t)}. " + " ".join(f"{k} {r.value(step[k])}." for k in LINE_KEYS7),
                        topic=SIM, kind="record"))
        if len(out[-1].text.replace(".", " . ").split()) > MAX_TOKENS7:  # 110 at most in seeds 1 to 100
            raise ValueError(f"the state line of seed {r.seed} step {t} is longer than {MAX_TOKENS7} tokens")
        if t % every == 0:
            out += [Line(f"{head} step {num(t)} {k}", r.value(step[k]), SIM, "fact") for k in LINE_KEYS7]
            if t + 1 < len(r.steps):
                nxt = r.steps[t + 1]
                out += [Line(f"{head} step {num(t)} next {k}", r.value(nxt[k]), SIM, "calc") for k in LINE_KEYS7]
    return out + [Line(f"{head} final {k}", r.value(v), SIM, "calc") for k, v in r.summary.items()]


# ---------------------------------------------------------------------------------------------
# lessons: questions that carry their inputs (topic ``predict_gravity``)
# ---------------------------------------------------------------------------------------------

RULES: dict[str, Rule] = {
    "free_fall_time": Rule(
        (Input("density", 0.01, 100, places=2),), free_fall_time,
        "time for a uniform sphere without pressure to fall together. sqrt of 3 pi over 32 density. g is 1"),
    "sphere_density": Rule(
        (Input("mass", 0.1, 2, places=2), Input("radius", 0.02, 2, places=2)), sphere_density,
        "mean density of a sphere. mass over four thirds pi radius cubed"),
    "softened_force": Rule(
        (Input("mass_1", 0.001, 1), Input("mass_2", 0.001, 1), Input("distance", 0.01, 2),
         Input("softening", 0.01, 0.2)), softened_force,
        "pull between two bodies. mass_1 times mass_2 times distance over the square root of distance squared "
        "plus softening squared cubed. g is 1"),
    "softened_potential": Rule(
        (Input("mass_1", 0.001, 1), Input("mass_2", 0.001, 1), Input("distance", 0.01, 2),
         Input("softening", 0.01, 0.2)), softened_potential,
        "potential energy of two bodies. minus mass_1 times mass_2 over the square root of distance squared "
        "plus softening squared. g is 1"),
    "spin_rate": Rule(
        (Input("spin", 0, 0.33, places=2), Input("potential", 0.3, 1), Input("inertia", 0.1, 0.5)), spin_rate,
        "angular velocity of the rotating start. sqrt of 2 spin potential over inertia. potential is the size "
        "of the potential energy. inertia is the sum of mass times squared distance from the axis"),
    "plane_speed": Rule(
        (Input("spin", 0, 0.33, places=2), Input("potential", 0.3, 1), Input("mass", 0.5, 2, places=2)),
        plane_speed,
        "random speed along each axis of the plane at the start. sqrt of one third minus spin times potential "
        "over mass. zero when spin is above one third"),
    "virial_ratio": Rule(
        (Input("kinetic", 0.05, 5), Input("potential", 0.1, 10)), virial_ratio,
        "support of a cloud. 2 kinetic over potential. 1 is balance. below 1 gravity wins"),
    "rotation_share": Rule(
        (Input("lz", 0.05, 0.6), Input("inertia", 0.01, 0.5), Input("potential", 0.3, 10)), rotation_share,
        "share of the binding held up by rotation. lz squared over 2 inertia potential"),
    "random_share": Rule(
        (Input("kinetic", 0.05, 5), Input("lz", 0.05, 0.6), Input("inertia", 0.01, 0.5),
         Input("potential", 0.3, 10)), random_share,
        "share of the binding held up by random motion. kinetic minus lz squared over 2 inertia. all over potential",
        valid=lambda kinetic, lz, inertia, potential: kinetic >= 0.5 * lz * lz / inertia),
    "jeans_mass": Rule(
        (Input("sigma", 0.01, 2), Input("density", 0.01, 100, places=2)), jeans_mass,
        "smallest mass that gravity pulls together against random speed sigma. pi squared times sqrt pi over 6 "
        "times sigma cubed over sqrt density. g is 1"),
    "jeans_number": Rule(
        (Input("mass", 0.1, 2, places=2), Input("sigma", 0.05, 2), Input("density", 0.01, 100, places=2)),
        jeans_number,
        "jeans masses in a cloud. mass over jeans_mass of sigma and density. it can break up when it holds several"),
    "toomre_q": Rule(
        (Input("sigma", 0.01, 2), Input("kappa", 0.1, 20, places=2), Input("surface_density", 0.01, 20, places=2)),
        toomre_q,
        "stability of a thin disc. sigma times kappa over 3 point 3 6 times surface_density. stable above 1. g is 1"),
    "bar_unstable": Rule(
        (Input("spin", 0, 0.33, places=2),), bar_unstable,
        "yes when the share of rotation in the binding is above 0 point 1 4. a rotating body in balance is then "
        "unstable and turns into a bar"),
    "centrifugal_radius": Rule(
        (Input("j", 0.01, 1), Input("mass", 0.1, 2, places=2)), centrifugal_radius,
        "radius of the circular orbit with angular momentum j per unit mass about mass. j squared over mass. g is 1"),
    "cooling_time": Rule(
        (Input("cooling", 0.1, 0.6, places=2),), cooling_time,
        "free fall times in which the toy drag takes the closing speed of an approaching pair of neighbours "
        f"down to 1 over e of it. 1 over {num(COOL_RATE7)} cooling. a toy rate. not a measured cooling time"),
    "drag_factor": Rule(
        (Input("cooling", 0, 0.6, places=2), Input("dt_tff", 0.0001, 0.1, places=4)), drag_factor,
        f"fraction of its closing speed an approaching pair of neighbours loses in one step. {num(COOL_RATE7)} "
        "times cooling times dt_tff. dt_tff is the step in free fall times. in a crowd a pair loses at most 1 "
        "over the number of approaching pairs of its busier particle"),
    "stage": Rule(
        (Input("radius", 0.2, 0.75), Input("flattening", 0.2, 1.1), Input("clumps", 0, 3, integer=True),
         Input("largest_clump", 0.06, 0.5), Input("radius_start", 0.55, 0.75)), stage7,
        "first that applies. multiple when clumps is 2 or more and largest_clump at least 0 point 2 5. star when "
        "clumps is 1 or more and largest_clump at least 0 point 2 5. fragmenting when clumps is 1 or more. disc "
        "when flattening is below 0 point 5. contracting when radius is below 0 point 7 5 radius_start. else "
        "cloud. largest_clump counts only when there is a clump"),
    "pattern": Rule(
        (Input("second_clump", 0, 0.15), Input("particles", 0, 256, integer=True), Input("flattening", 0.02, 0.8),
         Input("m1", 0, 0.6), Input("m2", 0, 0.6)), pattern7,
        "first that applies. pair when second_clump is at least 0 point 1. none when particles is below 4 0 or "
        "flattening at least 0 point 5. lopsided when m1 squared times particles is above 6 point 9 and m1 at "
        "least m2. bar when m2 squared times particles is above 6 point 9 and m2 above m1. else none"),
}
LESSONS = LessonGate(SIM, RULES)


def lesson_gate() -> LessonGate:
    """The gate of the lessons (``gravity predict <rule> <input> <value> ...``)."""
    return LESSONS


def run(seed: int, rules: int = 6, **params) -> Rollout:
    """``simulation().run``: round 6 by default, ``rules=7`` for the corrected level."""
    return Gravity().run(seed, rules=rules, **params)


def conserved(r: Rollout) -> Verdict:
    return Gravity().conserved(r)


def owns(prompt: str) -> bool:
    """True for the story of a seed (round 6 or ``rules 7``) and for a lesson."""
    return Gravity().owns(prompt) or LESSONS.owns(prompt)


def check(prompt: str, answer: str) -> Verdict:
    """The gate for every gravity prompt: a lesson by its rule, a story by replaying the seed."""
    if LESSONS.owns(prompt):
        return LESSONS.check(prompt, answer)
    return Gravity().check(prompt, answer)
