"""Level 1 of the toy universe: the first hour, when protons and neutrons become hydrogen and helium.

Big-Bang nucleosynthesis reduced to a handful of closed-form rules. Rounds 1-5 taught
things that *are*; this level teaches a process: the model watches an expanding, cooling universe
step by step and has to say what each number does next and how the run ends. The simulation is
its own gate: it replays any seed and judges the answer.

There are two rule sets. Round 6, described first, is the default and is frozen. ``rules=7``
corrects it (section "Rules 7" at the end): one universe near the measured one instead of a wide
box of them, the photons heated by electron-positron annihilation, a freeze-out that follows
from the neutron lifetime, and lessons that are the very functions the run calls.

Real physics kept (these are the formulas the code uses, nothing else):

* radiation era: ``T(t) = T1 / sqrt(t)`` with ``T1 = 1e10 K / sqrt(expansion_factor)`` and ``t``
  in seconds. The expansion rate ``H = 1 / (2 t)`` grows with ``T^2``, so a faster expansion
  (more relativistic species, ``expansion_factor > 1``) means a lower temperature at a given time.
* weak equilibrium: while the weak reactions are fast, ``n/p = exp(-Q / kT)`` with
  ``Q = m_n - m_p = 1.293 MeV`` (the ``plasma`` stage).
* freeze-out: the weak rate falls as ``T^5`` and the expansion rate as ``expansion_factor T^2``,
  so the ratio freezes at ``kT_freeze = kT(t_freeze, standard expansion) expansion_factor^(1/3)``,
  reached at ``t = t_freeze expansion_factor^(-5/3)``. With the defaults that is 0.79 MeV at
  1.2 s, where n/p = 0.19 (the textbook "about 1/5 to 1/6 at 0.8 MeV").
* free neutron decay ``n -> p + e- + anti-nu`` with lifetime ``tau_n`` (880 s here; the measured
  value is 878.4 s): ``n(t) = n_freeze exp(-(t - t_freeze) / tau_n)`` (the ``decay`` stage).
* the deuterium bottleneck: nothing heavier than hydrogen survives until the photons can no
  longer split deuterium. By the Saha equation the deuterium fraction grows as
  ``eta exp(B_D / kT)`` with ``B_D = 2.225 MeV`` (its slowly varying ``T^(3/2)`` prefactor is
  dropped), so the bottleneck opens at ``B_D / kT_d = B_D / kT_d0 - ln(eta_factor)`` with
  ``T_d0 = 8e8 K`` (0.069 MeV): a denser universe opens it a little earlier, at
  ``t_d = (T1 / T_d)^2`` (156 s for the defaults).
* helium: nearly every surviving neutron ends up in helium-4, so the helium mass fraction is
  ``Y = 2 n / (n + p)`` at the bottleneck, less the few percent that still decay while the
  nuclei form.

Toy parts (so that a run is thirty numbers and well under a millisecond):

* nothing is integrated: there are no reaction rates and no network of nuclear reactions, only
  the closed forms above evaluated at thirty times. The run is a cartoon of the textbook
  account of the first hour, not a nucleosynthesis code.
* one power law for the temperature. The real photon temperature gets a boost from electron-
  positron annihilation at 5e9 K, so the real bottleneck (about 0.07 MeV) comes at 200-300 s
  rather than this toy's 156 s.
* freeze-out is a sharp switch at a time that is a free parameter (``t_freeze``); in reality
  the ratio keeps slipping for a while and the freeze-out temperature follows from the strength
  of the weak interaction. Both this and the early bottleneck leave too many neutrons, which is
  why the toy's default helium fraction (0.263) sits above the real one (0.245 to 0.247).
* ``tau_n`` only sets how fast free neutrons decay. In reality the same weak coupling fixes
  the freeze-out temperature, so the real helium fraction reacts about three times more
  strongly to the neutron lifetime than this toy's does.
* once the bottleneck opens, the free neutrons are bound with one e-folding time
  ``tau_bind = 30 s`` (the real rise of helium-4 takes about 100 s, set by how fast deuterium
  builds up as the universe cools, not by the nuclear rates). Decay goes on meanwhile, so the
  share ``tau_n / (tau_n + tau_bind)`` of the neutrons at the bottleneck is bound and the rest
  becomes protons: ``n_free(t) = n_d exp(-(t - t_d) / tau_eff)``, ``1/tau_eff = 1/tau_bind +
  1/tau_n``, and every bound nucleus grows with the same ``progress = 1 - exp(-(t - t_d)/tau_eff)``.
* the traces are not computed from the reaction network. Their final number ratios to hydrogen
  are the standard yields at the measured baryon density, scaled with the baryon density by the
  fitted power laws (from memory, see :data:`CONSTANTS`): ``D/H = 2.5e-5 eta^-1.6``,
  ``He3/H = 1e-5 eta^-0.6``, ``Li7/H = 5e-10 eta^2``. They do draw their nucleons from the
  budget (a deuteron is one proton and one neutron), and they grow in step with the helium
  (the real deuterium first overshoots and is then burned down to its final value). The
  lithium law is the high-density branch; the real curve turns around near ``eta_factor`` 0.5
  and rises again below it, which this toy does not show.
* the random parameters span what-if universes far from ours (final helium from 0.12 to 0.49):
  the point is that the model learns the rule, not the one measured number.
* every nucleon weighs one unit: mass fractions are nucleon fractions, binding energy ignored.
* electrons, positrons, neutrinos and photons are not tracked. Charge is conserved because every
  lost neutron is a gained proton (and an unseen electron); once the free neutrons fall below
  1e-10 of all nucleons the run is set to its end state, so the budget stays exact.

Bookkeeping, per nucleon: free neutrons ``n``, hydrogen-1 ``p``, helium-4 ``he4`` (2p 2n),
deuterium ``d`` (1p 1n), helium-3 ``he3`` (2p 1n), lithium-7 ``li`` (3p 4n). Given the bound
neutrons ``N_b`` and all protons ``P`` at the end, the trace ratios ``r_x = x / p`` fix the end
state in closed form: ``p = (P - N_b) / (1 + r_he3 - r_li)``,
``he4 = (N_b - d - he3 - 4 li) / 2``.

Parameters (``DEFAULTS``; :func:`random_params` draws the first four from a seed):

    eta_factor        baryon-to-photon ratio relative to the measured 6.1e-10   0.3 .. 3 (log-uniform)
    tau_n             neutron lifetime in s                                     800 .. 960
    t_freeze          freeze-out time for the standard expansion, s             0.5 .. 2 (log-uniform)
    expansion_factor  expansion rate relative to the standard one                0.7 .. 1.4 (log-uniform)
    t1, t_d, tau_bind fixed: 1e10 K, 8e8 K, 30 s

The rollout of a seed is :func:`rollout` = ``run(seed, **random_params(random.Random(seed)))``;
``run(seed)`` alone uses the defaults (the seed is then only a label: all randomness is in the
parameters). Time runs from 0.1 s to 3000 s on a log grid of 30 steps (:data:`TIMES`, rounded to
three digits so the lines stay short): it starts early enough that every seed is still in weak
equilibrium for its first three steps or more (the earliest freeze-out of the box is 0.285 s)
and ends with at least five steps of the finished mix. Lines, with three significant digits
(:class:`NucleoRollout`):

    nucleo seed 1 params. eta_factor 0 point 4 0 9. tau_n 9 3 6. t_freeze 1 point 4 4.
        expansion_factor 0 point 8 3 5.                              (one line in the data)
    q nucleo seed 1 param tau_n. a 9 3 6.
    nucleo seed 1 step 2 2. time 2 4 9. temperature 6 point 9 4 e 8. n_p_ratio 0 point 1 1 3.
        free_neutrons 0 point 0 1 7 8. hydrogen 0 point 8 1 4. helium 0 point 1 6 8.
        deuterium 8 point 4 9 e minus 5. stage bottleneck.           (one line in the data)
    q nucleo seed 1 step 0 n_p_ratio. a 0 point 6 4 8.              (weak equilibrium at 0.1 s)
    q nucleo seed 1 step 2 2 helium. a 0 point 1 6 8.
    q nucleo seed 1 step 2 2 lithium. a 6 point 7 9 e minus 1 1.
    q nucleo seed 1 step 2 2 next helium. a 0 point 2 0 1.          (one step later)
    q nucleo seed 1 step 2 2 next stage. a helium_forming.
    q nucleo seed 1 final helium. a 0 point 2 0 2.                  (how it ends)
    q nucleo seed 1 final bottleneck_time. a 1 9 8.

The ``params`` line matters: the four numbers are all that tells one seed from another, and
every other number of the run follows from them by the rules above. Without it a model could
only learn each rollout by heart; with it the rule is there to be learned.

Stages, in order: ``plasma`` (before freeze-out), ``freeze_out`` (the first step at or after it),
``decay``, ``bottleneck`` (the first step with ``T <= T_d``), ``helium_forming`` (free neutrons
still above 1e-6), ``done``.

Plausibility targets (what a run must look like; ``tests/test_cosmos_nucleo.py`` holds them):

    default run       final helium 0.20 .. 0.32                       this toy 0.263, real 0.245
                      n/p at freeze-out 1/6 .. 1/5                    0.193
                      n/p at the bottleneck 1/8 .. 1/6                0.157, textbook 1/7
                      bottleneck opens between 100 s and 300 s        156 s, real 200-300 s
                      D/H 2.5e-5, He3/H 1e-5, Li7/H 5e-10             by construction
                      hydrogen + helium above 0.999 of the mass       0.99994
    every seed        all six stages appear, in order: at least 3 steps ``plasma``, one
                      ``freeze_out``, one ``bottleneck``, at least 5 steps ``done``
                      n/p starts at 0.57 .. 0.67 (weak equilibrium at 0.1 s) and only falls
                      final helium 0.10 .. 0.50, hydrogen the rest    0.12 .. 0.49 over the box
                      bottleneck opens between 100 s and 250 s        104 s .. 240 s
                      more expansion, a longer neutron life or an earlier freeze-out: more helium
                      more baryons: less deuterium and helium-3, more lithium, slightly more helium

Gates. :func:`conserved` checks every step: nucleons (free or bound) sum to 1 within 1e-9 (so
baryon number and, with the unseen electrons, charge are kept), protons never decrease (beta
decay only makes protons), time advances and the temperature strictly falls, mass fractions lie
in [0, 1], free neutrons and n/p never rise, helium never falls, the stages come in order; over
the run the helium never exceeds ``2 r / (1 + r)`` of the bottleneck ratio (two neutrons per
nucleus) and ends within 5 percent of ``tau_n / (tau_n + tau_bind)`` of it, the summary repeats
the last step, and for the default parameters the final helium lies between 0.20 and 0.32.
:func:`check` parses ``nucleo seed <s> step <t> <key>``, ``... step <t> next <key>``,
``nucleo seed <s> final <key>`` and ``nucleo seed <s> param <key>`` (seed and step written as
:func:`haishool.truth.num` writes them), replays the seed and compares numbers within 5 percent
(words and parameters exactly; an answer that is not a finite number is wrong).

Reproducibility. The run is plain float64 arithmetic in a fixed order with ``math.sqrt``,
``math.exp``, ``math.log`` and ``**``: no numpy, no array sums, no sorting, so nothing depends
on a numpy version or on the processor's vector units. The random parameters come from
``random.Random(seed)`` (the Mersenne Twister, the same on every Python). What can differ
between machines is the last bit of ``exp``, ``log`` and ``**`` with a broken exponent; the
tests show that for seeds 1 to 9998 no stage decision and no printed digit that depends on them
sits within 1e-12 of a flip. What does sit exactly on an edge is computed without them: a grid
time equal to the freeze-out time (``expansion_factor`` 1 and ``t_freeze`` on the grid; that
step is ``freeze_out``) and the final lithium ratio ``5e-10 eta^2`` (a plain product).

Rules 7
=======

Everything above is round 6 and stays as it is: the version-5 models were trained on those
rollouts, and ``run(seed)``, ``random_params(rng)`` and ``lines(rollout(seed))`` still give them
bit for bit (``tests/test_round6_frozen.py``). A review found round-6 rules that are wrong or
arbitrary. ``rules=7`` applies the corrections; nothing changes without it.

    run(seed, rules=7)                                  our universe (the seed is only a label)
    run(seed, rules=7, **random_params(rng, rules=7))   a what-if universe near ours
    rollout(seed, rules=7)                              the run a seed stands for in the lines
    lines(rollout(seed, rules=7))                       its lines; check() and owns() judge them
    records(rules=7)                                    the unit of every key, the stage order
    simulation(rules=7)                                 the same as an object
    LESSONS, lesson_gate()                              the lessons (they exist under rules 7 only)
    handoff()                                           what the next level starts from

A rules-7 rollout carries ``params["rules"] = 7`` (a round-6 rollout has no such entry) and its
story says ``rules 7`` after the seed, so no prompt has two truths; the lessons use the word
``nucleo7`` (three significant digits in a story, four in a lesson):

    nucleo seed 1 rules 7 params. eta_factor 0 point 7 6 1. tau_n 8 8 7. neutrinos 2.
    q nucleo seed 1 rules 7 param tau_n. a 8 8 7.
    nucleo seed 1 rules 7 step 2 2. time 2 4 9. temperature 8 point 6 4 e 8. n_p_ratio 0 point 1 3 5.
        free_neutrons 0 point 0 6 8 7. hydrogen 0 point 8 3 2. helium 0 point 0 9 5 8.
        deuterium 0 point 0 0 2 3 7. helium3 4 point 4 1 e minus 6. lithium 1 point 2 9 e minus 1 0.
        stage bottleneck_opens.                                      (one line in the data)
    q nucleo seed 1 rules 7 step 2 2 helium. a 0 point 0 9 5 8.
    q nucleo seed 1 rules 7 step 2 2 next stage. a helium_forming.
    q nucleo seed 1 rules 7 final helium. a 0 point 2 3 3.
    q nucleo seed 1 rules 7 final bottleneck_time. a 2 3 3.
    nucleo rules 7 key deuterium. unit nuclei_per_hydrogen.
    nucleo rules 7 stages. equilibrium. freeze_out. decay. bottleneck_opens. helium_forming. done.
    q nucleo7 predict freeze_kt expansion_factor 0 point 9 8 tau_n 8 4 2 point 9. a 0 point 7 7 1.

The corrections, each with the real-world value it follows and what stays a toy. Published
numbers are quoted from memory; where a number is this module's own it says so.

1.  **One universe, not a box of them.** Round 6 drew four parameters over wide ranges, with the
    freeze-out time as a free number: final helium ran from 0.13 to 0.48 over the seeds, and only
    a few percent of them came near the real value (observed 0.245 +- 0.003, standard prediction
    0.247). Under rules 7 the defaults (:data:`DEFAULTS7`) are our universe and
    ``random_params(rng, rules=7)`` draws, in this order, ``eta_factor`` log-uniform 0.7 to 1.3
    (three digits), ``tau_n`` uniform 870 to 890 s (three digits), and ``neutrinos`` from
    ``(2, 3, 3, 3, 4)``. Neither the freeze-out time nor the expansion rate is drawn any more: the
    first follows from the weak coupling (4.), the second from the number of light neutrino
    families, ``expansion_factor = sqrt(1 + 7 (N - 3) / 43)`` (the energy density before
    electron-positron annihilation is proportional to 10.75 + 1.75 (N - 3); measured: N_eff
    2.99 +- 0.17). Measured over seeds 1 to 9998: helium 0.229 to 0.264, mean 0.247; with three
    families 0.244 to 0.251. ``run(rules=7)`` refuses ``eta_factor`` outside 0.5 to 2, ``tau_n``
    outside 800 to 960 s and ``neutrinos`` outside 1 to 6, and any run that would not pass its
    own gate (a far-off ``t1``, ``t_d``, ``kt_freeze`` or ``tau_bind``). Toy: the seeded
    universes are still what-if universes (two or four neutrino families, a neutron lifetime
    off by up to 1.3 percent); only the default run is meant as ours, and the chain should hand
    that one on.
2.  **The photons are heated by electron-positron annihilation.** Round 6 used one power law,
    ``T = 1e10 K / sqrt(t)``, right at freeze-out but 22 to 25 percent too cold where the nuclei
    form, from 150 s on (the photons are 28 to 34 percent hotter than it says).
    :func:`temperature` multiplies the early law by :func:`photon_heating`,
    ``1 + (B - 1) / (1 + (T0 / 1.6e9 K)^2.5)`` with ``B = (10.75 / 3.3626)^(1/4) = 1.337``, the
    ratio of the late law ``1.333e10 K / sqrt(t)`` to the early one (after Kolb and Turner 1990:
    ``t = 132 s (0.1 MeV / kT)^2`` after the annihilation). ``B`` follows from the degrees of
    freedom; 1.6e9 K and the exponent 2.5 are a fit of this module to the integrated
    entropy-conserving relation, within 1.4 percent of it from 0.05 s to 5000 s (the test
    integrates that relation). Toy: ``B`` and the fit keep their three-family values for 2 or 4
    families, and one ``expansion_factor`` is used before and after the annihilation.
    :func:`time_at` inverts the law by bisection; both events take their time from it.
3.  **The bottleneck opens at 9e8 K, at about 209 s.** Round 6 had 8e8 K reached at 156 s; with
    the corrected cooling law 8e8 K comes at 268 s (274 s by the integrated relation) and would
    leave too little helium (0.231).
    Rules 7 use ``T_d0 = 9e8 K`` (0.078 MeV; Weinberg 1977 has deuterium holding together at
    0.9e9 K, textbooks give 0.07 to 0.08 MeV and n/p = 1/7 there) with the same Saha shift,
    :func:`bottleneck_temperature`. Default run: 208.7 s (the integrated relation reaches 9e8 K
    at 213 s, Weinberg's account at 226 s), n/p 0.147 (1/6.8). Toy: 9e8 K and the
    0.787 MeV of the freeze-out are calibrated *together* so that the default run gives the
    standard helium; neither is derived from reaction rates.
4.  **The neutron lifetime sets the freeze-out too.** The weak rate goes as ``T^5 / tau_n`` and
    the expansion rate as ``expansion_factor T^2``, so :func:`freeze_kt` is
    ``0.787 MeV (expansion_factor tau_n / 878.4 s)^(1/3)``, with the measured lifetime
    878.4 +- 0.5 s as the default (the Particle Data Group average, from memory; storage
    experiments agree with it, beam experiments give about 888 s; round 6: 880 s, and a
    lifetime that only set the decay). The
    helium now answers the lifetime as the real one does: ``d ln Y / d ln tau_n`` is 0.72 to 0.73
    here (real 0.73, after Cyburt, Fields, Olive and Yeh 2016; round 6 gave 0.21), and one more
    neutrino family adds 0.0130 (real about 0.013), one fewer takes 0.0152. Toy: the freeze-out
    stays a sharp switch (the real ratio keeps slipping), and the helium answers the baryon
    density only half as strongly as the real one (0.0045 per e-fold of ``eta_factor``, real 0.0096).
5.  **No prompt with two truths.** A rules-7 story says ``rules 7`` after its seed and its
    rollout carries ``params["rules"] = 7``; the round-6 prompts keep their round-6 answers.
6.  **Judged at the printed precision.** With universes this alike a 5 percent tolerance would
    let one constant answer pass for most seeds. A rules-7 number is right when it rounds to
    the three digits the run prints (:func:`_check7`); words and parameters exactly.
7.  **Helium-3 and lithium are in the state line**, not only in questions
    (:data:`STATE_KEYS7`; at most 89 tokens with a 7-digit seed).
8.  **The trace laws only where they hold.** Power laws of this kind are fits near the measured
    density, quoted for a baryon-to-photon ratio of 4e-10 to 8e-10 (after Kneller and Steigman
    2004, from memory), ``eta_factor`` 0.66 to 1.31; rules 7 draw 0.7 to 1.3 instead of 0.3 to 3
    (below about 0.5 the real lithium curve turns up again, which no branch is invented for).
    The traces also answer the neutrino families and the neutron lifetime (after Cyburt et al.
    2016, exponents from memory, the lifetime taken relative to 878.4 s so that the default run
    keeps the central values):
    ``D/H = 2.5e-5 eta^-1.6 (N/3)^0.395 (tau_n/878.4)^0.41``,
    ``He3/H = 1e-5 eta^-0.6 (N/3)^0.14 (tau_n/878.4)^0.15``,
    ``Li7/H = 5e-10 eta^2 (N/3)^-0.284 (tau_n/878.4)^0.43``. Toy: still fits, not a reaction
    network, and the leading numbers are round 6's rounded ones (the same paper has, from
    memory, 2.58e-5 eta^-1.60, 1.00e-5 eta^-0.59 and 4.65e-10 eta^2.11; deuterium is measured
    at 2.53e-5). The lithium is mostly made as beryllium-7 and part of the helium-3 as tritium; both
    are counted as what they decay to. The lithium value is the standard prediction, about
    three times what is observed in old stars.
9.  **Lessons.** Every formula of the rules-7 run is a module function, and each is a lesson of
    :data:`LESSONS` (``nucleo7 predict <rule> ...``, 18 rules): the run calls the very function
    that answers the lesson (``helium_ceiling`` as a guard on its end state: the bound takes no
    part in a number). Round 6 had one lesson (``cooling_temperature`` in
    ``haishool.cosmos.predict``), which states the single power law.
10. **The summary names the numbers the account turns on:** ``freeze_time``,
    ``freeze_temperature``, ``freeze_n_p_ratio``, ``bottleneck_time``, ``bottleneck_temperature``,
    ``bottleneck_n_p_ratio``, ``expansion_factor``, ``traces``, ``end_time`` next to the five
    final abundances (:data:`FINAL_KEYS7`). ``peak_n_p_ratio`` is gone: it was the n/p of the
    first grid step, a property of where the grid starts.
11. **Stage words.** ``plasma`` is ``equilibrium`` (the universe is a plasma for 380 000 years;
    what ends at freeze-out is weak equilibrium) and ``bottleneck`` is ``bottleneck_opens``: the
    first step at or after the opening, where the share ``1 - exp(-(t - t_bottleneck) / tau_eff)``
    of the neutrons is already bound (0.75 at 249 s in the default run). The grid and the
    switching rules are those of round 6, so a step number means the same time in both.
12. **The hand-off is exported.** ``traces`` is the mass fraction of deuterium, helium-3 and
    lithium-7, ``h (2 D/H + 3 He3/H + 7 Li7/H)`` (6.02e-5 for the defaults), and ``end_time`` the
    3000 s at which the run ends. :func:`handoff` gives the printed numbers the chain starts from
    (helium 0.247, traces 6.02e-5, hydrogen the rest, 3000 s = 9.51e-5 years).
13. **Deuterium is the stepping stone.** In round 6 the deuterium only rose to its final trace.
    Under rules 7 :func:`deuterium_in_transit` adds ``4 * 0.002 * p (1 - p)`` deuterons per
    nucleon at binding progress ``p``, taken from the helium-4 not yet made (two deuterons are
    one helium-4, so the nucleon and proton budgets are unchanged and helium still never falls):
    D/H passes through 1.2e-3 to 2.4e-3 at ``bottleneck_opens`` and falls to its final trace.
    Toy: the height 0.002 is this module's number of the order real codes show (a deuterium
    mass fraction of 1e-3 to 1e-2 near 0.07 MeV), the shape ``p (1 - p)`` is invented, and on
    the 30-step grid the peak shows on one step, ``bottleneck_opens`` (32 to 172 times the final
    trace; the step after it still has 2 to 9 times the final trace).
14. **Units.** ``hydrogen``, ``helium`` and ``traces`` are mass fractions, ``deuterium``,
    ``helium3`` and ``lithium`` number ratios to hydrogen (the convention of the field); the
    names do not say so, :func:`records` does, one record a key.

Kept from round 6: the early law ``1e10 K / sqrt(t)`` (exact for 10.75 degrees of freedom:
0.997e10), weak equilibrium ``exp(-Q / kT)``, the sharp freeze-out near 0.79 MeV, free decay
afterwards, the Saha shift with ``ln(eta_factor)``, the binding time ``tau_bind = 30 s`` with the
share ``tau_n / (tau_n + tau_bind)`` (:func:`bound_share`), the bookkeeping per nucleon, the
neutron floor and the ``done`` threshold, the grid, and the closed-form end state. Still true of
rules 7: nothing is integrated, there are no reaction rates, and a run is thirty numbers.

Plausibility targets under rules 7 (``tests/test_cosmos_nucleo_rules7.py`` holds them):

    default run       final helium 0.245 .. 0.250                     0.2473, real 0.245 to 0.247
                      hydrogen + helium + traces = 1                  0.7526 + 0.2473 + 6.02e-5
                      n/p at freeze-out 1/6 .. 1/5                    0.193 at 1.21 s, 0.787 MeV
                      n/p at the bottleneck 1/7.5 .. 1/6.5            0.147, textbook 1/7
                      bottleneck opens between 180 s and 250 s        208.7 s
                      d ln Y / d ln tau_n 0.65 .. 0.80                0.72 to 0.73, real 0.73
                      one more neutrino family: helium +0.011 .. +0.015    +0.0130, real 0.013
    every seed        helium 0.225 .. 0.270; with three families 0.243 .. 0.252
                      bottleneck 185 s .. 240 s, freeze-out 1.0 s .. 1.5 s
                      all six stages in order, at least 7 steps ``equilibrium``, 5 steps ``done``
                      deuterium peaks while the helium forms and ends at its final ratio

Gates under rules 7. :func:`conserved` runs the budget checks of round 6 on every step and
recomputes the chain with the lesson functions: every temperature is :func:`temperature` of its
time, no nucleus exists before the bottleneck opens, the two events are in order and their
temperatures are the cooling law at their times, ``freeze_n_p_ratio`` is the equilibrium ratio
at the freeze-out temperature, ``bottleneck_n_p_ratio`` is that ratio less the decayed neutrons,
the helium stays under :func:`helium_ceiling`, the bound neutrons are :func:`bound_share` of
those at the bottleneck (to 1e-9), hydrogen, helium and traces sum to 1, and the default helium
lies in 0.245 to 0.250. ``run(seed, rules=7, ...)`` puts its rollout through this gate before it
hands it back and raises ``ValueError`` where it fails. Reproducibility is that of round 6: plain floats in a fixed order; the
heating uses a product and a square root instead of a power, the bisection only ``sqrt``; for
seeds 1 to 9998 no grid time is within 3e-4 of an event and no printed digit within 1e-9 of a flip.
"""

from __future__ import annotations

import math
import random
from functools import lru_cache

from haishool.cosmos import Rollout, close, query_lines, state_line, summary_lines
from haishool.evo import Input, LessonGate, Rule, sig
from haishool.truth import Line, Verdict, num, parse_num

SIM = "nucleo"

#: physical constants and fitted numbers, each with where it comes from. All are from memory, so a
#: checker can target them; the simulation only ever reads ``value`` and ``exponent``.
CONSTANTS: dict[str, dict[str, float | str]] = {
    "k_mev_per_k": {"value": 8.617333262e-11,
                    "notes": "Boltzmann constant 8.617333262e-5 eV/K (CODATA 2018), from memory"},
    "q_np_mev": {"value": 1.29333236,
                 "notes": "neutron-proton mass difference m_n - m_p = 1.29333236 MeV (CODATA 2018), from memory"},
    "b_d_mev": {"value": 2.224566, "notes": "deuteron binding energy 2.224566 MeV, from memory"},
    "d_h": {"value": 2.5e-5, "exponent": -1.6,
            "notes": "standard BBN deuterium yield D/H at eta = 6.1e-10 and its sensitivity D/H ~ eta^-1.6, from memory"},
    "he3_h": {"value": 1.0e-5, "exponent": -0.6,
              "notes": "standard BBN helium-3 yield He3/H and its sensitivity He3/H ~ eta^-0.6, from memory"},
    "li7_h": {"value": 5.0e-10, "exponent": 2.0,
              "notes": "standard BBN lithium-7 yield Li7/H (predicted, three times the observed) and its "
                       "sensitivity on the high-eta branch Li7/H ~ eta^2, from memory"},
}
_K = float(CONSTANTS["k_mev_per_k"]["value"])
_Q = float(CONSTANTS["q_np_mev"]["value"])
_B_D = float(CONSTANTS["b_d_mev"]["value"])

#: parameter -> default. ``random_params`` draws the ones in :data:`RANGES`; the rest stay fixed
#: unless passed to ``run``.
DEFAULTS: dict[str, float] = {"eta_factor": 1.0, "tau_n": 880.0, "t_freeze": 1.2, "expansion_factor": 1.0,
                              "t1": 1e10, "t_d": 8e8, "tau_bind": 30.0}
PARAMS: dict[str, str] = {
    "eta_factor": "baryon-to-photon ratio relative to the measured 6.1e-10",
    "tau_n": "neutron lifetime in s",
    "t_freeze": "freeze-out time of n/p in s for the standard expansion rate",
    "expansion_factor": "expansion rate relative to the standard one (scales T1 by 1/sqrt and the freeze-out "
                        "temperature by the cube root)",
    "t1": "temperature in K at 1 s for the standard expansion rate",
    "t_d": "temperature in K below which deuterium survives, for eta_factor 1",
    "tau_bind": "e-folding time in s for binding the free neutrons into nuclei after the bottleneck",
}
#: parameter -> (low, high, scale) for :func:`random_params`
RANGES: dict[str, tuple[float, float, str]] = {
    "eta_factor": (0.3, 3.0, "log"), "tau_n": (800.0, 960.0, "lin"),
    "t_freeze": (0.5, 2.0, "log"), "expansion_factor": (0.7, 1.4, "log"),
}

#: metric -> meaning and unit, per step
KEYS: dict[str, str] = {
    "time": "seconds since the start, on a log grid from 0.1 s to 3000 s",
    "temperature": "temperature in K, t1 / sqrt(time)",
    "n_p_ratio": "neutrons per proton, bound or free",
    "free_neutrons": "free neutrons as a fraction of all nucleons",
    "hydrogen": "mass fraction of hydrogen-1 (free protons)",
    "helium": "mass fraction of helium-4",
    "deuterium": "deuterium nuclei per hydrogen nucleus (number ratio)",
    "helium3": "helium-3 nuclei per hydrogen nucleus (number ratio)",
    "lithium": "lithium-7 nuclei per hydrogen nucleus (number ratio)",
    "stage": "one word: plasma, freeze_out, decay, bottleneck, helium_forming or done",
}
#: summary metric -> meaning and unit (asked as ``nucleo seed <s> final <key>``)
FINAL_KEYS: dict[str, str] = {
    "helium": "final mass fraction of helium-4",
    "hydrogen": "final mass fraction of hydrogen-1",
    "deuterium": "final deuterium per hydrogen (number ratio)",
    "helium3": "final helium-3 per hydrogen (number ratio)",
    "lithium": "final lithium-7 per hydrogen (number ratio)",
    "bottleneck_time": "seconds at which the deuterium bottleneck opens",
    "bottleneck_n_p_ratio": "neutrons per proton when the bottleneck opens",
    "freeze_time": "seconds at which n/p freezes out",
    "peak_n_p_ratio": "the highest n/p of the run (the first step, in weak equilibrium)",
}
#: the keys of the state line (all keys get query lines); at most 70 tokens with a 7-digit seed
STATE_KEYS = ["time", "temperature", "n_p_ratio", "free_neutrons", "hydrogen", "helium", "deuterium", "stage"]
STAGES = ("plasma", "freeze_out", "decay", "bottleneck", "helium_forming", "done")
#: free neutrons below this fraction of all nucleons are counted as decayed
_NEUTRON_FLOOR = 1e-10
#: ``helium_forming`` while the free neutrons are above this fraction, then ``done``
_DONE_BELOW = 1e-6


def _round_sig(x: float, sig: int) -> float:
    return float(f"{x:.{sig}g}")


def _power(x: float, exponent: float) -> float:
    """``x ** exponent``, with the square as a plain product: that is the same bits on every
    machine, and ``5e-10 eta^2`` often sits exactly on the edge of the third digit (``eta`` 1.5
    gives 1.125e-9), where the last bit of a library ``pow`` would decide the printed number."""
    return x * x if exponent == 2 else x ** exponent


#: the output times in s: 30 log-spaced steps from 0.1 s to 3000 s (``0.1 * 30000 ** (i / 29)``),
#: rounded to three digits and written out so that the grid is the same on every machine
TIMES: tuple[float, ...] = (0.1, 0.143, 0.204, 0.291, 0.415, 0.591, 0.844, 1.2, 1.72, 2.45,
                            3.5, 4.99, 7.12, 10.2, 14.5, 20.7, 29.5, 42.1, 60.1, 85.8,
                            122.0, 175.0, 249.0, 355.0, 507.0, 724.0, 1030.0, 1470.0, 2100.0, 3000.0)
#: how many questions of one seed :func:`generate` asks before it moves to the next seed
QUERIES_PER_SEED = 20


class NucleoRollout(Rollout):
    """A rollout whose numbers print with three significant digits."""

    def value(self, text: float | int | str) -> str:
        return text if isinstance(text, str) else num(text, sig=3)


def _rule_set(rules: object) -> int:
    """6 (round 6, frozen) or 7 (the corrected rules); anything else is no rule set of this level."""
    if isinstance(rules, bool) or rules not in (6, 7):
        raise ValueError(f"nucleo knows the rules 6 (round 6, frozen) and 7 (corrected), not {rules!r}")
    return int(rules)


def random_params(rng: random.Random, rules: int = 6) -> dict[str, float]:
    """The four physics parameters drawn from ``rng`` (see :data:`RANGES`), rounded to three digits.
    With ``rules=7``: the three of :data:`RANGES7` (a universe near ours, see "Rules 7")."""
    if _rule_set(rules) == 7:
        return _random_params7(rng)
    out = {}
    for key, (lo, hi, scale) in RANGES.items():
        x = math.exp(rng.uniform(math.log(lo), math.log(hi))) if scale == "log" else rng.uniform(lo, hi)
        out[key] = _round_sig(x, 3)
    return out


def run(seed: int, rules: int = 6, **params: float) -> Rollout:
    """One rollout of thirty steps; missing parameters take their :data:`DEFAULTS`. With
    ``rules=7`` the corrected rules run instead (parameters and defaults: :data:`DEFAULTS7`) and
    the rollout says so in ``params["rules"]``; the round-6 statements below are untouched."""
    if _rule_set(rules) == 7:
        return _run7(seed, **params)
    unknown = set(params) - set(DEFAULTS)
    if unknown:
        raise TypeError(f"unknown parameter(s) {sorted(unknown)}; known: {sorted(DEFAULTS)}")
    p = {k: float(params.get(k, v)) for k, v in DEFAULTS.items()}
    if any(not math.isfinite(v) or v <= 0 for v in p.values()):
        raise ValueError("every nucleo parameter is a positive number")
    f, eta, tau_n, tau_bind = p["expansion_factor"], p["eta_factor"], p["tau_n"], p["tau_bind"]

    # the cooling law and the two events: freeze-out, then the bottleneck
    t1 = p["t1"] / math.sqrt(f)
    kt_freeze = _K * p["t1"] / math.sqrt(p["t_freeze"]) * f ** (1 / 3)
    t_freeze = p["t_freeze"] * f ** (-5 / 3)  # exactly p["t_freeze"] for the standard expansion
    ratio_freeze = math.exp(-_Q / kt_freeze)
    n_freeze = ratio_freeze / (1 + ratio_freeze)
    saha = _B_D / (_K * p["t_d"]) - math.log(eta)
    if saha <= 0:
        raise ValueError("eta_factor is so large that the deuterium bottleneck has no temperature")
    temp_d = _B_D / (_K * saha)
    t_bottleneck = _power(t1 / temp_d, 2)
    if not t_freeze < t_bottleneck:
        raise ValueError("n/p must freeze out before the deuterium bottleneck opens")
    n_bottleneck = n_freeze * math.exp(-(t_bottleneck - t_freeze) / tau_n)

    # the end state in closed form: what is bound, what decays meanwhile, how it is shared out
    tau_eff = 1 / (1 / tau_bind + 1 / tau_n)
    bound = n_bottleneck * tau_n / (tau_n + tau_bind)
    decayed = n_bottleneck - bound
    protons_total = 1 - n_bottleneck + decayed
    r_d, r_he3, r_li = (float(CONSTANTS[c]["value"]) * _power(eta, float(CONSTANTS[c]["exponent"]))
                        for c in ("d_h", "he3_h", "li7_h"))
    h_final = (protons_total - bound) / (1 + r_he3 - r_li)
    d_final, he3_final, li_final = r_d * h_final, r_he3 * h_final, r_li * h_final
    he4_final = (bound - d_final - he3_final - 4 * li_final) / 2

    # plain floats, one step at a time, every sum in a fixed order; no numpy, so no version of it matters
    steps: list[dict[str, float | int | str]] = []
    seen: set[str] = set()
    for ti in TIMES:
        temp = t1 / math.sqrt(ti)
        progress = 0.0
        if ti < t_freeze:
            ratio_eq = math.exp(-_Q / (_K * temp))
            n_free = ratio_eq / (1 + ratio_eq)
        elif ti < t_bottleneck:
            n_free = n_freeze * math.exp(-(ti - t_freeze) / tau_n)
        else:
            progress = 1 - math.exp(-(ti - t_bottleneck) / tau_eff)
            n_free = n_bottleneck * (1 - progress)
            # below the floor the run is at its end state: the last free neutrons are shared out
            # exactly as the end state shares them, so the proton count never steps back
            if n_free < _NEUTRON_FLOOR:
                progress, n_free = 1.0, 0.0
        he4, d, he3, li = progress * he4_final, progress * d_final, progress * he3_final, progress * li_final
        h = 1 - (n_free + 4 * he4 + 2 * d + 3 * he3 + 7 * li)
        neutrons = n_free + 2 * he4 + d + he3 + 4 * li
        if ti < t_freeze:
            stage = "plasma"
        elif "freeze_out" not in seen:
            stage = "freeze_out"
        elif ti < t_bottleneck:
            stage = "decay"
        elif "bottleneck" not in seen:
            stage = "bottleneck"
        elif n_free > _DONE_BELOW:
            stage = "helium_forming"
        else:
            stage = "done"
        seen.add(stage)
        # at the end state the trace ratios are the fitted ratios themselves (d / h is r_d there,
        # up to the last bit of h; the ratio itself has no such noise)
        traces = (r_d, r_he3, r_li) if progress == 1.0 else (d / h, he3 / h, li / h)
        steps.append({"time": ti, "temperature": temp, "n_p_ratio": neutrons / (1 - neutrons),
                      "free_neutrons": n_free, "hydrogen": h, "helium": 4 * he4,
                      "deuterium": traces[0], "helium3": traces[1], "lithium": traces[2], "stage": stage})
    last = steps[-1]
    summary = {"helium": last["helium"], "hydrogen": last["hydrogen"], "deuterium": last["deuterium"],
               "helium3": last["helium3"], "lithium": last["lithium"],
               "bottleneck_time": t_bottleneck, "bottleneck_n_p_ratio": n_bottleneck / (1 - n_bottleneck),
               "freeze_time": t_freeze, "peak_n_p_ratio": max(float(s["n_p_ratio"]) for s in steps)}
    return NucleoRollout(SIM, int(seed), p, steps, summary)


def _is_seed(seed: object) -> bool:
    return isinstance(seed, int) and not isinstance(seed, bool) and seed >= 0


@lru_cache(maxsize=4096)
def _replay(seed: int) -> Rollout:
    """The gate's own copy of a seed's rollout; never handed out, so nobody can change it."""
    return run(seed, **random_params(random.Random(seed)))


@lru_cache(maxsize=4096)
def _replay7(seed: int) -> Rollout:
    """The gate's own copy of a seed's rules-7 rollout."""
    return run(seed, rules=7, **random_params(random.Random(seed), rules=7))


def rollout(seed: int, rules: int = 6) -> Rollout:
    """The rollout a seed stands for in the lines and in :func:`check`: its random parameters
    (with ``rules=7``: its rules-7 parameters and the corrected rules).
    A fresh copy on every call; seeds are whole numbers from 0 (a prompt cannot name another)."""
    if not _is_seed(seed):
        raise ValueError(f"a nucleo seed is a whole number from 0, not {seed!r}")
    if _rule_set(rules) == 7:
        r = _replay7(seed)
        return NucleoRollout7(r.sim, r.seed, dict(r.params), [dict(s) for s in r.steps], dict(r.summary))
    r = _replay(seed)
    return NucleoRollout(r.sim, r.seed, dict(r.params), [dict(s) for s in r.steps], dict(r.summary))


def _fractions(step: dict) -> tuple[float, float, float, float, float, float]:
    """Mass (= nucleon) fractions of free neutrons, H-1, He-4, D, He-3, Li-7 at one step."""
    h = float(step["hydrogen"])
    return (float(step["free_neutrons"]), h, float(step["helium"]),
            2 * float(step["deuterium"]) * h, 3 * float(step["helium3"]) * h, 7 * float(step["lithium"]) * h)


def _protons(fractions: tuple[float, ...]) -> float:
    n, h, he4, d, he3, li = fractions
    return h + he4 / 2 + d / 2 + 2 * he3 / 3 + 3 * li / 7


def conserved(r: Rollout) -> Verdict:
    """The simulation's own gate (see the module docstring for the list)."""
    if r.sim != SIM or not r.steps:
        return Verdict(False, None, "not a nucleo rollout")
    if _carried_rules(r) == 7:
        return _conserved7(r)
    prev_temp, prev_protons, prev_time = math.inf, -1.0, -math.inf
    prev_stage, prev_n, prev_ratio, prev_he = -1, math.inf, math.inf, -1.0
    for i, s in enumerate(r.steps):
        if set(KEYS) - set(s):
            return Verdict(False, None, f"step {i} lacks {sorted(set(KEYS) - set(s))}")
        fr = _fractions(s)
        if any(not (0 <= x <= 1) for x in fr):
            return Verdict(False, None, f"step {i}: a mass fraction is outside [0, 1]")
        total = math.fsum(fr)
        if abs(total - 1) > 1e-9:
            return Verdict(False, None, f"step {i}: nucleons not conserved, fractions sum to {total!r}")
        protons = _protons(fr)
        if protons < prev_protons - 1e-12:
            return Verdict(False, None, f"step {i}: protons decreased from {prev_protons!r} to {protons!r}")
        if abs(float(s["n_p_ratio"]) - (1 - protons) / protons) > 1e-9:
            return Verdict(False, None, f"step {i}: n_p_ratio does not match the nucleon budget")
        if not float(s["temperature"]) < prev_temp:
            return Verdict(False, None, f"step {i}: temperature did not fall")
        if not float(s["time"]) > prev_time:
            return Verdict(False, None, f"step {i}: time did not advance")
        if s["stage"] not in STAGES or STAGES.index(s["stage"]) < prev_stage:
            return Verdict(False, None, f"step {i}: stage {s['stage']!r} out of order")
        if float(s["free_neutrons"]) > prev_n + 1e-12 or float(s["n_p_ratio"]) > prev_ratio + 1e-12:
            return Verdict(False, None, f"step {i}: free neutrons or n/p increased")
        if float(s["helium"]) < prev_he - 1e-12:
            return Verdict(False, None, f"step {i}: helium decreased")
        prev_temp, prev_protons, prev_time = float(s["temperature"]), protons, float(s["time"])
        prev_stage, prev_n = STAGES.index(s["stage"]), float(s["free_neutrons"])
        prev_ratio, prev_he = float(s["n_p_ratio"]), float(s["helium"])
    stages = [s["stage"] for s in r.steps]
    if stages.count("freeze_out") != 1 or stages.count("bottleneck") != 1 or stages[-1] != "done":
        return Verdict(False, None, "the run must freeze out once, open the bottleneck once and end done")
    last = r.steps[-1]
    if set(FINAL_KEYS) - set(r.summary):
        return Verdict(False, None, f"the summary lacks {sorted(set(FINAL_KEYS) - set(r.summary))}")
    if any(r.summary[k] != last[k] for k in ("helium", "hydrogen", "deuterium", "helium3", "lithium")):
        return Verdict(False, None, "the summary does not repeat the last step")
    if r.summary["peak_n_p_ratio"] != max(float(s["n_p_ratio"]) for s in r.steps):
        return Verdict(False, None, "peak_n_p_ratio is not the highest n/p of the run")
    y, ratio = float(r.summary["helium"]), float(r.summary["bottleneck_n_p_ratio"])
    most = 2 * ratio / (1 + ratio)
    if y > most * (1 + 1e-9):
        return Verdict(False, None,
                       f"final helium {y!r} needs more neutrons than the bottleneck ratio {ratio!r} leaves")
    tau_n, tau_bind = (float(r.params.get(k, DEFAULTS[k])) for k in ("tau_n", "tau_bind"))
    if not close(y, most * tau_n / (tau_n + tau_bind), rel=0.05):
        return Verdict(False, None, f"final helium {y!r} is not 2 r / (1 + r) of the bottleneck ratio {ratio!r}, "
                                    "less the neutrons that decay while the nuclei form")
    if all(float(r.params.get(k, math.nan)) == v for k, v in DEFAULTS.items()) and not 0.20 <= y <= 0.32:
        return Verdict(False, None, f"default parameters gave helium {y!r}, outside 0.20 .. 0.32")
    return Verdict(True)


def params_line(r: Rollout) -> Line:
    """``nucleo seed 1 params. eta_factor 0 point 4 0 9. tau_n 9 3 6. ...``: the drawn parameters."""
    fields = " ".join(f"{k} {r.value(r.params[k])}." for k in RANGES)
    return Line(f"{SIM} seed {num(r.seed)} params. {fields}", topic=SIM, kind="record")


def param_lines(r: Rollout) -> list[Line]:
    """``q nucleo seed 1 param tau_n. a 9 3 6.`` for each drawn parameter."""
    return [Line(f"{SIM} seed {num(r.seed)} param {k}", r.value(r.params[k]), SIM, "fact") for k in RANGES]


def _parse(prompt: str) -> tuple[int, str, int | None, str] | None:
    """``nucleo seed 7 step 1 5 next helium`` -> ``(7, "next", 15, "helium")``; ``None`` if not ours."""
    words = prompt.strip().rstrip(".").split()
    if words[:1] == ["q"]:
        words = words[1:]
    if words[:2] != [SIM, "seed"]:
        return None
    i = 2
    while i < len(words) and words[i].isdigit():
        i += 1
    seed = parse_num(words[2:i])
    if not isinstance(seed, int) or num(seed).split() != words[2:i] or i >= len(words):
        return None
    if words[i] == "final":
        key = " ".join(words[i + 1:])
        return (seed, "final", None, key) if key in FINAL_KEYS else None
    if words[i] == "param":
        key = " ".join(words[i + 1:])
        return (seed, "param", None, key) if key in RANGES else None
    if words[i] != "step":
        return None
    j = i + 1
    while j < len(words) and words[j].isdigit():
        j += 1
    step = parse_num(words[i + 1:j])
    rest = words[j:]
    if not isinstance(step, int) or num(step).split() != words[i + 1:j]:
        return None
    if len(rest) == 2 and rest[0] == "next" and rest[1] in KEYS:
        return seed, "next", step, rest[1]
    if len(rest) == 1 and rest[0] in KEYS:
        return seed, "step", step, rest[0]
    return None


def owns(prompt: str) -> bool:
    """True for a round-6 story question (``nucleo seed 7 step 3 helium``) and for a rules-7 one
    (``nucleo seed 7 rules 7 step 3 helium``); the lessons have their own gate, :data:`LESSONS`."""
    return _parse(prompt) is not None or _parse7(prompt) is not None


def _number(answer: str) -> float | None:
    """The finite number an answer spells, else ``None`` (``1 e 9 9 9`` overflows: not a number)."""
    got = parse_num(answer)
    try:
        x = None if got is None else float(got)
    except OverflowError:
        return None
    return x if x is not None and math.isfinite(x) else None


def check(prompt: str, answer: str) -> Verdict:
    """Replay the seed and compare: numbers within 5 percent (no absolute floor, lithium sits at
    5e-10), words and parameters exactly. An answer that overflows to infinity (``1 e 9 9 9``)
    is not a number. A prompt that says ``rules 7`` after its seed is replayed under the corrected
    rules and judged at the printed precision instead (:func:`_check7`)."""
    q = _parse(prompt)
    if q is None:
        q7 = _parse7(prompt)
        return Verdict(False, None, "not my question") if q7 is None else _check7(q7, answer)
    seed, kind, step, key = q
    r = _replay(seed)
    if kind == "param":
        ok = _number(answer) == r.params[key]
        return Verdict(ok, r.value(r.params[key]), "" if ok else "wrong parameter")
    if kind == "final":
        v = r.summary[key]
    else:
        at = step + (1 if kind == "next" else 0)
        if step < 0 or at >= len(r.steps):
            return Verdict(False, None, f"a nucleo rollout has steps 0 to {len(r.steps) - 1}")
        v = r.steps[at][key]
    expected = r.value(v)
    answer = answer.strip()
    if isinstance(v, str):
        return Verdict(answer == v, expected, "" if answer == v else "wrong word")
    got = _number(answer)
    if got is None:
        return Verdict(False, expected, "not a number")
    ok = close(got, float(v), rel=0.05, abs_=0.0)
    return Verdict(ok, expected, "" if ok else "off by more than 5 percent")


def lines(r: Rollout, keys: list[str] | None = None, horizon: int = 1, rules: int | None = None) -> list[Line]:
    """Parameter, state, query and summary lines of a rollout, which must be :func:`rollout` of
    its seed so that :func:`check` agrees with every answer. ``next`` is always one step later
    (``horizon`` 1): the prompt has no word for another distance, so :func:`check` could not
    judge it. The rule set is the one the rollout carries (``params["rules"]`` is 7, or absent
    for round 6); ``rules`` may name it and must then agree."""
    if r.sim != SIM:
        raise ValueError(f"not a nucleo rollout: {r.sim!r}")
    if horizon != 1:
        raise ValueError("nucleo's next is one step later; check() knows no other horizon")
    if rules is not None and _rule_set(rules) != _carried_rules(r):
        raise ValueError(f"the rollout was run under rules {_carried_rules(r)}, not {rules!r}")
    if _carried_rules(r) == 7:
        return _lines7(r, keys)
    if not _is_seed(r.seed) or r.params != {**DEFAULTS, **random_params(random.Random(r.seed))}:
        raise ValueError(f"lines() wants rollout({r.seed}): check() replays the seed's random parameters")
    mine = _replay(r.seed)
    if r.steps != mine.steps or r.summary != mine.summary:
        raise ValueError(f"lines() wants rollout({r.seed}) unchanged: its steps differ from the replay")
    out: list[Line] = [params_line(r), *param_lines(r)]
    for t in range(len(r.steps)):
        out.append(state_line(r, t, STATE_KEYS))
        out += query_lines(r, t, keys or list(KEYS), horizon)
    out += summary_lines(r)
    return out


def generate(rng: random.Random, n: int, rules: int = 6) -> list[Line]:
    """``n`` question lines (no records) over seeds drawn from ``rng`` (1 to 9998, each at most
    once): :data:`QUERIES_PER_SEED` questions of a seed, more only when ``n`` needs them.
    With ``rules=7`` the questions are those of the rules-7 rollouts."""
    if n <= 0:
        return []
    k = min(9998, -(-n // QUERIES_PER_SEED))
    out: list[Line] = []
    for seed in rng.sample(range(1, 9999), k):
        questions = [ln for ln in lines(rollout(seed, rules)) if ln.kind != "record"]
        out += rng.sample(questions, min(len(questions), -(-n // k), n - len(out)))
    return out


# ---------------------------------------------------------------------------------------------
# Rules 7: the corrected first hour (section "Rules 7" of the module docstring). Nothing above
# this line computes differently than in round 6: run(), random_params(), rollout(), conserved(),
# check(), owns(), lines() and generate() hand over to the functions below when the rule set is 7.
# ---------------------------------------------------------------------------------------------

#: the word of the rules-7 lessons (``nucleo7 predict freeze_kt ...``). A rules-7 *story* keeps
#: the word ``nucleo`` and says ``rules 7`` after its seed.
SIM7 = "nucleo7"

#: the numbers of the corrected rules, each with where it comes from. ``from memory`` marks a
#: published number quoted from memory, ``toy`` or ``fit`` one that is this module's own.
CONSTANTS7: dict[str, dict[str, float | str]] = {
    "tau_n_s": {"value": 878.4,
                "notes": "mean life of the free neutron, 878.4 +- 0.5 s (Particle Data Group average, 2022), from "
                         "memory; the storage experiments (material bottles and magnetic traps) agree with it, the "
                         "beam experiments give about 888 s, a discrepancy that is still open"},
    "g_before": {"value": 10.75,
                 "notes": "relativistic degrees of freedom before electron-positron annihilation: photons 2, "
                          "electrons and positrons 7/8 * 4, three neutrino families 7/8 * 6 (after Kolb and "
                          "Turner 1990), from memory"},
    "g_after": {"value": 3.3626,
                "notes": "after it: photons 2 and three neutrino families at (4/11)^(1/3) of the photon "
                         "temperature, 2 + 7/8 * 6 * (4/11)^(4/3) (after Kolb and Turner 1990), from memory"},
    "t_annihilation_k": {"value": 1.6e9, "exponent": 2.5,
                         "notes": "where and how sharply the photon temperature leaves the early law: a fit of "
                                  "this module to the integrated entropy-conserving relation, not a measured number"},
    "kt_freeze_mev": {"value": 0.787,
                      "notes": "k T at which n/p freezes for the standard expansion and neutron lifetime "
                               "(textbooks: 0.7 to 0.8 MeV); calibrated together with t_bottleneck_k so that the "
                               "default run gives the standard helium 0.247: a toy number"},
    "t_bottleneck_k": {"value": 9.0e8,
                       "notes": "temperature below which deuterium holds together at the measured baryon density "
                                "(Weinberg 1977 has 0.9e9 K, textbooks 0.07 to 0.08 MeV); calibrated together "
                                "with kt_freeze_mev: a toy number"},
    "tau_bind_s": {"value": 30.0, "notes": "e-folding time for binding the neutrons after the bottleneck: a toy number"},
    "d_peak": {"value": 2.0e-3,
               "notes": "deuterons per nucleon in transit at the height of helium formation: a toy number of the "
                        "order real codes show (a deuterium mass fraction of 1e-3 to 1e-2)"},
    "d_h_side": {"neutrinos": 0.395, "tau_n": 0.41,
                 "notes": "D/H ~ (N_nu / 3)^0.395 (tau_n / tau_0)^0.41 (after Cyburt, Fields, Olive and Yeh 2016), "
                          "exponents from memory"},
    "he3_h_side": {"neutrinos": 0.14, "tau_n": 0.15,
                   "notes": "He3/H ~ (N_nu / 3)^0.14 (tau_n / tau_0)^0.15 (after Cyburt, Fields, Olive and Yeh "
                            "2016), exponents from memory"},
    "li7_h_side": {"neutrinos": -0.284, "tau_n": 0.43,
                   "notes": "Li7/H ~ (N_nu / 3)^-0.284 (tau_n / tau_0)^0.43 (after Cyburt, Fields, Olive and Yeh "
                            "2016), exponents from memory"},
}
TAU_N = float(CONSTANTS7["tau_n_s"]["value"])
T1_K = 1e10  # K at 1 s before the heating, for the standard expansion (the exact value for g* 10.75 is 0.997e10)
#: the photon temperature after electron-positron annihilation over the early power law, at the
#: same time: (g_before / g_after)^(1/4) = 1.33716. Two square roots: the same bits on every machine.
HEATING = math.sqrt(math.sqrt(float(CONSTANTS7["g_before"]["value"]) / float(CONSTANTS7["g_after"]["value"])))
T_ANNIHILATION = float(CONSTANTS7["t_annihilation_k"]["value"])
KT_FREEZE = float(CONSTANTS7["kt_freeze_mev"]["value"])
T_BOTTLENECK = float(CONSTANTS7["t_bottleneck_k"]["value"])
TAU_BIND = float(CONSTANTS7["tau_bind_s"]["value"])
D_PEAK = float(CONSTANTS7["d_peak"]["value"])
SECONDS_PER_YEAR = 31557600.0  # a Julian year of 365.25 days

#: rules-7 parameter -> default: our universe. ``random_params(rng, rules=7)`` draws the three of
#: :data:`RANGES7`; the rest stay fixed unless passed to ``run``.
DEFAULTS7: dict[str, float] = {"eta_factor": 1.0, "tau_n": TAU_N, "neutrinos": 3, "kt_freeze": KT_FREEZE,
                               "t1": T1_K, "t_d": T_BOTTLENECK, "tau_bind": TAU_BIND}
PARAMS7: dict[str, str] = {
    "eta_factor": "baryon-to-photon ratio relative to the measured 6.1e-10",
    "tau_n": "neutron lifetime in s; it also sets the freeze-out (the same weak coupling)",
    "neutrinos": "number of light neutrino families; the expansion rate follows from it (expansion_factor)",
    "kt_freeze": "k T in MeV at which n/p freezes for 3 neutrino families and tau_n 878.4 s",
    "t1": "temperature in K at 1 s before the electron-positron heating, for 3 neutrino families",
    "t_d": "temperature in K below which deuterium survives, for eta_factor 1",
    "tau_bind": "e-folding time in s for binding the free neutrons into nuclei after the bottleneck",
}
#: parameter -> (low, high, how) for ``random_params(rng, rules=7)``: a universe near ours
RANGES7: dict[str, tuple[float, float, str]] = {
    "eta_factor": (0.7, 1.3, "log"), "tau_n": (870.0, 890.0, "lin"), "neutrinos": (2, 4, "choice"),
}
#: the neutrino draw: three families in three of five universes
NEUTRINO_DRAW: tuple[int, ...] = (2, 3, 3, 3, 4)
#: what ``run(seed, rules=7)`` accepts at all: outside, the fitted trace laws and the scaling of
#: the freeze-out no longer describe anything
LIMITS7: dict[str, tuple[float, float]] = {"eta_factor": (0.5, 2.0), "tau_n": (800.0, 960.0), "neutrinos": (1.0, 6.0)}

STAGES7 = ("equilibrium", "freeze_out", "decay", "bottleneck_opens", "helium_forming", "done")
#: metric -> meaning and unit, per step, under rules 7
KEYS7: dict[str, str] = {
    "time": "seconds since the start, on a log grid from 0.1 s to 3000 s",
    "temperature": "photon temperature in K, temperature(time, expansion_factor)",
    "n_p_ratio": "neutrons per proton, bound or free",
    "free_neutrons": "free neutrons as a fraction of all nucleons",
    "hydrogen": "mass fraction of hydrogen-1 (free protons)",
    "helium": "mass fraction of helium-4",
    "deuterium": "deuterium nuclei per hydrogen nucleus (number ratio); it peaks while the helium forms",
    "helium3": "helium-3 nuclei per hydrogen nucleus (number ratio)",
    "lithium": "lithium-7 nuclei per hydrogen nucleus (number ratio)",
    "stage": "one word: equilibrium, freeze_out, decay, bottleneck_opens, helium_forming or done",
}
#: summary metric -> meaning and unit (asked as ``nucleo seed <s> rules 7 final <key>``)
FINAL_KEYS7: dict[str, str] = {
    "helium": "final mass fraction of helium-4",
    "hydrogen": "final mass fraction of hydrogen-1",
    "deuterium": "final deuterium per hydrogen (number ratio)",
    "helium3": "final helium-3 per hydrogen (number ratio)",
    "lithium": "final lithium-7 per hydrogen (number ratio)",
    "traces": "final mass fraction of deuterium, helium-3 and lithium-7 together",
    "freeze_time": "seconds at which n/p freezes out",
    "freeze_temperature": "temperature in K at which n/p freezes out",
    "freeze_n_p_ratio": "neutrons per proton at freeze-out",
    "bottleneck_time": "seconds at which the deuterium bottleneck opens",
    "bottleneck_temperature": "temperature in K at which the deuterium bottleneck opens",
    "bottleneck_n_p_ratio": "neutrons per proton when the bottleneck opens",
    "expansion_factor": "expansion rate relative to the standard one, from the number of neutrino families",
    "end_time": "seconds at which the run ends and hands on (3000)",
}
#: every key is in the state line under rules 7: at most 89 tokens with a 7-digit seed
STATE_KEYS7 = list(KEYS7)
#: key -> unit, one dense word each (the model sees only names; :func:`records` says the units)
UNITS7: dict[str, str] = {
    "time": "seconds", "temperature": "kelvin", "n_p_ratio": "neutrons_per_proton",
    "free_neutrons": "fraction_of_nucleons", "hydrogen": "mass_fraction", "helium": "mass_fraction",
    "deuterium": "nuclei_per_hydrogen", "helium3": "nuclei_per_hydrogen", "lithium": "nuclei_per_hydrogen",
    "stage": "word",
    "traces": "mass_fraction", "freeze_time": "seconds", "freeze_temperature": "kelvin",
    "freeze_n_p_ratio": "neutrons_per_proton", "bottleneck_time": "seconds", "bottleneck_temperature": "kelvin",
    "bottleneck_n_p_ratio": "neutrons_per_proton", "expansion_factor": "times_the_standard_rate",
    "end_time": "seconds",
    "eta_factor": "times_the_measured_density", "tau_n": "seconds", "neutrinos": "families",
}


# the rules: every function here is called by the simulation and answers a lesson (LESSONS)

def expansion_factor(neutrinos: float) -> float:
    """Expansion rate relative to the standard one for ``neutrinos`` light neutrino families: the
    energy density before electron-positron annihilation is proportional to 10.75 + 1.75 (N - 3),
    and the rate to its square root, ``sqrt(1 + 7 (N - 3) / 43)`` (0.915, 1, 1.078 for 2, 3, 4)."""
    return math.sqrt(1 + 7 * (neutrinos - 3) / 43)


def radiation_temperature(time: float, expansion_factor: float, t1: float = T1_K) -> float:
    """Kelvin at ``time`` seconds by the early power law alone: ``1e10 / sqrt(expansion_factor time)``."""
    return t1 / math.sqrt(expansion_factor * time)


def photon_heating(temperature: float) -> float:
    """The factor by which electron-positron annihilation has heated the photons above the early
    power law, where that law says ``temperature``: 1 while it is hot, 1.337 in the end,
    ``1 + 0.3372 / (1 + (T / 1.6e9 K)^2.5)`` in between (a fit, see :data:`CONSTANTS7`)."""
    x = temperature / T_ANNIHILATION
    return 1 + (HEATING - 1) / (1 + x * x * math.sqrt(x))  # x^2.5 with a product and a root: no pow


def temperature(time: float, expansion_factor: float, t1: float = T1_K) -> float:
    """Photon temperature in K at ``time`` seconds: the early power law times the heating."""
    t0 = radiation_temperature(time, expansion_factor, t1)
    return t0 * photon_heating(t0)


def time_at(temperature_k: float, expansion_factor: float, t1: float = T1_K) -> float:
    """The first moment in s at which :func:`temperature` is at or below ``temperature_k``: the
    inverse of the cooling law by bisection on the geometric mean between 1e-4 s and 1e7 s
    (plain floats in a fixed order, at most 200 halvings)."""
    lo, hi = 1e-4, 1e7
    if not temperature(hi, expansion_factor, t1) <= temperature_k < temperature(lo, expansion_factor, t1):
        raise ValueError(f"no time between 1e-4 s and 1e7 s has the temperature {temperature_k!r} K")
    for _ in range(200):
        mid = math.sqrt(lo * hi)
        if not lo < mid < hi:
            break
        if temperature(mid, expansion_factor, t1) > temperature_k:
            lo = mid
        else:
            hi = mid
    return hi


def thermal_energy(temperature: float) -> float:
    """``k T`` in MeV for a temperature in K."""
    return _K * temperature


def equilibrium_ratio(kt_mev: float) -> float:
    """Neutrons per proton in weak equilibrium at ``k T`` in MeV: ``exp(-1.293 / kT)``."""
    return math.exp(-_Q / kt_mev)


def neutrons_per_nucleon(n_p_ratio: float) -> float:
    """The neutron share of all nucleons for ``n_p_ratio`` neutrons per proton: ``r / (1 + r)``."""
    return n_p_ratio / (1 + n_p_ratio)


def freeze_kt(expansion_factor: float, tau_n: float, kt_freeze: float = KT_FREEZE) -> float:
    """``k T`` in MeV at which n/p freezes out. The weak rate goes as ``T^5 / tau_n`` and the
    expansion rate as ``expansion_factor T^2``; they cross at
    ``0.787 MeV (expansion_factor tau_n / 878.4 s)^(1/3)``."""
    return kt_freeze * (expansion_factor * tau_n / TAU_N) ** (1 / 3)


def neutron_decay(free_neutrons: float, seconds: float, tau_n: float) -> float:
    """Free neutrons left after ``seconds`` of beta decay: ``n exp(-seconds / tau_n)``."""
    return free_neutrons * math.exp(-seconds / tau_n)


def bottleneck_temperature(eta_factor: float, t_d: float = T_BOTTLENECK) -> float:
    """Kelvin below which deuterium holds together: by the Saha equation (without its slowly
    varying prefactor) ``B_D / kT = B_D / (k 9e8 K) - ln(eta_factor)``, so a denser universe
    opens the bottleneck a little hotter: ``9e8 K / (1 - ln(eta_factor) / 28.68)``."""
    saha = _B_D / (_K * t_d)  # B_D / kT at eta_factor 1: 28.68
    shifted = saha - math.log(eta_factor)
    if shifted <= 0:
        raise ValueError("eta_factor is so large that the deuterium bottleneck has no temperature")
    return t_d / (shifted / saha)  # exactly t_d for eta_factor 1


def helium_ceiling(n_p_ratio: float) -> float:
    """The most helium-4 by mass that ``n_p_ratio`` allows, two neutrons a nucleus: ``2 r / (1 + r)``."""
    return 2 * n_p_ratio / (1 + n_p_ratio)


def bound_share(tau_n: float, tau_bind: float = TAU_BIND) -> float:
    """Share of the neutrons at the bottleneck that end up in nuclei; the rest decay while the
    nuclei form: ``tau_n / (tau_n + tau_bind)``."""
    return tau_n / (tau_n + tau_bind)


def bound_progress(seconds: float, tau_n: float, tau_bind: float = TAU_BIND) -> float:
    """How far the binding has come ``seconds`` after the bottleneck opened, 0 to 1:
    ``1 - exp(-seconds (1 / tau_bind + 1 / tau_n))``."""
    tau_eff = 1 / (1 / tau_bind + 1 / tau_n)
    return 1 - math.exp(-seconds / tau_eff)


def deuterium_in_transit(progress: float) -> float:
    """Deuterons per nucleon on their way to helium-4 at binding ``progress``, on top of the
    final trace: ``4 * 0.002 * progress * (1 - progress)``, at most 0.002 (a toy number)."""
    return 4 * D_PEAK * progress * (1 - progress)


def _trace_ratio(name: str, eta_factor: float, neutrinos: float, tau_n: float) -> float:
    main, side = CONSTANTS[name], CONSTANTS7[name + "_side"]
    return (float(main["value"]) * _power(eta_factor, float(main["exponent"]))
            * (neutrinos / 3) ** float(side["neutrinos"]) * (tau_n / TAU_N) ** float(side["tau_n"]))


def deuterium_ratio(eta_factor: float, neutrinos: float = 3, tau_n: float = TAU_N) -> float:
    """Final deuterium per hydrogen: ``2.5e-5 eta^-1.6 (N / 3)^0.395 (tau_n / 878.4)^0.41``."""
    return _trace_ratio("d_h", eta_factor, neutrinos, tau_n)


def helium3_ratio(eta_factor: float, neutrinos: float = 3, tau_n: float = TAU_N) -> float:
    """Final helium-3 per hydrogen: ``1e-5 eta^-0.6 (N / 3)^0.14 (tau_n / 878.4)^0.15``."""
    return _trace_ratio("he3_h", eta_factor, neutrinos, tau_n)


def lithium_ratio(eta_factor: float, neutrinos: float = 3, tau_n: float = TAU_N) -> float:
    """Final lithium-7 per hydrogen: ``5e-10 eta^2 (N / 3)^-0.284 (tau_n / 878.4)^0.43``."""
    return _trace_ratio("li7_h", eta_factor, neutrinos, tau_n)


def trace_mass(hydrogen: float, deuterium: float, helium3: float, lithium: float) -> float:
    """Mass fraction of deuterium, helium-3 and lithium-7 together, from the hydrogen mass
    fraction and the three number ratios to hydrogen: ``h (2 d + 3 he3 + 7 li)``."""
    return hydrogen * (2 * deuterium + 3 * helium3 + 7 * lithium)


RULES: dict[str, Rule] = {
    "expansion_factor": Rule(
        (Input("neutrinos", 1, 6, places=1),), expansion_factor,
        "expansion rate over the standard one with 3 neutrino families. sqrt of the sum 1 plus 7 over 4 3 times "
        "the difference neutrinos minus 3"),
    "radiation_temperature": Rule(
        (Input("time", 0.1, 3000, places=1), Input("expansion_factor", 0.9, 1.1, places=3)), radiation_temperature,
        "kelvin by the early power law before electrons and positrons heat the photons. 1 e 1 0 over sqrt of "
        "expansion_factor times time. time in seconds"),
    "photon_heating": Rule(
        (Input("temperature", 1e8, 1e10, places=-6),), photon_heating,
        "factor by which annihilating electrons and positrons have heated the photons. 1 plus 0 point 3 3 7 2 "
        "over 1 plus x to the power 2 point 5. x is temperature over 1 point 6 e 9. temperature is the kelvin "
        "of the early power law. a fit to the integrated relation"),
    "temperature": Rule(
        (Input("time", 0.1, 3000, places=1), Input("expansion_factor", 0.9, 1.1, places=3)), temperature,
        "photon temperature in kelvin. radiation_temperature times photon_heating of it. time in seconds"),
    "thermal_energy": Rule(
        (Input("temperature", 1e8, 4e10, places=-6),), thermal_energy,
        "k times temperature in mev. k is 8 point 6 1 7 3 e minus 1 1 mev per kelvin"),
    "equilibrium_ratio": Rule(
        (Input("kt_mev", 0.3, 3, places=3),), equilibrium_ratio,
        "neutrons per proton in weak equilibrium. exp of minus 1 point 2 9 3 3 over kt_mev. 1 point 2 9 3 3 mev "
        "is the mass a neutron has more than a proton"),
    "neutrons_per_nucleon": Rule(
        (Input("n_p_ratio", 0.05, 1, places=3),), neutrons_per_nucleon,
        "share of the nucleons that are neutrons. n_p_ratio over 1 plus n_p_ratio"),
    "freeze_kt": Rule(
        (Input("expansion_factor", 0.9, 1.1, places=3), Input("tau_n", 800, 960, places=1)), freeze_kt,
        "k times temperature in mev at which neutrons per proton freeze out. 0 point 7 8 7 times cube root of "
        "expansion_factor times tau_n over 8 7 8 point 4. tau_n in seconds. 0 point 7 8 7 is calibrated"),
    "neutron_decay": Rule(
        (Input("free_neutrons", 0.05, 0.3, places=3), Input("seconds", 0, 400, places=1),
         Input("tau_n", 800, 960, places=1)), neutron_decay,
        "free neutrons left after seconds of decay. free_neutrons times exp of minus seconds over tau_n"),
    "bottleneck_temperature": Rule(
        (Input("eta_factor", 0.5, 2, places=3),), bottleneck_temperature,
        "kelvin below which deuterium holds together. 9 e 8 over the difference 1 minus natural log of "
        "eta_factor over 2 8 point 6 8. a denser universe opens the bottleneck hotter. 9 e 8 is calibrated"),
    "helium_ceiling": Rule(
        (Input("n_p_ratio", 0.05, 0.4, places=3),), helium_ceiling,
        "most helium by mass that the neutrons allow. 2 times n_p_ratio over 1 plus n_p_ratio"),
    "bound_share": Rule(
        (Input("tau_n", 800, 960, places=1), Input("tau_bind", 5, 100, places=1)), bound_share,
        "share of the neutrons at the bottleneck that end in nuclei. tau_n over tau_n plus tau_bind. the rest "
        "decay meanwhile"),
    "bound_progress": Rule(
        (Input("seconds", 0, 300, places=1), Input("tau_n", 800, 960, places=1), Input("tau_bind", 5, 100, places=1)),
        bound_progress,
        "how far the binding has come seconds after the bottleneck opened. 1 minus exp of minus seconds times "
        "the sum 1 over tau_bind plus 1 over tau_n"),
    "deuterium_in_transit": Rule(
        (Input("progress", 0, 1, places=3),), deuterium_in_transit,
        "deuterons per nucleon on their way to helium on top of the final trace. 4 times 0 point 0 0 2 times "
        "progress times 1 minus progress. a toy number"),
    "deuterium_ratio": Rule(
        (Input("eta_factor", 0.7, 1.3, places=3), Input("neutrinos", 2, 4, integer=True),
         Input("tau_n", 860, 900, places=1)), deuterium_ratio,
        "final deuterium nuclei per hydrogen. 2 point 5 e minus 5 times eta_factor to the power minus 1 point 6 "
        "times neutrinos over 3 to the power 0 point 3 9 5 times tau_n over 8 7 8 point 4 to the power 0 point 4 1"),
    "helium3_ratio": Rule(
        (Input("eta_factor", 0.7, 1.3, places=3), Input("neutrinos", 2, 4, integer=True),
         Input("tau_n", 860, 900, places=1)), helium3_ratio,
        "final helium3 nuclei per hydrogen. 1 e minus 5 times eta_factor to the power minus 0 point 6 "
        "times neutrinos over 3 to the power 0 point 1 4 times tau_n over 8 7 8 point 4 to the power 0 point 1 5"),
    "lithium_ratio": Rule(
        (Input("eta_factor", 0.7, 1.3, places=3), Input("neutrinos", 2, 4, integer=True),
         Input("tau_n", 860, 900, places=1)), lithium_ratio,
        "final lithium nuclei per hydrogen. 5 e minus 1 0 times eta_factor squared times neutrinos over 3 to "
        "the power minus 0 point 2 8 4 times tau_n over 8 7 8 point 4 to the power 0 point 4 3"),
    "trace_mass": Rule(
        (Input("hydrogen", 0.7, 0.8, places=3), Input("deuterium", 1e-5, 6e-5, places=7),
         Input("helium3", 5e-6, 2e-5, places=7), Input("lithium", 1e-10, 1.2e-9, places=12)), trace_mass,
        "mass fraction of deuterium helium3 and lithium together. hydrogen times the sum 2 times deuterium plus "
        "3 times helium3 plus 7 times lithium. the three are nuclei per hydrogen"),
}
#: the gate of the rules-7 lessons: ``nucleo7 predict <rule> <input> <value> ...``, topic ``predict_nucleo7``
LESSONS = LessonGate(SIM7, RULES)


def lesson_gate() -> LessonGate:
    return LESSONS


def _value7(value: float | int | str) -> str:
    """A rules-7 number as it stands in a line: three significant digits (``sig`` then ``num``)."""
    if isinstance(value, str):
        return value
    if isinstance(value, int):
        return num(value)
    return num(sig(float(value)))


class NucleoRollout7(NucleoRollout):
    """A rules-7 rollout; its numbers print with three significant digits, rounded before they
    are written (so 1234.5 is ``1 2 3 0``, never a run of decimals)."""

    def value(self, text: float | int | str) -> str:
        return _value7(text)


def _carried_rules(r: Rollout) -> int:
    """The rule set a rollout was run under: 7 when its parameters say so, else round 6."""
    return 7 if isinstance(r.params, dict) and r.params.get("rules") == 7 else 6


def _random_params7(rng: random.Random) -> dict[str, float]:
    """A universe near ours, in this order: ``eta_factor`` log-uniform 0.7 to 1.3 and ``tau_n``
    uniform 870 to 890 s, both rounded to three digits, then the number of neutrino families
    from :data:`NEUTRINO_DRAW`."""
    lo, hi, _ = RANGES7["eta_factor"]
    eta = _round_sig(math.exp(rng.uniform(math.log(lo), math.log(hi))), 3)
    lo, hi, _ = RANGES7["tau_n"]
    tau_n = _round_sig(rng.uniform(lo, hi), 3)
    return {"eta_factor": eta, "tau_n": tau_n, "neutrinos": rng.choice(NEUTRINO_DRAW)}


def _run7(seed: int, **params: float) -> Rollout:
    """The corrected first hour on the grid of round 6 (:data:`TIMES`); missing parameters take
    their :data:`DEFAULTS7`. Every formula is one of the lesson functions above."""
    unknown = set(params) - set(DEFAULTS7)
    if unknown:
        raise TypeError(f"unknown rules-7 parameter(s) {sorted(unknown)}; known: {sorted(DEFAULTS7)} (the expansion "
                        "rate follows from neutrinos and the freeze-out from tau_n: neither is a free parameter)")
    p = {k: float(params.get(k, v)) for k, v in DEFAULTS7.items()}
    if any(not math.isfinite(v) or v <= 0 for v in p.values()):
        raise ValueError("every nucleo parameter is a positive number")
    for key, (lo, hi) in LIMITS7.items():
        if not lo <= p[key] <= hi:
            raise ValueError(f"rules 7 hold for {key} from {lo:g} to {hi:g}, not {p[key]!r}")
    eta, tau_n, tau_bind, families = p["eta_factor"], p["tau_n"], p["tau_bind"], p["neutrinos"]
    f = expansion_factor(families)

    # the two events: n/p freezes where the weak rate crosses the expansion rate, the bottleneck
    # opens where deuterium holds; both times are read off the one cooling law
    kt_freeze = freeze_kt(f, tau_n, p["kt_freeze"])
    temp_freeze = kt_freeze / _K
    t_freeze = time_at(temp_freeze, f, p["t1"])
    ratio_freeze = equilibrium_ratio(kt_freeze)
    n_freeze = neutrons_per_nucleon(ratio_freeze)
    temp_d = bottleneck_temperature(eta, p["t_d"])
    t_bottleneck = time_at(temp_d, f, p["t1"])
    if not t_freeze < t_bottleneck:
        raise ValueError("n/p must freeze out before the deuterium bottleneck opens")
    n_bottleneck = neutron_decay(n_freeze, t_bottleneck - t_freeze, tau_n)

    # the end state in closed form, as in round 6
    bound = n_bottleneck * bound_share(tau_n, tau_bind)
    decayed = n_bottleneck - bound
    protons_total = 1 - n_bottleneck + decayed
    r_d, r_he3, r_li = (deuterium_ratio(eta, families, tau_n), helium3_ratio(eta, families, tau_n),
                        lithium_ratio(eta, families, tau_n))
    h_final = (protons_total - bound) / (1 + r_he3 - r_li)
    d_final, he3_final, li_final = r_d * h_final, r_he3 * h_final, r_li * h_final
    he4_final = (bound - d_final - he3_final - 4 * li_final) / 2
    if not he4_final > 0:
        raise ValueError("so few neutrons are left at the bottleneck that the fitted traces alone would use them up")
    if not 4 * he4_final <= helium_ceiling(n_bottleneck / (1 - n_bottleneck)):
        raise ValueError("the end state holds more helium than the neutrons at the bottleneck allow")

    steps: list[dict[str, float | int | str]] = []
    seen: set[str] = set()
    for ti in TIMES:
        temp = temperature(ti, f, p["t1"])
        progress = 0.0
        if ti < t_freeze:
            n_free = neutrons_per_nucleon(equilibrium_ratio(thermal_energy(temp)))
        elif ti < t_bottleneck:
            n_free = neutron_decay(n_freeze, ti - t_freeze, tau_n)
        else:
            progress = bound_progress(ti - t_bottleneck, tau_n, tau_bind)
            n_free = n_bottleneck * (1 - progress)
            if n_free < _NEUTRON_FLOOR:
                progress, n_free = 1.0, 0.0
        # deuterium is the stepping stone: two deuterons in transit are one helium-4 not yet made,
        # so the nucleon and the proton budget are those of round 6
        transit = deuterium_in_transit(progress)
        he4 = progress * he4_final - transit / 2
        d = progress * d_final + transit
        he3, li = progress * he3_final, progress * li_final
        h = 1 - (n_free + 4 * he4 + 2 * d + 3 * he3 + 7 * li)
        neutrons = n_free + 2 * he4 + d + he3 + 4 * li
        if ti < t_freeze:
            stage = "equilibrium"
        elif "freeze_out" not in seen:
            stage = "freeze_out"
        elif ti < t_bottleneck:
            stage = "decay"
        elif "bottleneck_opens" not in seen:
            stage = "bottleneck_opens"
        elif n_free > _DONE_BELOW:
            stage = "helium_forming"
        else:
            stage = "done"
        seen.add(stage)
        traces = (r_d, r_he3, r_li) if progress == 1.0 else (d / h, he3 / h, li / h)
        steps.append({"time": ti, "temperature": temp, "n_p_ratio": neutrons / (1 - neutrons),
                      "free_neutrons": n_free, "hydrogen": h, "helium": 4 * he4,
                      "deuterium": traces[0], "helium3": traces[1], "lithium": traces[2], "stage": stage})
    last = steps[-1]
    summary = {"helium": last["helium"], "hydrogen": last["hydrogen"], "deuterium": last["deuterium"],
               "helium3": last["helium3"], "lithium": last["lithium"],
               "traces": trace_mass(last["hydrogen"], last["deuterium"], last["helium3"], last["lithium"]),
               "freeze_time": t_freeze, "freeze_temperature": temp_freeze, "freeze_n_p_ratio": ratio_freeze,
               "bottleneck_time": t_bottleneck, "bottleneck_temperature": temp_d,
               "bottleneck_n_p_ratio": n_bottleneck / (1 - n_bottleneck),
               "expansion_factor": f, "end_time": last["time"]}
    carried: dict[str, float | int | str] = dict(p)
    if families.is_integer():
        carried["neutrinos"] = int(families)
    carried["rules"] = 7
    out = NucleoRollout7(SIM, int(seed), carried, steps, summary)
    # a run that comes back has passed its own gate: with a far-off t1, t_d, kt_freeze or tau_bind
    # the thirty steps cannot show the six stages (or the budget breaks), and that is no rollout
    verdict = _conserved7(out)
    if not verdict.ok:
        raise ValueError(f"these parameters give no first hour that the thirty steps can show: {verdict.reason}")
    return out


def _conserved7(r: Rollout) -> Verdict:
    """The gate of a rules-7 rollout: the budget checks of round 6 on every step, and the chain of
    the two events recomputed with the lesson functions."""
    try:
        tau_n, tau_bind, t1 = (float(r.params[k]) for k in ("tau_n", "tau_bind", "t1"))
        f = expansion_factor(float(r.params["neutrinos"]))
    except (KeyError, TypeError, ValueError):
        return Verdict(False, None, "a rules-7 rollout names tau_n, tau_bind, t1 and neutrinos in its parameters")
    prev_temp, prev_protons, prev_time = math.inf, -1.0, -math.inf
    prev_stage, prev_n, prev_ratio, prev_he = -1, math.inf, math.inf, -1.0
    for i, s in enumerate(r.steps):
        if set(KEYS7) - set(s):
            return Verdict(False, None, f"step {i} lacks {sorted(set(KEYS7) - set(s))}")
        fr = _fractions(s)
        if any(not (0 <= x <= 1) for x in fr):
            return Verdict(False, None, f"step {i}: a mass fraction is outside [0, 1]")
        total = math.fsum(fr)
        if abs(total - 1) > 1e-9:
            return Verdict(False, None, f"step {i}: nucleons not conserved, fractions sum to {total!r}")
        protons = _protons(fr)
        if protons < prev_protons - 1e-12:
            return Verdict(False, None, f"step {i}: protons decreased from {prev_protons!r} to {protons!r}")
        if abs(float(s["n_p_ratio"]) - (1 - protons) / protons) > 1e-9:
            return Verdict(False, None, f"step {i}: n_p_ratio does not match the nucleon budget")
        if not float(s["time"]) > prev_time:
            return Verdict(False, None, f"step {i}: time did not advance")
        if not float(s["temperature"]) < prev_temp:
            return Verdict(False, None, f"step {i}: temperature did not fall")
        if not close(float(s["temperature"]), temperature(float(s["time"]), f, t1), rel=1e-12, abs_=0.0):
            return Verdict(False, None, f"step {i}: temperature is not the cooling law at this time")
        if s["stage"] not in STAGES7 or STAGES7.index(s["stage"]) < prev_stage:
            return Verdict(False, None, f"step {i}: stage {s['stage']!r} out of order")
        if float(s["free_neutrons"]) > prev_n + 1e-12 or float(s["n_p_ratio"]) > prev_ratio + 1e-12:
            return Verdict(False, None, f"step {i}: free neutrons or n/p increased")
        if float(s["helium"]) < prev_he - 1e-12:
            return Verdict(False, None, f"step {i}: helium decreased")
        if STAGES7.index(s["stage"]) < 3 and any(float(s[k]) != 0 for k in ("helium", "deuterium", "helium3", "lithium")):
            return Verdict(False, None, f"step {i}: a nucleus before the bottleneck opened")
        prev_temp, prev_protons, prev_time = float(s["temperature"]), protons, float(s["time"])
        prev_stage, prev_n = STAGES7.index(s["stage"]), float(s["free_neutrons"])
        prev_ratio, prev_he = float(s["n_p_ratio"]), float(s["helium"])
    stages = [s["stage"] for s in r.steps]
    if stages.count("freeze_out") != 1 or stages.count("bottleneck_opens") != 1 or stages[-1] != "done":
        return Verdict(False, None, "the run must freeze out once, open the bottleneck once and end done")
    last = r.steps[-1]
    if set(FINAL_KEYS7) - set(r.summary):
        return Verdict(False, None, f"the summary lacks {sorted(set(FINAL_KEYS7) - set(r.summary))}")
    if any(r.summary[k] != last[k] for k in ("helium", "hydrogen", "deuterium", "helium3", "lithium")):
        return Verdict(False, None, "the summary does not repeat the last step")
    y, h = float(r.summary["helium"]), float(r.summary["hydrogen"])
    d, he3, li = (float(r.summary[k]) for k in ("deuterium", "helium3", "lithium"))
    traces = float(r.summary["traces"])
    if traces != trace_mass(h, d, he3, li) or not abs(h + y + traces - 1) <= 1e-9:
        return Verdict(False, None, "traces is not the mass of deuterium, helium-3 and lithium that completes the mix")
    if r.summary["end_time"] != last["time"] or r.summary["expansion_factor"] != f:
        return Verdict(False, None, "end_time or expansion_factor does not match the run")
    t_f, temp_f, ratio_f = (float(r.summary[k]) for k in ("freeze_time", "freeze_temperature", "freeze_n_p_ratio"))
    t_b, temp_b, ratio_b = (float(r.summary[k]) for k in ("bottleneck_time", "bottleneck_temperature",
                                                          "bottleneck_n_p_ratio"))
    if not (0 < t_f < t_b and temp_f > temp_b > 0 and ratio_f >= ratio_b > 0):
        return Verdict(False, None, "the bottleneck must open after the freeze-out, colder and with no more neutrons")
    if not (close(temp_f, temperature(t_f, f, t1), rel=1e-9, abs_=0.0)
            and close(temp_b, temperature(t_b, f, t1), rel=1e-9, abs_=0.0)):
        return Verdict(False, None, "an event's temperature is not the cooling law at its time")
    if not close(ratio_f, equilibrium_ratio(thermal_energy(temp_f)), rel=1e-9, abs_=0.0):
        return Verdict(False, None, "freeze_n_p_ratio is not the equilibrium ratio at the freeze-out temperature")
    n_b = neutrons_per_nucleon(ratio_b)
    if not close(n_b, neutron_decay(neutrons_per_nucleon(ratio_f), t_b - t_f, tau_n), rel=1e-9, abs_=0.0):
        return Verdict(False, None, "bottleneck_n_p_ratio is not the freeze-out ratio less the neutrons that decayed")
    if y > helium_ceiling(ratio_b) * (1 + 1e-9):
        return Verdict(False, None,
                       f"final helium {y!r} needs more neutrons than the bottleneck ratio {ratio_b!r} leaves")
    if not abs(y / 2 + h * (d + he3 + 4 * li) - n_b * bound_share(tau_n, tau_bind)) <= 1e-9:
        return Verdict(False, None, f"the neutrons bound in nuclei are not bound_share of those at the bottleneck "
                                    f"(ratio {ratio_b!r}), less the neutrons that decay while the nuclei form")
    if all(float(r.params.get(k, math.nan)) == v for k, v in DEFAULTS7.items()) and not 0.245 <= y <= 0.250:
        return Verdict(False, None, f"default parameters gave helium {y!r}, outside 0.245 .. 0.250")
    return Verdict(True)


def _parse7(prompt: str) -> tuple[int, str, int | None, str] | None:
    """``nucleo seed 7 rules 7 step 1 5 next helium`` -> ``(7, "next", 15, "helium")``; ``None``
    for anything else, a round-6 prompt included."""
    words = prompt.strip().rstrip(".").split()
    if words[:1] == ["q"]:
        words = words[1:]
    if words[:2] != [SIM, "seed"]:
        return None
    i = 2
    while i < len(words) and words[i].isdigit():
        i += 1
    seed = parse_num(words[2:i])
    if not isinstance(seed, int) or num(seed).split() != words[2:i] or words[i:i + 2] != ["rules", "7"]:
        return None
    i += 2
    if i >= len(words):
        return None
    if words[i] == "final":
        key = " ".join(words[i + 1:])
        return (seed, "final", None, key) if key in FINAL_KEYS7 else None
    if words[i] == "param":
        key = " ".join(words[i + 1:])
        return (seed, "param", None, key) if key in RANGES7 else None
    if words[i] != "step":
        return None
    j = i + 1
    while j < len(words) and words[j].isdigit():
        j += 1
    step = parse_num(words[i + 1:j])
    rest = words[j:]
    if not isinstance(step, int) or num(step).split() != words[i + 1:j]:
        return None
    if len(rest) == 2 and rest[0] == "next" and rest[1] in KEYS7:
        return seed, "next", step, rest[1]
    if len(rest) == 1 and rest[0] in KEYS7:
        return seed, "step", step, rest[0]
    return None


def _check7(q: tuple[int, str, int | None, str], answer: str) -> Verdict:
    """Replay the seed under rules 7 and compare at the printed precision: a number is right when
    it rounds to the three digits the run prints (the universes are too alike for a 5 percent
    gate: one constant would pass for most seeds); words and parameters exactly."""
    seed, kind, step, key = q
    r = _replay7(seed)
    if kind == "param":
        ok = _number(answer) == r.params[key]
        return Verdict(ok, _value7(r.params[key]), "" if ok else "wrong parameter")
    if kind == "final":
        v = r.summary[key]
    else:
        at = step + (1 if kind == "next" else 0)
        if step < 0 or at >= len(r.steps):
            return Verdict(False, None, f"a nucleo rollout has steps 0 to {len(r.steps) - 1}")
        v = r.steps[at][key]
    expected = _value7(v)
    answer = answer.strip()
    if isinstance(v, str):
        return Verdict(answer == v, expected, "" if answer == v else "wrong word")
    got = _number(answer)
    if got is None:
        return Verdict(False, expected, "not a number")
    ok = _value7(got) == expected
    return Verdict(ok, expected, "" if ok else "not the three digits the run gives")


def _lines7(r: Rollout, keys: list[str] | None = None) -> list[Line]:
    """The lines of ``rollout(seed, rules=7)``: the same kinds as round 6, with ``rules 7`` after
    the seed, every key in the state line, and the rules-7 parameters and summary."""
    if not _is_seed(r.seed) or r.params != {**DEFAULTS7, **random_params(random.Random(r.seed), rules=7), "rules": 7}:
        raise ValueError(f"lines() wants rollout({r.seed}, rules=7): check() replays the seed's random parameters")
    mine = _replay7(r.seed)
    if r.steps != mine.steps or r.summary != mine.summary:
        raise ValueError(f"lines() wants rollout({r.seed}, rules=7) unchanged: its steps differ from the replay")
    head = f"{SIM} seed {num(r.seed)} rules 7"
    asked = keys or list(KEYS7)

    def line(prompt: str, answer: str = "", kind: str = "record") -> Line:
        return Line(prompt, answer, SIM, kind, {"rules": 7})

    out = [line(f"{head} params. " + " ".join(f"{k} {_value7(r.params[k])}." for k in RANGES7))]
    out += [line(f"{head} param {k}", _value7(r.params[k]), "fact") for k in RANGES7]
    for t, step in enumerate(r.steps):
        out.append(line(f"{head} step {num(t)}. " + " ".join(f"{k} {_value7(step[k])}." for k in STATE_KEYS7)))
        out += [line(f"{head} step {num(t)} {k}", _value7(step[k]), "fact") for k in asked]
        if t + 1 < len(r.steps):
            out += [line(f"{head} step {num(t)} next {k}", _value7(r.steps[t + 1][k]), "calc") for k in asked]
    out += [line(f"{head} final {k}", _value7(v), "calc") for k, v in r.summary.items()]
    return out


def records(rules: int = 6) -> list[Line]:
    """Round 6 has no table. Rules 7 says in what unit each key counts, because the names hide two
    kinds (hydrogen and helium are mass fractions, the traces number ratios to hydrogen), and
    names the stages in their order."""
    if _rule_set(rules) == 6:
        return []
    head = f"{SIM} rules 7"
    rows = [f"{head} key {k}. unit {UNITS7[k]}." for k in KEYS7]
    rows += [f"{head} final key {k}. unit {UNITS7[k]}." for k in FINAL_KEYS7]
    rows += [f"{head} param key {k}. unit {UNITS7[k]}." for k in RANGES7]
    rows.append(f"{head} stages. " + " ".join(f"{s}." for s in STAGES7))
    return [Line(row, topic=SIM, kind="record", meta={"rules": 7}) for row in rows]


def handoff(r: Rollout | None = None) -> dict[str, float]:
    """What the next level starts from, in the numbers a rules-7 run prints: ``helium`` and
    ``traces`` (mass fractions) to three digits, ``hydrogen`` as the rest (so the three sum to
    1), and the clock: ``end_time`` in s and ``end_time_years``. Without a rollout: our universe,
    ``run(0, rules=7)`` (one universe, many clouds)."""
    r = run(0, rules=7) if r is None else r
    if r.sim != SIM or _carried_rules(r) != 7:
        raise ValueError("handoff() wants a rules-7 nucleo rollout")
    helium, traces, seconds = sig(float(r.summary["helium"])), sig(float(r.summary["traces"])), float(r.summary["end_time"])
    return {"hydrogen": 1 - helium - traces, "helium": helium, "traces": traces,
            "end_time": seconds, "end_time_years": sig(seconds / SECONDS_PER_YEAR)}


class Nucleo:
    """The level-1 simulation as a :class:`haishool.cosmos.Simulation`; with ``topic``,
    ``records`` and ``generate`` it is also a :class:`haishool.truth.Gate` for the feedback loop.

    ``Nucleo()`` is round 6. ``Nucleo(rules=7)`` runs, draws and writes under the corrected rules
    unless a call names another rule set; ``conserved``, ``check`` and ``owns`` judge both kinds
    whatever the object was made with (a rollout and a prompt say which rules they are of)."""

    sim = SIM
    topic = SIM
    KEYS = KEYS
    FINAL_KEYS = FINAL_KEYS
    DEFAULTS = DEFAULTS
    rules = 6

    def __init__(self, rules: int = 6) -> None:
        if _rule_set(rules) == 7:
            self.rules = 7
            self.KEYS, self.FINAL_KEYS, self.DEFAULTS = KEYS7, FINAL_KEYS7, DEFAULTS7

    def _rules(self, rules: int | None) -> int:
        return self.rules if rules is None else _rule_set(rules)

    def run(self, seed: int, rules: int | None = None, **params: float) -> Rollout:
        return run(seed, self._rules(rules), **params)

    def rollout(self, seed: int, rules: int | None = None) -> Rollout:
        return rollout(seed, self._rules(rules))

    def random_params(self, rng: random.Random, rules: int | None = None) -> dict[str, float]:
        return random_params(rng, self._rules(rules))

    def conserved(self, r: Rollout) -> Verdict:
        return conserved(r)

    def check(self, prompt: str, answer: str) -> Verdict:
        return check(prompt, answer)

    def owns(self, prompt: str) -> bool:
        return owns(prompt)

    def lines(self, r: Rollout, keys: list[str] | None = None, horizon: int = 1,
              rules: int | None = None) -> list[Line]:
        return lines(r, keys, horizon, rules)

    def records(self, rules: int | None = None) -> list[Line]:
        """Round 6 has no table: every nucleo line comes from a rollout. Rules 7 says what unit
        each key has (:func:`records`)."""
        return records(self._rules(rules))

    def generate(self, rng: random.Random, n: int, rules: int | None = None) -> list[Line]:
        return generate(rng, n, self._rules(rules))

    def lesson_gate(self) -> LessonGate:
        return LESSONS


def simulation(rules: int = 6) -> Nucleo:
    """The level as an object: round 6 by default, the corrected rules with ``rules=7``."""
    return Nucleo(rules)
