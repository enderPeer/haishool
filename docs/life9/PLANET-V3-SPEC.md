# life9 v3: emergence only (design contract)

Owner's instruction (3 October 2026): let everything emerge. Write only the feedback-loop formulas that lead to
emergent behaviour. Nothing is pre-installed; the simulation creates it.

v3 lives in `haishool/life9/v3/`, with tests in `tests/life9/v3/`. It reuses version 2's verified physics modules
(`haishool/life9/planet/`: constants, chain, formation, materials, items, globe, climate, biosphere, crafting
physics), extends them where noted, and never edits them in ways that break their tests.

**How to read this record.** Each section keeps the design as written on 3 October 2026. Its **As built** part
says what the code does: `formation3`, `patch` and `optics`, `brain3`, `senses3`, `manipulate`, `body`, the world
module `world3`, the viewer `view3` and the command line. At the end of the fix stage (3 October) 352 tests passed.
Where the design and the As built part disagree, the As built part is the current truth and gives the reason.
- Section 11 lists the findings and the choices still open.
- Section 12 says what `world3` wires together.
- Section 13 records the owner's four decisions of 4 October 2026: the patch scale, a nitrogen share, an air-volume
  gene and the experiment protocol. The As built parts they touch say so.
- [PROTOCOL.md](PROTOCOL.md) is the binding experiment protocol. Every reported result follows it.

## 0. What is supplied, and nothing else

| Supplied (the irreducible base) | Why it cannot emerge inside this simulation |
|---|---|
| Laws of physics and chemistry, and their measured constants and material data (CODATA, NIST-JANAF, CRC, mineral and tissue properties) | They are the "starting math" of any universe |
| The world7 chain's starting conditions (star, elements, planet, era values) and v3's derivations from them | The chain is the simulation's own history |
| **One feedback loop: bodies that replicate with heritable random variation, and die when physics says so** (no fuel, too dry, too hot or cold, too damaged) | Evolution needs it. The origin of replicators is world7's earlier rungs, not v3's |
| A body's physical abilities: move and turn, open the mouth on what it touches, grip, release, push or strike with force, rub, press two held things together, set something down, vibrate the air, divide. All are motor outputs with physical costs | A body in physics can do these. What it does with them is not supplied |

Everything else must emerge or be absent:
- what to eat and drink, and when
- fleeing, hunting, calls and their meanings
- body size, lifespan, maturity, litter size, endothermy, brain size
- what counts as "good" for learning
- tool use, fire, cooking, crafting, cooperation and kinship
- where founders live

v3 removes every version-2 item in `world.WORLD_PROVENANCE`:
- `one_action` with named actions
- `time_budget` and background grazing
- `background_drink`
- `approach` (the engine walking an actor to its target)
- nearest-target `strike`
- kin-by-pedigree `share`
- the innate forage bias and `default_outputs` as a supplied policy
- `founder_habitat`
- the fixed `outcome` reward
- the allometric lifespan, maturity, gestation and litter formulas
- the Kleiber basal rate as a law
- the seed floor
- the CO2 cap as a hidden fix (it is reported if a planet still needs it)
- the slot cap as an ecological limit (it is reported; capacity is sized to avoid binding)

A test greps the v3 package for each forbidden mechanism (section 9).

### As built

The four rows, as the code has them:
- **Physics and chemistry.** Every constant in every v3 module has a `PROVENANCE` entry with a tag (reference,
  derived, new_rule, chain, or a mix). A test per module checks that the table is complete.
- **Chain starting conditions.** `formation3` reads the chain inputs: star, orbit, cloud composition, age and
  oxygen_pal. The chain's surface temperature is only a check.
- **The loop.** Division is a development process with a rate (fix stage, section 3). The divide output sets the
  share of the body's protein synthesis spent on its next child, and the child separates when its development is
  paid. The child's genome is `brain3.mutate3` of the parent's. Death comes only from the physical thresholds in
  `body.deaths`.
- **Abilities.** `brain3` has 20 motor outputs (11 scalars, 8 vocal, divide). `body` and `manipulate` turn them into
  physics, each with a cost.

Also supplied, stated honestly (each is tagged in `PROVENANCE`):
- The network architecture (version 2's recurrent net) and the form of the Hebbian rule with its evolved readout
  (section 5). Founders cannot learn, because their readout gains are 0.
- Physiological feedback rules whose strengths are genes. These are thermogenesis (gain x (setpoint - T_b)), growth
  once fat passes the `fat_store` level, urination above normal + `bladder`, salt excretion by the `kidney`, and the
  repair budget. The rules act automatically; how strongly they act evolves.
- Body-plan constants:
  - lean tissue is 84 % meat, 10 % bone and 6 % hide;
  - the body is a 2:1 prolate spheroid;
  - the mouth reaches one body radius;
  - the limb is 2.2 % of body mass and gets 25 % of sustained power;
  - sustained power is 30 % of peak;
  - each hidden unit stands for 1 mg of brain;
  - a litre of held air needs 0.21 kg of lung tissue, and blood and muscle hold 15 mL of O2 per kg of lean
    (4 October, section 3).
- Calibrations on Earth or on measured data:
  - the degassed share, on Earth's surface water;
  - the nitrogen degassed share, on Earth's N2 partial pressure (decided 4 October 2026, section 2);
  - the basin depth, on Earth's 70.9 % ocean;
  - the reference pressure P_cal, on the model Earth's 287.6 K;
  - the lake-keeping depth, on Earth's 3.7 % lake share;
  - the wear rate, on Rubner's lifetime energy;
  - the animal nose factor and the eye's contrast threshold.
- Starting conditions: the founders (section 6) and the plants' sparse, uniform sowing (section 7).
- Numerics that bound cost, not behaviour. These are the neighbour counts (K = 32, far_k = 16), the fire
  line-of-sight rounds, heat sub-steps, movement sub-steps and contact scan budgets. Each one reports when it
  binds: `bound`, `far_bound`, `beyond`, `fires_culled`, `overflow`, `no_item_slot`, `coarse_moves`,
  `travel_beyond_far`, `capacity_full` and `capacity_bound`.

Nothing rescues a world (owner decision, 4 October 2026; PROTOCOL.md rule 4):
- There is no synthetic fallback. A world7 seed that cannot be built is an outcome, not a reason to swap in a
  synthetic planet. The `synthetic_fallback` setting is removed.
- Patches are drawn by area over land, and founders sit at uniformly random positions. Nothing looks at
  habitability.
- Nothing is replenished. There is no seed floor and no seed rain. Founders are placed once, and an extinct arena
  stays empty.

None of the version-2 items listed above is in the v3 code. `brain3` imports only `think`, `init_live`, `init_state`,
`rows`, `install`, `genome_bytes`, `hebbian`, `to_bf16_stochastic`, `INIT_GAIN` and `WEIGHTS` from `planet.brain`, and
a test checks that list. So `planet.brain`'s ACTIONS, innate defaults, outcome and Kleiber constant cannot be reached
through v3.

The forbidden-mechanism grep runs per module: each test file greps its own module. `test_manipulate` also greps the
whole package for purpose-named functions (`make_fire`, `cook`, `knap`, `light_fire`, `start_fire`).

The package-wide gate is written (`tests/life9/v3/test_purity.py`, fix stage). It does four things:
- It greps every v3 source file for the section-9 mechanisms and their version-2 names. `ACTIONS` is matched as a
  whole word and case-sensitively, because "fractions", "interactions" and "reactions" contain it.
- It checks that `world3` builds the senses from `body.sense_view` only, which carries no lineage id.
- It permutes every lineage label among the living bodies. The two worlds must sense, act and end the day exactly
  alike, labels aside.
- It checks the rules in effect, not only the source text: the global biosphere's seed floor is 0, and the patches'
  nitrogen fixation follows growth, not a starting stock.

## 1. Scale: a real-size planet with true-scale habitat patches

- **Global layer (real size).** The planet has its real radius (5,674 km for 781). A cube-sphere grid (default G = 32,
  6,144 cells of about 180 km) runs version 2's climate and a coarse biosphere at real geometry. The terrain is the
  planet's real-scale relief (section 2).
- **Habitat patches (true scale).** Each world holds `patches` arenas (default 4).
  - Each arena is a square of side L = 2,048 m (config `patch_m`) at a global land cell. The cells are drawn at
    random by area on the setup RNG, with no habitability bias.
  - Fine grid 128 x 128 (16 m cells).
  - Inside a patch everything is at true scale: positions in metres, the horizon of the real radius (sagitta
    d²/2R), slopes and distances.
  - **Boundary:** periodic. The patch is a representative sample of its global cell, standing for homogeneous
    surroundings. This is the standard boundary condition of physics simulations for an infinite homogeneous
    medium, not a fence.
- **Coupling.**
  - The patch's air temperature, humidity, precipitation, light (insolation with the fine slope and aspect) and
    gas partial pressures come from its global cell each day.
  - Patch gas, carbon and water exchanges go back to the global ledgers.
  - Fine elevation = the global cell's elevation + periodic fractal detail with a real roughness spectrum
    (new_rule, stated). The lapse rate g/cp is applied per fine cell.
- **Time.** A day has `bouts` (default 4, i.e. 6 h) in which bodies sense, decide and act. Climate, global biosphere,
  patch hydrology and plants step once a day. Physiology integrates per bout. Day and night inside the day come from
  the sun angle per bout (no averaging), so nocturnality can emerge.

### As built (`patch.py`, `optics.py`, `world3.py`)

- **Arenas.** As designed: L = 2,048 m, n = 128, 16 m cells, periodic, with arena a = world x patches + p. Patch
  cells are drawn by area over land, with replacement (`choose_patch_cells`). Since the review of 4 October each
  world draws its terrain, patches, detail, deposits, sowing and founders from its own generator, seeded by the
  world's seed alone, so a seed's world does not depend on the other seeds of its sample or part.
  - **16 patches per world** by default, not 4 (fix stage). With 4, the draw by area alone often put every patch
    outside the bodies' temperature window (271-318 K), and the world died at founding. More patches sample the land
    better. There is still no habitability bias.
  - **A world without land gets sea patches** (review of 4 October 2026). `choose_patch_cells` draws them over all
    its cells, and its founders float or sink there; physics decides. Before, `world3` gave such a world no patch
    and recorded "no land", a habitability prior for land bodies (PROTOCOL.md rule 4). Its outcome row carries
    `no_land`.
- **Patch scale and slots (owner decision, 4 October 2026).** The patches stay 2 km. Each arena holds **4,096 body
  slots**, up from 1,024 at the fix stage (the design asked for 8,192).
  - A 4.2 km² patch could feed of order 1e8 bodies of 2-200 g. No slot count that fits in memory comes near that, so
    the cap binds once a lineage multiplies.
  - A birth into a full arena fails and loses its development. `capacity_full` counts these. The summary marks the
    world `capacity_bound` from the first day it happens.
  - `capacity_bound` is a numerical limit, not ecology. The run is kept and reported (PROTOCOL.md rule 1). After the
    binding day its population measures the cap.
  - 16 x 4,096 is four times the fix stage's slots per world. At 128 hidden units in float32 the weights take
    about 223 kB per slot, so about 14.6 GB per world (`World3.memory_bytes` reports it). bfloat16 weights halve that.
  - **4,096 slots bind too** (review of 4 October 2026). On the model Earth they bound on day index 4, about 2 days
    later than 1,024 slots (day index 2), at 4x the memory and about 1.6x the CPU time per day at equal population:
    35 s against 22 s on day 1, rising to 78 s by day 5 as the slots filled. Decision (1) assumed more headroom.
  - **GPU hosts.** The default world's float32 body state does not fit the RTX 3060 (12 GB) and barely fits the RTX
    4090 (24 GB). Whether GPU experiments use bfloat16 weights or fewer slots is an open owner decision (it changes
    the rules hash). The founders are drawn on the CPU per world and copied into the device's state, so set-up holds
    the state once; a resume loads it without copies.
  - **The item pool binds first.** In the default Earth run most arenas held 998-1,024 of their 1,024 item slots by
    day 3, and carcass parts went to the litter. `items_bound_day` and `fires_bound_day` are reported beside
    `capacity_bound`.
- **Fine elevation.** It is the global cell's elevation plus periodic fractal detail. The detail's rms comes from the
  planet's own relief between neighbouring global cells. A two-regime structure function carries it down: Hurst 0.5
  from the global spacing to a 5 km break, then 0.8 below it. The 5 km break is a stated pick inside the reported
  range.
  - On formation3's terrain (`make_terrain3`) the median detail rms is 11.6 m on 781 and 13.7 m on Earth (5-40 m
    from p10 to p90).
  - On version 2's `globe.make_terrain` it is only about 1 m. Use `make_terrain3`.
- **Depression conditioning.** Gaussian detail is not an eroded landscape: every local minimum is a closed pit, and
  raw detail puts 18.7 % of cells in lake basins. Basins shallower than 0.4 x the detail rms are filled (sediment
  infill). The 0.4 is calibrated so that 64 synthetic patches average Earth's 3.7 % lake share (Verpoorter 2014).
  The median patch has 0.9 %; many have none and a few hold 15-25 %.
- **Light.**
  - The sun direction is set per bout (8 samples per bout), with version 2's phase: noon at longitude 0 at t = 0.
  - The climate's daily mean is turned into instantaneous flux by flat ground's mean incidence over one whole solar
    day. Energy is therefore conserved over many days for any day length (12-48 h, or locked). On a 48 h day,
    whole sim days are dark.
  - Beam light falls per horizontal m² on the local slope. Diffuse light uses Erbs et al. 1982's daily diffuse share
    and the sky view factor, normalised over the patch.
  - `forcing_from_climate` needs the specs' day length (or `solar_day_s`) and raises without it.
  - Not modelled: terrain cast shadows, sky occlusion, and a daily air-temperature cycle. A fine cell's air is the
    global daily mean with the lapse rate g/c_p.
- **Coupling.**
  - The patch's net carbon exchange, including the sown carbon, is in each step's `exchange_mol`. `world3` books the
    change of the patch's carbon taken from the air, and the nitrogen fixed, into `climate.add_gas` every day.
  - The bodies' and the fires' O2 and CO2, and the water they give to the air, are booked to the global layer too.
  - Precipitation and evaporation in a patch are a sample of the global cell's and are not fed back.
  - Water, carbon and nitrogen that bodies take or give are booked in the patch's counters and in the bodies'
    float64 transit residuals.
  - **The air thins with height** (fix stage). The global partial pressures are at sea level. Each fine cell's air
    is reduced barometrically to its height. Mole fractions, and so fire's O2 limit, do not change with height.
- **Time (fix stage).** **24 bouts a day** (1 h), not 4. The brain senses and acts hourly, and the sun angle is
  resolved hourly, so dawn, dusk and night are distinct. Patch hydrology and plants step once a day; physiology steps
  per bout.
  - Every body rate is per second: development, digestion, drinking, the manipulation events and locomotion. So the
    bout length changes numerics, not physics.
  - **Movement sub-steps.** Within a bout each body moves in its own sub-steps, each at most 2 fine cells (32 m),
    at most 8 per bout. In each sub-step its mouth meets what lies along the path. A body that would need more is
    not slowed: it takes longer sub-steps, counted as `coarse_moves`. Bodies whose bout travel passes the senses'
    128 m far range are counted as `travel_beyond_far`.

## 2. Formation v3: derive what version 2 assumed (`v3/formation3.py`, extends `planet.formation`)

- **Water.** Accreted solids are hydrated according to the disc temperature at the planet's orbit:
  - T(r) = 278.3 K L^¼ / sqrt(r), which uses world7's own BLACK_BODY constant.
  - The water mass fraction of solids is anchored to meteorite classes (reference): enstatite chondrites about
    0.1 % (T > 400 K), ordinary chondrites about 0.5 % (300-400 K), CM/CI carbonaceous chondrites about 10 % (< 160
    K, ice-free), plus ice beyond the frost line. It is interpolated in log against T.
  - A degassed share is calibrated so that Earth's case gives Earth's surface water (new_rule, one constant).
  - This replaces `water_mass_fraction`.
- **Atmosphere.**
  - Outgassed C and N follow from the same solids: the bulk C and N of the chain's cloud, scaled by the
    condensation of carbonaceous and nitrogen-bearing solids at that disc temperature (reference anchors: CI
    chondrites C 3.5 % and N 0.3 %; enstatite chondrites C 0.4 % and N 0.04 %), times the same degassed share.
  - CO2 partitioning between air and carbonate rock is set by the carbon-silicate equilibrium at the chain's
    temperature (weathering = outgassing at steady state, Walker-Hays-Kasting). *(Superseded: see As built.)*
  - N2 and Ar follow from N and K-40 decay.
  - O2 comes from the chain's bodies era (as before).
  - The greenhouse then **results** from this CO2 and H2O, rather than being solved backwards from the chain's 288 K.
    The chain's T_s becomes a check, and the mismatch is reported. If a planet ends up much colder or hotter than
    world7 says, that is a finding.
- **Relief:** real scale. The peak relief is limited by rock strength, h_max ∝ σ_rock / (ρ g): Earth's g gives
  about 9 km (calibration, reference). Isostasy splits continents and ocean floor by the ocean volume.
- `check_spec` keeps every version-2 check, plus the water, C and N mass budgets and the disc temperature at the orbit.

### As built (`formation3.py`)

The code differs from the design in five ways:
1. The CO2 is a self-consistent fixed point, not weathering at the chain's temperature.
2. Weathering happens on land and on the seafloor.
3. K, U, Th and the heat flow scale per kg of rock.
4. The greenhouse adds a cloud placeholder, other gases and pressure broadening.
5. Four version-2 checks are superseded.

**Water, carbon and nitrogen.**
- All four meteorite classes are anchors. Mass fractions of H2O / C / N:

  | Class | H2O | C | N | Anchor temperature |
  |---|---|---|---|---|
  | EC | 0.1 % | 0.40 % | 0.04 % | 400 K |
  | OC | 0.5 % | 0.12 % | 0.003 % | 350 K |
  | CM | 10 % | 2.2 % | 0.08 % | 160 K |
  | CI | 17 % | 3.5 % | 0.3 % | 120 K |

  Contents are interpolated in log10 against the disc temperature and clamped beyond the outer anchors. The anchor
  temperatures are a stated choice taken from this spec's ranges.
- Beyond world7's frost line (169.4 K) ice joins the rock (ICE_FACTOR 3).
- One degassed share, 0.0149, applies to water and C. It is calibrated so that the model Earth gets Earth's surface
  water (2.3e-4 of its mass). Until 4 October it applied to N as well.
- C and N scale with the cloud's C and N per kg of rock, relative to the sun's.
- All degassed N goes to the air as N2. Carbon that is not in the air is carbonate.
- Argon uses Earth's own degassed share of 0.5. The water share would leave Earth with about 3 % of its argon.

**Nitrogen has its own degassed share (owner decision, 4 October 2026).** It is one more Earth calibration, stated
as such.
- The share (`nitrogen_degassed_share`) is calibrated so that the model Earth (`chain.earth_inputs`) gets Earth's
  real N2 partial pressure: the dry-air N2 share 0.78084 x 101,325 Pa = 79.1 kPa. It is solved together with P_cal
  as a fixed point of the model Earth's build.
- It comes out at 0.00635, which is 0.43 of the water's share. The bulk Earth is poorer in N than chondrites
  relative to H and C (Marty 2012, quoted from memory and flagged in `PROVENANCE`).
- Why: with the water share, chondritic N gave the model Earth 2.3 x Earth's N2 and 2.05 bar of air. Its O2 was then
  only about 10 % of the air, below fire's 15 % limit, so fire was impossible on the model Earth.
- Now the model Earth has 1.01 bar of dry air (1.03 bar with vapour). Its O2 mole fraction is 0.21. Fire is
  physically possible there.
- The same share applies to every planet. 781 now has 0.53 bar of dry air, with 4.7 % O2. Fire stays impossible
  there.
- The greenhouse's reference pressure P_cal is the model Earth's dry pressure, so it moves from 2.04 bar to
  1.013 bar. The tau calibration still gives the model Earth world7's 287.6 K.
- The earlier reading stays available as `nitrogen_calibration="water"` (one share for water, C and N), reported but
  not the default.

**K, U, Th and the heat flow.** These are set per kg of rock: the bulk silicate Earth's 240 ppm of K times the
cloud's lithophile scale (other metals per rock, relative to solar). For 781 the scale is 0.83. Its Z/Z_sun is 0.23,
but the planet is made of rock, not of the whole cloud.

**CO2.** Weathering equals outgassing.
- Outgassing scales with heat flow x degassed carbon per area, relative to the model Earth.
- Weathering (default `land_seafloor`) has two parts:
  - continental weathering by the WHAK law (beta 0.3, 13.7 K), times land / land_E;
  - seafloor weathering: 15 % of Earth's sink, times the heat flow, with pCO2^0.25 and 41 kJ/mol.

  A planet with no land weathers only on its seafloor.
- The weathering temperature is the planet's own T_s. The CO2 is the fixed point of CO2 -> greenhouse -> T_s ->
  weathering -> CO2 (`weathering_t="self_consistent"`).
- Every spec also reports the design's literal reading (weathering at the chain's T) and the published global WHAK
  law.

**Greenhouse.** The grey tau is computed forward and calibrated so that the model Earth gives world7's 287.6 K.
- Earth's tau is split vapour 0.50, CO2 0.20, clouds 0.25 and other gases 0.05 (Schmidt et al. 2010).
- CO2's exponent gives Myhre's 3.71 W/m² per doubling. Vapour's gives the IPCC AR6 vapour + lapse-rate feedback.
- Clouds are held at Earth's 25 % on every planet. This is a placeholder: formation has no cloud physics.
- Other gases (O3, CH4 and N2O, made by a biosphere) scale with the chain's oxygen_pal.
- Pressure broadening: absorber paths scale as (P / P_cal)^0.5. The exponent is chosen to match Goldblatt et al.
  2009's ~4.4 K for doubled N2: 0.5 gives 5.2 K, while 1.0 would give 13.3 K. P_cal is the model Earth's dry
  pressure: 1.013 bar since the nitrogen decision (2.04 bar before).
- Collision-induced absorption (N2-N2, CO2-CO2) is not modelled.
- Each spec reports the warming from clouds, from other gases and from broadening, in K.

**T_s and flags.** T_s is the lowest root of σT⁴ = σT_eq⁴ (1 + 0.75 tau(T)) above T_eq, with version 2's Bond albedo
of 0.30. Flags are reported and never gated:
- `frozen_mean`: T_s < 273.15 K. Formation has no ice-albedo feedback and does not shut weathering down on frozen
  surfaces. The v3 climate decides these planets, and must not take their formation T_s as a target.
- `runaway`: no root below 450 K.
- `pressure_beyond_fit`: dry P > 2 x P_cal.
- `co2_beyond_fit`: pCO2 > 1e4 Pa.
- `carbon_limited` and `no_land_weathering_assumed`.

**Relief.**
- Peaks are strength-limited: 8,848.86 m x g_E / g.
- Basin depth is D(g) = D_E (s g_E / g + 1 - s). The ridge share s = 2,600 / 3,682 scales with crust thickness
  (~1/g); the thermal-subsidence rest does not depend on g. D_E is calibrated on Earth's 70.9 % ocean.
- Ocean fraction = min(1, V / (4 π R² D)). The readings with D fixed and with D ~ 1/g are also reported.
- `relief_m` = peak + D.

**Terrain consumer.** `make_terrain3` builds the globe's terrain:
- its ocean area equals `spec.ocean_fraction` and its ocean volume equals the spec's;
- land heights are a truncated exponential with mean 840 m x g_E / g, up to the peak;
- ocean depths have shape k = 4.

Version 2's `globe.make_terrain` must not be given `relief_m`: it puts the sea elsewhere and spreads the land
linearly. Earth's shelf area and its share of ocean deeper than 5 km are not reproduced.

**Checks.** `check_spec3` keeps every version-2 check except the four in `SUPERSEDED`:
1. the grey T_s against the chain's whole-K T_s;
2. the backwards identity CO2 tau + residual = greenhouse tau;
3. the sign of the signed mismatch;
4. the grey target of a flagged runaway planet.

It adds:
- the water, C, N and Ar budgets;
- recomputed potassium, heat flow, outgassing and land share;
- the steady state, and the weathering temperature against the chosen reading;
- the tau components, every flag, and the relief.

Both fixed-point loops report `fixed_point_converged`.

**Results.** 32 planets were built: the model Earth, 781, 20 synthetic planets and 10 cached world7 seeds. All pass
`check_spec3`. The values are re-measured on 4 October with the nitrogen share. Values from 3 October, under the
single share, are in brackets where they changed.

| | Model Earth | 781 |
|---|---|---|
| Disc temperature | 278.3 K | 269.9 K |
| Surface water per planet mass | 2.30e-4 (calibrated) | 2.63e-4 |
| Ocean fraction: split D (D fixed; D ~ 1/g) | 0.716 (0.713; 0.717) | 0.513 (0.622; 0.478) |
| Heat flow | 46.6 mW/m² | 19.9 mW/m² |
| Outgassing, relative to the model Earth | 1 | 0.35 |
| pCO2 | 28.5 Pa (28.4) | 12.8 Pa (12.0) |
| pN2 | 79.1 kPa, Earth's (181.8 kPa) | 50.2 kPa (116.6) |
| pO2 | 21.2 kPa | 2.5 kPa |
| pAr | 985 Pa (968) | 579 Pa (575) |
| Total pressure, with vapour | 1.03 bar (2.05) | 0.54 bar (1.20) |
| O2 share of the air | 0.21 (0.10): fire possible | 0.047: no fire |
| T_s (chain's value) | 287.55 K (287.6 K) | **270.5 K (287.7 K): 17.2 K colder, frozen_mean** (270.8 K) |
| Warming from clouds / other gases / broadening | 11.8 / 2.3 / 0 K | 10.6 / 0.24 / -2.4 K (-2.1) |
| T_s with weathering at the chain's T / with global WHAK | 287.51 / 287.55 K | 266.0 / 272.4 K (266.3 / 272.7) |

The 31 planets besides the model Earth:
- **23 are fully water-covered**: 21 with D fixed, 23 with D ~ 1/g. Only their seafloor weathers, so they settle at
  287-312 K with 1.9e3 to 2e6 Pa of CO2.
- 4 are `frozen_mean`: 781 and synthetic 4, 14 and 15. None runs away and none is carbon-limited.
- 20 have more than twice the model Earth's dry pressure, up to 31 times (21 and 23 times before). The ratio grew
  because P_cal halved while CO2, which the nitrogen share does not touch, stayed. On 9 of the 20, CO2 is more than
  half the dry air. 20 have more than 1e4 Pa of CO2 (19 before). On these planets the greenhouse fits are
  extrapolated and collision-induced absorption is missing.
- Dry air runs from 0.28 to 32 bar.
- The mismatch against the chain runs from -27.4 to +23.7 K. Weathering at the chain's T, the design's literal
  wording, sends 14 planets to the 450 K bound (12 before).
- Anchor sensitivity (`METEORITE_CLASSES_SITES`, with EC at 278 K and OC at 190 K, where these classes formed):
  - water-covered planets: still 23;
  - coldest-orbit water per mass: 3.06 x Earth's (default 2.63 x);
  - degassed share: 0.23, which gives the model Earth 107 bar of air.

  One degassed share for water, C and N cannot hold for an Earth built from EC-like material. Since 4 October N has
  its own share; water and C still share one.

## 3. Bodies: physics, not allometry (`v3/body.py`)

State per arena slot `[A, N]`: `alive`, `pos` [A,N,2] in m, `heading`, `mass_kg` (lean tissue), `reserve_j`
(stored fuel), `water_kg`, `body_k` (body temperature), `damage` (0-1), `gut_kg` (contents by digestive class),
`held` [A,N,2] (item indices, two grips), `age_bouts`, lineage ids.

The genome holds the brain (section 5) and **body genes**, all heritable with random variation and with no target
values:
- `size_kg`: the mass at which growth stops
- `fur_m`: insulation thickness
- `skin_perm`: water permeability
- `thermo_gain` and `setpoint_k`: thermogenesis control. A gain of 0 is an ectotherm. Endothermy must evolve.
- `enz_plant`, `enz_meat`: digestive investment. Each costs maintenance, and they trade off by gut volume.
- `repair`: the share of fuel spent repairing damage
- `muscle`: power per kg; it costs maintenance
- `offspring_share`: the share of mass and reserve given at division
- `eye`, `ear`, `voice`: organ sizes, each with mass and maintenance cost

Energy and matter per bout (formulas with references; nothing allometric is supplied):
- **Maintenance.** Each tissue (lean, gut enzyme, muscle, organs, brain units) costs c_tissue W/kg at the reference
  temperature (reference: tissue-specific metabolic rates). The rate scales with the Arrhenius factor
  exp(-E/k (1/T_b - 1/T_ref)), E = 0.65 eV (reference: the metabolic theory of ecology; chemistry, not allometry).
- **Heat balance.** A body is an ellipsoid of its mass at tissue density 1,050 kg/m³.
  - Heat loss: A (T_b - T_air)/(fur_m/k_fur + 1/h). Convective h depends on size (forced convection for wind
    speed u, Nu correlations, reference).
  - Radiation is exchanged with the sky and ground.
  - Evaporation: skin_perm x A x (e_sat(T_b) - e_air) x L_v.
  - Heat production: maintenance, work inefficiency, and thermogenesis = thermo_gain x max(0, setpoint - T_b)
    (fuel).
  - T_b integrates with tissue cp 3,500 J/(kg K).
  - Death: T_b < 271 K (freezing) or T_b > 318 K (protein denaturation; reference).
- **Locomotion.** Mechanical work per metre = c_m M g (c_m = 0.1, reference: legged locomotion's mechanical cost)
  + M g Δh. Swimming costs ½ ρ_w C_d A_x v² per metre. Metabolic cost = work / 0.25 (muscle efficiency,
  reference). Top speed comes from muscle power: P_muscle = muscle x M_muscle against these costs. The aerobic
  capacity from pO2 limits sustained power (a Michaelis term on the O2 partial pressure, reference).
- **Ingestion.**
  - The mouth output opens on whatever is in contact at the mouth position: plant tissue in the fine cell, fresh
    water (soil above field capacity, or open water), an item, or another body (a bite: it removes tissue in
    proportion to the bite force against the tissue's toughness, and wounds).
  - Digestion: energy = Σ class mass x gross energy (reference: plant 18.5 MJ/kg dry, flesh 7 MJ/kg wet, fat
    39.5 MJ/kg) x enzyme efficiency enz/(enz + K).
  - Undigested matter goes to faeces (litter carbon). Salt from sea water raises osmotic damage.
  - Stones and metal give nothing and cost the gut's volume.
- **Water.** Losses are evaporation, respiration (∝ O2 used) and faeces; metabolic water comes from oxidation
  (chemistry). Death when water < 0.6 x the body's normal water (reference: lethal dehydration for mammals).
- **Growth and division.**
  - Surplus fuel builds tissue toward `size_kg` at a tissue cost of 7 MJ/kg plus a synthesis overhead (reference).
  - Division is a motor output. When it fires and the body can pay, the body splits off a child with
    `offspring_share` of its mass and reserve. The child's genome is the parent's plus random variation
    (section 6).
  - No maturity age exists: a body too small to give a viable child simply produces one that dies.
- **Damage and ageing.**
  - Damage rises with wounds and with oxidative wear ∝ O2 consumed x (1 - repair).
  - Repair costs fuel.
  - Damage reduces muscle power and senses.
  - Death at damage 1. Ageing therefore emerges from the repair/reproduction trade-off.
- **Death.** A dead body becomes a carcass item (flesh, fat, bone and hide by tissue mass) at its position. Carbon
  and water are booked.

All body formulas live in `body.py` with `PROVENANCE` tags. None names a behaviour.

### As built (`body.py`, with the gene table of `brain3.py`)

How the code differs from the design:
- five genes are added (`muscle_frac`, `fat_store`, `bladder`, `kidney`, and `air_l_kg` since 4 October);
- thermo_gain is per kg;
- fur is a spherical shell, and evaporation goes through a series path;
- temperature also acts on muscle, the jaws and the gut;
- locomotion adds the limbs' internal work, and `world3` moves bodies by a force-generation cost (below);
- division is a development process paid over time (fix stage), not a one-bout split;
- bodies have a density: they float or sink, and under water they live on an O2 store (fix stage and 4 October);
- wear is permanent and calibrated on Rubner's lifetime energy.

**State.** As designed, plus:
- `frame_kg`: the largest lean mass reached, scaled down at division;
- `n_kg`: free amino nitrogen; `salt_kg`;
- `wear`: the permanent, oxidative part of damage;
- `vel`, lineage ids, and float64 `transit` residuals for exchanges with the patch.

`gut` holds 5 classes (plant, protein, lipid, cooked, inert), each as kg dry, J, kg C and kg N. Heading is radians
from east, counter-clockwise, the same in `senses3` and `manipulate`.

**Genes.** All are heritable and mutated by `brain3.mutate3`, and none has a target.

| Gene | Meaning and bounds |
|---|---|
| size_kg | lean mass at which growth stops; 1 mg to 100 t |
| fur_m | coat thickness |
| skin_perm | the skin's own vapour conductance only |
| thermo_gain | W per kg of lean per K; founders sit at a negligible floor |
| setpoint_k | 271-318 K |
| enz_plant, enz_meat | gut tissue for each food class |
| repair | 0-1 |
| muscle | W per kg of muscle; 1-500 |
| muscle_frac | muscle mass share of the body |
| offspring_share | logit gene, 1e-6 to 1 - 1e-6 |
| eye, ear, voice | organ mass per body mass |
| fat_store (body.py) | kg of fat per kg of lean before fat builds tissue |
| bladder (body.py) | kg of water per kg of lean held above normal |
| kidney (body.py) | kg of kidney per kg of body |
| air_l_kg (body.py, 4 October) | litres of air held per kg of lean (lungs, air sacs, a swim bladder); 0 to 1 L/kg, linear; 0 is a body with no air store |

**Tissue budget.** lean = muscle + gut (enz_plant + enz_meat) + eye + ear + voice + brain (1 mg per active unit) +
fur + kidney + lung + rest. Fur mass is coat area x thickness x 65 kg/m³. Lung mass is 0.21 kg per litre of air
held. A genome that asks for more than the body gets every tissue scaled down in proportion
(`brain3.tissue_shares`). The coat costs mass but no maintenance. A body computes with no more brain units than its
brain tissue holds after that scaling (`body.active_units`, fix stage).

**Maintenance.** Per tissue, at measured rates, with no size scaling (Elia 1992):

| Tissue | Rate at 310 K |
|---|---|
| Muscle | 0.63 W/kg x (muscle / 100 W/kg) |
| Gut | 9.7 W/kg |
| Brain, eye, ear | 11.6 W/kg |
| Kidney | 21.3 W/kg |
| Lung (4 October) | 1.9 W/kg |
| Rest | 0.58 W/kg |
| Fat store | 0.22 W/kg |

The Arrhenius factor (E = 0.65 eV) acts on maintenance, muscle power, the jaw-cycle rate and gut passage. Its Q10
is about 2.2 over 280-305 K: x0.24 at 293 K and x1.8 at 318 K. Bite force does not depend on temperature
(isometric force).

**Heat balance.**
- The body is treated as the equal-volume sphere, with a spherical fur shell, (1/r1 - 1/r2)/(4π k_fur), and
  convection at the outer radius (Ranz-Marshall, within 20 % of Whitaker 1972). The result is scaled by the
  spheroid's area factor, 1.077.
- Radiation is linearised into an operative temperature: a Brutsaert sky, ground at air temperature, and shortwave
  on half the area.
- Evaporation goes through skin, fur and boundary layer in series. Respiratory water is computed per sub-step at
  that sub-step's T_b.
- Heat sources:
  - maintenance and repair;
  - thermogenesis, thermo_gain x lean x (setpoint - T_b), capped by the muscles' aerobic capacity at T_b minus
    what locomotion uses;
  - locomotion and muscle work (drag and climbing work leave the body);
  - growth (conversion and synthesis overhead) and digestion.
- The balance is integrated exactly per regime over 8 sub-steps. Tests check that a 20 g ectotherm cools to air
  temperature, and that an endotherm holds its setpoint at the fuel cost its insulation implies.

**Locomotion.**
- Power = |thrust| x sustained power (30 % of peak) x a(pO2) x muscle x muscle mass x (1 - damage) x the Arrhenius
  factor.
- The speed solves c_m M g v + the limbs' internal kinetic work (0.478 v^1.53 W per kg, Fedak et al. 1982) + air
  drag on land. In water it solves water drag.
- Climbing is paid from the same power. Metabolic cost = work / 0.25.
- `move(dt_s)` and `sum_loco` allow sub-steps within a bout.
- Measured at thrust 1 (muscle 100 W/kg, 40 % muscle):
  - 70 kg at 310 K: 4.47 m/s at 8.2 J/(kg m); Taylor 1982 measured 2.7.
  - 2 g at 300 K: 2.24 m/s at 7.2 J/(kg m); Taylor measured 73.

  The cost of transport comes out at about 7-9 J/(kg m) at every size. Small bodies move far too cheaply; large ones
  pay about 3 x too much (section 11).
- **In `world3` bodies move by `move_kt`, not `body.move` (fix stage).** The owner has not yet decided this contract
  change (section 11).
  - On land the cost is the force that carries the weight at each foot contact (Kram & Taylor 1990): each contact
    costs c M g, with c = 0.2 J/N (quoted from memory, to verify). The contact length is the body's own length,
    because the body plan has no separate leg. Air drag and climbing are added at 25 % efficiency.
  - A moving body spends its locomotion share of the bout at its full sustained power, and stands the rest.
  - Held items add their weight.
  - In water deeper than its height, a body denser than water walks the bottom with its weight in water. One that
    floats paddles at the surface at 25 % propulsive efficiency, against a bow-wave drag of up to 5 x.
  - The cost of transport comes within about 12 % of Taylor, Heglund & Maloiy 1982 from 2 g to 70 kg. The gates test
    the physics, not a fit to that curve.

**Ingestion.**
- The mouth sits at the front of the spheroid and reaches one equal-volume radius. It closes on a body first, then
  an item, then the ground.
- Bites:
  - Bite force = muscle stress 0.3 MPa x the jaw muscle's cross-section (3 % of muscle) x lever 0.4, set against
    the hide's cutting toughness.
  - The jaws take one gape-sized chunk (gape is 15 % of body length) per jaw cycle while contact lasts.
  - A prey that fits the gape is swallowed whole.
  - The bitten body's wound damage = removed tissue share / 0.3.
  - Items are bitten against their own species' toughness. Nothing harder than enamel (Mohs 5) is cut.
- Plants are cropped from the area the mouth sweeps in the bout: 2 x reach x path + π reach², at most the fine
  cell.
- Water from ponds, from soil above the bucket or from the sea is drunk at the gut's emptying rate (15 min). Sea
  water brings 3.5 % salt.

**Digestion.**
- Gut contents pass with a 12 h mean retention time, at the Arrhenius rate.
- Efficiency is enz / (enz + K) per class, with K = 0.05 for plant and 0.005 for meat.
- Cooked flesh (an item that reached 70 °C) digests 1.3 x better. Stones, metals and charcoal are inert.
- Absorbed energy becomes fat, limited by carbon. Surplus carbon leaves as CO2.

**Water and salt.**
- Water is lost by evaporation, respiration and faeces (2 kg of water per kg dry). Metabolic water comes from
  oxidising tripalmitin.
- Water above normal + bladder x lean leaves as urine.
- Salt leaves in urine at most at the kidney's concentration, 8.24 x kidney share kg/kg. That is 0.035 at the human
  share, and never more than NaCl's solubility of 0.36.
- Salt-carrying urine is also limited to 0.15 of the kidney's filtrate.
- Salt that cannot be flushed stays and harms: 1 % of body water held for 6 h is damage 1.
- Dilute urine and drinking have no limit beyond the emptying rate. There is no water-intoxication physics.

**Fat store and growth.** The store has no cap and is never burnt automatically.
- Fat above fat_store x lean, after the bout's committed costs, builds lean tissue toward size_kg (limited by
  nitrogen).
- Below that level, or once the body is at full size, fat accumulates.
- Founders start with 0.15 kg of fat per kg of lean.

**Division (fix stage: a development process with a rate).**
- The divide output, in [0, 1], is the share of the body's protein-synthesis capacity it spends on its next child.
  The capacity is 25 % of its own protein a day at 310 K, times the Arrhenius factor (Waterlow et al. 1978, quoted
  from memory).
- A child costs 0.8 MJ per kg: its protein is made anew from the parent's amino acids (3.6 MJ per kg of protein).
  The fuel is oxidised in the parent, and its heat is released there over the development time.
- The child separates when its development is paid. At full output and 310 K, a child of half the parent's lean
  takes at least 2 days.
- Development is a power, so births per day do not depend on the bouts per day. This replaces the Bernoulli draw
  per bout. It also removes the problem of finding 9: one bout's development heat would have cooked a small parent.
- The child takes offspring_share of the parent's lean, frame, store, water and free N.
- The parent keeps its gut, salt, grips, damage, wear and learned weights, and any development beyond this child's
  cost for its next one.
- The child starts with:
  - damage and wear 0 and an empty gut;
  - a mutated genome, live weights from its genome, and a zero hidden state;
  - a position touching the parent's rear.
- An offspring_share above one half means the fresh body gets most of the mass.
- A birth that finds no free slot (`capacity_full`) or is too small (`too_small`: child or remainder below 1 mg)
  fails. Its development is lost, as a real one would have been spent. `divide_fired` counts completed
  developments.

**Buoyancy (fix stage).**
- A body's density is its whole mass over lean / 1,050 + fat / 900 + (gut contents and water above normal) / 1,000
  kg/m³, plus the air it holds (below).
- A body denser than water, in water deeper than its height, lies on the bottom with its airway under water
  (`body.sunk`). A floating body holds its density's share under water.
- At the fix stage nothing in the body plan held air. Every founder was denser than water, and in ponds it drowned
  within about 5 minutes: damage rose by the time sunk over a fixed 300 s.

**Air volume and breath-holding (owner decision, 4 October 2026; revised by the review of 4 October).**
- A heritable gene, `air_l_kg`, sets the litres of air the body holds per kg of lean: lungs, air sacs or a swim
  bladder. Its bounds are 0 to 1 L/kg, about the lean's own volume. Founders draw it uniformly over those bounds.
- The air is held at the ambient pressure, and it lowers the body's density. Lean tissue with no fat floats once it
  holds about 0.048 L/kg. That is also the gene's mutation step. The air volume is the total lung capacity at all
  times: a body cannot breathe out to sink (a stated simplification).
- The air needs lung tissue: 0.21 kg per litre held (mammal lungs, Stahl 1967, quoted from memory). That tissue takes
  its share of the tissue budget and costs 1.36 W/kg (the human wet lung's own O2 use).
- **The lung is the exchange organ.** A body takes up at most 7.6e-3 mol O2/s per kg of lung at the reference
  inspired pO2 (a 1 kg mammal's VO2max over its lungs), in proportion to its own inspired pO2, plus what its skin
  passes (4.9e-10 mol/(s m^2 Pa), Krogh's constant over 30 um). The supply caps the sustained muscle power and
  thermogenesis; costs beyond it are an O2 debt (hypoxia). A 1 kg body without a lung suffocates; bodies up to
  about 10 g live on their skin.
- **The O2 carrier** is a heritable gene, `o2_carrier`: mL O2 bound per kg of lean (haemoglobin, myoglobin), 0 to
  298 (all the lean's protein), founders uniform over it. Its protein takes its share of the tissue budget and is
  re-made every 120 days.
- The O2 store is the air held from the alveolar pO2 (inspired x 0.75) down to 0.30 of it (4 kPa over 13.3 kPa,
  the human blackout over the normal alveolar pO2), plus the carrier unloading between the same two pressures by
  the aerobic law. The relative floor is one rule for breathing and breath-holding on every planet; it is a
  body-plan constant still to be decided by the owner.
- Under water a body draws its store; the debt is repaid at the next breath. A breath-hold within the store is
  harmless. When the store is spent the body is unconscious (it acts no more) and the debt beyond the store is
  anoxia: damage 1 after 180 s of its resting O2 use at 310 K. A death that anoxia dominates is booked as
  `anoxia`. There is no fixed time to drowning.
- **Sea water is 1,027 kg/m³**, ponds fresh. A body denser than the water treads it while it moves, when its
  sustained power covers the actuator-disc power of its weight in water; standing, it sinks in deep water.

**Damage and ageing.**
- Wounds (bites, strikes, burns, osmotic harm) and oxidative wear add to damage. Damage cuts muscle power and the
  eye's photon catch.
- Wear = O2 used x 3.19e-5 kg of protein per mol x (1 - the share the repair budget prevents). The rate is
  calibrated on Rubner's lifetime energy (0.92 MJ per g) and on 30 % of protein oxidised at damage 1. Wear is
  permanent.
- The repair budget is repair x the bout's running costs. It first heals wounds and osmotic harm at the cost of
  protein synthesis (0.24 MJ per unit of damage per kg of lean), then prevents wear.
- Measured at repair 0, a 20 g endotherm at 283 K lives about 3.7 years and a 70 kg body about 12 years. Lifespan
  varies smoothly with repair; there is no threshold.

**Death.**
- Causes: no fuel (lean below 0.6 x frame), dehydration (water below 0.6 x normal), freezing (271 K), denaturation
  (318 K), and damage 1. Damage comes from wounds, burns, osmotic harm, anoxia under water and wear.
- In `world3` a body can also heat to death in a fire: the fire's radiant flux on its cross-section, and the heat
  its own rubbing leaves in a bare limb, enter its heat balance. Charred skin is a burn wound (fix stage).
- The body becomes meat, bone, hide and fat items at its position. What finds no item slot goes to litter, and its
  remaining water goes to the soil.

**No-selection control (`selection=False`).** Physics still removes exactly its dead. First, though, each dying
body's genome and lineage are swapped with those of a random survivor.
- Measured, this cannot give drift only. A lethal genome is never purged and keeps killing its new bodies: the run
  went extinct in about 40 bouts, and 197 of 217 deaths came from genomes that had killed before.
- The drift baseline is therefore the neutral `Shadow` (Bedau & Brown 1999): gene-only genomes that take the run's
  own deaths and births per bout at random and mutate as bodies do.

**Ledgers.**
- Energy, carbon, water, nitrogen, inert matter (gut contents and bone mineral), salt and the gases of oxidation are
  kept in float64 per arena. They close to about 1e-8 relative over 400 bouts.
- Body-patch cross-books (water, sea, C, N, salt) close below 1e-6 over 100 days.
- Bone mineral made by growth is booked as diet ash. Mineral from burnt tissue, and salt excreted on land, leave the
  world, because the patch has no mineral or salt field.

**Measured** (before the fix stage: Bernoulli division, 4 bouts a day, no buoyancy).
- A 50-day mini-world (2 arenas, 96 founders each, random motors):
  - all 384 slots filled, with 1,143 births;
  - deaths: 481 from dehydration, 86 from no fuel;
  - selection is visible against the shadow: median skin_perm 10^-11.1 in the run, 10^-9.6 in the shadow.
- A full-size bout (2 x 8,192 slots) takes about 0.2 s on the CPU.

## 4. Senses: physical channels only (`v3/senses3.py`)

Inputs carry no class labels ("food", "stone", "predator", "kin" are never given). Per sector S = 8 around the
heading, out to the physical range of each organ:
- **Vision.**
  - Range and sharpness scale with the `eye` gene and the light level (the sun angle per bout, the patch's slope,
    night).
  - A target's apparent size is its angular diameter, giving visibility against the background by contrast. Its
    reflectance comes from the material and tissue optical table (reference albedos; plants green-dark, water
    dark, sand bright, fire bright and emitting).
  - Channels per sector:
    - nearest moving body's angular size and contrast
    - total angular cover of bodies
    - mean reflectance of the ground (plant cover darkens, water darkens, snow brightens)
    - items' angular size and brightness
    - emitted light (fire)
  - Terrain line of sight uses the fine elevation with the real-radius sagitta.
- **Hearing.**
  - Calls are D = 8 dimensional vectors with a loudness. Locomotion makes noise ∝ speed² x mass.
  - Propagation: spherical spreading 1/r², atmospheric absorption ∝ f² (reference), the speed of sound from the
    air, and nothing below 1 kPa. Terrain occlusion x 0.5.
  - Received per sector: D call features plus a broadband noise channel. Sensitivity scales with `ear`.
- **Smell (chemistry).** Volatile fields per fine cell diffuse and decay daily: plant volatiles ∝ plant
  biomass, carcass volatiles ∝ rotting flesh, smoke ∝ fire. The input is the concentration and its gradient along
  the heading. (Fields, not labels; a body has to learn what a smell means.)
- **Touch.** In contact under the mouth: mass of plant tissue, wetness, item hardness, item temperature, a body
  (its mass ratio). Held items: mass, hardness, sharpness, temperature.
- **Interoception.** reserve / capacity, water / normal, T_b, damage, gut fill, light level, own speed.
- **Neighbour search** uses a spatial hash on the fine grid (bucket per cell, 3 x 3 cells, at most K = 32
  candidates). No O(N²) per arena.

### As built (`senses3.py`)

How the code differs from the design:
- bodies and items share one set of object channels, so there is no animate/inanimate label;
- a second, coarse hash gives long range;
- absorption follows ISO 9613-1, not f²;
- the eye has a contrast threshold;
- there are four smell fields;
- body temperature is coded with a sign.

**Layout.** IN_DIM = 157 = 8 sectors x 16 + 8 smell + 6 touch + 8 held + 7 interoception.
- Sector 0 is straight ahead; the others follow clockwise.
- Each sector has 7 vision channels, 8 call features and a broadband `noise` channel. The vision channels are
  `size`, `contrast` and `motion` of the nearest seen object, plus `cover`, `flow`, `ground` and `emitted`.
- Every channel is dimensionless and of order 1. Dead slots are zero.

**Objects.** Bodies and items on the ground are one kind of thing to the senses. An object is a sphere of its own
volume with a reflectance (from the `optics.py` table), its own glow, a speed, a mass, a hardness and a temperature.
A still body with a stone's size, reflectance, hardness, mass and temperature gives an identical input (tested).

**Vision.**
- The eye's size is its gene's mass, as one sphere at tissue density. Its resolution is limited by diffraction or by
  receptor spacing. Its photon catch is scaled by (1 - damage).
- A target is seen when three tests pass:
  1. photon noise allows it (Rose criterion, k = 3);
  2. its contrast, diluted over the blur spot, clears c_min = 0.02 (Blackwell 1946);
  3. the terrain line of sight is clear (fine elevation and the real-radius bulge).
- Light per fine cell is the patch's bout light x the PAR share, plus a moonless night sky (0.002 lux), plus fires.
  Terrain blocks fire light just as it blocks sight.
- `ground` samples reflectance along the sector's centre line, out to 8 cells.
- `emitted` is log10(1 + x) / 8 decades, where x is the sources' photons over the background behind them. The
  background is counted over the larger of the source's image and the blur spot. A 1,100 K flame (luminance 0.12
  W/m²/sr) reads 0 by day against sunlit ground (12.7) and about 0.71 at night.

**Hearing.**
- A call's acoustic power is `call_w` when the caller passes it: `body.call_power_w` gives loudness x 1 W per kg of
  voice organ. Otherwise senses3 falls back to its own formula, loudness x 0.005 x muscle x voice mass.
- The call's frequency is the quarter-wave resonance of the voice organ, a tube of aspect ratio 4.4. A 70 kg body
  with 3 g/kg of voice organ calls at about 504 Hz; a 20 g founder with voice 1e-3 calls at 11 kHz.
- Locomotion noise ∝ mass x speed²: 45 dB at 1 m for 70 kg at 1.4 m/s.
- Intensity at the listener:
  - spherical spreading;
  - ISO 9613-1 absorption (classical plus O2 and N2 relaxation, from the arena's pressure, temperature, humidity and
    composition), which reproduces ISO 9613-2's table within 1.5 %;
  - x 0.25 in intensity (x 0.5 in pressure) behind terrain;
  - nothing below 1 kPa of air.
- The ear's threshold falls as its collecting area grows.
- `observe` requires the arena's `Air`, built from the climate's partial pressures and vapour. There is no silent
  Earth default.

**Two tiers of neighbours (sight and hearing).**
- The fine tier is the hash of 3 x 3 fine cells, keeping the K = 32 nearest.
- The coarse tier has cells of 8 x 8 fine cells and scans 3 x 3 of them, with far_k = 16 candidates. It holds only
  targets whose upper-bound range (for the arena's best eye or ear) reaches beyond one fine cell. Candidates are
  ranked by distance over range.
- A far candidate counts only when it is detected on its own.
- Range is physical and isotropic out to at least 128 m.
- Measured at full scale with loud random founders:
  - calls carry hundreds of metres to kilometres;
  - far_k binds for every body (`far_bound` 1.0);
  - `beyond` (range over 128 m) is 0.5 at night and 1.0 by day;
  - each body hears 18.9 sources and sees 6.9 bodies by day.

**Fires.**
- At each body the fires that matter are kept: those giving at least c_min of the sky's light, or seen by the eye.
- They are ranked per sector and line-tested 2 at a time, for at most 3 rounds.
- Untested fires still light the point, unoccluded, but are left out of the emitted sum.
- Measured error: 1.6-8 % of sectors, with 64-256 fires in rugged terrain (`fires_culled`).

**Smell.** Four chemical fields (`patch.Volatiles`):
- plant volatiles;
- sulfur volatiles of rotting animal matter (carcasses and faeces);
- smoke (the CO tracer of fires);
- CO2 from point sources: bodies' metabolism and fires. The ecosystem's own exchange is not in it.

Each field gives a concentration code, log10(1 + c / c_th) / 3 decades, and its gradient along the heading. c_th is
0.01 x the human-panel threshold (an animal nose). A 10 kg carcass codes 0.23 in its own cell and 0.11 three cells
away; dense plants code about 0.8.

**Touch.**
- Under the mouth:
  - leaf cover (fAPAR);
  - damp (soil water over the bucket);
  - wet (free water depth over body radius);
  - for the nearest object in contact: mass ratio, hardness (Mohs / 10) and temperature, tanh(dT / 10 K).
- Held items: mass ratio, hardness, sharpness and temperature.
- There is no taste. Sea water and fresh water feel alike; salt acts only through physiology. `water_at`'s is_sea
  flag never reaches the senses.

**Interoception.** In `brain3.INTERO_NAMES` order, as the modulator's input. The table is in section 5.

**Cost at full scale** (4 x 8,192 bodies, 4,096 items per arena): 1.0 s per observe at night and 1.35 s by day
without fires; 2-5 s at night with 8-256 fires.

## 5. Brain and learning: nothing innate (`v3/brain3.py`)

- **Network.** The recurrent network of version 2 (`planet.brain`), with hidden size `hidden` (default 128) and
  active units k as a gene with a maintenance cost per unit. Founders have small random weights and no biases toward
  any output.
- **Motor outputs** (continuous, physically scaled): turn, thrust, mouth, grip_left, grip_right, release, force
  (push or strike with what is held, or a bare limb), rub (rubs the held item against what is in contact; friction
  work becomes heat), press (presses both held items together, so binding can happen if a binder is between them),
  place (sets the held item down at the mouth position, so into a fire if one is there), loudness, the vocal vector
  [D], and divide.
- **Learning.** Hebbian plasticity is modulated by an **evolved readout** m = tanh(Σ g_i x interoceptive_i):
  - the readout gains g and the plasticity rate eta are genes
  - founders start with g = 0 and eta = 0 (no learning, no notion of good)
  - if learning helps, evolution turns it on and decides what it values
- **The loop.** Physics gives some bodies more energy, water and intact tissue. Those divide more. Their genomes
  spread. That is the only feedback loop.

### As built (`brain3.py`)

**Network.**
- Version 2's recurrent tanh network, with hidden size 128. Weights fold back by reflection at ±3.
- Active units k run from 2 to 128.
- Founders: k is uniform in [2, 32], weights are N(0, gain² / fan-in) at version 2's gains using each founder's own
  k, and every bias is 0.
- Each unit is 1 mg of brain tissue at the nerve rate (body.py).

**Outputs (OUT_DIM 20).**
- turn and thrust lie in [-1, 1] (tanh). Thrust is signed: forward or backward.
- mouth, grip_left, grip_right, release, force, rub, press, place, loudness and divide lie in [0, 1] (sigmoid).
- vocal [8] lies in [-1, 1].
- A raw output of 0 maps to the middle of its range. Over 12 seeds x 8,192 random founders, every decoded output
  sits within 0.01 of mid-range.
- In the world every output acts at a rate per second, so its daily frequency does not depend on the bout length
  (fix stage). Divide is a development rate (section 3). Grip, release, place and press are continuous-time events
  at rates set by their intensities, drawn on the world generator (`manipulate.EVENT_RULE`). Force, rub and thrust
  are continuous shares of the bout. `brain3.fire`, a Bernoulli draw that raises without a generator, remains only
  as the fallback when no bout length is given.

**Plasticity.** dW = eta x m x outer(pre, post), on each body's own active block, clipped.
- m = tanh(Σ g_i x intero_i).
- eta is a log gene on [1e-6, 0.1] per bout. Founders sit at the floor of 1e-6, not at exactly 0, because a log gene
  needs a positive floor.
- g is a linear gene on [-8, 8], one per channel. Founders have g = 0, so m = 0 exactly, and their live weights stay
  bit-identical.
- No bias channel and no felt-change channel is supplied.

**The evolved modulator's interoceptive channels** (coded by `senses3`):

| Channel | Code | Range |
|---|---|---|
| reserve | fat store / (0.3 kg fat per kg lean x 39.5 MJ/kg) | from 0, and above 1 once the store passes that scale (it has no cap); founders start at 0.5 |
| water | body water / normal | from 0.6 (death) to above 1 (the bladder) |
| body_temp | (T_b - 294.5 K) / 23.5 K | -1 at freezing, +1 at denaturation; signed, so a gain can weigh cold against hot |
| damage | damage | 0 to 1 |
| gut | dry gut contents / gut capacity | 0 to 1 |
| light | log10(1 + E / E_night) / 8 decades at the body (sky, plus fires with a clear line) | about 0 to 1 (noon about 0.96) |
| speed | Fr / (1 + Fr), with Fr = v / sqrt(g x 2 r) | 0 to 1 |

The G_MAX = 8 bound assumes order-1 channels, and all seven are of order 1. Water, which sits near 1 most of the
time, can serve as an evolvable offset.

**Speed and memory.**
- `think` for 8,192 bodies takes 5.4 ms on the CPU at k ≤ 32 and 25 ms at k = 128. `hebbian` takes 3.3 ms.
- `mutate3` on 8,192 bodies takes about 2.2 s, but only the dividing bodies are mutated.
- At capacity a patch holds about 1.3 GB of genome plus 0.54 GB of live weights in float32. bfloat16 weight storage
  (stochastic rounding, tested) roughly halves this.

## 6. Variation and founders

- **Mutation at division.**
  - Every gene and weight gets Gaussian noise (scale gene `mut`, itself heritable and bounded).
  - k changes by ±1 with a small probability.
  - Organ genes are clamped to their physical bounds.
- **Founders** (new_rule, stated: the starting population of the loop):
  - `founders` per patch (default 2,048), at uniformly random positions with random headings.
  - Each has a random genome. Body genes are drawn broadly within physical bounds: size 2-200 g, organ sizes small.
  - Bodies start at their size with half a reserve and normal water.
  - The founders' scale (small, so fast-dividing) is the chain's bodies era expressed physically: about 256
    cells, i.e. tiny animals. It sets only the starting range, not a target.
- **Capacity.** Capacity per patch (default 8,192 slots) is sized above the patch's carrying capacity. When it is
  reached, divisions fail and this is counted and reported (`capacity_full`). The cap is never a silent ecological
  limit.

### As built (`brain3.mutate3`, `body.found`, `body.divide`)

**Mutation.**
- Every float gene takes one Gaussian step in its own coordinate:
  - lin: the value itself;
  - log: its logarithm;
  - logit: its log-odds.
- The step size is mut x the gene's unit, and the result is mirror-reflected into the bounds. The step never
  depends on the gene's value. So with no selection each gene drifts only toward its stated neutral law, which is
  uniform in its coordinate. Measured after 300 neutral generations, every coordinate median stays between 0.497
  and 0.502.
- mut itself takes log steps of fixed size MUT_TAU = 0.1, within [1e-3, 0.5]. Founders draw it log-uniform on
  0.01-0.1. The child's new mut scales all of its other steps.
- k changes by ±1 with probability 0.05, within [2, 128].
- Weights of inactive units never mutate. A unit switched on is drawn fresh at founder scale.
- bfloat16 weights are rounded stochastically.

**Bounds.** Bounds are physics or measured records, never targets:
- setpoint: 271-318 K;
- skin_perm: 1e-12 to 7.15e-5 kg m^-2 s^-1 Pa^-1, from free water down to ten times below the most waterproof
  arthropod cuticle;
- size: 1 mg to 100 t;
- muscle up to 500 W/kg, fur up to 0.1 m, thermo_gain up to 200 W/(kg K): measured records;
- air_l_kg: 0 to 1 L per kg of lean, from no air store to about the lean's own volume (4 October);
- o2_carrier: 0 to 298 mL O2 per kg of lean, from no carrier to all the lean's protein (review of 4 October);
- tissue shares up to 1: mass conservation, closed by `tissue_shares`.

`bound_report` gives the share of a population sitting on each bound, so no bound acts silently.

Genes that start at a negligible floor (eta and thermo_gain) leave it by unbiased drift at about mut e-folds per
generation. Crossing 5 e-folds at mut = 0.03 takes about 1e4 generations by drift alone.

**Founders (`body.found`).** F per arena, at uniformly random positions and headings. Genomes come from brain3, plus
body.py's extra genes, drawn broadly:

| Gene | Founder draw |
|---|---|
| size | 2-200 g, log-uniform |
| muscle | 10-200 W/kg |
| muscle_frac | 0.05-0.5 |
| enzymes | 1e-3 to 0.05 |
| organs | 1e-5 to 2e-3 |
| fur | 1e-5 to 2e-3 m |
| skin_perm | its whole range |
| setpoint | uniform |
| repair | 0-1 |
| offspring_share | 0.01-0.99, logit-uniform |
| fat_store | 0.03-0.6 |
| bladder | 1e-3 to 0.1 |
| kidney | 1e-3 to 1e-2 |
| air_l_kg (4 October) | 0-1 L per kg lean, uniform over its bounds: zero is included |
| o2_carrier (review of 4 October) | 0-298 mL O2 per kg lean, uniform over its bounds |
| thermo_gain, eta | at their floors |
| g | 0 |

The air and carrier genes are drawn after the headings, so adding them left every earlier founder draw unchanged.

Each founder starts at its size, with 0.15 kg of fat per kg of lean and normal water.

**Founders and slots in `world3`.** 512 founders per arena (the design said 2,048). Since 4 October each arena has
4,096 slots (1,024 at the fix stage; the design said 8,192). Founders are placed once. An arena whose bodies all die
stays empty.

**Capacity.** A division into a full arena fails and is counted (`capacity_full`). The cap binds: in the 50-day
mini-world (96 founders and 192 slots per arena), and on the Earth reference by day 2 at 1,024 slots. A 2 km patch
could feed far more 2-200 g bodies than any slot count that fits in memory. So binding is reported as
`capacity_bound`, a numerical limit, not ecology (section 1).

## 7. Plants and the patch ecology (`v3/patch.py`, extending `planet.biosphere`)

- **Fine-grid plants** per patch cell: soft tissue, wood and litter, using version 2's production equations with
  the patch's light (including slope and aspect), temperature, soil water and CO2.
- **No seed floor.** Plants spread by **seed dispersal**:
  - Each cell exports a share of its production as seeds to its 8 neighbours (a dispersal kernel, new_rule
    constants with a reference range).
  - Seeds germinate where production would be positive.
  - Spin-up: seeds are sown sparsely and uniformly, then plants grow for `veg_spin_days`. Plants end up where they
    can grow.
- **Soil water and runoff** per fine cell come from the global cell's precipitation, with runoff routed downhill to
  the lowest cells. Ponds form where water collects (fresh water exists where it physically accumulates).
- **Global coupling.** The patch's net CO2, O2 and H2O exchanges go to the world's air. The coarse global biosphere
  (version 2, with the fixed seed rule) runs the rest of the planet.

### As built (`patch.py`)

**Plants.** Each fine cell holds soft tissue, wood, litter, mineral N and a seed bank. They follow version 2's
production equations at the cell's light, temperature, soil water and CO2, with two exceptions:
- fAPAR follows leaf area: LAI = SLA 20 m²/kg x leaf share 0.5 x dry soft tissue, with extinction 0.5. Its initial
  slope is 10.6 m² per kg C (version 2's was 2). Seedlings grow at about 0.06 per day, inside Grime & Hunt 1975's
  range of 0.05-0.3.
- Standing water drowns production (x exp(-depth / 0.1 m)), and nothing germinates under more than 1 cm.

**Seeds.**
- 10 % of growth becomes seed. Half of it lands in the 8 neighbouring cells, weighted by 1/distance.
- Germination e-folds in 10 days, wherever a seedling's net production would be positive.
- The seed bank halves each year.
- The spin-up sows 1/64 of land cells uniformly at 1 g C/m². The sown carbon is taken from the air and reported in
  the next step's exchange.
- There is no minimum stock anywhere.
- The spin-up length a canopy needs (`veg_spin_days`) must be re-measured after the fAPAR fix. The earlier estimate,
  1,500-2,000 days, was for the slower law.

**Water.**
- Each fine cell has a soil bucket (150 kg/m²), snow and open water.
- Runoff follows the depression hierarchy (fill-spill-merge, Barnes et al. 2020). Each closed depression keeps its
  own water and level until it is full to its sill, then spills into its neighbour. Neighbours share one level only
  above their merge sill. A nested-pit test checks this, and water is conserved to 1e-12.
- Water leaves at the sea and at each arena's lowest cell, a sink at its own level.
- Ponded water seeps away at 2 mm per day.
- A temperate year starting from empty lakes (288 K, 2.7 kg/m²/day of rain):
  - 2.1-2.3 % of cells end up under more than 1 cm of water, with a mean depth of 0.3-1.5 m;
  - 25 % of the rain leaves the patch and 67 % evaporates.
- There are no flowing streams. Fresh water exists only in lakes, ponds and wet soil.

**Nitrogen.** A bite removes all the eaten N from the cell. Bodies carry it and return it in urine, faeces and
carcasses at their own position. Seeds carry N.
- Fixation follows growth (fix stage). Symbionts fix `bnf_share` = 0.5 of the N the day's growth asks for, where
  the mineral pool limits it (Cleveland et al. 1999 and Vitousek et al. 2013, quoted from memory).
- This replaces version 2's rule, which refilled the gap to the starting stock. That was a replenishment with a
  target, which section 0 forbids. The purity test checks it is gone.
- The N2 fixed is taken from the global air.

**Smells.** Four near-surface concentration fields (section 4), following dc/dt = K∇²c - c/τ + E/h. They are solved
exactly in Fourier space on the periodic grid, for any time step. Carcass emission uses 5 % of the flesh per day.

**Ground look.** Layered from bare ground through litter cover, canopy (fAPAR) and ponds (ice when frozen) to snow.

**Ledgers.** Carbon, water and nitrogen are kept per arena in float64. Over 100 days for 2 worlds x 2 patches, with
taking and giving, water closes below 1e-9 of the stores and C and N below 1e-6. The step's evaporation is checked
against the physical field to within float32 rounding.

**Cost.** `make_geometry` for 16 arenas takes about 0.2 s. One patch day takes about 40 ms at the default size, and
routing takes 18 ms per day for 16 arenas.

## 8. Crafting physics inside patches (`v3/manipulate.py`)

This module reuses `planet.crafting` and `planet.materials`. Items and fires carry patch coordinates (metres), and
the existing transform, fire and knapping physics apply unchanged. Only the triggers change, from named actions to
physical contact:
- **Grip** picks up the item under the mouth (if light enough for muscle power).
- **Force** with a held item on an item or body in contact delivers ½ m v² from muscle power and the lever. On a
  brittle item this knaps; on a body it wounds.
- **Rub** turns work (force x sliding distance per bout) into heat in both items. If a fuel item reaches its
  ignition temperature in sufficient O2, it ignites: a fire is born. No `make_fire` exists. *(Superseded: a fire is
  born from an ember on tinder; see As built.)*
- **Press** binds two held items when a binder (fibre, resin, hide strip) is held or in contact, producing the
  assembly fields.
- **Place** sets an item down at the mouth position, so into a fire, onto a hot item, or into water.
- **Heat** follows contact with fire and transforms follow their physical conditions, as in version 2.
- Cooking follows from flesh heated above its cooking temperature (the same transform physics), and digestion
  gives the cooked gain.

### As built (`manipulate.py`)

How the code differs from the design:
- a fire is born only from an ember on tinder;
- the contact temperature is capped;
- force is many strokes over the bout, not one blow;
- press needs a separate binder item, and the grips set the tool's orientation;
- striking the ground is the source of loose material.

**Time and power.**
- A bout's time is shared by press (600 s per press), rub, force and locomotion (`time_split`). When together they
  ask for more than the bout, they are scaled down together. body.move must use the returned `thrust_share`.
- Peak power (`body.muscle_peak_w`) drives the brief acts: lifting a grip and each stroke.
- Sustained power drives the long acts: rubbing, pressing and the spacing of strokes. The limb gets 25 % of
  sustained power, from version 2's arm cranking against whole-body work.
- Work is returned as mechanical work; body.py pays for it at efficiency 0.25.
- Rubbing with no partner does no friction work. The free limb's oscillation is still charged (about 0.2 W at
  70 kg).

**Grip.** It lifts the nearest loose item in contact if the item's weight is at most peak power / 1 m/s, minus what
is already held. The strongest claimant wins a contested item. Hot items can be gripped (version 2's refusal is
gone), and their temperature reaches touch and burns.

**Force.**
- Each stroke runs at constant peak power, v³ = 3 P a / m_eff, which a test checks against numerical integration.
  Every held item is a rod of its own length, and the limb is a rod.
- Strokes per bout = time share x contact time / stroke cycle, limited by the limb's sustained power.
- On a brittle item softer than the striker, flakes come off (blow / 500 J per kg) and the edge sharpens. The flakes
  of one bout form one heap item.
- On a body, strokes wound. 9 % land on the head (a Binomial draw). Damage is 1 - exp(-wound / (k x M)), with
  k = 0.7 J/kg on the head (skull fracture, Yoganandan 1995) and 4.0 J/kg on the body (thoracic impact, Kroell
  1974). One 69 J stone blow gives 0.30 on the head and 0.06 on the body.
- With nothing in contact, strokes break pieces off the ground under the mouth, in proportion to the reachable mass:
  sticks from standing wood, fibre from litter, and clasts from the top 5 cm of the deposits.
- Besides carcasses, this is the only source of loose items. Founder-size blows rarely detach anything.

**Rub: the ember-tinder-fire chain.**
1. 10 % of the rubbing work becomes heat at a 5 mm spot (version 2's drill values).
2. The spot's temperature comes from conduction into both partners. On the partner the spot stays on, it is a
   stationary disc with its transient. On the partner it slides over (at 0.5 m/s), it is a moving-source (Peclet)
   flash.
3. The temperature is capped at the lower decomposition or melting temperature of the two partners: wood and fibre
   pyrolyse at 673 K, tissue chars at 573 K, minerals melt. Heat above the cap goes in one of three ways:
   - it chars tissue, which goes to the soil;
   - it burns a bare limb's skin (`skin_char_j` and `skin_char_kg`, for body.py);
   - it goes into the bulk of a pyrolysing or melting partner.
4. A fuel partner strictly above its ignition temperature, in at least 15 % O2, on a dry spot, makes an ember of 2 g
   of its fuel.
5. The ember becomes a new fire only on tinder touching it: the igniting partner, the other partner, or the nearest
   loose item at the mouth. The tinder must be at least 50 % plant fibre. Without tinder the ember dies.
6. The fire then follows version 2's `fire_step`: burning, heating, transforms and charring. 30 g of tinder does not
   light a 0.5 kg board by itself; bigger fuel needs kindling.

Measured:
- A wood stick on a board needs 48.9 W of limb rubbing for a whole bout (196 W of whole-body sustained power) to make
  an ember. Doing it in 60 s needs 155 W.
- A bare limb on wood reaches only the 573 K skin cap and burns. It makes no fire.
- Random founders (2-200 g) made 0 embers in 8 bouts with more than 100 rubbing contacts. Random 0.5-80 kg bodies
  made 0-1 fires from 82-99 rubs.

**Press.** It joins the two held items only when a separate binder item (fibre, resin, hide) lies loose at the
mouth. The left grip's item is the handle end and keeps the joint; the right grip's item is the head, whatever
their hardness. The lever comes from the items' own lengths.

**Place, release and heat.**
- Place and release set items down at the mouth, inside a fire's bed when the mouth is in one.
- `heat_step` runs version 2's `fire_step` on patch coordinates. Items in a bed, held ones included, are in the
  fire; all other items get the fires' radiation. It reports the mouth's fire temperature and the radiant flux for
  body.py's burns.
- Cooking is flesh that has been above 70 °C. A big fire chars flesh away to the soil. Clay in a hot enough bed
  becomes ceramic.

**Not built.** Push; litter on the ground as a rubbing or fuel partner (litter becomes items only when struck);
quenching in water; contact heating from one item to another.

**Ledgers.** Species and element masses close below 1e-6 relative over random bouts, including heat, decay, bites,
carcasses and ground pieces. Rubbing reports where its energy went. A crowded bout (4 x 2,048 bodies, 8,192 items,
64 fires) takes about 0.6 s on the CPU.

Behaviour statistics must use energies (blow x strokes, wound, heat). A partner index is reported only where a
stroke happened.

## 9. Gates (tests)

- **Forbidden-mechanism grep** over `haishool/life9/v3`: no named action list with purposes (`ACTIONS = ("rest",
  "forage"...)`), no `innate`, no `forage_bias`, no `background_drink`, no `time_budget`, no `pedigree`/`relatedness`
  inside senses or actions, no `seed_floor`, no `LIFESPAN`/`MATURITY` allometry, no reward constants.
- **Ledgers:** energy, carbon, oxygen, water and items close over 100 days for 2 worlds x 2 patches.
- **Physics unit tests:**
  - A 20 g ectotherm cools to air temperature.
  - An endotherm genome with gain holds its setpoint at a fuel cost matching A ΔT / R_insulation.
  - Dehydration, freezing and starvation kill at the physical thresholds.
  - Rubbing dry wood with enough work ignites it at 21 % O2 and not at 3 %.
  - A bite transfers flesh and wounds.
  - Division conserves mass, energy and water.
- **Formation v3:** water, C and N budgets for 781, Earth and 20 synthetic planets. Earth's water comes out at
  about Earth's (calibration). The derived T_s is reported against the chain's.
- **Determinism and exact resume** on the CPU. GPU statistics match the CPU within noise (bench script).
- **No-selection control:** `selection=False` (random survival of the same number of deaths) must give genome
  drift only. It is used in analysis to separate selection from drift.

### As built

- At the end of the fix stage (3 October) 352 tests pass on the CPU: formation3 37, patch 49, brain3 41, senses3 47,
  manipulate 56, body 55, world3 23, view3 39 and the package-wide purity gate 5. Version 2's 372 still pass. The
  four decisions of 4 October add their own tests.
- **Forbidden-mechanism grep.** It runs per module, plus the package-wide gate `test_purity.py` (section 0).
- **Ledgers.** Each module closes its own:
  - patch over 100 days for 2 x 2 patches;
  - body over 400 bouts, plus a 100-day body-patch test on 4 arenas;
  - manipulate over random bouts.

  `world3` closes every ledger of every layer over 10 days on a small world, with the books crossing to the global
  layer checked (`air_booked`, `water_booked`). The 100-day world gate of the design is still to be run.
- **Physics unit tests in place:**
  - the 20 g ectotherm, and the endotherm at its insulation cost;
  - thermogenesis capped by aerobic capacity;
  - the dehydration, freezing and no-fuel thresholds;
  - a bite transfers flesh and wounds;
  - division conserves mass, energy and water, and the parent survives;
  - locomotion against measured speed and cost;
  - cold muscles move, bite (the jaw rate) and digest less.
- **The fire gate** now reads: rubbing dry wood with enough work makes a fire on tinder at 21 % O2 and not at 3 %, and
  an ember without tinder dies.
- **Formation.** 32 planets, more than the 20 synthetic ones asked for, pass `check_spec3` and close their budgets.
  Outcomes (runaway, frozen, water-covered, the mismatch) are reported, never gated.
- **Determinism and exact resume** on the CPU are tested in every module that draws random numbers, and for the whole
  world (`test_determinism_and_exact_resume`). `index_add_` is non-deterministic on GPU, so only statistical
  agreement is expected there. The first v3 runs (3 October) ran on the RTX 4090, the R9700 and the RTX 3060 as well
  as the CPU, and agreed within noise (PLAN.md).
- **Lineage labels are invisible.** Permuting them among the living gives the same senses and the same world
  (`test_purity.py`).
- **No-selection control.** The neutral `Shadow` replaces it as the drift baseline (section 3). `selection=False`
  remains as a mode, but it melts down.

## 10. What results will mean

- **Measured:** population per patch, lineages, gene trajectories (size, fur, enzymes, thermo gain, brain units,
  eta, readout gains), behaviour statistics (mouth on plant / water / body / item, movement relative to smells and
  sounds, rubbing, gripping, fires born, transforms), call-to-situation information (version 2's measures, adapted),
  and selection against drift by the control.
- **Honesty:** a patch going extinct is a result. Behaviour that appears only at the start (from random weights) is
  not emergence; emergence is a change sustained across generations, measured against the no-selection control.

### As built

- The drift baseline is the neutral `Shadow`, not `selection=False`. Neither removes fecundity selection: births
  still depend on the genome in both.
- Read these bound and budget reports alongside any result:
  - `brain3.bound_report`: the share of bodies at each gene bound, k at 128, and weights at the clip;
  - `capacity_full`, and `capacity_bound` with its day;
  - senses3's `bound`, `far_bound`, `beyond` and `fires_culled`;
  - manipulate's `no_item_slot` and `overflow`;
  - the movement sub-steps' `coarse_moves` and `travel_beyond_far`.
- Behaviour statistics come from energies, not from partner indices (section 8).
- **A behavioural baseline** (fix stage). 64 founders' networks per arena are kept unchanged. Once a day the living
  bodies' networks and these baseline networks are run on the same bodies' inputs. The mouth's intensity on plants
  and on water, against off them, is reported for both. Behaviour counts as changed only against this baseline and
  the shadow.
- **Viability** (fix stage). The summary reports births and deaths per day, turnover, the food energy taken against
  the energy spent, the fat runway, the arenas without plants and the lineage counts.
- **Reporting follows [PROTOCOL.md](PROTOCOL.md)** (owner decision, 4 October 2026). Every sampled seed gets one
  outcome: no star, no planet, no habitable planet, no bodies era (an exclusion by the chain's outcome, counted
  apart), extinct at a day, or alive; a seed that cannot be resolved is an error, not an outcome. The
  nothing-happened counts are reported first, over the whole sample.

## 11. Findings and open choices for the owner

### What the physics forced (results, not bugs)

1. **781 is colder than world7 says.** Its formation mean is 270.5 K (270.8 K before the nitrogen decision), 17 K
   below the chain's 287.7 K, with 13 Pa of CO2 (the model Earth has 28). It is old and metal-poor, so its rock
   makes less heat and outgasses less. Formation has no ice feedback, so the v3 climate decides whether 781 freezes
   over, and it must not use 270.5 K as a target. Plants there will see little CO2. Even without the broadening
   penalty it stays below freezing.
2. **Most planets are ocean worlds.** 23 of the 31 non-Earth planets are fully covered by water. Cold orbits give
   more water per kg, and big planets have shallower basins. Since the review of 4 October a world without land gets
   sea patches and founders, and physics decides (PROTOCOL.md rule 4).
3. **The model Earth had 2.05 bar of air** under one degassed share: chondritic N/H2O gave 2.3 x Earth's N2. Its O2
   was then 10 % of the air, so fire was impossible on it. **Resolved by the owner's decision of 4 October:**
   nitrogen has its own share, and the model Earth has 1.01 bar of dry air with 21 % O2 (section 2).
4. **Landless planets weather only on the seafloor.** They settle at 287-312 K with bars of CO2. The design's
   "weathering at the chain's temperature" sends 14 planets to 450 K. The self-consistent loop is now the contract.
5. **Fire needs a big body and a chain of fuels.** An ember needs about 50 W of limb rubbing for a whole bout, and
   the limb gets a quarter of sustained power. By our estimate the smallest body that can make an ember weighs a
   few kg if it has the strongest muscles, and about 20 kg with typical ones (100 W/kg, 40 % muscle). The ember
   dies without fibre tinder, and tinder alone does not light a board. Founders cannot make fire (tested).
6. **Calls carry far once absorption is right:** hundreds of metres to kilometres. At full scale the 16-place far tier
   is full for every body, so hearing is limited to the 48 most reachable sources.
7. **Dehydration was the main killer** in the first mini-world (481 of 567 deaths). Selection drove skin permeability
   down within 50 days, visibly against the shadow.
8. **A no-selection control with expressed genes cannot give drift.** A lethal genome keeps killing. The shadow is the
   baseline.
9. **Division cannot pay development heat within one 6 h bout.** The heat would cook a small parent. **Resolved at
   the fix stage:** development is a rate, paid over days (section 3).
10. **The 6 h bout is coarse for contact.** Two bodies touching for a whole bout with any force output nearly kill each
    other, because contact time comes from bout-mean velocities. Locomotion has the same problem. **Addressed at the
    fix stage:** 24 bouts a day, movement sub-steps of at most 32 m, and every rate per second (section 1).
11. **Synthetic terrain has pits everywhere.** Without conditioning, 19 % of every patch was lake. The conditioning is
    calibrated to Earth's lake share.
12. **Plants spread slowly from sparse seed** (about 0.06 per day per seedling), so the spin-up must be long.
13. **Broken stone never decays.** Items pile up wherever large bodies strike the ground. (`world3` now rots items
    with `manipulate.decay_step` each day; broken stone still does not decay.)

**First `world3` runs (3 October 2026, fix stage, before the four decisions).** These are a case study (781) and a
control (the Earth reference), not a sample (PROTOCOL.md).

14. **Planet 781 freezes at once.** Its derived climate has a 267.6 K mean with 54 % ice. All 4 patches of that run
    were at 247-260 K, and 3 were on the night side. Every founder died in the first hour. This is a valid "nothing
    happens" result.
15. **On the Earth reference** 4 of 16 patches lost all founders in the first hour. The rest grew and hit the
    1,024-slot cap by day 2. Food was far from limiting: a 2 km patch could feed millions of 20 g bodies.
16. **Fire was impossible on the model Earth.** Its 2.05 bar of air held only 10.4 % O2, below the 15 % limit (finding
    3; resolved by the nitrogen decision).
17. **Every founder was denser than water** and drowned in ponds within about 5 minutes. Nothing in the body plan
    held air (resolved by the air-volume gene).
18. **Speed** at 16 patches x 1,024 slots and 24 bouts a day: 3.7 s per day on the RTX 4090, 4.1 on the R9700, 10 on
    the RTX 3060 and 22-24 on this PC's CPU. The engine is bound by kernel launches, so more worlds per process
    amortise that. At 4,096 slots on this PC's CPU (review of 4 October): 35, 38, 47, 60 and 78 s for days 1-5 of the
    model Earth, and the slots bound on day index 4 (section 1).

### Decided on 4 October 2026

The owner decided four of the open choices. Section 13 lists them with where each is described.
- the patch scale: keep 2 km patches, 4,096 slots per arena;
- a separate degassed share for nitrogen;
- a heritable air-volume gene;
- the five-rule experiment protocol ([PROTOCOL.md](PROTOCOL.md)).

### Choices still open

- **Locomotion cost.** `world3` already moves bodies by Kram & Taylor's force cost (section 3), with c = 0.2 J/N
  quoted from memory. The owner has not yet decided this contract change. The details are below.
- **The planet pick (PROTOCOL.md rule 2)** is settled by the review of 4 October: chain version 3 takes world7's
  physical pick, the innermost temperate terran planet with liquid water, and records the bridge's old pick beside
  it. In all 66 seeds of the regenerated cache the two agree (planet 1).
- **'No bodies era' seeds.** v3 founds no bodies on a planet whose chain made no bodies era. That conditions the body
  stage on an earlier stage's outcome while rule 3 is deferred, so these rows are labelled excluded and counted apart.
  The protocol-consistent alternative is to build them and let physics decide.
- **Breathing constants.** The O2 floor of the store (0.30 of the alveolar pO2), the anoxia tolerance (180 s of
  resting O2 use) and the lung's uptake per kg (a 1 kg mammal's VO2max) are body-plan constants; any could become a
  gene.
- **The N2 target.** The model Earth is calibrated to 79.1 kPa of N2 (dry standard air). The owner asked for about
  78 kPa; Earth's N2 inventory would give 75.5 kPa on the model Earth. Choose a partial pressure or a mass.
- **GPU precision and slots.** The float32 default world does not fit every cluster GPU (section 1).
- **Carrying state between stages (PROTOCOL.md rule 3)** is deferred. The chemistry layer and inventory-carrying
  hand-offs are the planned remedy.
- **The meteorite anchor temperatures** (EC 400, OC 350, CM 160, CI 120 K, taken from this spec's ranges). Anchors at
  the classes' formation sites leave the water-world count at 23, but break the single degassed share of water and
  C (107 bar on the model Earth). Either choose the anchors, or give each volatile its own share, as N now has.
- **Clouds.** They are held at Earth's 25 % of the greenhouse on every planet, which is worth 10.6 K on 781. This is a
  placeholder until a cloud model exists. Dropping it would freeze 781 hard.
- **Collision-induced absorption is missing.** 20 planets have more than twice the model Earth's dry air, up to 31
  times. Their temperatures are uncertain and probably too cold. Adding it needs a cited grey coefficient.
- **The pressure-broadening exponent** is 0.5, chosen to match Goldblatt's doubled-N2 warming. An exponent of 1.0
  would nearly triple that warming.
- **Basin depth.** The split rule is the default. On 781 it gives a 51 % ocean; fixed D gives 62 % and D ~ 1/g gives
  48 %.
- **Locomotion cost, in detail.** The spec's c_m = 0.1 and 25 % efficiency, plus Fedak's internal work, give 7-9
  J/(kg m) at every size. The measured values are 73 at 2 g and 2.7 at 70 kg. Small bodies were 5-10 x too cheap and
  covered 48-88 km per 6 h bout, wrapping round the 2 km arena many times; large ones paid about 3 x too much. The
  fix stage did both options:
  - movement sub-steps within shorter (1 h) bouts;
  - Kram & Taylor's force cost in `world3.move_kt`, which comes within about 12 % of the measured curve from 2 g to
    70 kg. It is a contract change waiting for the owner's decision.
- **The wear rate** is calibrated on Rubner's lifetime energy (0.92 MJ/g), which is a datum about lifespans. At repair
  0 it gives lifespans of years, so selection on repair will be weak within a run. The alternative is an in-vivo
  protein-damage rate.
- **A child is born with no damage or wear,** although its tissue is the parent's own. Physically it would carry the
  parent's wear share. As built, putting most of the mass into the child rejuvenates a lineage. Decide which rule
  to keep.
- **Interoception has no constant (bias) channel and no change-per-bout channel,** so the evolved readout can weigh
  only levels. Adding either is a design decision, and it would grow n_intero.
- **Brain cost** is 1 mg per hidden unit at any body size. That is negligible for large bodies (128 units weigh
  0.13 g) and up to 1.6 % of a 2 g founder.
- **Smaller calibrations to watch.** Each is tagged new_rule in `PROVENANCE`:
  - the lake-keeping depth (0.4 x rms) and the 5 km relief break;
  - the nose factor (0.01) and the contrast threshold (0.02);
  - the voice-organ aspect ratio (4.4);
  - the limb's share of sustained power (0.25);
  - friction efficiency 0.1 for every rubbing pair (a plough rub may dissipate more);
  - the 2 g ember, with plant fibre as the only tinder;
  - knapping at 500 J/kg;
  - item capacity, since heaps never decay;
  - every air store charged as mammal lung (0.21 kg per litre); bird air sacs and swim bladders weigh less.
- **Citations to verify.** Some were written from memory and are marked in `PROVENANCE`: Goldblatt 2009, Brady &
  Gislason 1997, Alt & Teagle 1999, Gaillardet 1999, Stein & Stein 1992, Eakins & Sharman 2012, Lillywhite 2006,
  and the meteorite contents (Lodders 2003, Piani 2020). Added at the fix stage and on 4 October:
  - Kram & Taylor's c = 0.2 J/N (the law's form is confirmed, the number is not);
  - the protein synthesis rate (Waterlow et al. 1978) and the paddling efficiency (Fish 1984);
  - nitrogen fixation (Cleveland et al. 1999, Vitousek et al. 2013) and Earth's missing nitrogen (Marty 2012);
  - the lung mass per litre (Stahl 1967), the breath-hold O2 floor (Lindholm & Lundgren 2009) and the blood and
    muscle O2 store (Kooyman 1989, Ponganis 2011);
  - from the review of 4 October: VO2max (Taylor et al. 1981), Krogh's diffusion constant, Huefner's number, the
    red cell's life, Szpilman 2012, sea water's density (EOS-80), Churchill 1983's free convection, the wind drift
    of the sea surface (Wu 1983), Trenberth & Smith 2005 and the low-pressure flammability limit.

## 12. What the world module must wire

- **Terrain.** Build the globe with `formation3.make_terrain3`, never `globe.make_terrain(relief_m)`.
- **Climate.**
  - Do not use `t_surface_target_k` as a target for frozen_mean planets.
  - Give the v3 climate formation3's outgassing (`outgassing_rel_earth`) and weathering mode. Version 2's dynamic
    carbon cycle outgasses by heat alone and weathers per m² of land, with no pCO2 term.
- **Patch.**
  - Pass `specs=` to `forcing_from_climate` so it gets the day length.
  - Forward `diag['exchange_mol']` to `climate.add_gas(gas_delta(...))` every step.
  - Build an `Air` per arena from the climate's partial pressures and vapour; `observe` requires it.
- **Senses.**
  - Build senses3's bodies with `senses3.Bodies.from_mapping(body.sense_view(b, loudness, calls))`. That passes
    body's call power, so the power paid equals the power heard. senses3's own fallback formula is different.
  - Deposit metabolic CO2 (via `metabolism`'s `metabolic_w` and `patch.metabolic_co2_emission`), and the smoke and
    CO2 from fires, into the volatile fields.
  - Radiated call power does not yet depend on the air's impedance.
- **Manipulation.**
  - `manipulate.bout` takes `peak_w` and `sustained_w`. `body.manipulation_view` still returns a single `power_w`,
    already reduced by |thrust|. Until that view is updated, the world must build the two powers itself:
    `body.muscle_peak_w`, and SUSTAINED_SHARE x aerobic x peak, with no |thrust| cut, because `time_split` already
    shares the bout.
  - Pass `vel`, `body_pos` and `mouth_pos`.
  - Feed `thrust_share` into `body.move`, and `work_j` into `body.bout(work_j=...)`.
  - Add force's wound damage to `b.damage`.
  - Turn rub's `skin_char_j` and `skin_char_kg` into burns, and heat_step's `mouth_fire_k` and `radiant_w_m2` into
    burns or absorbed heat (`Env.heat_w`).
  - Pass `ground=ground_stock(...)`, and call `take_ground` for broken pieces.
  - Use `take_items` = `manipulate.remove_mass` and `spawn_items` = `manipulate.spawn_items`, so that the item ledger
    stays closed.
- **Movement.** Sub-step `move(dt_s)` with re-sensing, then pass `loco=` to `body.bout`.
- **Ledgers.**
  - Add `body.transit_kg(b)` to the world's water, C and N books.
  - Wood items carry Ca and K that the patch does not track.
  - Bone mineral and salt on land leave the world.
- **Capacity.** Size the body slots and the item pools, and count `capacity_full` and `no_item_slot`.

### As built (`world3.py`, fix stage)

`world3` wires all of the above except the items below.

Wired:
- the terrain from `make_terrain3`;
- no climate target for `frozen_mean` or `runaway` planets: their climate keeps the spec's albedo and finds its own
  state;
- `forcing_from_climate` with the specs, and an `Air` per arena from the climate's partial pressures, reduced to the
  arena's height;
- the patches' carbon and nitrogen, and the bodies' and fires' gases and water, booked to the global layer each day;
- the senses built from `body.sense_view`, so the call power paid is the power heard;
- metabolic CO2, smoke and fire CO2 into the smell fields;
- peak and sustained power built by the world, `thrust_share` into locomotion, `work_j`, wounds, burns, the fires'
  heat and the hands' heat into the bodies;
- ground pieces and items through `take_ground`, `remove_mass` and `spawn_items`, so the item ledger closes;
- `transit_kg` in the world's books; items leaving to the ground counted by element;
- `capacity_full`, `no_item_slot`, `no_fire_slot` and `capacity_bound` counted and reported.

Still open:
- **The climate's carbon cycle is version 2's.** It outgasses by radiogenic heat alone and weathers per m² of land
  with no pCO2 term and no seafloor term. formation3's outgassing and weathering mode are not passed on. So the
  run's CO2 can drift from formation3's steady state, and a landless planet has no weathering sink in the run. The
  drift is slow on run time scales (Earth's outgassing is about 1e-4 of its air's CO2 a year), but it is a mismatch.
- **Movement is sub-stepped without re-sensing.** The brain's command holds for the whole 1 h bout. The bout was
  shortened instead (section 1).
- **Radiated call power** does not yet depend on the air's impedance.

## 13. The owner's decisions of 4 October 2026

| Decision | What changes | Where |
|---|---|---|
| **Patch scale** | Patches stay 2 km. Each arena holds 4,096 body slots (was 1,024). Binding is still reported as `capacity_bound`, a numerical limit, not ecology | Sections 1 and 6 |
| **Nitrogen** | Nitrogen gets its own degassed share, one more Earth calibration, so the model Earth has Earth's 79 kPa of N2. Its air comes to about 1 bar with 21 % O2, so fire is physically possible on Earth | Section 2 |
| **Buoyancy** | A heritable air-volume gene (lungs, air sacs): litres of air per kg of lean, with lung tissue mass and maintenance. It lowers body density and holds O2 for breath-holding. Time to drowning follows from the O2 store and the O2 use, not a fixed 300 s. Founders draw it from a broad range that includes zero | Section 3 |
| **Experiment protocol** | Five rules: keep every outcome; sample by a rule fixed before the biological outcome is known; carry state between stages (deferred, a known gap); no automatic rescue or replacement; fix the rules before each experiment | [PROTOCOL.md](PROTOCOL.md); sections 0 and 10 |

What the protocol changes in the code:
- The synthetic fallback is removed. A world7 seed that gives v3 no planet is recorded with its outcome (no star, no
  planet, no habitable planet, no bodies era). It is neither skipped nor replaced.
- Every sampled seed gets one outcome row, in the sampled order.
- An extinct arena is never re-founded.
- The rules hash (`world3.rules_identity`) covers the source of `haishool/life9/v3` and `haishool/life9/planet` and
  of every in-repo module they import, the config, the rule dataclasses in use, every module constant in effect
  (private ones too) and the Earth reference. The command line writes it into the experiment's manifest before the
  first step, and refuses to resume under a different hash; the checkpoint carries it too.
- Every seed's starting conditions are pinned in `DIR/inputs.json` before the first step; errors go to
  `DIR/errors.json`; every run and resume to `DIR/runs.jsonl` (review of 4 October 2026).
