# life9 experiment protocol (binding)

The owner adopted this protocol on 4 October 2026. It binds every life9 v3 experiment whose results are
reported. [PLANET-V3-SPEC.md](PLANET-V3-SPEC.md) section 0 says what the simulation may contain. This document
says how an experiment with it is set up, run and reported.

A run outside the protocol is a **test**: the unit tests, `describe`, `bench`, smoke runs and debugging runs.
A test is never reported as a result.

## The five rules

| # | Rule | In one line |
|---|---|---|
| 1 | **Keep every outcome** | No star, a barren planet, a frozen planet, no land, dead founders and extinction are all results |
| 2 | **Follow every formed planet, or sample by a rule fixed before the biological outcome is known** | Never choose seeds or planets by what lived on them |
| 3 | **Carry state between stages** | What one stage made is what the next stage starts from |
| 4 | **No automatic rescue or replacement** | No synthetic fallback, no placement by habitability, no replenishment that bypasses a modelled process |
| 5 | **Fix the rules before each experiment** | A rule change is a new experiment |

Rules 1, 2, 4 and 5 are enforced by the code (below). Rule 3 is **not yet satisfied**, and the "no bodies era"
exclusion depends on it (rule 1). Every report must say so.

## Rule 1: keep every outcome

Every sampled seed ends with exactly one recorded outcome. Nothing is dropped, retried until it works, or replaced
by something that worked. "Nothing happened" is an outcome.

The outcomes, in chain order (`world3.World3.outcomes`; the kind is in brackets):
1. **no star** (`no_star`): the star era did not fuse.
2. **no planet** (`no_planet`): no planets era, or no planet formed.
3. **no habitable planet** (`no_habitable_planet`): no temperate rocky planet, or none with liquid water.
4. **no bodies era** (`no_bodies_era`): the chain's planet did not reach the bodies era. v3 founds no bodies there,
   because section 0 makes world7's earlier rungs the origin of v3's replicators. This is **not a physical outcome**:
   it conditions the body stage on an earlier stage's biological outcome while rule 3 is deferred (v3's founders
   carry nothing from the chain). The rows are labelled `excluded` ("excluded by the chain's outcome") and counted
   apart (`excluded_by_chain_outcome` in `outcomes.json`). Building these planets and letting physics decide is the
   protocol-consistent alternative, an open owner decision.
5. **extinct at day d** (`extinct`): every arena of the world is empty. Each arena's day of extinction is kept, so
   founders that all died in the first day are visible.
6. **alive at day d** (`alive`): with the population and the arenas alive.

A world without land is not a non-run: its patches are sea, its founders float or sink there, and physics decides
(rule 4). Its row carries `no_land`. Every day in the outputs counts the **days completed** when it happened
(`day_base`): "extinct at day 1" is an arena empty at the end of its first day.

The first four are worked out from the chain's own eras (`world3.chain_outcome`) and cached next to the chain
cache. `planet_inputs` reads the caches first (the planet, then the absent outcome), so a host without scipy runs
every cached seed. Each row also carries every arena's extinction day, the founding (the share of founders dead in
the first bout), `capacity_bound` and its first day per arena, the first day the item and fire pools overflowed,
the non-finite counts, the spin-up's replenishment, the planet pick, `check_spec3`'s failures, and the hash of the
seed's chain inputs. A `capacity_bound` world is kept as a result: after that day its population measures the slot
cap, not the ecology.

**Errors are not outcomes, and they are not dropped.** A crash, a non-finite value or a missing chain cache is
recorded as an error with its cause. An experiment with errors is incomplete. A seed may be rerun only when the
error came from the environment (a lost host, out of memory), only under the same rules hash, and the rerun is
reported.

How the code records them: a seed whose chain inputs cannot be resolved gets an `error` row with its cause; a failed
set-up or day is written to `DIR/errors.json` (seed, stage, day, cause, traceback); `outcomes.json` is then
`complete: false`. Non-finite locomotion is counted per world in the rows (`nonfinite`). Every run and resume is
appended to `DIR/runs.jsonl` with its environment, mode and days; a seed pinned as an error that a later run
resolves is recorded there and in `inputs.json` as a rerun.

How the code enforces it:
- A seed that gives v3 no planet is an outcome row, not an exception and not a skipped seed.
- `DIR/outcomes.json` lists every sampled seed in the sampled order, whatever happened to it.
- The summary keeps every world, those with no arena included.
- Extinct arenas stay in the run. They are never re-founded (rule 4).

## Rule 2: the sample is fixed before the biological outcome is known

`run` needs a sampling rule (`--sample`). The rules are:
- `range:LO:HI`: every seed from LO to HI, inclusive.
- `hash:SALT:N:LO:HI`: the N seeds of LO..HI with the smallest SHA-256 of "life9-v3-sample:SALT:seed". It is a draw
  that no outcome can steer.
- `earth:K`: K replicates of the Earth reference (a control; the seed varies only the world's draws).

For `range` and `hash` the source (`world7` or `synthetic`) must be given. It is the experiment's choice and is never
substituted. `--part K:N` splits a sample into contiguous parts, one manifest per part.

Not allowed:
- choosing seeds from the chain cache by the rung they reached (version 2 ran only planets that reach the bodies
  era);
- dropping a seed after a look at it;
- extending or shrinking a range after seeing results, without a new manifest.

**Case studies.** An explicit seed list (`--seeds`) may have been picked from known outcomes, so `run` refuses it
unless `--explicit WHY` declares it a separate experiment that is not pre-registered. Planet 781 is such a case
study: earlier rounds followed it because its chain climbed to groups. Case studies are never pooled with a sample.

**One planet per seed.** world7 forms a whole planetary system. Its chain climbs every temperate rocky planet that is
terran with liquid water, and the hand-off to life9 (`planet.chain`) gives one planet per seed. v3 therefore cannot
yet follow every formed planet. Its sampling rule is: **every seed in a pre-registered range, one planet per seed,
chosen by world7's physical habitable-zone rule** (the innermost rocky planet in the temperate flux zone that is
terran with liquid water).

**The pick is the physical rule** (chain version 3, review of 4 October 2026). Version 2 took the life8 bridge's
planet (the most links up to groups), which depends on the chain's biological outcome. Version 3 takes the innermost
temperate terran planet with liquid water and records the bridge's pick beside it (`planet_index_bridge`,
`pick_agrees_with_bridge`, `temperate_ocean_planets`; each outcome row carries them as `pick`). The cache was
regenerated under version 3 for seeds 0-63 and every seed cached before (95 seeds, 66 with a planet): the two picks
agree in all of them. Hosts that hold a version-2 cache need the new one.

**The chain cache is part of the experiment.** Seeds not yet cached are computed on a host with scipy (world7's
cosmos levels need it). All hosts then share the one cache, because world7 differs in the last bit between Windows
and Linux. A seed whose cache is missing on a host is an error, not an outcome. Each seed's starting conditions are
pinned in `DIR/inputs.json` before the first step, and every resume and part checks them (rule 5).

**A world depends on its seed alone.** Each world's set-up draws (terrain, patches, detail, deposits, sowing,
founders) come from its own generator, seeded by its seed and the config's `rng_seed` salt. Re-splitting a range
into other parts gives every seed the same world. The run's own draws (events, division, mutation) are shared by the
worlds stepped together, so a part's dynamics are statistically, not bitwise, independent of the partition.
`--set rng_seed` redraws every world, so `run` refuses it unless `--explicit WHY` declares a separate experiment.

**Synthetic samples are habitable by construction.** `--source synthetic` draws temperate liquid-water planets with a
bodies era, so such a sample cannot give the first four outcomes. The manifest and `outcomes.json` say so
(`habitability_conditioned`).

## Rule 3: carry state between stages (not yet satisfied)

What happens today:
- **world7 to v3.** The hand-off carries era values: the star, the orbit, the cloud's element mix, the age and the
  bodies era's oxygen level. Some are the eras' rounded records (the whole-kelvin surface temperature, the 3-digit
  era flux, the oxygen level). Others are read unrounded from the level rollouts (orbit, mass, flux, luminosity).
  The chemistry era's molecule inventory and world7's cells and bodies are not carried over.
- **v3 starts with new founders.** They are random genomes at random places, not world7's bodies.
- **Inside v3** the stages run in one process and pass their full state on: formation, then the climate spin-up,
  the global biosphere, the patches and their plants, then the founders. Two spin-up conditions are imposed: the
  global biosphere spins up with the chain's air held, and the patch plants are sown once, sparsely and uniformly.

The planned remedy:
- a chemistry layer, so the chain's chemical inventory becomes v3's ledgers;
- hand-offs that carry inventories (matter, organisms) instead of rounded era values.

Until then every v3 result states: "Rule 3 not satisfied: the chain's state is handed over as era values, and the
founders are new."

## Rule 4: no automatic rescue or replacement

| Rescue | Status in v3 |
|---|---|
| Synthetic fallback (a synthetic planet where a world7 seed fails) | Removed (owner decision, 4 October 2026). A seed that gives no planet is an outcome (rule 1) |
| Placement by habitability | None. Patches are global land cells drawn at random by area; a world without land draws them over all its cells (sea patches). Founders sit at uniformly random positions. Version 2's `founder_habitat` is gone, and the purity test greps for it |
| Seed rain or a seed floor | None. The global biosphere's seed floor is 0 (tested). Patch plants spread only by seed from plants that exist |
| Refilling bodies after deaths or extinction | None. Founders are placed once. An extinct arena is never re-founded |
| Nutrient top-up | None. Nitrogen fixation follows the plants' growth, not a starting stock (tested) |
| CO2 cap | None. formation3 reports `co2_capped` False |
| Matter from nowhere | Every ledger is float64 and closes. Anything that appears needs a booked source |
| Set-up replenishment | The global biosphere spins up with the chain's air held and jumps its wood and litter to steady state: carbon is not conserved there. The carbon made and the air's moles before and after are reported per world (`spin_up_replenishment`, in the rows) |

What remains, stated as calibrations, not rescues. None of them looks at life:
- the Earth calibrations of formation: the degassed shares of water and C, and of N; the basin depth; the
  greenhouse's reference pressure; the lake depth;
- the climate's cloud albedo, fitted per planet toward formation3's T_s on planets that are neither `frozen_mean`
  nor `runaway`;
- the spin-up conditions under rule 3;
- body-plan constants of breathing that the owner has not yet decided: the O2 store's floor (0.30 of the alveolar
  pO2), the anoxia tolerance (180 s of resting O2 use) and the lung's uptake per kg.

How the code enforces it: the synthetic fallback is gone. `tests/life9/v3/test_purity.py` greps every v3 source for
the forbidden mechanisms and checks that the rules in effect have no floor and no target. The ledger tests close
every book.

## Rule 5: fix the rules before each experiment

An experiment is defined by its **manifest**, `DIR/manifest.json`. `run` writes it before the worlds are built, so
before the first step. It holds:
- the protocol version and the start time;
- the sampling rule, its seeds and the part;
- the source, the full config and the days to run;
- the **rules identity** (`world3.rules_identity`);
- the environment: Python, torch and numpy versions, platform, host, CPUs, device, CUDA or HIP, threads and the
  command line;
- a note that rule 3 is deferred, a known gap.

The rules identity is a SHA-256 over:
- the source of every `.py` file in `haishool/life9/v3` and `haishool/life9/planet` and of every in-repo module they
  import, found statically (the formula parser, cosmos level 3, life9's brain, the chain's world7 path), with line
  endings normalised so a Windows and a Linux checkout agree;
- the config;
- the rule dataclasses as `World3` uses them;
- every upper-case module constant of those packages and of `life9.brain`, `truth.formula` and `cosmos.planets` in
  effect, private ones too (caches excepted), so a constant changed at run time changes the hash;
- the Earth reference that calibrates formation3. It is computed from the code (it needs no scipy); no per-host
  cache file is read.

The chain inputs are the starting conditions, not rules. Each seed's inputs are pinned in `DIR/inputs.json` before
the first step and hashed in its outcome row. The checkpoint carries the rules hash it was saved under.

Consequences:
- A run directory that holds a manifest is never reused for a new experiment.
- `--resume` continues the experiment up to the manifest's days. It refuses to run under a different rules hash (the
  manifest's and the checkpoint's), with a different sample, source, config, length or starting conditions, or on
  another device or host unless `--allow-device-change WHY` declares it. A complete experiment is not run again; a
  resume without a checkpoint keeps the earlier partial outputs aside.
- A `--set` change is a config change, so it is a new experiment.
- A bug fix is a rule change. The fixed run is a new experiment. The old results stay reported, with the bug
  stated, as version 2's seed floor was.
- Results with different rules hashes are never pooled.

GPU runs are not bit-exact (PLAN.md section 6). A GPU run and a CPU run with the same rules hash are the same
experiment on different devices. They are compared as statistics, never row by row.

## How the code enforces the protocol

| Mechanism | Rules | What it does |
|---|---|---|
| Manifest before running | 2, 5 | `DIR/manifest.json`, written before the worlds are built: sample, source, config, days, rules identity, environment |
| Inputs pinned | 1, 5 | `DIR/inputs.json`: every seed's starting conditions, before the first step; checked on every resume |
| Errors and runs | 1 | `DIR/errors.json` and `DIR/runs.jsonl`; `outcomes.json` incomplete with errors |
| Rules hash | 5 | `world3.rules_identity`; `--resume` refuses a different hash, sample, source, config or length |
| `--sample` | 2 | `range:LO:HI`, `hash:SALT:N:LO:HI` or `earth:K`; `--seeds` only with `--explicit WHY`, as a separate experiment |
| No fallback | 4 | The synthetic fallback is gone; a seed with no planet is an outcome |
| Outcomes kept | 1 | `DIR/outcomes.json`: one row per sampled seed; extinct arenas are never re-founded |
| Purity and ledger tests | 4 | No forbidden mechanism in the source; nothing appears without a booked source |

## How results are reported

1. **Name the experiment.** Give the manifest, the rules hash, the sampling rule and the sample size n.
2. **Count every outcome, starting with the nothing-happened counts.** Every count is "k of n seeds" over the whole
   sample, in this order: no star, no planet, no habitable planet, no bodies era (excluded by the chain's outcome,
   counted apart), extinct (with the days, and how many worlds lost every founder in the first day), alive at the
   end; and how many worlds had no land (sea patches).
3. **Then report what happened** in the worlds that lived, with the bound reports of PLANET-V3-SPEC section 10
   beside it: `capacity_full` and `capacity_bound` with its day, the item and fire pools' first overflow day
   (`items_bound_day`, `fires_bound_day`) and their counts, the senses' bounds, the movement sub-steps' bounds, and
   `brain3.bound_report`.
4. **Measure selection against the neutral shadow**, and behaviour against the founders' baseline networks.
5. **List errors separately**, with their causes and any reruns.
6. **Label case studies and controls** (781, the Earth reference). Never pool them with a sample.
7. **State the known gaps**: rule 3 is not satisfied; 'no bodies era' seeds are excluded by the chain's outcome;
   the global biosphere's spin-up holds the chain's air (`outcomes.json` lists them as `known_gaps`).

A report that leaves out seeds, worlds or arenas where nothing happened breaks the protocol.

## Earlier results under this protocol

Results from before 4 October 2026 are **pre-protocol** and are labelled so:
- **Version 2 runs** (3 October). Seeds came from the cached chain planets that reach the bodies era (a rule 2
  breach). The engine had a synthetic fallback, founders were placed on habitable cells, and the seed floor fed
  barren planets (rule 4 breaches).
- **First v3 results** (3 October: 781 freezes; the Earth reference fills its slots by day 2). These are a case study
  and a control, made before the four decisions. They are not a sample.
