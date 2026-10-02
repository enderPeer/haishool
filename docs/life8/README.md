# life8: Haishool round 8, "living worlds" (foundation)

life8 is the simulation engine for Haishool round 8. Round 7 (`haishool/evo/`, world7)
took toy worlds from the first minutes to a society, but its upper rungs (signals, society)
were scripted summaries. life8 replaces those rungs with a simulation of persistent
individuals. Each one has an inherited body, senses, movement, energy, injuries, age,
reproduction and its own learning controller. The world also holds material objects with
physical properties, local noisy signals that cost energy, sharing, imitation and teaching.
Nothing in it is scripted: no dictionary, no grammar stage, no technology tree, and no
reward for a symbol, a tool or a named behaviour. A learner's reward is only its own
energy and health change.

This foundation stage does not claim language, technology or society. It vendors the
engine, fixes the bugs an independent review found, makes many runs affordable and
re-measures the review's key learning result. The behaviours the user wants (lineages
that develop signalling, useful tools, real cooperation) still have to be built and
measured in later stages.

## Where it came from

- Source: Matrix (`C:/Users/end/Desktop/Matrix`), same owner, MIT licence, commit
  `2cf045a419b871202de783f5da55807d27a77827` (branch `codex/matrix-foundation`).
  Matrix itself was not modified.
- `haishool/life8/ORIGIN.json` records the source commit and the sha256 of every copied
  file as stored at that commit (engine, viewer, tests, licence).
  `haishool/life8/LICENSE-matrix.txt` is the unchanged MIT notice.
- Copied: `matrix_sim/*.py` and `viewer.html` into `haishool/life8/`, and the 6 test files
  into `tests/life8/`. The engine already used relative imports, so only the tests
  needed their imports rewritten (`matrix_sim` to `haishool.life8`).
- Before any change, all **127 copied tests passed** (`python -m pytest tests/life8 -q`,
  Python 3.11.9, 4.35 s).
- The review is `matrix-review/assessment.md` in the session scratchpad. Its section 7
  lists the bugs fixed below.

## Use

```
python -m haishool.life8 run --seed 1 --steps 600 --out runs/seed1 --log compact
python -m haishool.life8 resume runs/seed1/checkpoint.json --steps 400 --out runs/seed1b
python -m haishool.life8 verify runs/seed1/checkpoint.json
python -m haishool.life8 serve --root runs
python -m haishool.life8.analysis.founders --seeds 1-6 --ticks 600 --population-limit 256
python -m pytest tests/life8 -q                    # 276 passed, 7 slow ones skipped, about 8 min (2026-10-02)
LIFE8_RUN_SLOW=1 python -m pytest tests/life8 -q   # also runs the slow experiment tests
```

In Python, use `World.create(seed, Config(...))`, then call `world.step()` and
`world.to_dict()`. `World.from_dict(state)` continues a run exactly.

## What changed from Matrix, and why

### Bug fixes (review section 7; each has a regression test in `tests/life8/test_fixes.py`)

Each regression test was run against an untouched copy of Matrix 2cf045a: 19 of the 24
tests in that file fail there and pass on life8. The other 5 are new coverage (the two
noise and range tests), a positive control for hard food, the teaching control and the
provenance check.

| # | Bug | Fix |
|---|-----|-----|
| 1 | Each birth's 4 + 12 energy was booked as the outcome of whatever action the parent took that tick (usually `eat`). | The birth transfer stays in the energy ledger but is left out of the learning reward. Events report it as `birth_transfer`. `Config.birth_transfer_in_reward=True` restores the old booking, as an intervention control only. |
| 2 | A receiver that came into range late got the message stamped with the delivery tick, so its age read 0 and it stayed audible for up to 2 x TTL. | The heard record keeps the creation tick (`tick`) and the delivery tick (`heard_tick`). Age is now the true age, and the message expires on time. |
| 3 | A share that moved 0 energy was still recorded as `energy_shared` in social memory and edges. | No energy moved means no social record. |
| 4 | `tool_damage_gain` counted strikes on soft food, which change nothing. | The gain counts only damage the target can still absorb (opening left on food, durability on objects, health on agents), compared with a bare hand. Events carry `effective_damage`. |
| 5 | The validated `effort` was ignored by combine (0.8), dismantle (0.8) and heat (1.0). | The requested work is spent and used, capped at 5.0 per action. Defaults are unchanged. |
| 6 | Combine appended the new composite at the end of the inventory, but tools are taken from position 0. | The new composite goes first, so it is the held tool. |
| 7 | Demonstrations carried the demonstrator's private energy and health bands, social-memory features and heard symbol, and were scored as one-step outcomes, so a demonstrated step toward food was taught as harmful. | A demonstration now holds only what a neighbour could see (`world.DEMO_FEATURES`: the demonstrator's situation and what it holds). It also carries the demonstrator's next observable situation, and `learning.observe_demo(..., next_observation=...)` uses the same bootstrapped target as the learner's own experience. Validation refuses demo memories that contain private fields. |
| 8 | `imitate` wrote into the passive demonstrator's social memory as if it had acted. | Imitation is recorded as `imitations` from the imitator, in the imitator's memory only. `teach` still records `demonstrations` on both sides. |
| 10 | Voice and hearing (on/off organs) flipped at the full 8% mutation rate. | They use their own rate, `Config.organ_flip_rate` (default 0.005). |
| 11 | `mean_action_concentration` read 0.0 after extinction, and it mostly measures how often individuals eat. | It is `None` when no living individual has acted. `summary()["dominant_actions"]` shows which action dominates, which is usually `eat`, so this number is not a specialization index. |
| 13 | `test_world.py`'s `..._cost_range_noise_...` test checked neither range nor noise, and the noise branch had no test. | Renamed. New tests check that the noise rate (0, 0.25 and 1) is applied, that substitutes are uniform over the other three symbols, that the `noisy` flag is consistent, and that reach is a hard cutoff. |

Not fixed here (outside this task, still open): 9 (voice power comes from the sensing
gene), 12 (the controller's memory buffers are still write-only logs), 14 (metabolism has
no benefit) and 15 (minor). The review's design gaps (no sender credit route, no
inherited behaviour, one-step credit assignment, constructions worse than raw stone,
society labelled rather than measured) are untouched. They are the next stages.

### Checkpoint and schema versions

The new state fields (`pending_heard`, `heard_tick`, demo `next_observation`, the
`imitations` kind) and the changed dynamics make old states incompatible. All versions
were bumped: state `life8-state-v2` / model `life8-persistent-ecology-v2`, checkpoint
`life8-checkpoint-v2`, run manifest `life8-run-v2`, controller
`life8-factored-q-controller-v3`. Matrix checkpoints and states (`matrix-*-v1`) are
refused with an explicit error (`test_matrix_and_old_life8_states_are_refused`).
`checkpoint.source_identity()` now hashes only the engine modules at the package top
level, so editing scripts in `haishool/life8/analysis/` does not invalidate checkpoints.

### Speed and size for many runs

- **Spatial grid** (`haishool/life8/spatial.py`). Observation and message delivery look
  only at 2x2 cells near the query (radius plus 1e-6) instead of scanning everything. The
  distance function, radius filter and (distance, id) order are unchanged, so results are
  identical. The grid is built before each read-only pass and is never saved.
  `test_spatial_hash.py` compares state hashes with the grid on and off over 300 ticks
  for seeds 1-3 (and 80 ticks in full-log mode). They are identical. Setting
  `world.spatial_index = False` falls back to the scan.
- **Exact speedups in the learner.** The observation is projected once per call. Feature
  strings are memoized. The JSON round trip is skipped for plain scalar features. Action
  values are summed with `math.fsum` over only the features that have weights, which is
  exact because fsum is correctly rounded. The hashes of three 300-tick worlds were the
  same before and after these changes.
- **`Config.log`**: `"full"` (as in Matrix: every observation, action and transition) or
  `"compact"`. Compact keeps births, deaths, capacity pauses, thermal contacts and:
  - `signal`: sender, symbol, message id, range, and the sender's bands (energy, health,
    food direction and reach, object and neighbour near, target hardness, tool) at the
    moment it chose to call;
  - `heard`: receiver, sender, message id, heard and sent symbol, noise flag, age,
    distance, the receiver's bands at its next decision, and the **receiver's next
    action**. It is emitted at the start of the next tick and kept in the checkpoint
    (`pending_heard`), so continuation stays exact;
  - `act`: collect, drop, the strikes, combine, dismantle, heat, share, imitate and teach,
    each with its outcome detail.

  `run_world` writes snapshots every K = `--sample-every` ticks. Compact mode changes only
  the records, never the dynamics (`test_compact_log_changes_records_not_dynamics`).
  Experience export refuses compact runs.
- `run_world` records `bytes_written` in the manifest.

Measured on this machine (`docs/life8/results/speed-and-size.json`). Times are seconds
per 100 ticks at a fixed population, the engine plus JSON for every event:

| agents | Matrix 2cf045a | life8 full log | life8 compact log |
|---:|---:|---:|---:|
| 25 | 1.55 | 0.77 | 0.54 |
| 50 | 3.43 | 1.68 | 1.17 |
| 100 | 7.89 | 3.84 | 2.55 |
| 128 | 10.76 | 5.13 | 3.57 |

The engine alone (no event JSON) went from 1.34 / 3.03 / 6.81 / 9.15 s to 0.59 / 1.17 /
2.71 / 3.56 s, about 2.3-2.6x faster. Event volume at 128 agents fell from 134.6 MB to
2.8 MB per 100 ticks.

Bytes written by `run_world` per 1000 ticks (food-regulated world, `food_patches=30`,
about 46-51 individuals, K = 20):

| seed | full events.jsonl | compact events.jsonl | snapshots.jsonl | compact total |
|---:|---:|---:|---:|---:|
| 1 | 377.2 MB | 5.87 MB | 4.07 MB | 9.94 MB |
| 2 | 333.2 MB | 5.05 MB | 3.77 MB | 8.82 MB |

Determinism: the same seed gives the same state hash on Python 3.11.9 and 3.12.14
(checked for a 150-tick full-log and a 150-tick compact world). 3.13 and 3.14 were not
available on this machine to test.

## Re-measured: founders alive at age 300 (review bug 1)

The measurement uses default Config, seeds 1-6 and 600 ticks. It counts the 24 founders
per world not dead before tick 300, and P(eat | food in reach) for founders aged 50-99
(`haishool/life8/analysis/founders.py`). The intervention runs the same seeds with
`birth_transfer_in_reward=True`.

| engine | founders alive at age 300 | founder P(eat \| food in reach), age 50-99 |
|---|---:|---:|
| Matrix 2cf045a, untouched (reproduces the review exactly, seed by seed) | 44 / 144 | 0.35 |
| review's intervention (birth cost added back to the reward, review number) | 105 / 144 | 0.77 |
| **life8**, capacity raised to 256 so every run reaches tick 300 | **105 / 144** | **0.77** |
| life8 with `birth_transfer_in_reward=True` (same seeds) | 57 / 144 | 0.36 |

life8 beat its own intervention on 6 of 6 paired seeds: 16/9, 22/12, 17/13, 17/11, 18/7
and 15/5. At the default capacity of 128, one life8 world (seed 5) reaches the capacity
pause at tick 298, before age 300 is defined. On the other five seeds it is 87/120.
Raising the cap does not change the trajectory before the cap is reached. Results are in
`docs/life8/results/founders-*.json`. The slow test
`test_birth_fix_raises_founder_survival_against_the_same_seed_intervention` repeats this.

What this shows: removing the misattributed birth cost lets individuals learn to eat
(P(eat | food in reach) rises from 0.36 to 0.77) and roughly doubles founder survival
against the same seeds with the cost put back. The intervention still keeps the other
fixes, which is why it scores 57 rather than Matrix's 44. This is a toy ecology. The
numbers are counts in these runs, not probabilities for any real system.

## Rules this engine keeps

- **Exact JSON checkpoint continuation.** A resumed run equals an uninterrupted one by
  state hash, in full and compact mode.
- **Closed ledgers.** Energy, food, mass and heat ledgers are validated on every save.
- **Local observations only.** Every action has a cost, and runs end with an explicit
  stop reason.
- **Honest learning reward.** The learner's reward is only its energy and health change,
  minus the birth transfer, which is bookkeeping and not an outcome of the action.
- **Pure standard library** in the engine. Analysis code may use numpy.
- **Same seed, same run.** Determinism was checked on Python 3.11 and 3.12.

Signals are not called language, social edges are not called institutions, and toy
outcomes are not called real probabilities.

## Files

- `haishool/life8/`: `world.py`, `learning.py`, `materials.py`, `entities.py`,
  `config.py`, `checkpoint.py`, `cli.py`, `dataset.py`, `server.py`, `viewer.html` (from
  Matrix); `spatial.py` (new); `analysis/founders.py` (new); `ORIGIN.json`;
  `LICENSE-matrix.txt`.
- `tests/life8/`: the 6 copied Matrix test files (imports rewritten, one test renamed),
  `test_fixes.py`, `test_spatial_hash.py`, `test_spatial_log.py`, `test_experiments.py`,
  and `conftest.py` (the `slow` marker; slow tests are skipped unless `LIFE8_RUN_SLOW=1`
  or `-m slow`).
- `docs/life8/results/`: the measurements quoted above.

## Discovery and lineage harness (round 8, measurement stage)

`haishool/life8/discovery.py` and `haishool/life8/lineage.py` measure, with interventions,
whether signalling carries anything. `Config.channel` switches the channel: `"scrambled"`
replaces each delivered symbol by a uniform one from a separate RNG kept in the
checkpoint (the world RNG stream is untouched), `"masked"` delivers the call but hides
its symbol from the controller. Costs, range and delivery are the same in all three.

- Sender information: I(symbol; joint sender state food_near, food_seen, energy_band,
  neighbor_near) from compact `signal` events, minus a permutation null, with p-value.
- Receiver information: I(heard symbol; next action) in the normal run, minus the same in
  a paired scrambled run of the same seed.
- Fitness: paired branches from one checkpoint (normal vs scrambled, 8 replicates with
  the same reseeded world RNG in both arms): population, food energy per agent-tick,
  survival of those alive at the branch.
- Verdict `all_three` needs sender excess >= 0.05 bits with p <= 0.01, receiver excess
  >= 0.02 bits with p <= 0.01 and >= 0.02 bits above scrambled, and a positive food-energy
  effect with sign-test p <= 0.05 over >= 6 branches. `all_three` would still not be
  evidence of language.
- Lineages: a persistent table (founder, parent chain, depth, birth/death; dead
  individuals stay), 8 neutral marker genes inherited with the world's mutation rate,
  convention agreement within vs across founder lineages (label-shuffle null), and
  persistence of a lineage's situation-to-symbol mapping after its founder died.

Controls (two-player game, life8 learner, seeds 1-6,
`results/discovery-signalling-game-controls.json`): with a shared payoff, sender
information is 0.76-0.80 bits (default learner) and 0.59-0.66 (one-step learner); with
the sender paying alone it is 0.03-0.10 and 0.01-0.20. In the review's framing (food
east/west, one-step learner): 0.60-0.66 vs 0.005-0.098. The default learner's
egocentric frame makes "food east" and "food west" look the same to a sender, so in that
framing it reaches only 0.001-0.003 bits; the default control uses a frame-free cue.

Real worlds (seeds 1-6, 1500 ticks, `food_patches=30`, window 1000-1500,
`results/discovery-real-worlds.json`): verdict null in 6/6. Sender excess -0.0003 to
+0.022 bits (seed 5: 0.022, p 0.005, below the 0.05 threshold); receiver normal minus
scrambled -0.002 to +0.007 bits; normal minus scrambled food energy per agent-tick
-0.0019 +- 0.0036 (2 of 6 positive). Lineages (`results/lineage-real-worlds.json`):
within-lineage agreement 0.246-0.255 vs across 0.249-0.254 (chance 0.25), label-shuffle
p 0.30-0.83; 0 of 45 lineage mappings persisted after their founder died; body mass
against the neutral markers z = -1.36 to +2.07 over 1500 ticks (no clear selection in
runs this short). Commands: `python -m haishool.life8.discovery --seeds 1-6 --config
'{"food_patches": 30}'`, `--game` for the controls, and `python -m haishool.life8.lineage`.

## Bridge from round 7 (`bridge.py`)

`bridge.world_config(world7_world, base=Config())` returns `(Config, founders)` for a world7
planet that reached the rung `bodies`; `bridge.create_world(seed, config, founders)` builds the
living world. Every number is a documented toy formula of one world7 era value (module
docstring), and `founders["sources"]` names the era behind each output:

| life8 | from world7 | formula |
|---|---|---|
| `Config.visibility` (new field, multiplies sight radius; 1.0 keeps Matrix exactly) | `surface_k.flux` | 0.5 + 0.5 min(1, S) |
| `action_cost` | `surface_k.mass` (gravity g = M^0.46) | 0.08 g^0.5 |
| `base_metabolism` | gravity, `surface_k.temperature` | 0.045 g^0.25 (1 + 0.01 abs(T - 288)) |
| `food_regrowth` | `bodies_k.light` | 0.10 clip(sqrt(light/200), 0.5, 1.5) |
| founders' body_mass, metabolism | `bodies_k.consumer_size` n | 1 + 0.1 (log2 n - 7); (n/128)^-0.25 |
| founders' manipulation | `bodies_k.consumer_types` | 0.6 + 0.08 types |
| founders' sensing, voice, hearing | `senses_k` eyes, ears, talkers | 0.6 + 0.16 max(eyes, ears); talkers; min(1, ears) |
| object materials | `chemistry_k` mix, organic, h2o | weights stone 1, wood/fiber 0.5 clip(organic/50), clay 0.5 min(1, h2o/5000), metal 0.25 if reducing |

Genes are drawn from center x (0.75, 1.25) by the bridge's own RNG (the world stream is
untouched). An earth twin keeps life8's defaults exactly. Predators, injury and the oxygen
limit on body mass are not life8 mechanisms yet: they are in `founders["ecology_hints"]`
(`"connected": False`) for the ecology stage.

`bridge.candidate_worlds(n, start_seed, workers=)` scans world7 seeds in order with world7's
own chain cut after `senses` (eras equal the prefix of `world7.run(seed).steps`, tested).
Seeds 0-299: 49 qualify (as round 7 reported), 116 s with 16 processes on a loaded machine
(`results/bridge-scan-0-299.json`); the collapse level (about 1.2 s) dominates every seed.
`bridge.synthetic_worlds(n)` draws planets directly from ranges that cover those 49 (tested),
labelled `"source": "synthetic"`; they reproduce neither world7's correlations nor its
frequencies.

200 ticks, population 24, compact log (`results/bridge-runs-200.json`): all 49 bridged
worlds ran (one reached the population cap at tick 198); mean alive 83.1 vs 84.0 for the
same life8 seeds with the default config, founders alive 1029 vs 1067 of 1176, bridged lower
in 25 of 49 pairs. So within 200 ticks the hand-off changes survival little; this is a
check that the mapped worlds are viable, not a result about them.

## Heritable behaviour (`genome.py`, review gap 2 and gap 18)

Each organism carries a genome next to its anatomy (`Agent.genome`, `{}` = default):
innate starting weights over 25 local features (bias, food in reach, food direction,
energy and health band, neighbour in reach, held tool, heard symbol 0-3) for all 26
actions, including the signal rows; a learning rate and a base exploration rate; and,
with `age_effects`, a lifespan and a maturity age. The newborn's controller starts from
those weights (`learning.controller_from_biases`); learning then changes them like any
other weight. A child's genome is built from its parent's genome only, never from the
parent's controller, so nothing learned is inherited (`test_learned_weights_are_never_inherited`).

| Config | default | meaning |
|---|---|---|
| `genome` | `"frozen"` | `"frozen"`: newborns start blank, as in Matrix (the control). `"evolving"`: biases (rate `bias_mutation_rate` .05 per entry, Gaussian `bias_mutation_scale` .25, bound ±3), alpha and epsilon (`mutation_rate`/`mutation_scale`) are inherited with mutation |
| `reproduction` | `"asexual"` | `"sexual"`: a fertile partner within interaction range is required (willing = automatic reproduction, or its last action was `reproduce`); each parent pays the full birth cost and half the offspring energy; every anatomy gene, bias row and scalar gene comes from one parent at random |
| `age_effects` | `False` | ability (movement, eating efficiency, strike force) starts at 0.5 and grows to the adult level at maturity, falls linearly to 0.5 over the last 30 % of life; later maturity gives a stronger adult ((maturity/35)^0.25, within 0.8-1.2); a longer heritable lifespan costs metabolism x (lifespan/max_age)^0.5 |

Genome draws use their own RNG (string-seeded from world seed and child id), so the
default (frozen, asexual, no age effects) keeps the world RNG stream and every
trajectory: a frozen and an evolving world with random actions have identical
trajectories (`test_frozen_default_keeps_newborns_blank_and_the_world_rng_untouched`).
Voice and hearing keep their own `organ_flip_rate`. Checkpoints of evolving, sexual,
ageing worlds continue exactly.

**Measured** (`python -m haishool.life8.analysis.heritable --seeds 1-8 --ticks 4000`,
food_patches 30, population_limit 256, the learner profile active at the time;
`results/heritable-genome.json`). Newborns followed are those born at least 50 ticks
before the end. The `neutral` control evolves the same genome with the same draws but
does not express it (newborns stay blank), so it measures drift.

| | newborns | died before age 50 | born 0-999 | born 3000-3999 | innate P(eat \| food in reach), gen 1-3 / 4-7 / 8-11 / 12-15 / 16-19 |
|---|---|---|---|---|---|
| frozen | 5209 | 35.9 % | 38.2 % | 32.5 % | 0.038 at every generation (blank: 1/26) |
| neutral (drift) | 5209 | 35.9 % | 38.2 % | 32.5 % | 0.038 / 0.030 / 0.037 / 0.030 / 0.017 |
| evolving | 4867 | **19.4 %** | 27.3 % | 14.1 % | 0.039 / 0.086 / 0.100 / 0.117 / 0.099 (n=60) |

Evolving had lower newborn mortality than frozen on 8/8 paired seeds (per seed
12.3-32.0 % vs 32.8-38.9 %). The innate eat bias roughly triples over 12-15 generations
while the unexpressed copy drifts flat or down: a Baldwin-type effect in this toy world.
The 20.5 % reference is the review's number for Matrix's default ecology; in this
food-regulated ecology the blank-newborn baseline is 35.9 %, so the fair comparison is
the paired one. Newborns' actual P(eat | food in reach) at ages 0-19 (learning included) went from
0.11-0.21 in ticks 0-499 to 0.11-0.44 in ticks 3500-3999 in the evolving runs (higher in
7 of 8 seeds, flat in 2 of them), while frozen runs stayed at 0.21-0.36 throughout; so
evolving newborns start below frozen ones and catch up. Pooled over the run they eat
less often when food is in reach (0.225 vs 0.261) yet die less, so much of the
mortality gain probably comes from avoiding costly actions; this measurement does not
separate the two.

## Multi-step credit (`learning.py`, controller v4, review gap 4)

The default learner profile is now `multistep`. `learning.use_profile("one_step")` gives
back the old rule (one-step TD, constant epsilon 0.15) for intervention runs. Every new
parameter is stored in the controller, so checkpoints carry it.

| Mechanism | What it does | Parameter (multistep / one_step) |
|---|---|---|
| Eligibility traces | Naive Q(lambda) with replacing traces over the last `trace_steps` own decisions (12 at lambda 0.8). Emptied at death. Demonstrations never enter it. | `lam` 0.8 / 0 |
| Replay | After each own outcome, re-learns `replay` transitions from the 32-entry memory (this makes the memory read, gap 17). Entries are picked by a fixed stride, so no random numbers are used. | `replay` 4 / 0 |
| Conjunctions | food_direction x distance band, food_near x energy band, a (bearing, distance, hardness, energy) tile, and heard_symbol x heard_bearing as a hook. The bearing hook activates once a heard message carries `bearing`; the world does not supply one yet. | `conjunctions` on / off |
| Egocentric frame | Weights see direction bins and move actions relative to the visible food's bearing, so a step toward food learned in one compass direction carries over to the other seven. Actions, memories and q tables stay in the world frame. | `egocentric` on / off |
| Decaying exploration | epsilon x (floor + (1 - floor) x h / (h + decisions)): 0.30 at birth, 0.165 after 200 decisions, toward 0.03. | `epsilon` .3, `epsilon_floor` .1, `epsilon_halflife` 200 / constant .15 |

The reward is unchanged: the engine's energy and health change only.
`test_reward_treats_action_names_alike` checks that relabelling actions in the experience
only relabels the learned weights. Newborns start blank. For the genome,
`learning.controller_from_biases(bias_weights, alpha=, epsilon=)` builds a controller with
innate starting weights. With the egocentric frame, ("food_direction", "n") x move_n
means "toward the food".

Measured with `python -m haishool.life8.analysis.credit corridor|sparse|default`. Every
row compares the same seeds with one mechanism switched off.

- **Corridor** (reward 1 only for eating within reach, 10 actions, so chance is 0.1;
  40 seeds; `results/credit-corridor-10.json` and `-40.json`). Greedy P(step toward)
  after 10 episodes: multistep **0.90** (35/40 seeds above 0.5), one_step **0.16**
  (6/40). Lambda 0.5, 0.7, 0.8 and 0.9 all gave 0.90-0.91, so the default of 0.8 is not
  distinguished from its neighbours. Replay (0.71 without it) and the trace itself
  (traces_only 0.51) do the work. After 40 episodes one_step also gets there (1.00,
  against 0.93 for multistep). The gain is speed. The old rule is not stuck at chance in
  a clean corridor.
- **Sparse world** (the review's probe: 16 founders, 15 patches, no automatic
  reproduction, tools, communication or culture, 1500 ticks, seeds 1-6, 16 enabled
  actions so chance is 1/16; `results/credit-sparse.json`). Founders starved:
  multistep **83/96**, one_step 88/96, heuristic **7/96**. Without replay 81, without
  the egocentric frame 89, without conjunctions 81, without traces 84. **This is a
  null.** The gap to the heuristic is not closed. The learners also reproduce (775
  births with multistep, 430 with one_step, 0 for the heuristic, which never chooses
  `reproduce`). Each birth costs the parent 16 energy, so founder starvation mixes
  foraging with reproductive investment. Final population: 170 vs 102 (one_step).
  Founder P(toward | food visible, out of reach) at age 100+: **0.097** vs 0.046
  (one_step, below chance). It rises with age to 0.13 at 400-1600. Without the
  egocentric frame it is 0.069, and without conjunctions 0.070.
- **Default worlds** (seeds 1-4, 600 ticks, population_limit 256;
  `results/credit-default.json`; 26 actions, so chance is 0.038). P(toward | visible,
  out of reach) at age 100+: multistep **0.073** (0.12 at age 400-800), one_step
  **0.024**. Without replay it is 0.017, and without the egocentric frame 0.047. Cost:
  P(eat | food in reach) at age 100+ falls from 0.80 (one_step) to **0.50**. Most of the
  fall comes from replay (0.75 without it). Final population is 414 vs 454, and 63 vs 59
  founders starved. The world is not better off in this setting. The old learner's main
  failure here was "eat" chosen out of reach (P = 0.68 at age 100+ in a 300-tick run).
  Replay is what removes it (0.15).
- Cost per tick (default world, 4 seeds in parallel): 45 s per 600-tick run vs 22 s
  for one_step. In the sparse world: 29 s vs 5 s.

Slow claim test: `test_close_soft_food_approach_rises_against_the_one_step_rule_in_the_sparse_world`.
With the new default learner, the foundation's slow founder test
(`test_experiments.py`) still shows the survival gain, but its P(eat | near) margin
falls to 0.198, just under the 0.2 it asserts.

## Critic fixes before the cluster search (B1, B2, F1, F2)

The round-8 critic (`matrix-review/round8-critic.md` in the session scratchpad) found the
engine sound but the search not ready. These are fixed, each with a regression test:

| Finding | Problem | Fix | Test |
|---|---|---|---|
| B1 | The documented search ran plain worlds (no kin credit, frozen genome, no predators or blooms), where signalling cannot pay, and its sender state ignored danger and blooms. | `search --preset living` merges `config.LIVING` under the `--option` values (plan and run; recorded in `search.json`). The documented example and `scripts/life8_search.sh` use it. With the ecology on, the sender state and the convention situation gain `danger_seen`, `danger_direction` and `bloom_seen` (`search.ECOLOGY_BANDS`, as `analysis/ecology.py`). New source `bridged` alternates world7 worlds that reach bodies and synthetic planets; with the ecology on they get their planet's predators, damage, visibility and temperature (`bridge.connect_ecology`). | `test_search.py::test_living_preset_reaches_the_search_rows_and_its_sender_state` (plan and row `kin_credit` > 0, ecology bands in the sender state), `::test_bridged_plan_alternates_world7_and_synthetic_and_splits_by_host` |
| B2 | The sender criterion passed on the pooled test, which individual and lineage habits pass (masked and scrambled runs gave significant excess). | The criterion uses the within-sender excess (symbols shuffled only inside each sender's calls) in `discovery.verdict` and `search.criteria`. `search.ARMS` gains `masked`. In paired rounds the normal arm's within-sender excess must beat the masked arm's (scrambled if masked is missing) by `sender_over_control_bits` = 0.02, and the paired score's sender part is that excess over the control. `discovery.discover` passes its scrambled run as the control (`verdict(..., sender_control=...)`), and so does `analysis/ecology.py`. | `test_search.py::test_planted_sender_habits_fail_the_sender_criterion` (planted per-sender habits: sender = False in both `discovery.verdict` and `search.criteria`, screen and paired; the same mapping against an uninformative control passes), `test_discovery.py::test_verdict_needs_all_three_with_explicit_thresholds` |
| F1 | Kin credit was added to whatever the sender did during the kin window (89% landed on non-signal actions). | The credit goes to the call that caused the hearing. A pending hearing is kept per (call, relative). When a call is learned, its decision (weight keys, signal action, memory position) is stored in the ecology record (`calls`). Each tick a call earns `kin_credit x r x` the hearer's advantage (reward minus its running mean, `KIN_BASELINE_RATE` 1/16, so metabolic drift is not credited), discounted by gamma per tick since the call, and `learning.credit_decision` moves only that decision's (feature, signal action) weights by alpha x credit / n, and adds the credit to the remembered call transition so replay keeps it. Every `learn()` gets the organism's own reward only. Full logs get a `kin_credit` event (call, message id, credit); the `kin_credit` field of `action` events is gone. | `test_ecology.py::test_kin_credit_lands_on_the_signal_transition_only`, `::test_credit_decision_moves_only_that_decisions_weights`, `::test_kin_credit_in_a_living_world_reaches_only_calls`, plus the rewritten amount, stranger and symbol tests |
| F2 (high) | The convention statistic passed on inherited symbol habits (evolving genome). | `lineage.convention_agreement` and `search.convention_agreement` test the situation-specific difference: each pair's agreement minus the agreement of its situation-blind symbol distributions, within minus across. `within`/`across` stay the raw agreements; `difference_raw` is the old statistic. | `test_search.py::test_inherited_symbol_habits_are_not_a_convention`, `test_lineage.py::test_inherited_habits_without_a_situation_mapping_are_not_a_convention` |

Measured after F1 (LIVING, full log, 250 ticks): seed 1, 2901 credited calls; seed 2, 4302;
all of them signal decisions (before: 11% of credited transitions were calls). The
ecology record grew by `calls` (at most 44-46 open call records) and `reward_mean`; a
LIVING state from before F1 with kin credit pending is refused, one without is migrated.
Exact continuation is unchanged (the LIVING continuation tests pass, normal and scrambled).
Still open from the critic: F3 (no multiple-comparison guard, no minimum sample sizes or
confirmation round: read the counts of passed criteria against worlds x alpha), F4
(libm differences across platforms: each host is exact with itself), F5-F7.

## Many-world search (`search.py`, `scripts/life8_search.sh`)

`python -m haishool.life8.search run --out DIR --worlds N --rounds R --ticks T --keep F --workers W [--source plain|synthetic|world7|mixed|bridged] [--preset living] [--option key=value ...]` runs a successive-halving search over living worlds; `report --out DIR` summarises it and `collect --out MERGED DIR...` merges several folders. World options are a pass-through dict of Config fields (JSON values), so fields added by later stages work without changes. The module is standard library only: the hosts have no numpy, so the measurements are pure-python re-implementations, and a test checks them against `discovery.py` and `lineage.py` (same MI to 1e-9, same sign test, same within/across convention agreement to 1e-4; the language cut-offs equal `discovery.THRESHOLDS`).

How a search runs:

| Step | What happens |
|---|---|
| Round 0 (screen) | Every candidate runs `T` ticks from tick 0 with the compact log in a process pool. Only measurements from that single run are used. |
| Selection | Worlds are ranked by the composite score (ties by id). The best `ceil(F x alive)` are kept. Worlds that stopped (extinction, capacity) or failed keep their row but cannot continue. The kept worlds' end states become `checkpoints/<id>.r<round>.json.gz`. Everything else is deleted, and a `selection` row records the kept and dropped ids and the cut-off. |
| Round r >= 1 | Kept worlds continue from their checkpoints for `T x growth^r` ticks (growth defaults to 2). Paired branches then start from each end state, for kept worlds only: `--reps` replicates (default 6) of `--branch-ticks` (default 200) in four arms: normal, `channel=masked` (calls delivered and paid, symbols hidden: the sender control), `channel=scrambled` and `tools=False`. The world RNG is reseeded identically per replicate in every arm. Each world's row is written as soon as its own branches are in. |
| Files | `results.jsonl` holds one `world` row per world per round, failures included, plus the selection rows. `search.json` holds the plan, arguments, status and stop reason. Checkpoints exist only for the kept worlds of the latest round. No per-tick logs are written. |
| Resume | Re-running `run` on the folder skips finished rounds and worlds. A kept world whose end state was lost is re-simulated deterministically. A test checks that an interrupted search ends with rows and checkpoints equal to an uninterrupted one. |
| Output cap | `--max-output-mb`: before each round, the search projects the folder size (current files plus one end state per running world, sized by the largest checkpoint, at least 1 MB). If the projection is over the cap, it stops with `stop_reason: output_cap`. It can be resumed with a larger cap. |

Measurements use the last half of each round's ticks:

- **sender**: I(symbol; food_near, food_seen, energy_band, neighbor_near, and with the ecology on danger_seen, danger_direction, bloom_seen) against a permutation null, plus a within-sender null that shuffles only inside each sender's calls. The within-sender null removes individual habits that merely correlate with state; a test shows such habits give pooled excess > 1.5 bits and within-sender excess < 0.01. The criterion uses the within-sender excess (>= 0.05 bits, p <= 0.01) and, in paired rounds, needs the normal branches' within-sender excess to beat the masked branches' by >= 0.02 bits (fix B2). Screen score: the smaller of pooled and within excess; paired score: the excess over the masked control (scale 0.02).
- **receiver**: I(heard; next action) against a permutation null. In paired rounds the score uses the normal-branch excess minus the scrambled-branch excess.
- **scramble fitness**: normal minus scrambled food energy per agent-tick, with a sign test (normal minus masked is reported as well).
- **convention**: situation-specific symbol agreement (pair agreement minus the agreement of the pair's situation-blind symbol distributions, fix F2) within founder lineages minus across them, against a label-shuffle null. Situation: food_seen, energy_band (+ the ecology bins).
- **tools**: paired rounds use normal minus tools-off food energy per agent-tick. Round 0 uses only the engine's `tool_damage_gain` per agent-tick, an unpaired screen.
- **cooperation**: share reciprocity against a proximity-matched null. For each share A to B: did B give to A in the previous 100 ticks? This is compared with a recipient drawn uniformly from everyone within interaction range of A at that tick.

Composite score = sum over the six components of `min(2, max(0, x)/scale) x f`. f is 1 when the component's p <= 0.05 and 0.25 otherwise; the round-0 tool screen always gets 0.25. The scales are sender 0.05 bits, receiver 0.02 bits, scramble fitness 0.004, convention 0.05, tools 0.004 (screen 0.01) and cooperation 0.05. An unmeasurable component (no calls, no shares) scores 0. The score is for ranking only. The pass flags (`criteria` in each row) use stricter cut-offs (`search.CRITERIA`). A pass is a measured dependency or paired effect in this toy world, not language, technology or an institution.

**Local tiny search** (8 worlds, mixed plain and synthetic, 2 rounds, 200 then 400 ticks, keep 0.5, 2 branch reps of 100 ticks): it ran in 32 s with 8 workers. In round 1, all 4 kept worlds had language verdict `null`, and no world passed any criterion. The best round-1 sender excess was 0.030 bits, below the 0.05 cut-off. The test `test_tiny_search_keeps_the_best_half_and_saves_only_kept_checkpoints` runs the same shape.

**Cost of one world per 1000 ticks** at the search defaults (`food_patches=30`, `population_limit=256`, about 41-44 individuals). The measurement is `run_segment` (engine, tracking and all round measurements with 100 permutations), seed 1 alone, then one world per core at once:

| Host | Python | nproc | 1 world alone (s / 1000 ticks) | All cores busy (s per world) | Throughput (world-ticks per hour) |
|---|---|---|---|---|---|
| this Windows machine (shared with other agents) | 3.11 | 32 | 34.9 | 45.5 (16 processes) | 1.07 M (16 processes) |
| adler40 | 3.14.4 | 24 | 20.4 | 40.7 | 1.83 M |
| knecht24 | 3.14.4 | 24 | 40.0 | 66.0 | 1.15 M |
| specht32 | 3.14.4 | 12 | 31.7 | 58.1 | 0.66 M |
| falke64 | 3.14.4 | 12 | 17.1 | 29.7 | 1.31 M |

The default Config (`food_patches=60`, about 93 individuals) costs 60.6 s per 1000 ticks alone on this machine. Per-tick cost grows faster than linearly with population. In a search, each kept world also costs 3 x reps x branch_ticks ticks per round (3600 at the defaults).

Example: 400 worlds per host with `--ticks 500 --keep 0.25 --rounds 3`. That is 200k ticks in round 0, 100k + 360k (branches) in round 1, and 50k + 90k in round 2, about 0.8 M world-ticks per host. Expected wall time is about 0.45 h on adler40, 0.6 h on falke64, 0.7 h on knecht24 and 1.2 h on specht32, when nobody else is using those machines.

Disk: a kept world's gzipped checkpoint is about 0.36-0.5 MB at about 45 individuals; uncompressed state is about 9.8 MB. During a round, each running world holds one such file until selection, so 400 worlds need about 200 MB at peak. `results.jsonl` takes about 2-3 KB per world-round.

**Hosts** (`scripts/life8_search.sh start TAG START_SEED "ARGS" HOST:WORKERS:WORLDS[:CAP_MB] ...`):

- One plan of sum(WORLDS) worlds from `START_SEED` is built locally (`search plan`, with the `--preset` and `--option` values of ARGS) and cut into consecutive parts in host order (`search split`), so no world7 world is searched twice. Bridged worlds need `haishool.evo` and numpy, which adler40, specht32 and falke64 lack. The hosts need only python3 3.11 or later and tmux.
- The code, the host's plan part and a launcher `run.sh` are copied with tar/cat over ssh into `~/life8-search/TAG/`. Each search runs under `nice -n 10` in a detached tmux session `life8-search` (it survives disconnects), with its own `out/` folder and `search.log` (UTC timestamps on every round line). A host with an existing `life8-search` session or `TAG/out` is refused. `stop TAG HOST...` kills only that session.
- `CAP_MB` (default 1500) becomes `--max-output-mb`. Give knecht24 an explicit cap; the task says it has about 2 GB free, though `df` showed 31 GB free on 2026-10-02.
- `resume` relaunches a search, `status` shows progress and disk, and `collect` copies each host's `results.jsonl`, `search.json` and log into `runs/life8-search/TAG/HOST/` and merges them into `merged/report.json`.
- The first version was tested with a local ssh stand-in (`LIFE8_SSH`). The tmux version ran on the real hosts: a 64-world benchmark (`r8-bench`, session `life8-bench` via `LIFE8_SESSION`) and the search `r8a` below.

### Search r8a on the cluster (launched 2026-10-02 16:41 UTC, running at hand-off)

Hand-off record: `docs/life8/search-run.json` (hosts, folders, sessions, commands, sizes, cost
measurements, expected finish, how to collect and report).

- **Command:** `bash scripts/life8_search.sh start r8a 0 "--preset living --source bridged --rounds 4 --ticks 500 --growth 2 --keep 0.25 --reps 6 --branch-ticks 200 --perm-reps 100" adler40:22:1800:4000 knecht24:22:920:1500 specht32:10:620:3000 falke64:10:1100:3000`
- **Worlds:** 4440 (2220 world7 worlds that reach bodies, seeds 0-12121 scanned, and 2220 synthetic planets), living preset with each planet's predators (0-3), split adler40 1800, knecht24 920, specht32 620, falke64 1100.
- **Shape:** 4 rounds of 500, 1000, 2000 and 4000 ticks (worlds end at tick 7500), keep 0.25, paired branches in rounds 1-3 for kept worlds (6 x 200 ticks x 4 arms).
- **Where:** `~/life8-search/r8a/` on each host (code, plan.json, run.sh, out/, search.log), tmux session `life8-search`, nice 10, 22/22/10/10 workers, output caps 4000/1500/3000/3000 MB. No GPU is used.
- **Cost measured first** (`r8-bench`: 64 worlds, 300 + 600 ticks, all workers busy): 39 / 77 / 52 / 29 s per world per 1000 ticks on adler40 / knecht24 / specht32 / falke64. The first 10 minutes of r8a ran at 50 / 85 / 72 / 36 s, so the expected wall time is 3.3-4.1 h.
- **Expected finish:** about 20:00 UTC (knecht24) to 20:50 UTC (specht32), that is 22:00-22:50 Berlin time; range 19:45-21:30 UTC.
- **Collect and report:** `bash scripts/life8_search.sh status r8a adler40 knecht24 specht32 falke64` until every host shows `complete rounds_done`, then `bash scripts/life8_search.sh collect r8a adler40 knecht24 specht32 falke64` (writes `runs/life8-search/r8a/merged/report.json`). Report passed criteria next to worlds x alpha (F3 is open) and confirm the best final-round worlds from their checkpoints with fresh seeds before any claim.

**Cross-platform caveat (measured):** the same code and seed (300 ticks) gives the same population and measurements on Windows/Python 3.11 and Linux/Python 3.14, but the state hash differs. Two `bond_strength` values differ in the last bit. They come from `math.expm1` in `materials.combine_items`, and the C maths library differs between the two platforms. Each search is exact on its own host, and a resume must run on the same platform. Results from different hosts are independent samples, not bit-identical reruns.

## Living-world ecology (round 8, ecology stage: `world.py`, `config.py`)

Goal: an ecology where signalling *can* pay, so meaning could emerge without being supplied. Every
mechanism is a `Config` field that is neutral at its default (default worlds keep their trajectories;
the whole suite passes). `config.LIVING` / `config.living_config(**overrides)` is the measured preset.
Ecology draws come only from `Random(seed ^ ECOLOGY_SALT)`, saved as `eco_rng_state`; the record lives in
`world.eco` (state key `ecology`) and continues exactly from a checkpoint (tested, normal and scrambled).

| Mechanism | Fields (LIVING value) | What it does |
|---|---|---|
| Rich short-lived patches (approach) | `bloom_interval` 8, `bloom_amount` 16, `bloom_ttl` 25, `bloom_richness` 2, `bloom_partners` 1 | A bloom appears at a random place every 8 ticks, gives 2x energy per unit and spoils after 25 ticks. Ledger keys `food_bloomed` / `food_spoiled` keep the food ledger closed. `bloom_partners`=2 makes it a cooperative resource. Observation bin `bloom_seen`. |
| Predators (flee) | `predators` 2, `predator_speed` .6, `predator_mass` 1.5, `predator_damage` .25, `predator_sight` 5, `predator_visibility` .6, `predator_cooldown` 15 | Chase the nearest individual; wound = damage x 2 m_p/(m_p + body_mass); then sated and wandering. Prey see them only within 0.6 x their own (health-scaled) sight radius: bins `danger_seen`, `danger_direction`. Deaths are `predation`. Hazards without an energy budget. |
| Real injuries | `permanent_injury` .25, `health_ability` True, `strike_injury` .2, `injury_mass_scaling` True | A quarter of every wound lowers the health ceiling for good (rest heals only to it); health scales eating, striking and sight by 0.5 + 0.5 health; organism strikes wound by strike_injury x damage x attacker/target mass. |
| Calls beyond sight, own gene | `call_power` True, `call_reach` 8 | Voice power is its own heritable gene (founders 0.75-1.25, mutated at birth with mutation_rate/scale): reach = 8 x power (sight at sensing 1 is 5), cost = signal_cost x (1 + .3 power). Fixes review bug 9 in this mode. |
| Sender credit route (kin) | `kin_credit` 1, `kin_window` 8 | For 8 ticks after a relative hears a call, that call (its decision's features and signal action, not whatever the sender does later; fix F1) is credited kin_credit x r x the hearer's advantage (reward minus its running mean), discounted by gamma per tick since the call. (Before F1, and in the measurements below: the sender's learning reward at each tick of the window included kin_credit x r x the hearer's own reward.) r = 2^-(d_a+d_b) through the closest shared ancestor in an in-engine pedigree (8 generations; parent-child .5, siblings .25). Independent of the symbol (tested with all 4 symbols); never in the ledger or in `last_outcome`. |
| Receiver features | `receiver_features` True | Heard message gets an 8-way `bearing` to the call's origin (the learner's documented hook, with the heard_symbol x heard_bearing conjunction), `heard_age_band`, `heard_kin_band` (close/kin/distant/stranger), and the sender's `own_call` (its live call's symbol). |
| Reciprocity | `reciprocity_features` True | Directed per-peer tallies (`given`, `received`) in social memory; bin `neighbor_balance` (i_owe / they_owe / even / none) for the nearest neighbour. |
| Food regulation | `food_patches` 30 | Population 20-67 over 2000 ticks, far below the 128 cap; but 2/12 seeds went extinct (see below). |
| Other | `ambient_temperature` 20, `genome` "evolving" | Objects cool toward the ambient temperature; the evolving genome lets lineages inherit signal biases. |

`learning.py` got one additive hook (owned by the learning stage): `ECOLOGY_FEATURES` are read from the
observation when present, and `danger_direction` is rotated in the egocentric frame like `food_direction`.
`bridge.connect_ecology(cfg, founders)` maps the round-7 hints: predators = round(10 x predator_density),
predator_damage = 0.1 + 0.6 x attack_injury, predator_visibility = clip(1 - 0.5 x predator_pressure, .4, 1),
ambient_temperature = ambient_temperature_c; it marks the hints connected. Oxygen / max_body_mass are not connected.

### Measured: 12 seeds x 2000 ticks (`docs/life8/results/ecology-living-12x2000.json`)

`python -m haishool.life8.analysis.ecology --seeds 1-12 --ticks 2000 --jobs 12`. Arms on the same seeds:
normal, scrambled channel, and kin_credit=0. Window ticks 1000-2000; 8 paired 200-tick branches from tick 1500.
Cost: 240-640 s per seed for all arms (12 processes on a shared machine), about 3-4 s per 100 ticks at 30-50 individuals.

| Test (seeds 1..12) | Result |
|---|---|
| discovery verdict | null 11/12, partial 1/12 (seed 10: default-state sender test passed, 0.053 bits, then the world went extinct) |
| Sender I(symbol; food_near, food_seen, energy, neighbour) excess | 0.004-0.056 bits, p <= 0.005 in 10/12; at or above 0.05 bits only in seeds 5 (p 0.09) and 10 |
| Sender I(symbol; danger_seen, bloom_seen) excess | -0.003 to 0.036 (mean 0.0059); p <= 0.01 in 6/12. kin_credit=0 arm: -0.001 to 0.006 (mean 0.0016); kin minus no-kin positive in 8/12 (sign p 0.39). Scrambled arm: up to 0.11 (seed 7), so it is not caused by the content being useful |
| Receiver I(heard; next action), normal minus scrambled | -0.018 to +0.010 bits, positive 7/12 (sign p 0.77): null |
| Food energy per agent-tick, normal minus scrambled branches | positive in 9/11 seeds (sign p 0.065); no per-seed sign test reaches 0.05; survival and population differences mixed |
| Lineage agreement, situation (danger_seen, bloom_seen) | within 0.250-0.323 vs across 0.234-0.383; within > across with p <= 0.015 in 6/12 seeds, but only 0.01-0.05 above chance 0.25. Inherited symbol habits (evolving genome) explain this as well as a convention; persistence was not tested |
| Information asymmetry (diagnostic) | of deliveries from a sender that sees danger, the hearer does not see it in 65-87%; for blooms 45-60%. Calls do carry what hearers lack |
| Symbol use by situation (seed 1) | danger and bloom calls spread over all 4 symbols in about chance proportions |
| Populations / stops | 2/12 extinct by tick 2000 (seeds 5, 10); others 20-67 alive. Starvation dominates deaths; predation 23-54 per run (about 10-13%); strike injury 0-2 |

**Verdict: no meaning emerged.** Senders' calls carry a little, significant information about their state,
slightly more with the kin route on, but receivers do not act on the content and scrambling costs nothing
measurable. Not language, not a convention.

### Why: the oracle-sender probe (`docs/life8/results/ecology-oracle-probe.json`)

Diagnostic intervention only (a supplied mapping, never an engine mechanism): every voiced individual that
sees a predator calls `signal_0`, one that sees a bloom calls `signal_1`. Seeds 1-6 x 1200 ticks, normal vs
scrambled channel (`python -m haishool.life8.analysis.ecology_probe`). Even with perfectly informative calls,
receivers did not use them: receiver information over scrambled -0.009 to +0.008 bits; predator strikes per
1000 agent-ticks, normal minus scrambled, -6.1 to +4.9 (lower in 4/6); bloom meals +3.3, +4.3, -1.3, +2.3,
-1.7, +3.9; population +15, +8, -36, -4, +14, +22. So the bottleneck is on the **receiver side and in the
value of the information**, not only in the sender credit: a predator strikes the nearest individual, usually
the caller itself, so hearers 5-10 units away are rarely at risk, and a bloom 8 units away is often gone before
a hearer arrives. No sender credit route can select informative calls while no receiver response pays.

### Next change (not done)

1. Make the information worth acting on for hearers: predators that hunt across call range (packs, or strikes
   that hit groups) and blooms that last long enough to reach and need partners (`bloom_partners`=2). Rerun
   the oracle probe until receivers profit from oracle calls in a paired test; the probe is the gate.
2. Only then rerun the 12-seed measurement with the kin route.
3. Lower the preset's extinction risk (2/12) before long evolution runs, e.g. bloom_interval 6 or one predator.
