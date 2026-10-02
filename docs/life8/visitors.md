# life8 visitors: experience data and the visitor model

A **visitor** is an inhabitant of a twin copy of a search planet whose actions come from a
small GPT trained only on the experience of life8 inhabitants: no human text and no other
Haishool data. A visitor sees only its own local observation, acts only through its own body
and the normal actions at their normal costs, and can die. Its offspring are ordinary
inhabitants: nothing learned is inherited. Visitor worlds are a separate arm, always paired
with an untouched twin (same planet, same seed). They never enter the search's emergence
results.

There are two variants:

- **traveller**: the model never saw the planet it visits.
- **elder**: the model saw the planet's first 1,000 ticks and is placed into its later history.

This page covers the experience data and the model: the format, the data set, training,
measurements and the export. The visitor runtime and the paired arms are separate files
(`visitor.py`, `visitor_arms.py`), summarised below together with the results of the trained
model as a visitor on the 40 TRAVELLER and 40 ELDER planets.

## The line format (`haishool/life8/experience.py`, the single source of truth)

```
q life8 see <feature> <value> <feature> <value> ... want <outcome>. a <action>.
```

- **Features.** The features are what the inhabitant's own learner decides from:
  `learning.observation_features(observation, extras=True)` (`experience.features_of`). They
  come in the fixed order `experience.FEATURE_ORDER`: the 19 state bins, the 7 living-world
  bins, `heard_symbol`, `food_distance_band` and `heard_bearing`.
  - A feature with value `None` (no held tool, nothing heard, no food seen) is left out.
  - Booleans are written `yes`/`no`, integers as digit tokens (`haishool.truth.num`), and
    strings as one lowercase token.
  - An unknown feature name, or a value that is not one dense token, is an error.
- **Outcome.** The outcome is `thrive`, `live` or `suffer`. It comes from the individual's own
  learning reward (energy change + 8 x health change, birth transfer excluded, as in
  `World.step`), summed over the 10 ticks that start with the decision. If the individual
  dies inside those 10 ticks, the sum is over the ticks it lived.
  - A decision whose window runs past the simulated ticks while the individual is alive is
    dropped.
  - The cuts are the 33rd and 67th percentiles of the calibration set, which is every kept
    line of the 320 train planets (6,259,415 lines): `suffer` below **-1.5893**, `thrive`
    above **0.1048**, `live` in between (bounds inclusive).
  - At inference a visitor asks `want thrive`.
- **Action.** The action is the engine's action name as one token (`experience.ACTION_WORDS`
  = `learning.ACTIONS`, 26 actions). The engine chose these actions as `{"type": name}`, so
  share, teach, imitate, strike and the others used the engine's default target choice.
- **Checks.** Every line passes `haishool.truth.is_dense`. A line with all 29 features stays
  under 120 tokens (tested). Real lines average 47.6 tokens with the end token.

API for the visitor runtime:

- `features_of(observation)` turns an engine observation into features.
- `prompt_for(features, want="thrive")` gives `q life8 see ... want thrive. a`.
- `action_of(answer)` returns the engine action, or `None`.
- `build_world(entry, log)` builds an r8a plan entry, as `search.build_world` does.

Example line (train):

```
q life8 see food_direction n food_near no object_near no neighbor_near no energy_band high tool_available no health_band well inventory_count 0 target_hardness_band middle target_open_band low neighbor_familiar no neighbor_last_exchange none danger_seen no danger_direction none bloom_seen no neighbor_balance even food_distance_band close want suffer. a signal_2.
```

## Data set

`python -m haishool.life8.experience generate --out DIR --raw RAW --workers 30 --keep 0.25 --elder-keep 0.5`
was run on this PC (32 cores, a process pool of 30). Each planet ran in its own process with
the full log. Lines were taken from `World.step`'s `transition` records in process. No run
archive was written.

- **Raw files.** Each planet's complete decisions went to a gzip file under the session
  scratchpad (`visitors/raw/`, 440 planets). These are intermediate files, not life8
  archives.
- **Subsample.** The subsample is deterministic per (planet, tick, individual), using crc32.
  It was applied when the data set was assembled.
- **Data set location.** The data set itself is in `scratchpad/visitors/data/` (3.3 GB) and
  `~/haishool-visitor/data/` on adler40.

### Planets (from `runs/life8-search/r8a/stage/plan.json`, the r8a plan of 4,440 planets)

`experience.select_planets` makes a deterministic draw (seed 20261002). For each source it
samples 30+30+300 world7 planets and 10+10+100 synthetic planets:

| group | world7 | synthetic | ticks | data |
|---|---:|---:|---:|---|
| traveller (reserved) | 30 | 10 | none | **none at all** |
| elder (reserved) | 30 | 10 | 1,000 | topic `predict_life8_elder` only |
| train | 240 | 80 | 2,000 | `predict_life8_experience-train` |
| dev | 30 | 10 | 2,000 | `predict_life8_experience-dev` |
| sealed | 30 | 10 | 2,000 | `predict_life8_experience-sealed` |

The experience splits are split by planet. The elder topic covers the 40 elder planets. It
keeps only decisions whose 10-tick window ends by tick 1,000, so no line uses any reward from
tick 1,001 on. Its train/dev/sealed split is by individual (80/10/10, crc32 of planet and
individual), so the model has seen every elder planet, and its dev and sealed sets are whole
lives it did not see.

**Traveller planets (no data):** synthetic-1735 synthetic-185 synthetic-2399 synthetic-2597
synthetic-3775 synthetic-3993 synthetic-4111 synthetic-791 synthetic-851 synthetic-911
world7-10081 world7-10632 world7-11163 world7-11439 world7-11543 world7-1191 world7-12095
world7-1698 world7-1906 world7-2140 world7-2488 world7-2593 world7-2862 world7-3155
world7-3198 world7-3552 world7-356 world7-423 world7-4627 world7-4808 world7-6036
world7-6143 world7-6439 world7-7102 world7-7672 world7-7958 world7-8348 world7-8510
world7-9006 world7-9996

**Elder planets (lines from ticks up to 1,000 only):** synthetic-1771 synthetic-1847
synthetic-1965 synthetic-241 synthetic-2675 synthetic-2725 synthetic-2921 synthetic-4249
synthetic-4275 synthetic-997 world7-10536 world7-10753 world7-10917 world7-1119 world7-1200
world7-213 world7-3254 world7-4104 world7-4162 world7-4336 world7-4396 world7-5109
world7-5273 world7-5351 world7-5762 world7-5890 world7-6037 world7-6577 world7-7373
world7-7682 world7-7849 world7-7862 world7-834 world7-8613 world7-8769 world7-8854
world7-8926 world7-9102 world7-9410 world7-9961

**world7-7849 went extinct at tick 880.** It has no later history, so an elder can be placed
on only 39 of the 40 elder planets. The lists are also in the data report
(`report.json["planets"]`) and in the export manifest.

### Rate and size (measured)

- **Pilot.** 30 train planets x 2,000 ticks with 30 workers took 252 s wall. That is
  91,089 decisions per planet on average, 132 s per planet and 688 decisions/s per process.
  The data planets average about 79,500 decisions each, or about 25 M decisions for the 320
  train planets. A keep rate of 0.25 was chosen for the experience topic, which gives about
  6.3 M train lines, inside the 3-8 M target. The elder topic used 0.5, because it has only
  40 planets x 1,000 ticks.
- **Full run.**
  - The other 410 planets took 3,153 s wall with 30 workers, so 3,405 s in total.
  - Per-planet time had a median of 195 s and a maximum of 688 s, with the machine fully
    loaded.
  - 33,468,417 decisions over 788,415 planet-ticks, about 9,800 decisions/s for the whole
    pool.
  - Assembly from the raw files took 239 s.
- **Stops.** 72 of 440 planets went extinct before their last tick: 59 train, 3 dev,
  9 sealed and 1 elder. The others reached their tick count. Mean population on the data
  planets was 40.8.

| topic / split | lines | tokens | planets |
|---|---:|---:|---:|
| experience train | 6,259,415 | 297,775,574 | 320 |
| experience dev | 939,210 | 44,647,608 | 40 |
| experience sealed | 719,234 | 33,141,298 | 40 |
| elder train | 672,089 | 31,912,968 | 40 |
| elder dev | 78,529 | 3,720,100 | 40 |
| elder sealed | 78,224 | 3,735,936 | 40 |

Outcomes in experience train are suffer 2,065,607, live 2,128,201 and thrive 2,065,607.
Actions in experience train are dominated by `eat` (2,375,455, 38%), then `rest` (241k),
`combine` (181k), `drop` (180k), `collect` (177k) and `reproduce` (174k). The rest are each
about 2.5-3%: the learners' epsilon exploration spreads over all 26 actions.

`report.json` uses the curriculum format that `train_final.input_manifest` reads. Each topic
has `topics.<topic>.splits.<split>.{lines,tokens,sha256}`. The report also records the cuts,
the planet groups, per-planet stats and the action and outcome counts.

## Training (adler40, RTX 4090, GPU 0)

- **Where.** The run used `~/haishool-visitor/` on adler40. That is a new folder holding a
  copy of `haishool/` and `data/`, sent with tar over ssh. Nothing else on adler was touched.
  - It ran in tmux session `haishool-visitor` with `run-visitor.sh`, which trains, evaluates
    and exports, and can be restarted phase by phase.
  - It ran on GPU 0 only (`CUDA_VISIBLE_DEVICES=0`), as one process under nice 5, with
    torch 2.14 from `/home/ender/homunculi/.venv`.
  - Before the run, GPU 0 held another user's idle ComfyUI process (7.4 GB, 0% utilisation).
    It was shared, not stopped. GPU 1 was not used.
- **Command.** `python -m haishool.life8.experience train --data data --out runs/visitor
  --steps 140000 --batch 128 --ctx 128 --layers 6 --width 384 --heads 8`. This is
  `train_final.train(mix="sim")`, with holdout 0 and peak lr 1e-3 (the default for a model
  trained from scratch).
- **One change, made in the wrapper and not in `train_final.py`.** Mix sim requires
  stories (`legacy_*`) for 20% of the tokens and splits the lesson share equally between
  topics. Visitor data has no stories, and an equal split would give the 40-planet elder
  topic half of all tokens. The wrapper therefore sets the predict share to the whole and
  weights the two topics by their tokens: experience 0.9032, elder 0.0968 (observed in the
  samples: 0.9032 / 0.0968).
- **Rate measured first.** A rate test used a 30-planet pilot data set, 1,500 steps each:
  85 steps/s at batch 48 and 48 steps/s at batch 128, so batch 128 moves 1.5 times more
  tokens. 140,000 steps at batch 128 was chosen for about 50 minutes.
- **Result.**
  - Training took **2,940 s (49 min)** at 47.6 steps/s, after 2.8 min of token encoding. It
    ran 18:21-19:12 UTC.
  - 2.29 B tokens were sampled, which is 7.0 passes over the 329.7 M single-copy tokens.
  - The model has a vocabulary of 105 words and 10.71 M parameters.
  - Loss per token stayed between 0.33 and 0.36 from step ~5,000 on (0.357 at step 1,000,
    0.3455 at the end), averaged over all tokens. Most tokens are observation tokens.
  - No line was dropped as too long.

## Measurements (dev and sealed)

`python -m haishool.life8.experience evaluate` scores every dev and sealed line, 100 s on
GPU 0. For each line the model gets the line's own prompt (`... want <its outcome>. a`) and
answers with the most likely of the 26 action words at the next position.

- **Unrestricted answer.** The model's unrestricted top token was an action word on 100% of
  lines.
- **Baselines.** All baselines are computed from the train files of both topics (6.93 M
  lines, 1,766,867 distinct observations).
  - Majority action per observation: the exact `see` text. An observation not seen in train
    falls back to the overall majority.
  - Majority action per (observation, outcome).
  - The overall majority, which is `eat`.
- The life8 learner's own greedy action is not available offline and was skipped.

| split | lines | model | majority per observation | majority per (observation, outcome) | always `eat` | thrive lines | model, thrive | majority per observation, thrive |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| experience dev (unseen planets) | 939,210 | **0.3817** | 0.3524 | 0.3423 | 0.3815 | 312,122 | **0.6392** | 0.5963 |
| experience sealed (unseen planets) | 719,234 | **0.3789** | 0.3529 | 0.3420 | 0.3786 | 235,320 | **0.6295** | 0.5939 |
| elder dev (unseen lives, seen planets, ticks <= 1000) | 78,529 | **0.3535** | 0.3263 | 0.3182 | 0.3536 | 23,158 | **0.6189** | 0.5790 |
| elder sealed | 78,224 | **0.3514** | 0.3223 | 0.3135 | 0.3506 | 24,030 | **0.6031** | 0.5604 |

77-79% of experience dev and sealed observations (74-75% for elder) occur verbatim in train.

**What this shows.**

- **No action learned beyond `eat`.** The model beats the per-observation majority by 2.6-3.0
  points (3.6-4.3 on thrive lines). But it is **equal to "always eat"** to within 0.1 point
  on every split.
- **Its argmax is almost always `eat`.** On experience dev it chose `eat` on 929,130 of
  939,210 lines. The rest were `rest` 6,551, `collect` 990 and the four signals 2,513
  together.
- **Why `eat` wins.** The behaviour in the data is about 38% `eat`, while the other 25
  actions each get about 2-4%: they are mostly the learners' epsilon exploration (floor
  0.1). With that spread, `eat` is the most likely action in nearly every state.
- **Elder.** The elder rows are lower only because the elder planets' early ticks hold less
  `eat` (35% in elder dev). Having seen the planet's own early history did not make the
  model better than "always eat" there either.
- **Verdict.** As a predictor of the exact next action, the model has learned the action
  distribution's mode and nothing more.

**Probe of the conditional distribution** (20,000 experience dev lines; script and result
in scratchpad `visitors/probe.py` and `visitors/model/probe.json`):

- **Which states it separates.**
  - P(`eat`) depends on the state and on the asked outcome: P(`eat` | food in reach, want
    thrive) = 0.68, against 0.21 with no food in reach. In the data P(`eat`) is 0.48 and
    0.17.
  - Under want suffer the same two values are 0.18 and 0.14.
- **Why the argmax is still `eat`.**
  - Under want thrive the argmax is `eat` in 100% of lines, with or without food in reach,
    because no other single action gets more mass.
  - The argmax differs between want thrive and want suffer on 1.4% of lines.
- **Alternative rule, not built.** The rule argmax log P(a | thrive) - log P(a | suffer)
  picks `eat` in 99.8% of the food-in-reach lines. In the no-food lines it picks `eat` 78%,
  then `move_sw` 16% and `move_s` 5%. The bias toward the south may be an artifact. This is
  a measurement of the model, not a decision rule the visitor uses. The visitor runtime
  decides that, and the paired arms measure it.

## Export

`python -m haishool.life8.experience export` writes float16 weights with the keys
`model_state`, `gpt_config` and `itos`. `student.load` reads it.

- **Files.** The export is at
  `scratchpad/visitors/model/visitor-6x384-fp16.pt`: 21,518,437 bytes, sha256
  `052729506c926e51d75c3b4d4b86481e25322e2b5fdea542f5f3506553f84116`, copied back and
  hash-checked. The largest difference from the float32 weights is 0.00092.
- **Manifest.** `visitor-6x384-fp16.json` holds the config (6 layers, width 384, 8 heads,
  ctx 128, vocab 105), the outcome cuts and the planet groups. On adler the float32
  checkpoint is `~/haishool-visitor/runs/visitor/student.pt`.
- **Copied with it.** `eval.json`, `probe.json`, `training-report.json`, `run-visitor.sh`
  and its log.
- **How to ask the model.** Ask `experience.prompt_for(experience.features_of(observation),
  "thrive")`, with the prompt preceded by `<eos>` as in `student.generate`. Read the next
  token, restricted to `experience.ACTION_WORDS`.

## Visitor runtime and paired arms (summary)

`haishool/life8/visitor.py` drives visitors without any engine change. `Visitors.decide` draws
every organism's decision itself, in `World.step`'s order and with the world RNG, and passes all
of them as overrides. A visitor's own native draw is made and discarded, so the RNG stream is the
untouched twin's, and a policy that returns the native choice gives a bit-identical twin
(tested). `TorchPolicy` encodes `<eos>` + `prompt_for(features_of(observation), "thrive")`, reads
the next-token log-probabilities and plays the highest-scoring action word that the body can do
at that tick (`allowed_actions(..., "body")`), else `rest` (a fallback).
`haishool/life8/visitor_arms.py` runs the arms on copies of r8a plan entries through
`search.build_world` (same seed, config and founders): `baseline` (the untouched twin), `one`,
`tenth` (round(n/10) = 2 of 24 founders), `all` (every founder) from tick 0, and `elder` (one
living individual drawn at tick 1,000). `report` pairs each arm with its baseline per planet;
`compare` subtracts a control folder's per-planet arm effects.

## Results: the trained model as visitor (2,000 ticks)

### Setup

- **Machine.** This PC (32 logical cores), with CPU torch 2.14.1+cpu installed for this run and
  `OMP_NUM_THREADS=1` per worker. The r8a search on the cluster was not touched.
- **Model.** `scratchpad/visitors/model/visitor-6x384-fp16.pt` (sha256 `0527295...6f84116`) loads
  with `student.load` in 0.11 s (cast to float32, 10.71 M parameters). All 26 action words are in
  the vocabulary and no prompt word was unknown. Real prompts are about 62 words.
- **Decision time on CPU** (median, one process): 12 ms for one prompt with 1 thread, 4.7 ms
  with 4 threads. With 1 thread, a padded batch of 4 or more prompts costs about 33 ms per
  prompt. `torch.set_flush_denormal(True)` brings that back to 10 ms up to batch 8, so the cause
  is most likely denormals in the padded positions. This was measured, not changed. In the arm
  runs the model added about 17 ms per visitor decision (`all` arm: 227 s per run against
  109 s for the baseline).
- **Planets.** The 40 TRAVELLER planets (`--group traveller`, never in the model's data; the
  runner checks the export manifest) ran with arms baseline, one, tenth and all. The 40 ELDER
  planets (`--group elder`) ran with baseline and elder at tick 1,000. Configs are the plan's
  own: the living preset with the search's predator variations (predators, predator_damage
  and predator_visibility differ from `config.LIVING` on most planets), with asexual
  automatic reproduction.
- **The elder model is the same single model.** It was trained on the 320 train planets plus
  the elder planets' first 1,000 ticks. It is not one model per planet.
- **Commands.** These two ran at the same time, with 0 errors:
  - `python -m haishool.life8.visitor_arms run --out scratchpad/visitors/model-traveller --group traveller --arms baseline,one,tenth,all --policy .../visitor-6x384-fp16.pt --ticks 2000 --workers 20`: 160 runs in 1,312 s wall.
  - `... run --out scratchpad/visitors/model-elder --group elder --arms baseline,elder --policy fake:rest --elder-policy .../visitor-6x384-fp16.pt --elder-at 1000 --ticks 2000 --workers 10`: 80 runs in 1,165 s wall.
- **Random control at the same length.** The same planets, arms and ticks were run with
  `fake:random` visitors (uniform over the body's allowed actions), in
  `scratchpad/visitors/random2000-traveller` and `random2000-elder`.
  - Their baseline rows were copied from the model folders, because the untouched twin does
    not depend on the policy. Only the 160 visitor runs were simulated (623 s and 551 s wall).
  - `compare` gives model minus random per planet (`model-*/compare-random2000-*.json`).

### What the visitor did

**Every one of the 312,790 model decisions was `eat`**: one 10,827, tenth 20,817, all
270,988 and elder 10,158, with 0 fallbacks. `eat` is always in the allowed set and was always
the top allowed word.

- **The offline result holds.** Offline the argmax was `eat` on 99% of dev lines, and the
  worlds show the same.
- **It is an always-eat rule.** An always-`eat` fake policy would give bit-identical twins,
  because the visitors' actions are the only difference between the twins.
- **What it never does.** It never moves, calls, teaches or collects.
- **It still has offspring.** Reproduction is automatic and asexual in these configs.
- **Overlap with the body's own choice.** 35-41% of its decisions matched what the body's own
  controller drew. Natives eat about that often.

### Paired statistics over planets

How to read the table:

- Differences are arm minus the untouched twin of the same planet, over the visitors' window
  (from placement to tick 2,000).
- Each cell is mean +- SE over planets, then (positive/negative planets), then the two-sided
  sign test p.
- "Visitor - native" compares the visitor with the native that had the same id in the twin.
- A p printed as 0 in the JSON is < 1e-6.

| measure | one (40) | tenth (40) | all (40) | elder (39-40) |
|---|---|---|---|---|
| visitor - native survival, ticks | -43.6 +- 29.3 (17/22, p .52) | -32.7 +- 24.8 (19/20, p 1) | **-32.3 +- 7.6 (11/29, p .006)** | **+72.3 +- 22.9 (20/5, p .004)** |
| visitor - native offspring | +0.28 +- 0.47 (20/12, p .22) | +0.04 +- 0.36 (19/16, p .74) | +0.01 +- 0.11 (19/21, p .87) | **+1.44 +- 0.37 (17/2, p .0007)** |
| visitor - native demonstrations | -2.1 +- 0.7 (10/24, p .024) | -2.9 +- 0.6 (6/27, p .0003) | -3.5 +- 0.2 (0/40) | -0.4 +- 0.6 (9/10, p 1) |
| visitor - native calls | -8.7 (0/17) | -9.3 (0/21) | -10.6 (0/33) | -4.2 (0/9) |
| population mean | +2.1 +- 1.6 (23/17, p .43) | -0.8 +- 2.0 (19/21, p .87) | -3.5 +- 2.1 (14/26, p .081) | -3.0 +- 1.4 (16/23, p .34) |
| population at the end | +1.7 +- 3.9 (p .18) | -0.5 +- 4.2 (p .73) | -1.2 +- 4.0 (p .86) | -2.6 +- 1.8 (p .60) |
| food energy per agent-tick | -0.001 +- 0.003 (p .27) | +0.003 +- 0.004 (p .87) | -0.011 +- 0.004 (14/26, p .081) | +0.011 +- 0.008 (p .34) |
| extinctions (arm / baseline) | 6 / 7 | 9 / 7 | 13 / 7 | 8 / 8 (1 before placement) |
| call adoption (moved, persists) | no visitor calls: undefined (n 0) | n 0 | n 0 | n 0 |

Slot values behind the table (one / tenth / all / elder):

- **Survival.** Model visitors lived 271 / 260 / 282 / 260 ticks; the native slot lived
  314 / 293 / 315 / 188.
- **Offspring.** Visitors had 2.43 / 2.11 / 2.32 / 2.46; the native slot had
  2.15 / 2.08 / 2.31 / 1.03.
- **Deaths, all arm.** Founder visitors died of starvation 583, predation 204, age 172 and
  injury 1. The natives died of starvation 569, predation 225, age 163 and injury 3.
- **Deaths, elder arm.** Visitors died of age 23, predation 10 and starvation 6. The native slot
  died of starvation 18, predation 11 and age 10.
- **The elder individual** was 3-630 ticks old at placement (median 192), and 8 were not yet
  mature.

**Model minus random visitor** (same planets, arms and ticks; `compare`):

| measure | one | tenth | all | elder |
|---|---|---|---|---|
| visitor - native survival | +130 +- 28 (33/6, p 1e-5) | +119 +- 20 (38/2) | +133 +- 8 (40/0) | +194 +- 34 (32/3) |
| visitor - native offspring | +2.2 +- 0.5 (28/0) | +1.8 +- 0.3 (33/2) | +2.0 +- 0.1 (40/0) | +2.3 +- 0.4 (22/0) |
| population mean | -0.7 +- 1.7 (p .64) | -2.7 +- 1.8 (p .081) | **+12.6 +- 2.6 (35/5, p 1e-6)** | -0.1 +- 1.1 (p 1) |
| food energy per agent-tick | -0.000 (p .081) | +0.005 (p .15) | +0.077 +- 0.015 (28/12, p .017) | +0.006 (p .20) |

The random visitor itself, at 2,000 ticks (one / tenth / all / elder):

- **Against the native slot.** Survival was -174 / -152 / -165 / -122 ticks and offspring
  -1.9 / -1.8 / -2.0 / -0.9, all p < 1e-4.
- **World.** In the all arm the population mean was -16.2 +- 2.6 (5/35).
- **Extinctions.** 7 / 8 / 21 of 40 planets (baseline 7); elder 9 (baseline 8).

### Answers

- **Do visitors live longer and have more offspring than the natives they replace?**
  - TRAVELLER: no. A founder that only eats lives about 33-44 ticks less than the native in its
    slot. This is significant only with the 960 visitors of the all arm (p .006). It has the
    same number of offspring (all differences within +-0.3, p >= .2). It is far better than a
    random visitor: +119 to +133 ticks and +1.8 to +2.2 offspring.
  - ELDER: yes. Placed at tick 1,000, the always-eat body lived 72 ticks longer (20/5 planets,
    p .004) and had 1.44 more offspring (17/2, p .0007) than the same individual under its own
    controller, mostly by not starving (6 against 18 starvation deaths).
  - The elder model behaves exactly like the traveller model (only `eat`). So the elder
    advantage is an effect of "stay and eat" late in a planet's history, **not** of having seen
    the planet's early ticks.
- **Do worlds with visitors do better or worse?**
  - With one or two visitors: no measurable difference. Population, food and extinctions all
    have p >= .18.
  - One visitor perturbs the whole world chaotically: a single founder shifts a planet's
    population mean by anywhere from -24 to +31. The random control shows the same spread.
  - With every founder a visitor, the world is slightly worse than its twin: population mean
    -3.5 (p .081), food per agent-tick -0.011 (p .081), and 13 against 7 extinctions. It is
    still much better than with random founders (+12.6 population).
  - Elder: no difference (p >= .34).
- **Do natives imitate visitors?**
  - Yes, about as often as they imitate the native slot. Per 100 visitor life-ticks there were
    1.24 / 0.93 / 0.72 / 1.36 successful imitations, against 0.88 / 0.85 / 0.90 / 1.21 for the
    native slot.
  - The copied action was `eat` every time (2,413 copies).
  - The visitor never teaches (0 against 106-2,625 native teach acts), which is why its
    "demonstrations" are lower.
  - The elder arm's +1.26 imitations per slot (19/2, p .0002) comes with its longer life. Per
    life-tick the rates are about equal.
- **Do visitors' calls spread and persist after their death?**
  - Null by construction. The model visitor never called (0 calls in 312,790 decisions), so it
    has no mapping to adopt and every adoption measure is undefined (n = 0).
  - For reference, the random visitor called 403 / 842 / 10,882 / 122 times. Its adoption
    "moved" was -0.030 (p .33), -0.036 (3/18, p .0015), +0.035 (20/9, p .061) and +0.045 (n 9).
    It never persisted (|persists| <= 0.016, p >= .29).
  - These are echo effects of a random caller, not adoption.

### Per planet: TRAVELLER

Column key:

- **d pop**: population mean, arm minus twin.
- **surv v-n** and **offs v-n**: visitor minus native slot, averaged over the arm's visitors.
- **ext**: the planet went extinct.

| planet | baseline end | one end | one d pop | one surv v-n | one offs v-n | tenth end | tenth d pop | tenth surv v-n | tenth offs v-n | all end | all d pop | all surv v-n | all offs v-n | random one surv v-n | random tenth surv v-n | random all surv v-n |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| synthetic-1735 | 2000 | 2000 | -6.1 | +0 | +4.0 | 2000 | +3.0 | +35 | +2.5 | 2000 | -14.6 | -59 | -0.6 | -516 | -272 | -190 |
| synthetic-185 | 679 ext | 2000 | +14.7 | +1 | +1.0 | 1489 ext | +3.1 | -24 | +1.0 | 2000 | +6.5 | -36 | -0.4 | -14 | -30 | -162 |
| synthetic-2399 | 2000 | 2000 | +15.9 | -7 | +1.0 | 2000 | +18.9 | +57 | +0.0 | 2000 | +17.0 | +30 | +0.3 | -52 | -27 | -192 |
| synthetic-2597 | 973 ext | 2000 | +16.0 | +1 | +0.0 | 811 ext | -1.3 | +42 | +0.5 | 964 ext | -2.7 | -46 | +0.4 | +1 | +34 | -118 |
| synthetic-3775 | 2000 | 2000 | +0.8 | -25 | +1.0 | 2000 | +3.6 | +124 | +2.0 | 2000 | +7.4 | -1 | +1.0 | -76 | -44 | -202 |
| synthetic-3993 | 2000 | 2000 | -2.4 | -126 | +1.0 | 2000 | -16.2 | +110 | +5.5 | 2000 | -8.5 | -86 | -0.6 | -73 | -272 | -210 |
| synthetic-4111 | 2000 | 2000 | +7.1 | -514 | -5.0 | 2000 | +6.1 | -288 | -2.0 | 2000 | -9.2 | -73 | -0.8 | -584 | -333 | -236 |
| synthetic-791 | 2000 | 2000 | -8.2 | -146 | +0.0 | 2000 | -14.9 | -85 | -1.5 | 1591 ext | -24.9 | -44 | +0.2 | -190 | -150 | -154 |
| synthetic-851 | 2000 | 2000 | +15.9 | -98 | -1.0 | 2000 | -5.0 | -30 | -0.5 | 2000 | -8.5 | -54 | -0.3 | -68 | -65 | -135 |
| synthetic-911 | 2000 | 2000 | +1.0 | -161 | -5.0 | 2000 | -4.5 | -140 | -4.5 | 2000 | -9.5 | -87 | -1.8 | -241 | -221 | -207 |
| world7-10081 | 1992 ext | 2000 | +6.6 | -109 | +1.0 | 1961 ext | -9.8 | -236 | +0.0 | 2000 | +17.9 | -38 | +0.7 | -159 | -254 | -200 |
| world7-10632 | 2000 | 1798 ext | -7.5 | +85 | +3.0 | 2000 | -6.2 | -29 | -0.5 | 1308 ext | -16.3 | -72 | -0.1 | -103 | -59 | -134 |
| world7-11163 | 2000 | 2000 | -0.0 | -103 | -3.0 | 2000 | +0.4 | +44 | +0.0 | 2000 | -0.6 | +69 | +0.5 | -84 | -164 | -128 |
| world7-11439 | 2000 | 2000 | -3.0 | +100 | +1.0 | 2000 | +5.6 | -42 | +0.5 | 2000 | -6.6 | -18 | +0.4 | -424 | -338 | -204 |
| world7-11543 | 2000 | 2000 | -1.5 | -131 | -3.0 | 2000 | +1.1 | -112 | -4.0 | 2000 | +8.6 | -23 | +0.0 | -145 | -292 | -142 |
| world7-1191 | 2000 | 2000 | +12.6 | +111 | +6.0 | 2000 | -4.8 | +54 | +0.5 | 2000 | -18.3 | -120 | -0.2 | -304 | -130 | -189 |
| world7-12095 | 2000 | 2000 | +7.6 | -409 | -6.0 | 2000 | +7.2 | -306 | -4.0 | 2000 | +10.6 | +0 | -0.2 | -492 | -348 | -203 |
| world7-1698 | 2000 | 2000 | +10.9 | +15 | +0.0 | 1577 ext | -20.5 | +0 | +0.0 | 1765 ext | -16.2 | -64 | +0.6 | +1 | -48 | -217 |
| world7-1906 | 1423 ext | 903 ext | -7.2 | -87 | +1.0 | 2000 | +24.3 | -138 | -1.0 | 1009 ext | -0.2 | -69 | +0.4 | -206 | -173 | -166 |
| world7-2140 | 895 ext | 1365 ext | +4.2 | +260 | +1.0 | 2000 | +8.0 | +80 | +0.0 | 696 ext | -0.4 | -1 | +0.1 | +27 | +34 | -113 |
| world7-2488 | 2000 | 2000 | -0.3 | -6 | +1.0 | 1650 ext | -8.9 | -9 | +0.5 | 1011 ext | -20.0 | +23 | +0.5 | -7 | +10 | -96 |
| world7-2593 | 2000 | 2000 | +3.0 | -377 | -3.0 | 2000 | -0.9 | -336 | -4.0 | 2000 | +7.6 | -11 | -0.5 | -352 | -396 | -142 |
| world7-2862 | 2000 | 2000 | +6.6 | +128 | +4.0 | 2000 | -8.3 | +142 | +3.5 | 2000 | -8.0 | -26 | +0.2 | -374 | -362 | -213 |
| world7-3155 | 2000 | 2000 | +2.1 | +85 | +1.0 | 2000 | +4.1 | +340 | +4.0 | 2000 | +12.5 | +42 | +1.0 | -73 | +2 | -203 |
| world7-3198 | 2000 | 1211 ext | -6.5 | +397 | +7.0 | 1292 ext | -4.9 | +210 | +3.0 | 775 ext | -12.4 | -43 | -0.4 | +7 | +46 | -99 |
| world7-3552 | 1735 ext | 2000 | +7.5 | -393 | -3.0 | 1358 ext | -0.0 | -174 | -1.5 | 1304 ext | -4.1 | -24 | -0.7 | -460 | -260 | -182 |
| world7-356 | 2000 | 1648 ext | -14.3 | -144 | -5.0 | 2000 | +5.9 | +20 | -1.0 | 1533 ext | -12.9 | -68 | -0.7 | -214 | -31 | -166 |
| world7-423 | 1129 ext | 962 ext | +1.5 | -78 | +1.0 | 890 ext | -1.5 | +4 | +0.5 | 1604 ext | +7.5 | +6 | +1.2 | -114 | -74 | -92 |
| world7-4627 | 2000 | 2000 | +12.7 | +74 | +0.0 | 2000 | +2.8 | +6 | +0.5 | 1328 ext | -9.4 | -33 | -0.4 | -17 | +20 | -111 |
| world7-4808 | 2000 | 2000 | -8.3 | +194 | +6.0 | 2000 | -7.2 | +110 | +4.0 | 2000 | -3.6 | -52 | -0.1 | -372 | -292 | -193 |
| world7-6036 | 2000 | 2000 | +30.7 | -329 | +0.0 | 2000 | +26.3 | -228 | -1.0 | 2000 | +32.4 | +14 | +0.9 | -359 | -260 | -155 |
| world7-6143 | 2000 | 2000 | -3.2 | +32 | +0.0 | 2000 | -4.9 | +36 | +1.5 | 2000 | -5.9 | +37 | -0.1 | -7 | -48 | -132 |
| world7-6439 | 2000 | 2000 | -3.5 | +231 | +3.0 | 2000 | -25.8 | +92 | +1.5 | 2000 | -17.4 | -43 | -1.0 | -99 | -70 | -158 |
| world7-7102 | 2000 | 2000 | -19.2 | -27 | +2.0 | 2000 | -18.4 | -230 | -3.0 | 2000 | -14.4 | -99 | -0.4 | -194 | -267 | -223 |
| world7-7672 | 2000 | 2000 | +4.4 | -181 | +0.0 | 2000 | -2.4 | -444 | -2.5 | 2000 | +3.2 | -161 | -1.5 | -487 | -468 | -252 |
| world7-7958 | 2000 | 2000 | +6.5 | +35 | +0.0 | 2000 | +15.1 | -32 | -1.0 | 2000 | +3.0 | -56 | -0.3 | +52 | -152 | -184 |
| world7-8348 | 2000 | 2000 | +12.9 | -50 | -1.0 | 2000 | +18.7 | +4 | +1.0 | 2000 | -10.7 | -67 | -0.1 | +19 | -141 | -176 |
| world7-8510 | 2000 | 2000 | +1.9 | -100 | -2.0 | 2000 | +3.9 | -12 | -0.5 | 1561 ext | +1.7 | +35 | +1.5 | -151 | -70 | -96 |
| world7-9006 | 2000 | 2000 | -2.2 | +20 | -1.0 | 2000 | +10.4 | +101 | +1.0 | 2000 | +11.8 | +26 | +1.1 | -39 | -32 | -197 |
| world7-9996 | 2000 | 2000 | -23.8 | +88 | +3.0 | 1852 ext | -34.7 | -24 | +0.5 | 2000 | -33.4 | +5 | +0.5 | -17 | -84 | -29 |

### Per planet: ELDER (model visitor placed at tick 1,000)

| planet | baseline end | elder end | elder d pop | elder surv v-n | elder offs v-n | random elder surv v-n |
|---|---|---|---|---|---|---|
| synthetic-1771 | 2000 | 1758 ext | -5.5 | +209 | +2.0 | -95 |
| synthetic-1847 | 2000 | 2000 | +8.2 | +416 | +4.0 | -71 |
| synthetic-1965 | 1664 ext | 2000 | +7.6 | +0 | +0.0 | +0 |
| synthetic-241 | 1672 ext | 1569 ext | -0.6 | +7 | +0.0 | -5 |
| synthetic-2675 | 2000 | 2000 | -3.6 | +0 | +0.0 | +2 |
| synthetic-2725 | 2000 | 2000 | -0.4 | +90 | +3.0 | -420 |
| synthetic-2921 | 2000 | 2000 | -6.1 | +81 | +0.0 | -9 |
| synthetic-4249 | 2000 | 2000 | -3.7 | -35 | +0.0 | -35 |
| synthetic-4275 | 1306 ext | 1339 ext | +0.3 | +0 | +0.0 | +1 |
| synthetic-997 | 2000 | 2000 | -0.6 | -63 | +0.0 | -108 |
| world7-10536 | 2000 | 2000 | +7.8 | +0 | +1.0 | -500 |
| world7-10753 | 2000 | 2000 | -7.0 | +2 | +0.0 | -4 |
| world7-10917 | 2000 | 2000 | -4.8 | +0 | +0.0 | +0 |
| world7-1119 | 2000 | 2000 | +2.8 | +0 | +0.0 | +0 |
| world7-1200 | 2000 | 2000 | -3.2 | +154 | +5.0 | +76 |
| world7-213 | 2000 | 2000 | -7.8 | +291 | +3.0 | -3 |
| world7-3254 | 1318 ext | 1484 ext | +0.7 | +153 | +0.0 | -254 |
| world7-4104 | 2000 | 2000 | +5.0 | +98 | +0.0 | -20 |
| world7-4162 | 2000 | 2000 | -31.6 | +0 | +0.0 | -12 |
| world7-4336 | 2000 | 2000 | +5.4 | -17 | +0.0 | -7 |
| world7-4396 | 2000 | 2000 | -10.7 | +0 | +0.0 | -258 |
| world7-5109 | 2000 | 2000 | -5.1 | +10 | +0.0 | -22 |
| world7-5273 | 2000 | 2000 | -4.1 | +0 | -2.0 | -179 |
| world7-5351 | 1206 ext | 1652 ext | +0.4 | +0 | +0.0 | -1 |
| world7-5762 | 2000 | 2000 | +5.4 | +18 | +0.0 | -1 |
| world7-5890 | 2000 | 2000 | +3.2 | +1 | +0.0 | -10 |
| world7-6037 | 1311 ext | 1789 ext | +1.6 | +57 | +1.0 | +45 |
| world7-6577 | 2000 | 2000 | -13.7 | +0 | +1.0 | -12 |
| world7-7373 | 2000 | 2000 | +1.3 | +30 | +3.0 | -488 |
| world7-7682 | 2000 | 2000 | -29.2 | +175 | +7.0 | -222 |
| world7-7849 | 880 ext | 880 ext | +0.0 | - | - | - |
| world7-7862 | 1786 ext | 2000 | +5.3 | -7 | +0.0 | -113 |
| world7-834 | 2000 | 2000 | -6.0 | +469 | +7.0 | -23 |
| world7-8613 | 2000 | 2000 | -8.6 | +0 | +3.0 | -661 |
| world7-8769 | 2000 | 2000 | -2.8 | +215 | +5.0 | -321 |
| world7-8854 | 2000 | 2000 | +1.0 | +503 | +6.0 | -26 |
| world7-8926 | 2000 | 2000 | -2.5 | -168 | -2.0 | -197 |
| world7-9102 | 2000 | 1729 ext | -1.5 | +129 | +4.0 | -107 |
| world7-9410 | 2000 | 2000 | +1.3 | +0 | +3.0 | -471 |
| world7-9961 | 2000 | 2000 | -19.2 | +0 | +2.0 | -211 |

world7-7849 went extinct at tick 880, before placement, so it has no elder.

### 3D pages (one TRAVELLER tenth twin and its baseline)

The planet is **world7-10081**, the first in the traveller group order
(`select(group="traveller")`, with world7 and synthetic alternating). It was chosen by that
rule, not for its result. Both runs were recorded with `visitor.write_archive(...,
sample_every=1)` for 1,200 ticks, giving 1,201 snapshots each; the archives are 199 MB and
115 MB. Both pages were built with `view3d.build(..., every=1)`.

- **Pages.**
  - `scratchpad/visitors/pages/world7-10081-tenth.html` (2.0 MB): visitors 13 and 21 (flagged
    `traveller` in the page), 332 decisions, all `eat`, 0 fallbacks. Population at tick 1,200
    is 6.
  - `scratchpad/visitors/pages/world7-10081-baseline.html` (2.9 MB): the untouched twin.
    Population at tick 1,200 is 19.
- **Same worlds as the statistics.** Both populations equal the arm runs' trackers at tick
  1,200 (6 and 19). Over 2,000 ticks this tenth twin went extinct at tick 1,961 and its
  baseline at tick 1,992. The visitors lived 236 ticks less than their native slots.
- **Predators are shown only at their strikes.** view3d's replay uses
  `World.create(seed, config)`, but plan planets are built with their founders
  (`bridge.create_world`). The replay therefore did not match even the baseline (individual 1
  differs at tick 1).
- **Not checked in a browser.** The browser pane here could not open local files.

### Limits and open points

- **Always eat.** The model is an always-eat rule in these worlds. Every effect above is the
  effect of "never move, always try to eat", compared with the natives' learned controllers.
  A visitor that does anything else would need a different decision rule (the probe's
  thrive/suffer ratio, or sampling) or data without exploration noise.
- **No adoption test.** Call adoption cannot be tested with this model, because it never
  calls.
- **The elder advantage is not knowledge of the planet.** The same behaviour gave the traveller
  founders no advantage. Why "stay and eat" helps an individual alive at tick 1,000 but not a
  founder is not resolved here. The elder slot's natives starve more: 18 of 39.
- **Arm size.** The tenth arm is 2 of 24 founders (round).
- **Censoring.** Survival is censored at tick 2,000. In these runs every visitor and every
  native slot died before the end.

## Open points (data and model)

- **Effectively always `eat`.** The model's greedy policy is effectively "always eat".
  - A visitor that takes the argmax will try to eat where no food is in reach. That action
    fails and pays the action cost.
  - Whether this is better or worse than an ordinary inhabitant is for the paired arms to
    measure. A visitor's survival against its twin is the test, not this offline accuracy.
    Measured since: see "Results: the trained model as visitor" above.
- **Possible ways forward (not done).**
  - Rules that use the state dependence the probe shows, such as the thrive-over-suffer
    ratio, or sampling instead of argmax.
  - Data without the exploration noise, for example only decisions the learner took
    greedily. Those are not marked in the step records today.
- **The elder variant is one model.** It is the same model as the traveller, trained on
  other planets and on the elder planets' first 1,000 ticks. It is not 40 models trained
  only on their own planet. Offline it did not beat "always eat" on unseen lives of its own
  planets.
- **Extinctions.** 72 of the 440 simulated planets went extinct within their run. The
  traveller planets were not simulated here, so their fate is unknown.
