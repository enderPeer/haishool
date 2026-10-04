# Update 9 plan: 3D worlds on the GPUs, brains that grow, listeners that profit

Prepared 2 October 2026 while r8a's long continuations (`life8-long` on knecht24, specht32
and falke64) and the world7-781 long-tick test were running. **Nothing running was touched.**
The hosts were only inspected read-only (`nvidia-smi`, `lspci`, `tmux ls`). life8 was not
edited, so its `checkpoint.source_identity()` and every running checkpoint stay valid.

What exists now: a new package, `haishool/life9/`. It is a working prototype engine with tests and a
3D replay page. This document is the audit that motivated it and the plan for running it.

## Version 2 (3 October 2026): the planet engine `haishool/life9/planet`

The owner asked for update 9 to be a real GPU/CPU world close to the real one: a spherical planet with gravity,
layers, chemistry, craftable materials and technology, and a richer environment. All of it is to be derived
from the start of the simulation chain, with bigger brains. The design contract and the as-built record are in
[PLANET-SPEC.md](PLANET-SPEC.md). In short:

| Part | What it does | Where it comes from |
|---|---|---|
| star | mass, luminosity and age; T_eff (4,320 K for 781); the photosynthetic share of the light (Planck integral) | world7 chain, then derived |
| planet | radius, iron core, mantle and crust, g, escape speed (781: 5,674 km, g 7.54 m/s²) | chain element mix (Fe/Mg/Si/O), then derived. The mass-radius rule is new_rule |
| air | N2, O2, CO2, Ar and H2O; pressure, scale height, speed of sound (781: 0.89 bar, 339 m/s) | O2 from the chain's bodies era. The other gases are new_rule plus the greenhouse calibration |
| orbit and rotation | year from Kepler's law; tidal lock decided by the Kasting 1993 radius | 781 is locked, with a permanent day side and night side |
| climate | energy balance on the sphere: grey greenhouse, CO2 forcing, lapse rate, ice, clouds, water cycle (evaporation, rain, soil, runoff, snow), carbon-silicate weathering | reference physics, calibrated so a planet without life reproduces the chain's 288 K |
| biosphere | plant, wood, litter and nutrients; production limited by light, temperature, water, CO2 and nitrogen; decomposition; ocean plankton; O2 and CO2 exchange | reference ecology |
| bodies | Kleiber metabolism, cost of transport, climbing, swimming drag, keeping warm and cool, water turnover, oxygen limits, allometric life history; corpses become meat, fat, bone and hide | reference allometry |
| brains | recurrent networks of up to **128 units** (48 at the start); size is an evolved gene that costs energy; 14 actions plus a call vector; within-life learning from the individual's own outcome | flat life9, enlarged |
| materials and crafting | 31 species with real hardness, toughness, density, melting point and heat data; collect, knap, haft, fire, heat, cook; smelting and firing as physical transformations (clay to ceramic, limestone to lime, malachite to copper, cassiterite to tin, iron bloom, bronze, steel, glass); fire needs O2 ≥ 15 % | reference data (CRC, NIST-JANAF, archaeometallurgy) |
| scale | a 10 km habitat globe carrying the planet's real gravity, air, light and chemistry; one tick is one day; 13,824 cells | the one compression, stated in the spec |

- **Tests:** 371 pass on Windows and on adler40 (Linux, RTX 4090), plus the GPU device tests on the 3060, the
  R9700 and the 4090.
- **Ledgers:** carbon, oxygen, water, energy, element and deposit ledgers close to about 1e-7 relative.
- **Cross-platform:** world7 differs in the last bit between Windows and Linux. The chain inputs are therefore
  cached once and shared by all hosts.

**Speed at G=48 with 1,024 slots per world** (`docs/life9/results/planet/bench-*.json`):

| Device | 2 worlds | 8 worlds | 16 worlds |
|---|---|---|---|
| RTX 4090 | 0.054 s/day | 0.084 | 0.117 (about 490 k world-days/h) |
| R9700 | 0.092 | 0.124 | 0.191 |
| RTX 3060 | 0.154 | 0.286 | 0.418 |
| adler40 CPU, 24 threads | 0.188 | | |

16 worlds take 5.4 GB on the GPU.

**Honest limits of version 2:**
- **Planet 781 cannot make fire.** Its O2 is about 3 % of the air, so smelting and ceramics are physically
  impossible there until the biosphere raises oxygen. The `earth8` control exists to show what crafting does
  where fire is possible.
- **Grazing is partly automatic.** Individuals graze in the background for the hours their chosen action leaves
  free.
- **`share` finds relatives by pedigree,** which is a supplied fact rather than a perception.
- **The climate is too even.** The equator-to-pole contrast is too small and runoff is low.
- **The habitat globe is a compression.**
- **Some reference values are cited from memory** and are flagged in the code.
- **The 1,024-slot cap binds.** Several worlds and all Earth controls reached it within a year. The GPUs have
  memory for 2,048-4,096 slots per world.

### First version-2 runs (started 3 October 2026, about 04:40 UTC; 20,000 days each; tmux session `life9-run`)

| Host / GPU | Run | Worlds |
|---|---|---|
| adler40 4090 | `chain16a` | 781, 85, 6, 7, 9, 10, 29, 35, 40, 54, 56, 63, 65, 66, 74, 76 |
| adler40 4090 | `earth8` | Earth reference control x 8 (`--source earth`) |
| falke64 R9700 #0 | `chain16b` | 77, 79, 88, 90, 100, 101, 102, 110, 111, 120, 152, 160, 167, 171, 172, 178 |
| falke64 R9700 #1 | `p781x16` | planet 781 x 16 independent histories |
| knecht24 3060 #0 | `chain9c` | 182, 197, 207, 213, 221, 223, 225, 230, 232 |
| knecht24 3060 #1 | `p85x8` | planet 85 x 8 |

- **Layout:** each host keeps the code in `~/life9-planet/code`, the results in `~/life9-planet/runs/<name>/`
  (summary.jsonl, state.pt every 2,000 days, view.html for world 0) and the log in `runs/<name>.log`.
- **Resume:** `bash ~/life9-planet/run9.sh NAME GPU_ENV --resume ../runs/NAME/state.pt --days N`.
- **Seeds:** all world7 seeds are among the 41 cached chain planets that reach the bodies era
  (`runs/life9/chain-cache`).
- **First year:** 5 of 32 distinct world7 planets died out. Several worlds, and every Earth control, hit the
  1,024 cap.
- **Free:** knecht24 GPU 2 is free.

### Results of the first version-2 runs (finished 3 October 2026, about 05:30-06:15 UTC)

20,000 days per world, 73 worlds. The per-world table is in `runs/life9-planet/analysis.json`, and each run's
`summary.jsonl`, `gates.json` and `view.html` are in `runs/life9-planet/<run>/`.

| Finding | Numbers |
|---|---|
| ledgers | every checked ledger closed over 55 years (worst 8.6e-6 relative, typically 1e-7 to 1e-8) |
| survival | 67 of 73 worlds alive at day 20,000. Extinct: 6, 40, 56 and 171 (all founders died of thirst within 50 days), 9 (starved by day 100), 152 (day 2,450) |
| the slot cap | almost every surviving world sat at the 1,024-slot cap for most of the run. The cap, not the ecology, set the population size |
| generations | 6-20 in 55 years (founders weigh 30 kg and mature at about 1.5 years) |
| brains | mean active units 48.0 at the start and 47.7-48.2 at the end in every world: **no evolution of brain size** |
| behaviour | the 13 non-forage actions stayed at about 5-6 % each, close to their starting share. Forage held its innate-bias share (18-41 %) and moved by at most 3 points. **Behaviour did not change over 55 years** |
| diet | 0.04 to 0.05-0.08 toward meat; meat eaten only 153-1,477 times per world |
| violence | 0.77 million successful strikes per world. Injury killed 35-5,001 per world |
| crafting on world7 planets | no fire and no transformation anywhere: O2 is 1.2-6.5 kPa, below the 15 % fire limit |
| crafting on the Earth controls | 339-409 fires per world, out of about 1 million make_fire tries. Clay to ceramic 3-9 kg; copper up to 1 kg; tin up to 1.6 kg; lime up to 0.45 kg; bloomery iron 0.05 kg (world 4). This is chance from random action choice, not a skill: nothing was repeated or improved |

**Bug found in the results (fixed, but these runs were made with it):** the biosphere's seed floor refilled every
lit land cell to 1 g C/m² of plants **every day**, whatever the temperature or water. Its own description says
"a year's seed rain". That is up to 365 g C/m² a year of food that no photosynthesis made. The carbon was booked
from the air, so every ledger still closed. About half of the surviving world7 planets have plant production at
or below zero, among them planet 85 in all 9 histories, and 76, 230, 120, 232 and others. Those populations
lived on this leak. Worlds 76 and 230 crashed only after they had grazed even the floor's growth away, and their
"shrinking" bodies were starvation, not evolution. Results on barren planets are invalid. The worlds with real
production (781, the Earth controls, and 88, 90, 100, 111, 167 and others) used little of it.

The fix (biosphere.py) refills seeds at a year's rate and only where a seedling could grow that day. The
regression test `test_seed_rain_is_not_a_daily_food_supply` fails on the old rule. The suite gives 372 passed.

### Corrected version-2 reruns (`*-fix`, finished 3 October 2026 about 08:00-08:45 UTC)

Same seeds and 20,000 days, with the seed floor reduced to a year's seed rain where seedlings can grow. Results are in
`runs/life9-planet/*-fix/` and `analysis-fix.json`.

**36 worlds have real plant production** (NPP > 1 g C/m²/yr): 781 (17 histories), the 8 Earth controls, and 7, 54,
88, 90, 100, 102, 111, 152, 160, 167, 172. Their results hardly changed:
- the population sits at the 1,024-slot cap
- brain units stay at 48
- behaviour stays at its starting distribution
- crafting on the Earth controls is chance: 339-408 fires, 3-10 kg of products

**31 worlds live on almost no production.** Even a year's seed rain, 1 g C/m²/yr across the whole land, still feeds
about 1,000 animals on a barren planet. That is about 6x what they need on planet 85. The supplied seed rain
therefore remains a food source on barren worlds, with starvation now far higher (planet 85: 489 to 9,659 deaths).
v3 removes it completely: seeds come only from plants that exist.

Collapsed or extinct with the fix: 76 (6 left), 230 (4), 223 (158), 101 (extinct at day 3,500). 6, 9, 40, 56 and
171 died out in the first 100 days, as before.

## Version 3 (3 October 2026): emergence only (`haishool/life9/v3`, contract [PLANET-V3-SPEC.md](PLANET-V3-SPEC.md))

Built, reviewed for physics and for emergence purity, fixed, and tested: 351 v3 tests pass, and version 2's 372
still do. What v3 contains:
- **Real-size planet:** water, CO2, N2 and Ar derived from the chain's disc temperature and the meteorite classes;
  the greenhouse results from these instead of being fitted.
- **Patches:** 16 true-scale 2 km patches per world, with periodic boundaries, plant seed dispersal, downhill
  water into ponds, and smell fields.
- **Bodies:** physical heat, water, fuel, damage and buoyancy. Division is a protein-synthesis rate driven by an
  output; there is no maturity age and no lifespan law.
- **Senses:** physical only, with no labels.
- **Action:** contact-only manipulation; fire can only come from rubbing tinder hot enough.
- **Brain:** nothing innate; the learning signal is evolved and starts at zero.
- **Controls:** a drift-only control, and purity tests that grep for anything pre-installed.

**First results (CPU and GPU agree within noise)**
- **Planet 781 freezes at once.** Its derived climate has a 267.6 K mean with 54 % ice. All 4 patches are at
  247-260 K, and 3 are on the night side. Every founder dies in the first hour. This is a valid "nothing happens"
  result.
- **The Earth reference:** 4 of 16 patches lose all founders in the first hour; the rest grow and hit the
  1,024-slot cap by day 2. Food was far from limiting: a 2 km patch could feed millions of 20 g bodies.
- **Fire is impossible on the model Earth.** Its chondritic nitrogen gives 2.05 bar of air, so O2 is only 10.4 %,
  below the 15 % fire limit.
- **Every founder is denser than water** and drowns in ponds within about 5 minutes; nothing in the body plan
  holds air.

**Speed:** 3.7 s per day on the RTX 4090, 4.1 on the R9700, 10 on the RTX 3060 and 22-24 on this PC's CPU, at 16
patches x 1,024 slots and 24 bouts per day. The engine is bound by kernel launches; more worlds per process
amortise that.

**Decisions open for the owner:**
- the patch scale against the slot cap
- a separate nitrogen degassing calibration for Earth
- an air-volume (lung) gene for buoyancy
- the locomotion force-cost constant (Kram & Taylor c = 0.2 J/N, cited from memory)
- adopting the five-rule experiment protocol

## 1. Where life8 limits emergent behaviour (audit)

### A. The listener problem: hearers have no reason and little means to respond

| # | Weakness | Evidence |
|---|---|---|
| A1 | **Information is not worth acting on.** A predator strikes the nearest individual, usually the caller. A bloom 8 units away is gone before a hearer arrives. | Oracle probe (perfect calls): hearers' gain over scrambled calls was -0.009 to +0.008 bits; predator strikes were lower in only 4/6 seeds (`docs/life8/README.md`, "Why: the oracle-sender probe"). |
| A2 | **Speakers are rare.** Voice is a binary organ. Founders get it with the planet's `talkers` share, and it flips at 0.005 per birth. Voiceless individuals can still choose `signal_*`, which pays the action cost and does nothing. | world7-781 at tick 7,500: **7 of 99 have a voice, 99 of 99 can hear.** Over the run: 74,500 `signal_*` choices, 1,905 calls actually sent (97 % wasted). See `world.py:612` (cost always paid) and `world.py:769` (voice check). |
| A3 | **The hearer sees one call, flattened.** Only the freshest message becomes `heard_symbol` (4 values), for up to `message_ttl`=8 ticks. It is stale, it hides multiple callers, and nothing sequential survives. | `learning.py:351` |
| A4 | **Direction meanings shift with food.** The egocentric frame rotates every direction bin by the *food's* bearing. "Call from the north-east" means a different weight depending on whether food is in view, so "move away from the call" must be learned twice, in inconsistent frames. | `learning.py:395-411` |
| A5 | **Tiny, rare credit.** Each TD error is split over about 30 active features. A heard symbol is present in few decisions, so its weights move very slowly within a life of about 200-600 decisions. | `learning.py:576-602`. world7-781: individuals alive at the end averaged 215 decisions. |
| A6 | **No innate route for listening.** The genome's innate features are bias, food, energy, health, neighbour, tool and heard_symbol. Danger, call bearing and kin are missing, so evolution cannot hard-wire "turn away from a call". | `genome.py:49-58` |
| A7 | **A flat open plane.** Everyone near a predator can see it, apart from distance. There is no vantage point, no blocked view and no reason why one individual knows what another cannot. | Section C |

### B. The brain capacity cap

| # | Weakness | Evidence |
|---|---|---|
| B1 | **The cap that binds is the model class, not `max_features`.** The learner is linear over one-hot bins plus 4 hand-listed conjunctions. It cannot represent an interaction nobody listed, for example "a call heard while food is behind me". | world7-781: weights per individual 43-133 (mean 82). The 256-feature LRU is **never reached**, so raising it changes nothing. |
| B2 | **No internal state.** Decisions are Markov in the current bins. The 32-entry memory is a replay buffer, not working memory. | `learning.py` (memory used only by `_replay`) |
| B3 | **Each life starts nearly blank.** It has innate biases over 25 features and nothing else. Lifetimes are about 600 ticks. | `genome.py`. heritable-genome results: innate P(eat) rose 0.04 → 0.12 in 15 generations. |
| B4 | **Exploration noise dominates behaviour.** An epsilon floor of 0.1, uniform over 26 actions, means most non-`eat` actions are noise. The visitor GPT trained on this data learned "always eat" and nothing more. | `docs/life8/visitors.md` |
| B5 | **Brain size is not a trait.** No individual can have more or less capacity, and capacity costs nothing, so it cannot evolve. | — |

### C. No third dimension

| # | Weakness |
|---|---|
| C1 | The world is a 32 x 24 flat torus. The 3D page draws heights "only to make things visible" (`docs/life8/viewer3d.md`). |
| C2 | So there is no **vantage** (high ground sees further), no **occlusion** (hills hide predators and food), no **terrain energetics** (climbing costs), and no **habitat structure** (fertile valleys, poor ridges). These are exactly the conditions that make one individual's knowledge valuable to another (A1, A7). |

### D. Compute

| # | Weakness |
|---|---|
| D1 | The engine is pure Python with per-agent dicts: 17-40 s per world per 1000 ticks. r8a needed about 4 h on 70 CPU cores for 4,440 short worlds. The GPUs were idle the whole time. |
| D2 | There are too few replicates per claim (critic finding F3 is still open). Paired branches cost 3 x reps x branch_ticks per kept world. |

## 2. Update 9: `haishool/life9` (prototype, implemented)

It is a separate engine in PyTorch. It is batched: `cfg.worlds` independent worlds advance in
one step, and every world is a paired replicate. It runs on the CPU (exact and tested), on
CUDA (adler40 and knecht24), and on ROCm (falke64 and specht32, once a ROCm torch is
installed there).

| Weakness | life9 mechanism | File |
|---|---|---|
| C1, C2, A7 | **Height-field terrain on a torus**, periodic, with relief 6 by default. z is the real ground. **Line of sight** is tested against the terrain. Sight grows with height (`height_sight_gain`). Hills fully block sight and halve sound (`sound_occlusion`). Climbing costs energy (`climb_cost`). Food grows mainly in valleys (fertility (1 - h/relief)^2). | `terrain.py`, `world.py` |
| A1 | **Ambush predators.** They are stealthy: prey see them only within `predator_detect` x sight. Their stamina is limited (`predator_stamina` 12 ticks, speed 1.1 against prey speeds up to 1.2 x gene), so prey that start fleeing early escape and late ones do not. A predator hunts the nearest prey *it* can see. That is usually someone in a valley who cannot see it, not the caller on the ridge. Warning others now has a physical value. Nothing about calls is coded. | `world.py::_predators` |
| A2 | **Voice is a continuous gene** (0-1.5) for every individual. It is inherited with mutation and scales loudness. Silence is free, and loudness costs `call_cost` x loudness. | `brain.py`, `world.py` |
| A3, A6 | **Calls are continuous vectors** (`vocal_dims` 4) with a loudness, not 4 symbols. Hearing is egocentric. Each of 8 sectors around the hearer's own heading gets the summed call vector, attenuated by distance and terrain. Several callers and their directions are all heard at once. | `world.py::sense` |
| A4 | **Body frame.** Every sense is relative to the individual's own heading, so a call's direction means the same thing whatever else is in view. | `world.py::_sectors` |
| B1, B2 | **A recurrent network per individual.** Inputs are 62 sensor values; there are up to `hidden` units and 8 outputs (turn, speed, eat, loudness, call vector). The hidden state carries what was heard or seen in earlier ticks. All brains run as one batched matrix multiply. | `brain.py` |
| B5 | **Brain size is an evolved, paid-for gene.** `k` active units (2..`hidden`) mutates by ±1, and every unit costs `neuron_cost` per tick. Capacity is selected for or against, not fixed. The ceiling `hidden` is a config value, and reaching it is visible in `mean_hidden`. | `brain.py`, `world.py` |
| B3, A5 | **Evolution of all weights plus lifetime plasticity.** The genome holds the full starting network, so evolution can hard-wire "turn from a call" (fixes A6). Within a life, recurrent weights change by reward-modulated Hebbian learning, driven by the individual's own energy and health change minus its running mean (life8's honest reward). Offspring start from the genome, never from learned weights (tested). | `brain.py::hebbian` |
| B4 | **No epsilon noise.** Behaviour is the network's output, so recorded experience is policy, not exploration. | — |
| D1, D2 | **Batched worlds.** A paired test uses W worlds as replicates in one process. | `world.py`, `probe.py` |
| A1 gate | **Listener gate** (`probe.py`). In the `oracle` arm, callers who see a predator emit a fixed vector. This is a supplied meaning, used for diagnosis only. The `oracle_deaf` arm makes the same calls and pays for them, but nobody hears. **Pass** requires two things: predation is lower with hearing in a significant majority of worlds (sign test p <= 0.05), and hearers that cannot see the predator move away from it more often than non-hearers. **No emergence run is read until this gate passes.** | `probe.py` |

Rules kept from life8: there is no dictionary, no meaning for any call vector, and no reward for calling,
listening or cooperation. Learning uses only the individual's own outcome. Nothing learned is inherited.
Energy ledgers are closed and checked; with float32 they close to about 0.002 energy per world over 400 ticks.
The same seed gives the same run, and resume is exact on CPU (tested). The config is recorded with every run.

### Measured so far (this PC, CPU only; torch 2.14.1+cpu)

- Tests: `python -m pytest tests/life9 -q` gives 10 passed in 5 s. They cover the terrain period, a ridge
  blocking sight symmetrically, same seed giving the same world, exact resume, the energy ledger, silent unused units, brain
  size costing energy, offspring that start from the genome, the deaf and oracle channels, arms sharing terrain and
  founders, and the sign test.
- Speed: 8 worlds x 128 slots gives **0.74 M world-ticks per hour on CPU**. That is already about one whole life8 host
  (0.66-1.83 M). GPU numbers are still to be measured (phase 1).
- Viability: 4 worlds x 96 slots for 400 ticks kept 37-58 alive, with 27-48 births and 6-15 predation deaths per
  world. Random founder brains survive long enough to evolve.
- Listener gate on CPU (16 worlds x 2,000 ticks): **failed** in both the default and a harsher ecology (section 5). It is a first look, not the gate run.

## 3. GPUs (inventory, read-only, 2 Oct 2026 20:21 UTC)

r8a has finished: no `life8-search` session is left on any host. All four hosts' **CPUs are fully busy** with
`life8-long`, the world7-781 long continuations: f0 to tick 100,000 on adler40, and futures f1-f13 to tick 102,500
(adler40 f1-f4, knecht24 f5-f9, falke64 f10-f11, specht32 f12-f13). They run under nice 10, started at 19:38 (f0)
and 19:58 UTC, and write one 5,000-tick segment every 8-13 minutes. If that rate holds, they end at about 22:40
(falke64, specht32) to 23:50 UTC (adler40, knecht24).

PyTorch is already installed in `~/homunculi/.venv` on every host (Python 3.12.14, torch 2.14.0). adler40 and
knecht24 have the CUDA 13.0 build, and falke64 and specht32 have the ROCm 7.2 build, so **no installation is
needed**. The system `python3` (3.14) has no torch.

| Host | GPUs | State at 20:21 UTC | Use in update 9 |
|---|---|---|---|
| knecht24 | 3 x RTX 3060 12 GB | 0 % busy; about 0.5 GB each held by idle `homunculi-serve` processes | **free**: one probe or sweep per GPU |
| falke64 | 2 x Radeon AI PRO R9700 32 GB | 3 % busy; 1.4 GB and 6.9 GB held by `homunculi serve` processes | **free** (about 25-31 GB each). Check ROCm on RDNA4 with `bench` first |
| adler40 | RTX 4090 24 GB, RTX 4080 16 GB | 4090: 0 %, 7.4 GB held by another local account's ComfyUI. 4080: 12.4 GB held by 11 homunculi processes. Disk 96 % full (36 GB free) | 4090 usable alongside ComfyUI (about 17 GB free). 4080 **not free**. Keep outputs small. |
| specht32 | RX 9070-class 16 GB, RX 9060 XT 16 GB | 13.2 and 14.2 GB held by `ggml-rpc-server` (llama.cpp RPC over Vulkan, session `ender-rpc-specht-amd-pair`) | **not free** |

A GPU process also needs about one CPU core. While `life8-long` runs, every life9 job takes a little from it.

Rendering: the 3D replay (`view.html`) runs on the *viewer's* GPU through WebGL. That costs nothing on the
cluster. For offline films, GPU Blender (Cycles OptiX on adler40, HIP on falke64) can render frames from the
same frame records. That is phase 6, and only after the science gates.

### Stopped on the user's instruction (2 Oct 2026, about 20:58-21:03 UTC)

- **`life8-long` stopped on all four hosts.** This was the world7-781 long continuations: `tmux kill-session -t
  life8-long`, then a check that no `analysis.longrun` process or worker was left. Saved state at the stop:
  - adler40: f0 7 segments, 2 rows; f1-f4 6 segments, 3 rows each
  - knecht24: f5-f9 4 segments, 1 row each
  - specht32: f12 5 segments, 2 rows; f13 4 segments, 1 row
  - falke64: f10-f11 7 segments, 4 rows each

  The runs are resumable from their last saved segment. Run `bash ~/life8-long/w781/run2.sh WORKERS FUTURE...` with
  the same futures on the same host: f0 and f1-f4 on adler40, f5-f9 on knecht24, f10-f11 on falke64, and f12-f13 on
  specht32. Each run may only resume on the platform it started on.
- **ComfyUI on adler40 stopped.** This is another local account's systemd user service `comfyui-4090.service`, stopped with
  `sudo systemctl --user -M <account>@ stop comfyui-4090.service` on the owner's instruction. Its unit file is
  `disabled` and it has `Restart=on-failure`, so it stays stopped until someone starts it again. The RTX 4090 went
  from 7.4 GB used to 1 MiB.
- **Not stopped**, because they are services rather than simulations:
  - the public chat apps on adler40 (`haishool-chat`, `haishool-hops`, `v5_app`, `visitor-chat`)
  - the homunculi model servers (11 processes holding 12.2 GB of adler40's RTX 4080; about 0.5 GB on each knecht24
    3060; a few GB on the falke64 R9700s)
  - specht32's `ggml-rpc-server` (llama.cpp RPC, 13-14 GB on each AMD card)
  - an archive copy from adler40 to falke64

## 4. Phases and gates

| Phase | What | Where | Gate before the next phase |
|---|---|---|---|
| 0 (done) | Engine, tests, viewer, CPU numbers | this PC | tests pass |
| 1 (done) | Speed grid on one GPU per host type, and a check that CPU and GPU give the same statistics (section 4a) | knecht24 GPU 0, falke64 GPU 0, adler40 4090 | **passed**: throughput recorded; CPU and GPU agree within noise |
| 2 | **Listener gate** at 64 worlds x 10-20 k ticks. If it fails, change the *ecology*, never the brains or the rewards: predator stealth or stamina, relief, food layout. | knecht24 | the gate passes in two fresh seeds |
| 3 | **Emergence runs without an oracle.** Arms: normal, masked, scrambled, deaf. Port the r8a measures to vector calls: sender information after clustering call vectors, with the within-sender null; receiver information; fitness cost of scrambling; lineage conventions. Thresholds are fixed beforehand and corrected for multiple comparisons, with worlds as replicates (closes F3). | all free GPUs | stated in advance in a results file before the runs |
| 4 | Bridge world7 planets: relief, predators, light, gravity into `Config9` (as `life8/bridge.py`) | CPU | — |
| 5 | Bring back life8's material and social layer: objects, tools, sharing, teaching. Then experience data for a new visitor model. Its lines are policy, not epsilon noise (B4). | — | phase 3 shows calls carrying information |
| 6 | Long recorded runs and rendered films | adler40 or falke64 | — |

## 4a. Phase 1 result: GPU speed (measured 2 Oct 2026, 20:24-20:45 UTC)

Each test was one process per GPU, with one CPU thread at nice 10 in `~/life9-bench`, using the hosts' existing
`~/homunculi/.venv` (torch 2.14.0: CUDA 13.0 on the NVIDIA cards, ROCm 7.2 on the R9700). `life8-long`
filled every CPU throughout and kept running. The script is `scripts/life9_bench_grid.py`. The raw results are in
`docs/life9/results/bench-<host>.json` and `check128-<host>.json`. Compute is dense over all slots, so the cost
does not depend on how many slots are alive. M = million world-ticks per hour.

| worlds x slots (brain units) | RTX 3060 (knecht24) | R9700 (falke64) | RTX 4090 (adler40) | this PC's CPU (8-32 threads) |
|---|---|---|---|---|
| 16 x 128 (32) | 46.5 ms/tick, 1.2 M | 15.1 ms, 3.8 M | 32.3 ms, 1.8 M | |
| 64 x 128 (32) | 47.1 ms, 4.9 M | 16.6 ms, 13.9 M | 33.4 ms, 6.9 M | |
| 64 x 256 (32) | 85.8 ms, 2.7 M | 43.3 ms, 5.3 M | 32.7 ms, 7.0 M | |
| **256 x 128 (32)** | 97.5 ms, **9.5 M** | 47.2 ms, **19.5 M** | 33.3 ms, **27.7 M** | 8 x 128: 24 ms, 1.2 M; 32 x 128: 64 ms, 1.8 M |
| 256 x 256 (32) | 318 ms, 2.9 M | 174 ms, 5.3 M | 111 ms, 8.3 M | |
| 64 x 128 (128) | 56.5 ms, 4.1 M | 23.4 ms, 9.8 M | 36.6 ms, 6.3 M | 8 x 128: 33 ms, 0.9 M |
| peak GPU memory, 256 x 256 | 3.0 GB | 3.1 GB | 3.0 GB | |

How to read the table:
- **One GPU equals many CPUs here.** The same 128 worlds x 500 ticks took 158 s on this PC's CPU, 27.0 s on the 3060
  (5.9x), 12.9 s on the R9700 (12x) and 9.1 s on the 4090 (17x). At the best batch size, the three 3060s, two R9700s
  and the 4090 together would do about 95 M world-ticks per hour. The whole r8a cluster did about 5 M per hour with
  life8, but life8 worlds are richer: they have objects, tools and social actions.
- **Small batches are wasted.** Below 256 worlds the time per tick hardly changes: about 33 ms on the 4090, 47 ms on
  the 3060 and 15 ms on the R9700. That floor is Python launching a few hundred small GPU kernels per tick on one CPU
  core, and here that core was shared with `life8-long`. So runs should use 256 or more worlds per process. Later,
  CUDA graphs or `torch.compile` could lower the floor.
- **Population is the expensive axis.** Going from 128 to 256 slots per world costs 3-4x, because every pair checks
  a sight line. A brain of 128 units instead of 32 costs 1.1-1.4x.
- **Memory is not the limit.** Peak use is at most 3.1 GB, and the 3060 has 12 GB, the 4090 about 17 GB free, the
  R9700 about 25 GB free.

**GPU against CPU statistics.** All 128 worlds x 500 ticks from seed 1, compared per world with Welch t-tests.
Every difference is within noise:

| | CPU (this PC) | 3060 | R9700 | 4090 |
|---|---|---|---|---|
| alive at tick 500 (mean of 128 worlds) | 50.2 | 49.4 (p 0.55) | 50.6 (p 0.76) | 48.5 (p 0.19) |
| births | 46.1 | 44.9 (p 0.43) | 46.8 (p 0.61) | 44.7 (p 0.34) |
| starvation deaths | 32.1 | 32.0 (p 0.82) | 32.3 (p 0.69) | 32.2 (p 0.86) |
| predation deaths | 11.7 | 11.5 (p 0.48) | 11.8 (p 0.76) | 11.9 (p 0.38) |
| calls | 23,780 | 23,906 (p 0.74) | 24,095 (p 0.39) | 23,526 (p 0.51) |
| largest energy-ledger error | 0.0064 | 0.0069 | 0.0060 | 0.0060 |

The random streams differ between devices, so the worlds are different draws and only their statistics can match.
An earlier 32-world comparison (48.0 alive on CPU, 52.8 on the R9700) was noise, and it disappears at 128 worlds.

**What this means for phase 2.** One gate setting at 256 worlds per arm x 20,000 ticks takes about 22 min on the 4090,
about 32 min on an R9700 and about 65 min on a 3060 (two arms, one after the other). A sweep of 10 ecology settings x
2 seeds then takes about 2 h across the six free GPUs. On this PC's CPU, the same sweep at only 64 worlds would
take about 35 h.

## 5. Listener gate, first CPU look

Seed 1, 16 worlds x 96 slots x 2,000 ticks per arm, on the CPU. Each run took 3-6 minutes, two at once on a shared machine.
The result files are in the session scratchpad: `gate2-default.json` and `gate2-harsh.json`.

| Ecology | predation lower with hearing (worlds) | sign p | reached individuals moved away, oracle / deaf | flee more with hearing (worlds) | predation share of deaths | extinct worlds (oracle / deaf) | gate |
|---|---|---|---|---|---|---|---|
| default (2 predators, damage 0.55) | 7 of 16 | 0.80 | 0.483-0.503 / 0.491-0.504 | 8 of 16 | 19-31 % | 0 / 0 | **fail** |
| harsh (4 predators, one strike kills) | 5 of 16 | 0.21 | 0.447-0.495 / 0.474-0.493 | 8 of 16 | 69-97 % | 7 / 7 | **fail** |

**Not passed.** Calls that are perfectly informative do not make hearers move away from the predator. Hearers move
away about half the time in both arms. Mean brain size stayed at the starting 8 units, so it did not move in
2,000 ticks. The harsh ecology wiped out 7 of 16 worlds in each arm, so it is too harsh as set.

One measurement fix came out of this. The first version compared hearers with non-hearers inside the oracle arm.
That comparison showed hearers moving away *less* (1 of 16 worlds, p = 0.0005), but it was confounded: callers stand
near predators, so hearers do too. The gate now compares individuals a call reached, between the hearing arm and the
deaf arm, world by world (`probe.py` docstring).

What this means for phase 2. 2,000 ticks is about 10 generations from random brains, which is short. Predation
causes only a quarter of the deaths in the default ecology. The gate needs longer runs and more worlds on the GPUs,
plus an ecology sweep between default and harsh: predator stamina, detection fraction, relief, and the share of
food on low ground. If hearers still do not respond, test the lifetime learner on its own: a fixed sound in a fixed
sector before each strike, in a small corridor test like life8's `credit corridor`. This will show whether Hebbian
plasticity can learn a response at all, or whether only evolution can.

## 6. Limits of the prototype (stated plainly)

- **The space is a height field.** That is a 3D surface: z is real, and so are sight lines, sound shadows and climbing.
  There is no flying, swimming, caves or overhangs. A volumetric world (water or air layers) is a later option.
- **GPU runs are not bit-exact.** `scatter_add` and matrix multiplies reorder floats. CPU runs are exact and resume
  exactly. GPU runs will be reproducible as statistics, and `torch.use_deterministic_algorithms` is not enforced yet.
- **Lifetime learning is Hebbian, not Q-learning.** Lifetime learning may be weaker than life8's learner at first;
  evolution carries more of the load. A per-individual gradient learner (batched, own-reward) is a later option.
- **Only part of life8 is ported.** life9 has terrain, food, predators, calls, evolution and plasticity. life8's
  objects, tools, sharing, teaching and kin credit are not ported yet. life8 stays the reference engine for those.
- **The predator model is new and untested.** The stealth and stamina numbers are first guesses. Phase 2 exists to
  test them.
- **The capacity cap is a config value.** `capacity` (slots per world) and `hidden` (maximum brain units) are hard
  ceilings. `capacity_full` and `mean_hidden` show when they bind.

## Commands

```
python -m pytest tests/life9 -q
python -m haishool.life9 bench --worlds 64 --capacity 256 --ticks 50 --device cuda
python -m haishool.life9 probe --seed 1 --ticks 10000 --worlds 64 --device cuda --out gate-s1.json
python -m haishool.life9 run --seed 1 --ticks 3000 --worlds 16 --device cuda --out runs/life9/s1 --view-every 2 --view-ticks 1000
python -m haishool.life9 run --resume runs/life9/s1/state.pt --ticks 3000 --out runs/life9/s1b
python -m haishool.life9 run ... --option predator_stamina=8 --option relief=8.0
```
