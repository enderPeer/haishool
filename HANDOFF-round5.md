# Round 5 hand-over notes

## Inhabitant inspection upgrade — 2 October 2026

At the user's request, Codex added a separate render/export observer without
editing active `world7.py`, `senses.py`, `bodies.py`, or other simulation core files:

- `haishool/evo/inhabitants.py`: deterministic recorded traits, phenotypes and
  SVG primitives; senses snapshot IDs and actual bodies lineage/parent IDs.
- `haishool/inspect_world.py`: frozen-source subprocess runs, verified archives,
  offline HTML export, and local Run world server.
- `docs/inhabitants.html`: planet, population, generation, genotype/agent and
  body-lineage inspector. No generated images or client randomisation.
- `scripts/start-inhabitant-inspector.ps1`: hidden localhost launcher.

For the next render-enabled run:
`python -m haishool.inspect_world run --seed 85 --out runs/inhabitants/<new-run-name>`.
Or use `http://127.0.0.1:8653/` and its Run world button. An existing builder can
call `export_inhabitants(world)` after creating a WorldRollout; the exporter does
not rerun or mutate it. Existing simulation run functions remain unchanged.

World 85's first archive is `runs/inhabitants/world-85-render-v1`; World 86 also
completed through the live UI, with no bodies/senses and an honest empty result.
The source snapshot protects runs from concurrent Opus edits. Archive verification
checks artifact hashes, source inventory, dependency receipt and seed consistency.
All 47 new tests passed on Adler, including symlink/path checks. Local tests:45
passed,2 symlink-permission skips. No new training or public chat changes occurred.

Scope: geometry is explicitly a display encoding, not physically evolved anatomy;
sensor levels are not organ counts. Senses records do not contain persistent
individual ancestry, and society does not contain individual citizens. These
limits are labelled in exports and the UI. Future biological shape genes or
individual lineage tracking need a simulator change, not silent renderer invention.
Do not train the renderer's arbitrary colours/shapes as biological observations.

## User's next objective — world-only learning

On 1 October 2026, after the v5 comparison, the user clarified the intended
direction: Fable is extending life forms and society; future learning should use
only simulated worlds and their generations, with mathematical, biological,
chemical and linguistic capabilities developing from that experience. The old
Haishool facts, relationship paths and chatbot benchmark are not the target of
this next experiment. The completed v5 models remain preserved baselines.

This is a recorded objective, not a claim that a new trainer has been implemented
or launched. Fable's active `haishool/evo` code was inspected read-only. The next
training design must not silently reuse `train_final.py`'s mandatory old-fact/hop
mixture. Simulation laws and modelling assumptions should remain explicit.

To test this stronger objective, retain chronological interaction episodes:
agent identity and lineage, available perceptions, actions, emitted and received
messages, listener responses, and consequences for resources and survival. Keep
complete simulator state for auditing, separate from what an agent could know.
Learning a finished dictionary or a formula's answer is a useful baseline, but
is not evidence that a learner inferred those meanings or procedures from lived
interaction. Distinguish an external predictive learner from a learner acting
inside the world.

The current draft has emergent signal conventions and bounded language learning,
with a supplied meaning inventory, two allowed grammatical orders and scheduled
phases. Society's named technologies currently unlock at fixed skill thresholds;
the `mathematics` label does not demonstrate invented arithmetic. These are
prototype assumptions to assess against the new objective, not claims that the
unfinished upgrade has already achieved unrestricted emergence.

The user's ambition includes "every possible language". A measurable research
target is broad language diversity and rapid learning of unfamiliar communication
systems in unseen worlds, not a claim of exhaustive coverage. Invented world
languages do not automatically supply the historical conventions of English,
German or other human languages. Evaluation should hold out entire worlds,
communities, vocabularies and meaning combinations, measure successful action
and prediction, and compare against removed or shuffled communication. A world
may fail to develop life or useful communication; those failures remain evidence.

## Completion addendum — 1 October 2026

The Codex continuation completed the final pipeline on Adler at 18:59:35 UTC.
It did not change either public service. Both sizes, feedback passes, candidate
selection, sealed evaluation and production refinements are finished. The larger
selected candidate scored 39.42% exact / 44.78% gate-accepted over the eleven
nonlegacy sealed topics; the smaller scored 34.48% / 39.88%. These are research
results with substantial remaining numerical and retention weaknesses, not a
recommendation to replace v4b automatically.

All four float32 model exports and 36 reports are in `model/final-v5/` locally.
`comparison.json` contains the detailed results and caveats; `manifest.json` binds
exported hashes to the original checkpoints. Exports were checked tensor for
tensor, loaded/generated on Adler, and hash-verified after transfer. The selected
8x512 checkpoint is `haishool-v5-8x512-selected.pt`; the production checkpoint is
separate and must not inherit its sealed score. README and the local world page
now include measured results. No public-model switch or message to another
session was made by this continuation.

Public routing can use the existing gate `owns/check` interfaces. Worked forces
and substances append `steps`; explicit-input simulation forms and their grammar
are in `haishool/cosmos/predict.py`. The exported checkpoints preserve the
`{model_state, gpt_config, itos}` interface expected by `student.load`.

The historical notes below describe the earlier handover and are preserved.

### V5 preview routing connected

At the user's follow-up request, the LAN preview at `http://<lan-host>:8652/`
now runs `haishool.v5_app` from the isolated directory
`/home/ender/haishool-v5-preview-20261001`. New modules `science_routes.py` and
`v5_app.py`, plus `docs/v5-chat.html`, route English maths/science to the trained
gate prompt formats and display a separate rule check. Existing `app.py`, public
ports 8650/8651, and the completed training runtime were not changed. 100 route/API
tests passed on Adler; browser tests confirmed 12+7=19, carbon protons=6, and a
wrong molar-mass model answer visibly marked against the reference value. The
temporary test port 8653 was stopped after deployment. The preview still uses the
same selected 8x512 checkpoint; this work did not retrain it.

Written 1 October 2026, 20:35 Berlin time, by the session that built rounds 4, 4b and the
round-5 gates and simulations, for whoever finishes the final training. Everything here is
something the code and reports do not already say. Delete this file when round 5 is committed.

## Who owns what

Two other Claude sessions share this checkout and the training host.

- **"Adler training status" session** owns the public chat: `haishool/app.py`, `links.py`,
  `translate.py`, `search.py`, `docs/index.html`, `docs/chat.html`, `docs/endpoint.json`, and
  on adler40 the folder `~/haishool` with tmux `haishool-chat` (port 8650, model v4b). Ask it
  before touching those. It is waiting for the exact round-5 prompt forms to route arithmetic
  and element questions from the public chat.
- It depends on four things in `haishool/student.py`: the signatures of `load`, `generate`,
  `answer`, `describe`; the leading `<eos>` that `generate()` puts before every prompt;
  checkpoints saved as `{model_state, gpt_config, itos}`; and that a grown vocabulary keeps
  every old token id. `tests/test_student.py` covers them.
- **This session** owns `haishool/relations.py`, `hops_app.py`, `docs/hops.html`,
  `docs/world.html`, `docs/endpoint-hops.json`, and on adler40 the folder `~/haishool-r4b`
  with tmux `haishool-hops` (port 8651, the test chat behind `docs/hops.html`).

## What is running where

| Place | What |
|---|---|
| adler40 `~/haishool-final-20261001` | the final pipeline (GPU 0), status in `runs/final-v5/status.json` |
| adler40 `~/haishool`, port 8650 | public chat, version 4b, not to be changed by the training |
| adler40 `~/haishool-r4b`, port 8651 | hops test chat, version 4b |
| this PC, `cloudflared` | quick tunnel to port 8651; its address is in `docs/endpoint-hops.json` and changes when the tunnel restarts |
| adler40 `~/haishool-r5`, `~/haishool-r5-dev`, `~/haishool-r4` | finished pilot, experiment and scratch runs; safe to delete (about 2 GB) |

adler40's disk is 91 % full (86 GB free). `scipy` was added to the shared `~/homunculi/.venv`
for the gravity level; `mendeleev` is not installed there, so the element table and atomic
weights can only be rebuilt on the Windows PC. The gates themselves read the committed jsonl.

## Measured so far

- `model/r5-pilot-eval-report.json`: v4b data plus 29,000 repeated round-5 lines. Old facts
  0.999 trained and 0.35 held-out, hop paths 0.66 held-out, table lookups 0.80 to 0.95, calc
  0.25, solve 0.17, physics formulas 0.0, molar mass 0.07.
- `model/exp-maths-stream-6x384.json` and `-8x512.json`: maths only, 1.5 million distinct lines
  seen about 11 times: 0.93 and 0.956 held-out, trained equals held-out. Bare multiplication
  0.56, with worked steps 0.98 to 1.0.
- `data/truth-v5/build-review.json`: the integration report and the critic's 20 findings,
  12 gaps and 8 code defects on the first round-5 build.

## Checked at 20:45 against the running pipeline

- The simulation tests pass on adler40 (311 of 311 with numpy 2.5.3, the Windows PC has 2.4.6),
  including the pinned worlds: rollouts are the same on both machines.
- The 36 worlds embedded in `docs/world.html` are identical to what the current code computes.
- First development scores of the 6x384 held-out run (`runs/final-v5/6x384-eval/dev.json`,
  30,000 steps, 368.6 million sampled tokens): maths 0.64 (worked 0.97, direct 0.59),
  reactions 0.82, elements 0.74, substances 0.37, forces 0.25, predict_world 0.92 by the gate,
  predict_life 0.59, predict_chem 0.14, predict_gravity 0.01, predict_planets 0.01,
  predict_nucleo 0.0; old facts recalled 0.929, where v4b and the pilot had 0.995 to 0.999.
- Reading: the run is short for this corpus. The maths-only experiment spent all 368 million
  tokens on maths (about 11 passes over 33.7 million tokens) and reached 0.93; here maths gets
  a share of the 40 % truth allocation, about one to two passes. The trainer does about
  39 steps a second on the 4090, so ten times the steps is about two hours a model. The drop
  in old-fact recall has the same cause: old data gets 40 % of the same token budget that used
  to be all its own.

## Still open, not started by anyone

1. `usa` and `united_states` are two nodes in the v4b graph (126 links on `usa`); one alias
   line in `data/hops-seed-r4.jsonl` fixes it, but it changes the hops data, so it needs a
   new hops build.
2. Yes/no on near misses: every trained "no" uses a random partner, so the model answers yes
   to "does turkey border israel". Fix in `relations.yes_no_lines`: partners from the same
   type within two hops.
3. Hop paths are still a fixed file (19,041 pairs repeated). The maths result says a stream of
   many more pairs should raise the 66 % held-out exact rate.
4. Round 5 is not linked into the round-4 graph (its keys were named to avoid `made_of` and
   `found_in`), so the hops explorer cannot walk from water to hydrogen.
5. Numbers have two token forms: digit tokens in round 5, whole words (`1971`, `47`) in
   rounds 1 to 3.

## When the training has finished

The user asked for, in this order: results reported per topic against the sealed sets;
round-5 code, data and reports committed (rounds 4 and 4b were committed as "the exact data
the model was trained on, reproducible byte for byte", then the next round on top);
`docs/world.html` "Where it stands" updated with the real scores; the prompt forms sent to the
Adler session; and the private overview page "Haishool Explorer"
(https://claude.ai/artifact/A94Mb4ge3FfGPLD21pvQLD) updated. Switching the public chat to a
new model is the user's decision, not part of the training.

The world explorer in `docs/world.html` embeds 36 rollouts (seeds 1 to 36) computed with
`haishool.cosmos.world` as of 19:10. If `world.py` or a level changes, those embedded worlds
are stale and must be regenerated before the page is republished.

## Round 7 started at 21:55 (this session)

The user asked for the toy rules to be corrected and the ladder continued past replicators.
A build is running that edits `haishool/cosmos/{nucleo,gravity,planets,chem,life}.py` and adds
`haishool/evo/` (stars, cells, bodies, senses, signals, society, world7), `haishool/curriculum7.py`
and tests. Two rules protect round 5/6 and the version-5 models:

- Corrections are options: `run(seed, rules=7)`. Without `rules` every round-6 rollout and line is
  bit-identical; `tests/test_round6_frozen.py` (56 pinned hashes) is the guard.
- `haishool/cosmos/world.py`, `predict.py`, `curriculum.py`, `train_final.py`, `final_run.py`,
  `round5.py` and `student.py` are not edited by the round-7 build.

If you commit round 5/6 while this is running, commit the five level modules as of their state
before 21:55 or wait for the build to finish; a half-applied `rules=7` edit would still pass the
frozen test only once its agent is done.

## Stored anatomy stage — 2 October 2026

The new additive developmental model is `haishool/evo/anatomy.py`; archive and
render orchestration is `haishool/anatomy_run.py`. It consumes a saved phenotype
from `runs/inhabitants/world-85-render-v1` without rerunning or changing the
source world. The current anatomy destination is
`runs/anatomy/world85-dominant-v2`.

```bash
python -m haishool.anatomy_run build --archive runs/inhabitants/world-85-render-v1 \
  --out runs/anatomy/world85-dominant-v2 --seed 85
python -m haishool.anatomy_run verify runs/anatomy/world85-dominant-v2
python -m haishool.anatomy_run render runs/anatomy/world85-dominant-v2 --samples 64 --threads 8
```

Build requires a new output directory. The saved anatomy includes exact cell
positions/radii, binary division history, growth snapshots and supplied-volume
ledger, tissue assignments, contact distances and assumed neural connections.
The sample preserves 256 cells, 64 neural cells, seven cell types and zero eyes.
The seven types include unnamed spatial subtypes; their identities and functions
were not supplied by the earlier evolution model. Its actual sphere-intersection
graph is connected, so the renderer does not need to add bridges or body parts.

Validation: 52 anatomy/archive tests passed. The final 4K CPU render completed in
220 seconds and was visually inspected. Its 3840 × 2160 dimensions, delivered
PNG path, source identity and every output hash were verified. Outputs are `render/anatomy-4k.png`,
`render/saved.blend` and `render/surface.glb`. The renderer derives the surface
only from the saved cells and records mesh approximation and display settings.
`delivery-verification.json`, `blend-inspection.json` and `mesh-inspection.json`
record the final checks. Reopening the Blender scene verified all 256 stored
cell instances against the source coordinates, radii and tissue roles. The
portable mesh is closed and consistently wound; its voxel surface volume is
1.9% above the sum of the original sphere volumes, so it remains an approximate
display surface, not a volume-conserving physical model. The source cells remain
unchanged. A later PNG-path identity guard is present in the live wrapper; the
frozen v2 wrapper predates that guard, and the delivered PNG was checked explicitly.

This is a new developmental model conditioned on recorded traits, not recovered
historical anatomy, an atomistic reconstruction or an identified society
citizen. Layout, tissue functions and connection choices remain explicit model
assumptions. Units are cell radii, not metres. The stage is not yet connected to
survival, selection or training; no active core evolution files were edited.
