# Haishool

A small language model that learns **facts, not language**. It is trained from scratch only on
dense lines of facts (everyday objects, people, countries, religions, history, markets, science,
software and pop culture); fixed rules turn its answers into English.

**Try it:** https://enderpeer.github.io/haishool/ · **Training data:** https://enderpeer.github.io/haishool/data.html

```
your question ──► parser (rules) ──► dense query ──► model ──► dense answer ──► translator (rules) ──► English
"what is the capital of japan"       q japan capital. a        tokyo                 "The capital of Japan is Tokyo."
```

## How it works

1. **Facts.** A local teacher model (Qwen2.5-14B-Instruct) lists things and writes short facts
   about each. Everyday objects get nine attributes (kind, color, shape, size, made of, parts, use,
   place, alive). Since round 3, other things have a type with their own keys: a person has a role,
   country, era and what they are known for; a country has a capital, languages, currency, number of
   states and government; an event has a time, place, cause, result and people; and there are types
   for religions, concepts, works (software, games, films, music, books), lists and offices.
   ```
   spoon. kind utensil. color silver. shape curved. size hand. made metal plastic. parts handle bowl. use eating_soups. place kitchen_dining_room. alive no.
   japan. kind country. known_for technology_electronics. continent asia. capital tokyo. language japanese. currency yen. states 47. government parliamentary_constitutional_monarchy. type country.
   ```
2. **Model.** A 16-million-parameter GPT (6 layers, width 384, word-level vocabulary) learns only
   these lines and the matching questions (`q japan capital. a tokyo.`). It never sees an English
   sentence.
3. **Parser.** Rules find the thing a question is about (including aliases such as the German
   *Kanzler*) and which key is asked; the record's type decides the key ("when" is the time of an
   event, the year of a film, the era of a person).
4. **Translator.** Rules turn the dense answer into a sentence. They add no facts: if the model is
   wrong, the sentence is wrong in the same way, and the page shows the raw model output under
   every answer.
5. **Hand-written records.** A few records are written by hand and say so in the data:
   - who it is: called Homunculi, runs locally in Berlin, no desires or feelings of its own;
   - questions it cannot know ("which shoe am I wearing?"): it says it cannot know that and why;
   - "make me rich": no quick safe way and no personal investment advice, only general habits;
   - current office holders, with "as far as my data goes";
   - a few common lists (German federal states, the big religions, the planets).

## Results (version 3)

| | |
|---|---|
| Records | 3,080 plus identity: 1,632 everyday objects, 410 concepts, 366 people, 256 works, 196 countries, 170 events, 27 lists, 19 religions, 4 offices |
| Facts | 22,341 |
| Facts recalled exactly | 1,000 of 1,000 sampled trained facts |
| Held-out facts guessed exactly | 22 % (1,877 facts held back in a separate run; 26 % of their words right). Version 2: 25 % of 1,253 |
| Model | 15.7 million parameters, word vocabulary of 13,226 |
| Training | 14,000 steps, about 10 minutes on one RTX 4090 |

What it cannot do: answer about things outside its table (it says so), hold a conversation, or
know more than the keys of each record. The facts come from the teacher model and have not been
checked by a person; some are wrong or oddly worded, and they show up exactly that way.

## Version 4: links between things

Round 4 (built in a parallel session) adds a small graph of 52 things and 135 hand-checked links
(54 kinds: borders, capital of, founded, holy to, influenced, ...). Disputed links keep who holds
them instead of one flat answer. The model learned each thing's links, each side's view and the
paths between any two things:

```
q turkey borders. a greece syria.
q jerusalem capital_of. a israel palestine contested.
q jerusalem capital_of according_to palestine. a palestine.
q abraham hop led_zeppelin. a abraham father_figure_of christianity according_to christianity ... signed led_zeppelin.
```

In the chat: "what borders Turkey", "what is the capital of Israel" (contested, with the sides),
"... according to Palestine", "how is Abraham connected to Led Zeppelin" (a six-hop path), and
descriptions of linked things combine their facts with their links (`haishool/links.py`).

The round-4 data (`data/hops-v4/`) and its build and training code (`haishool/relations.py`,
`student.py --extra-lines`) are in the repository; rebuilding the data reproduces it byte for byte.

| | Version 4 | Version 3 |
|---|---|---|
| Trained facts recalled exactly | 1,000 of 1,000 | 1,000 of 1,000 |
| Held-out facts guessed exactly | 21 % of 1,877 | 22 % of 1,877 |
| Relations recalled exactly | 98 % of 213 | n/a |
| Views ("according to") recalled exactly | 19 of 19 | n/a |
| Paths between linked things it never saw, exactly | 96 % of 279 (path length 97 %) | n/a |
| Parameters | 15.8 million | 15.7 million |

## Version 4b: everything linked

Version 4b links all 3,081 records into one graph instead of 52 hand-picked things: 3,840 things
and 14,502 links, every thing reachable from every other (at most 9 hops between sampled pairs).
The links come from the hand-written seed (122, with views and reasons), from record values that
name another record (5,075, e.g. `spoon made_of metal`; only harmless spelling slips such as
`fire_station` / `firestation` are resolved), from shared values (`europe`, `1970s`, `kitchen`)
and from a few rules that follow from the others (`located_in`, `shares_food`, `bandmate`). A link
the seed holds under a view is never added again as a plain fact from the teacher. The model also
learns to say no (`q turkey borders colombia. a no.`) and each thing's links
(`hagia_sophia links. built_by byzantine_empire. ...`).

| | Version 4b | Version 4 | Version 3 |
|---|---|---|---|
| Trained facts recalled exactly | 995 of 1,000 | 1,000 of 1,000 | 1,000 of 1,000 |
| Held-out facts guessed exactly | **35 % of 1,877** | 21 % | 22 % |
| Relations / views / yes-no recalled | 99 % / 19 of 19 / 100 % | 98 % / 19 of 19 / n/a | n/a |
| Unseen paths: exact / only real links / reach the target | 66 % / 89 % / 99 % (600 checked) | 96 % / 99 % / 100 % | n/a |
| Training lines added to the facts | 66,357 (1.07 M tokens per pass) | 5,588 | n/a |
| Steps (one RTX 4090) | 24,000, about 37 minutes | 14,000 | 14,000 |

Linking the facts raised the share of unseen facts the model guesses exactly from 22 % to 35 %.
Known weaknesses: its "no" examples use random partners, so it answers yes to near misses ("does
Turkey border Israel?"); the test chat (`haishool/hops_app.py`) checks every step and every yes or
no against the links and marks the model's mistakes. Paths can be real but pointless where a
teacher record is wrong, and `usa` is still a separate hub from `united_states` in this build.
Data: `data/hops-v4b/` (`hops-train-r4.txt` sha256 12013c56...), model: `model/haishool-v4b.pt`.

In the public chat (`haishool/links.py`, version 4b since 2026-10-01): yes/no questions ("does
Turkey border Greece") answer from the links the model was trained on and say when the model's own
guess disagrees; a missing link is answered as "not in what I learned", not as a hard no; paths work
by partial names ("how is Einstein connected to Newton"), and two things that only share a kind are
described as such instead of a path through the hub.

## How it answers every question

It never refuses. If a question names no record, Homunculi still answers from its training data,
trying these in order and saying under the answer how it found the record:

1. a rare word of a record's name ("who is **einstein**"), also with a typo fixed ("einstien");
2. the question's words inside another record's facts ("who invented the **telephone**" finds the
   person whose facts mention it; "which country has the capital **tokyo**" finds Japan);
3. small talk ("hello", "how are you") goes to its identity record;
4. otherwise the record whose name is spelled most alike, said plainly: "I learned nothing about
   quasar. The closest thing I know is Qatar: ...".

The model writes every fact in the answer; the search only picks which record to ask.

## How it compares

| Model | Parameters | Training text | Training compute (≈ 6 × parameters × tokens) |
|---|---|---|---|
| **Haishool v3** | 15.7 M (6 layers, width 384, context 96 words) | 1.3 MB: 22,341 facts, 284 k word tokens, repeated (172 M tokens processed) | ≈ 1.6 × 10¹⁶ FLOP, 10 minutes on one RTX 4090 |
| TinyStories models (2023) | 1 to 33 M | about 2 million very simple short stories | small, hours on one GPU |
| GPT-2 small (2019) | 124 M | about 40 GB of web text | |
| Homunculi a 0.0.2 (our general model) | 218 M | about 21 B characters (4.6 B tokens), 24 hours on one RTX 4090 | ≈ 6 × 10¹⁸ FLOP |
| GPT-3 (2020) | 175 B | about 300 B tokens | ≈ 3 × 10²³ FLOP |
| Qwen2.5-14B (the teacher) | 14.7 B (48 layers, width 5,120, long context) | about 18 T tokens | ≈ 1.6 × 10²⁴ FLOP |

Haishool is about a thousand times smaller than its teacher and saw tens of millions of times less
text; the teacher's training took roughly a hundred million times more compute. It is the same kind
of network (a decoder-only transformer), only tiny. What makes it work at that size is the split:
the model only has to store facts in one regular format, and fixed rules do the language. The price
is that it knows nothing outside its table, cannot reason over several facts or hold a conversation,
and is only as right as the teacher's facts (22 % of held-back facts it guesses exactly, the rest it
must have seen).

## Run it yourself

```bash
pip install torch fastapi uvicorn
python -m haishool.app --model model/haishool-v4b.pt --records data/records-r3-all.jsonl --hops data/hops-v4b --port 8650
# version 4: --model model/haishool-v4.pt --hops data/hops-v4; version 3 without links: model/haishool-v3.pt, no --hops
python -m haishool.hops_app --model model/haishool-v4b.pt --records data/records-r3-all.jsonl --port 8651
# version 4b test chat: paths, yes/no and views, every step checked against the links
```

Train your own (the teacher needs [ollama](https://ollama.com) with `qwen2.5:14b-instruct-q4_K_M`):

```bash
python -m haishool.teacher objects --out data/objects.txt                       # rounds 1-2: everyday objects
python -m haishool.teacher records --objects data/objects.txt --out data/records.jsonl
python -m haishool.teacher3 entities --out data/entities-r3.txt                 # round 3: typed records
python -m haishool.teacher3 records --entities data/entities-r3.txt --out data/records-r3.jsonl
python -m haishool.combine data/all.jsonl data/records.jsonl data/records-r3.jsonl data/manual.jsonl data/identity.jsonl
python -m haishool.student train --records data/all.jsonl --out runs/full --steps 14000
python -m haishool.relations build --seed data/hops-seed-r4.jsonl --records data/records-r3-all.jsonl --out data/hops-v4b   # round 4b: links
python -m haishool.student train --records data/records-r3-all.jsonl --extra-lines data/hops-v4b/hops-train-r4.txt --out runs/r4b-full --steps 24000
python -m pytest tests                                                          # parser and translator rules
```

## Files

| Path | Content |
|---|---|
| `haishool/` | schema, teachers, merge, model, student training, parser and translator, chat app |
| `model/` | weights of versions 1 to 5 (half precision) and their training and evaluation reports |
| `data/` | the training facts, one JSON line per record: `records-r1/r2/r3.jsonl` per round, `manual.jsonl` and `identity.jsonl` by hand, `records-r3-all.jsonl` merged (what v3 was trained on) |
| `docs/` | the website (landing page with chat, data browser) |
| `tests/` | rule tests for parser, translator and merge |
| `LESSONS.md` | what we learned before this, and why it is built this way |

## Rounds 5–6: checked calculations and toy worlds

The local follow-up adds five truth gates (`haishool/truth`) and a six-level toy
simulation chain (`haishool/cosmos`). The simulations produce reproducible training
examples; they are not a physical forecast of the universe or an explanation of
the real origin of life. The browser's `docs/world.html` contains saved example
worlds, not a model generating a live world.

`python -m haishool.round5 build` builds the compact integration dataset, including
per-topic training and sealed files, rollout records, token counts and prompt lists.
Direct calculations, worked calculations, checks and judgments share a semantic
split key, so one form cannot reveal a held-out answer in another form.

The final curriculum adds checked worked steps for all 18 force formulas and for
molar masses and mass percentages. `haishool.cosmos.predict` supplies explicit-input
questions about analytical substeps and conditional expectations: cooling,
accretion, chemical temperature schedules, replication probabilities, conserved
resources and world handoffs. Seed-only rollout questions remain separate recall
tasks; they are not counted as generalization to new physical states.

On the training host, run:

```bash
python -m pip install -r requirements.txt
python -m haishool.final_run pipeline --root runs/final-v5 --data data/final-v5 --workers 12
```

This builds a bounded corpus targeting 1.5 million unique maths lessons, 300,000
force lessons, additional table/chemistry lessons, explicit-input prediction
lessons and 3,456 rollouts. Actual counts, saturation, hashes and exclusions are
reported in `data/final-v5/report.json`. It is a large generated corpus streamed
from disk, not an infinite online generator. Only training files enter the model's
vocabulary and token cache. Development and sealed question families are reserved
by semantic hash; simulation seeds are disjoint across splits and levels.

`haishool.train_final` uses one disk-backed token copy per source and a fixed token
mixture: 40% old facts/links, 40% truth topics, 15% explicit-input predictions and
5% legacy rollout recall. Within a topic, feedback receives 20% of its existing
allocation. Training uses 256-token context for worked chemistry, bf16 on CUDA,
atomic checkpoints every 1,000 steps, and configuration/data/RNG checks on resume.

The pipeline trains both 6-layer/384-width and 8-layer/512-width models for 30,000
steps with old-data holdouts. Each gets a verified feedback pass and 3,000 further
steps. Development accuracy and old-fact retention select the candidate before
one sealed evaluation. A separate 3,000-step production refinement restores old
held-out facts; its results are kept distinct from the evaluated checkpoint.
The pipeline does not update the public chat. It writes progress and failures to
`runs/final-v5/status.json`, phase logs/checkpoints under that directory, and a
`summary.json` when complete. Running the same command resumes completed phases
and valid checkpoints; it rejects changed inputs rather than silently mixing runs.

The saved earlier maths-only experiments scored 93.0% (6×384) and 95.6% (8×512)
on 3,000 held-out questions. These are not final combined-model scores; those are
reported separately below because the dataset and training objective differ.

### Final combined run, completed 2026-10-01

The successful Adler pipeline ran from 18:09:53 to 18:59:35 UTC (49 minutes 42
seconds); the dataset build took another 4 minutes 31 seconds. It completed both
model sizes, verified feedback, candidate selection, sealed evaluation, and separate
production refinements. There were 2,272,426 training lines / 60,871,214 tokens,
4,352 development and 4,352 sealed questions across 17 topics, and 3,456 rollouts.
Independent file and semantic-split audits found no mismatches or leakage.

The main score below covers eleven truth/input-conditioned topics, 256 sealed
questions each (2,816 total). Six legacy seed-recall topics are separate in the
reports. Exact accuracy is the primary comparison: gate acceptance allows each
topic's own numerical tolerance and is not a uniform precision standard.

| Selected evaluation checkpoint | 6×384 | 8×512 |
|---|---:|---:|
| Parameters | 16.25 M | 32.68 M |
| Sealed exact accuracy | 34.48% (971/2,816) | **39.42% (1,110/2,816)** |
| Sealed gate acceptance | 39.88% | **44.78%** |
| Old trained facts recalled | 929/1,000 | **969/1,000** |
| Feedback checkpoint selected | No: development acceptance fell 40.80% → 40.20% | Yes: 43.79% → 44.32% |

| Sealed topic, exact (256 questions per row) | 6×384 | 8×512 |
|---|---:|---:|
| Maths | 63.67% | 74.61% |
| Elements | 73.05% | 75.00% |
| Substances | 32.42% | 40.23% |
| Reactions | 81.64% | 80.08% |
| Forces | 25.00% | 37.11% |
| Nucleosynthesis inputs | 0.78% | 0.00% |
| Gravity inputs | 1.17% | 1.56% |
| Planet inputs | 0.78% | 1.56% |
| Chemistry inputs | 12.11% | 11.72% |
| Life inputs | 60.55% | 73.05% |
| World handoff inputs | 28.13% | 38.67% |

Worked maths was 34/36 correct for the small model and 36/36 for the wide model.
Worked forces remained 22/128 and 37/128; worked substances 38/141 and 49/141.
World handoffs accepted 87.50% and 97.66% under a 5% numerical tolerance, which
explains their large gap from exact accuracy. This is approximate calculation of
the specified handoff formulas, not reliable prediction of whole worlds.

Each feedback pass asked 2,816 fresh questions and wrote 5,632 answer/judgment
lines, correcting 1,647 small-model answers and 1,542 wide-model answers. Feedback
helped only the larger candidate under the declared selection rule.

Production checkpoints are separate. Their development exact/accepted scores were
35.30%/40.52% (small) and 40.06%/44.67% (wide); old-fact recall was 90.7% and 95.2%.
Restoring held-out material changes that recall sample, so these are not paired
recall drops from the evaluation checkpoints. Both production refinements used
correction files, including the small run whose feedback candidate was rejected.
Neither production checkpoint inherits its selected candidate's sealed score.
The pooled old held-out scores (47.57% and 50.51%) combine 1,877 fact questions and
3,862 hop questions and must not be compared with historical fact-only inference.

All four float32 exports in [model/final-v5](model/final-v5) were checked tensor for
tensor against their original checkpoints, loaded and generation-tested on Adler,
then hash-verified locally along with 36 reports. The [comparison](model/final-v5/comparison.json)
and [artifact manifest](model/final-v5/manifest.json) identify every model and score.
The strongest measured result is `haishool-v5-8x512-selected.pt`. The public chat
remains v4b: the new models are experimental and have not preserved its near-perfect
recall of trained facts. Next priorities are explicit fact-retention targets,
worked intermediate calculations for weak numeric predictions, and a production
selection rule that does not reintroduce rejected feedback without verification.

### V5 maths/science chat preview

The separate v5 preview now routes supported English maths and science questions
before the legacy fact/link parser. `haishool/science_routes.py` translates input
into an existing gate-owned prompt; `haishool/v5_app.py` asks the neural model,
checks its answer, and returns the rule result separately. No model retraining or
public v4b service change is needed. Unsupported calculations fail explicitly
instead of turning into an unrelated nearest-record answer.

```bash
python -m haishool.v5_app --model model/final-v5/haishool-v5-8x512-selected.pt \
  --records data/records-r3-all.jsonl --hops data/hops-v4b --port 8652
```

Examples: `What is 12 plus 7?`, `How many protons does carbon have?`,
`Molar mass of water`, `Balance H2 + O2 -> H2O`, `pressure f=100 area=2`,
and `calc 4 7 times 6 steps`. Explicit-input simulation forms are also supported,
for example `life predict expected_mutations length 8 mu 0 point 2`.
Named force operands use SI units; unsupported units are rejected, not silently
converted. Each question is independent. The UI in `docs/v5-chat.html` preserves
the model's raw output and labels agreement or disagreement with the gate.

The live LAN preview is `http://192.168.178.171:8652/`, running in the isolated
`/home/ender/haishool-v5-preview-20261001` directory on Adler. Its launcher is
`scripts/run-v5-preview-adler.sh`. The completed training directory and both public
v4b services remain separate. Router and API verification: 100 tests passed on
Adler, followed by live API and browser checks. The routing fix exposes capabilities;
it does not make a wrong model prediction correct.

### Render and inspect inhabitants from a world run

`haishool.inspect_world` is an optional observer for round 7. It does not change
the simulator's dynamics or training lines. Each run captures a stable source
snapshot, runs the engine in a fresh subprocess, checks world consistency, and
saves `world.json`, `inhabitants.json`, `manifest.json`, and an offline `viewer.html`.
The archive includes effective parameters, attempt seeds, dependency versions,
source hashes and the versioned drawing geometry.

```bash
python -m haishool.inspect_world run --seed 85 --out runs/inhabitants/my-world-85
python -m haishool.inspect_world verify runs/inhabitants/my-world-85
python -m haishool.inspect_world serve --root runs/inhabitants --port 8653
```

The local inspector at `http://127.0.0.1:8653/` can run another seed and browse
planets, saved generations, genotypes, individual snapshot records and body
lineages. On Windows, `scripts/start-inhabitant-inspector.ps1` starts it hidden
and avoids duplicate servers. An existing archive is never overwritten.

Drawings are deterministic schematics of recorded traits: zero eye/calling genes
produce no eye/calling marker; sensor levels remain capacities rather than organ
counts. Body size is a cell count, not a physical length. Colours, outlines and
marker placement are explicitly versioned display choices, not evolved anatomy.
The same saved phenotype renders the same way without image generation.

Body lineages preserve the simulator's real lineage and parent IDs. Senses IDs
are local to a snapshot because the simulator records genomes, not lifelong
individual identities; generation 295 is the last pre-reproduction snapshot in
the current level. Society remains aggregate context, not fabricated citizens.
Empty or extinct worlds are shown without invented inhabitants. Other builders
can call `haishool.evo.inhabitants.export_inhabitants(world)` on a stored world
without rerunning it. Keep display-only geometry separate from biological
training targets.

Validation: 47 exporter/archive tests passed on Adler (45 passed and two
Windows-symlink tests skipped locally). A real World 85 export passed world
conservation and archive checks; it retained 60 sensory snapshots, 200 final agents
and ten final genotypes. Live and offline viewer interactions were checked in the
browser. The observer and viewer are separate from Opus's active biology files
and the completed v5 runtime.

### Stored cell anatomy before rendering

`haishool.anatomy_run` adds a developmental stage conditioned on a phenotype
already saved by the world inspector. It grows an exact cell count through
recorded binary divisions, relaxes the cells in three dimensions, and saves
their positions, radii, ancestry, tissue assignments, measured contacts and an
explicit growth-volume ledger in `anatomy.json`. Source and renderer code,
parameters, seed, dependency versions and hashes are archived for verification.

```bash
python -m haishool.anatomy_run build --archive runs/inhabitants/world-85-render-v1 \
  --out runs/anatomy/world85-dominant-v2 --seed 85
python -m haishool.anatomy_run verify runs/anatomy/world85-dominant-v2
python -m haishool.anatomy_run render runs/anatomy/world85-dominant-v2 --samples 64 --threads 8
```

The source archive must already exist. Choose a new output directory for a new
build; existing archives and render destinations are not overwritten. The cell
model uses NumPy and SciPy. Rendering uses Blender on the CPU; use `--blender`
to specify its executable when the default Windows installation path differs.
The renderer can produce `render/anatomy-4k.png` (3840 × 2160),
`render/saved.blend` and `render/surface.glb`, with receipts linking each output
to the stored cells. The surface comes from those cells, with no added anatomical
features. Lighting and material finish remain rendering choices.

The archived World 85 specimen preserves **256 cells, 64 neural cells, seven cell
types and zero eyes**. Spatial layout, tissue functions, unnamed subtype
identities and neural connection choices are new model assumptions. This is not
recovered historical anatomy, atom positions or a society citizen's identity.
Lengths use mature-cell-radius units, not metres. Anatomy is not yet connected
to world survival, selection or training, and no core evolution rules are changed.
Validation: 52 anatomy and archive tests passed. The first full render in
`runs/anatomy/world85-dominant-v2` was verified at 3840 × 2160, with matching
output hashes and all 256 cell instances checked after reopening the Blender
scene. Rendering took 220 seconds on eight CPU threads. The derived surface is
a voxel approximation; its volume diagnostics are recorded in the render receipt.

## Licence

Code: MIT (`LICENSE`). Model weights and data: CC BY 4.0. The facts were generated with
Qwen2.5-14B-Instruct (Apache-2.0).
