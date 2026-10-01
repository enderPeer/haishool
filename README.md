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

## Licence

Code: MIT (`LICENSE`). Model weights and data: CC BY 4.0. The facts were generated with
Qwen2.5-14B-Instruct (Apache-2.0).
