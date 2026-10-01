# Haishool

A small language model that learns **facts, not language**. It is trained from scratch only on
dense lines of facts about everyday objects; fixed rules turn its answers into English.

**Try it:** https://enderpeer.github.io/haishool/ · **Training data:** https://enderpeer.github.io/haishool/data.html

```
your question ──► parser (rules) ──► dense query ──► model ──► dense answer ──► translator (rules) ──► English
"what color is a banana"           q banana color. a          yellow green          "A banana is usually yellow or green."
```

## How it works

1. **Facts.** A local teacher model (Qwen2.5-14B-Instruct) lists everyday objects and writes nine
   short attributes for each: kind, color, shape, size, made of, parts, use, place, alive.
   ```
   spoon. kind utensil. color silver. shape curved. size hand. made metal plastic. parts handle bowl. use eating_soups. place kitchen_dining_room. alive no.
   ```
2. **Model.** A 13-million-parameter GPT (6 layers, width 384, word-level vocabulary) learns only
   these lines and the matching questions (`q spoon color. a silver.`). It never sees an English
   sentence.
3. **Translator.** Rules turn the dense answer into a sentence. They add no facts: if the model is
   wrong, the sentence is wrong in the same way, and the page shows the raw model output under
   every answer.
4. **Identity.** One record says who it is: it is called Homunculi and runs locally in Berlin.

## Results (version 2)

| | |
|---|---|
| Objects | 1,634 (858 in round 1, 776 in round 2), plus the identity record |
| Facts recalled exactly | 1,000 of 1,000 sampled trained facts |
| Held-out facts guessed exactly | 25 % (1,253 facts held back in a separate run; 30 % of their words right). Version 1: 27 % of 667 |
| Model | 13.2 million parameters, word vocabulary of 6,570 |
| Training | 9,000 steps, about 5 minutes on one RTX 4090 |

What it cannot do: answer about objects outside its table (it says so), hold a conversation, or
know anything beyond the nine attributes. The facts come from the teacher model and have not been
checked by a person.

## Run it yourself

```bash
pip install torch fastapi uvicorn
cat data/records-r1.jsonl data/records-r2.jsonl > data/all.jsonl
python -m haishool.app --model model/haishool-objects-v2.pt --records data/all.jsonl --port 8650
```

Train your own (the teacher needs [ollama](https://ollama.com) with `qwen2.5:14b-instruct-q4_K_M`):

```bash
python -m haishool.teacher objects --out data/objects.txt
python -m haishool.teacher records --objects data/objects.txt --out data/records.jsonl
cat data/records.jsonl data/identity.jsonl > data/all.jsonl
python -m haishool.student train --records data/all.jsonl --out runs/full
```

## Files

| Path | Content |
|---|---|
| `haishool/` | schema, teacher, model, student training, translator, chat app |
| `model/` | weights of versions 1 and 2 (half precision) and their training and evaluation reports |
| `data/` | the training facts, one JSON line per object, per data round |
| `docs/` | the website (landing page with chat, data browser) |
| `LESSONS.md` | what we learned before this, and why it is built this way |

## Licence

Code: MIT (`LICENSE`). Model weights and data: CC BY 4.0. The facts were generated with
Qwen2.5-14B-Instruct (Apache-2.0).
