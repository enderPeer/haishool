# What we learned before Haishool

Haishool is the minimum viable start that came out of a longer project: training small language
models from scratch on a home GPU cluster. These are the lessons that shaped it. Every number was
measured in that project; nothing here is estimated.

## 1. Fluent is not the same as knowing

A 210-million-parameter model trained for 24 hours on 4.6 billion tokens of broad English text
(web, encyclopedia, synthetic textbooks) predicted text clearly better than its predecessor (0.13
to 0.19 bits per character lower on every held-out source). It writes grammatical English. It
still drifts off topic after a few words and invents facts. Better text prediction did not turn
into knowledge.

## 2. A small model does not pick up facts from prose

We repeated about 400 facts some 200 times each, written in many different ways, inside the
training text. Afterwards the model answered questions about facts that had only appeared in
prose correctly 4 times out of 392, against 2 without the extra text. Facts that were also trained
as question and answer pairs rose from 12 to 35 out of 390. Repetition in running text was not
enough.

## 3. Fine-tuning teaches form, not knowledge

Chat fine-tuning taught the models to answer in the right format and to stop. It added almost no
knowledge, and it cost some: the language the model had learned before got measurably worse unless
old text was mixed back in (replay). One fine-tune broke the limit we had set for that loss by
nearly three times; with 40 % replay it stayed just inside.

## 4. The architecture matters more than the clock

With the same 30 minutes on comparable cards, a model reading byte-pair tokens with rotary
positions reached 1.21 bits per character; the older model reading single characters with learned
positions reached 1.50 in 38 minutes. In a longer comparison at equal steps, rotary positions
beat learned ones (0.951 against 0.972) but ran about a quarter slower per token.

## 5. Looking things up beats remembering

A plain retrieval system that quotes the best sentence from a source found the right article for
90 % of development questions and answered 69 % of them correctly, without generating anything.
No trained model came close on facts.

## 6. Separate the facts from the language (the Haishool idea)

If the model only has to learn facts in a compact, regular form, it learns them: Haishool's
12-million-parameter model reproduces every trained fact (1,000 of 1,000 sampled), and when ten
percent of the facts were held back it guessed 27 % of them exactly from similar objects. The
English comes from fixed rules that cannot add anything the model did not say, so a wrong answer
stays visibly wrong instead of being dressed up.

## 7. Measure as if you wanted to be proven wrong

- Write down what counts as success before looking at results.
- Keep one test set sealed and open it once.
- Count every attempt, including the failed and the duplicate ones; a search over many variants
  finds something by chance.
- Report failures with the same care as successes; most of this list comes from failures.

## 8. Small practical things that cost hours

- Check the time zone of every data source: one feed we assumed to be in New York time was in
  UTC, which would have shifted every session-based result by five hours.
- A training script that freezes "the checkpoint at the last step" fails when the last step is not
  a checkpoint step. Fall back to the final state.
- Watch disk space and memory per worker before a long run, not during it.
- Good, simple, consistent data beats a lot of mixed data (the TinyStories result: tiny models
  write coherent stories when the training text is simple and consistent).
