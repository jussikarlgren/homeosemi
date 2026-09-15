# Student project notes: How you encode words changes what a small model can learn

*yarns-and-loss project — a starting point for a term project*

## The big idea, in one sentence

Before a language model does anything clever, it has to turn each word into a vector —
and the *way* you choose those starting vectors can matter as much as the model
architecture, especially when data is scarce.

## Background you need

Every token (word or character) enters a language model as an **embedding**: a vector.
Normally these are random and learned during training. But you don't *have* to learn
them — you can hand the model a fixed, structured encoding that already reflects
something you know about language. This project asks which structures help, and when.

We use a deliberately tiny word-level GPT (~4.6M parameters) on **BabyLM** (child-language
text), and we often cap the data at 100,000 tokens on purpose, because the scarce-data
regime is where your *assumptions* about language matter most.

The key linguistic idea: **the lexicon is not uniform.** Words fall into functional
categories that behave differently in grammar. We use five:

| category | examples | what we do with it |
|---|---|---|
| referential | pronouns, nouns, adjectives | learn (trainable) |
| structural | prepositions, conjunctions, determiners | **freeze** (grammatical scaffold) |
| predicative | verbs, auxiliaries | learn |
| clause-adverbial | negation, modals, hedges | **freeze** |
| residual | punctuation, unknown words | learn |

By *freezing* the "grammatical machinery" categories, we stop the model from overwriting
them with noise when data is scarce — they act as a fixed scaffold.

## What we already found (so you don't repeat it)

- **Structured beats learned at low data — robustly.** At 100k tokens, the category-aware
  encoding beats a plain learned embedding by ~7% bits-per-character, averaged over 3
  seeds, with no overlap between the seed groups. The mechanism is *regularisation*: it
  stops the tiny model from overfitting.
- **The advantage has a crossover.** By ~500k–800k tokens the learned baseline catches
  up, and past ~1M tokens the fixed prior actually *hurts*. Structured priors are a
  low-data tool.
- **Small alphabets prefer distinguishability over similarity.** At the character level
  (only 65 symbols), encodings that make similar characters point in similar directions
  *hurt* — they "collapse the rank" of the input and the model wastes effort separating
  symbols again. Orthogonal/random encodings do better there.
- **Coarse categories are the sweet spot.** Splitting "structural" into finer
  sub-categories (prepositions vs. determiners vs. conjunctions) *hurt* at this scale —
  finer distinctions need more data.

## Project challenges you could take on

Pick one; each is genuinely open.

1. **Find the crossover precisely, and explain it.** We know the structured prior wins
   below ~500k tokens and loses above ~1M. Map the crossover curve carefully (say, 50k /
   100k / 200k / 400k / 800k tokens) and build a simple theory that *predicts* where it
   sits from properties of the corpus (vocabulary size, Zipf slope). Bonus: does the
   crossover move if you change model size?

2. **Design a better freeze policy.** We freeze two categories for the whole run. Should
   the freeze be *gradual* — frozen early, released once the model has seen enough? (This
   connects directly to the sibling "curriculum + restarts" project.) Measure whether a
   scheduled unfreeze beats a permanent freeze.

3. **Which distinctions actually earn their weight?** We found coarse categories help but
   fine ones don't, *at this scale*. Re-open that at larger data or model size: is there a
   data budget where the finer split finally pays off? Build the curve.

4. **A new structured encoding of your own.** Try encoding a different linguistic
   dimension — e.g. tense/aspect, animacy, or argument structure — as a frozen overlay and
   test it the same way. Does your chosen distinction help in the low-data regime, and
   does it also show a crossover?

5. **Capacity vs. prior.** A companion experiment found that making the model *bigger*
   did not dissolve a representational trade-off — if anything it sharpened it. Test the
   analogous claim here: does a bigger model make the structured-encoding advantage
   disappear (because it can now learn the structure itself), or persist? Predict first,
   then run.

## What you'd build and measure

- The harness (`experiments/train_experiment.py`, `experiments/encoders.py`) already
  supports the encodings, per-category freezing, an overlay strength `--overlay_alpha`,
  and data-budget caps. Adding a new encoding means writing one function in `encoders.py`.
- Your deliverable is a **seed-banded BPC plot** (mean ± std over ≥3 seeds) against your
  chosen variable, with the learned baseline clearly marked. Bits-per-character is the
  metric because it's comparable across tokenizations.
- State a hypothesis *before* you run, include a control that could prove you wrong, and
  report null results honestly — in the low-data regime the honest null is often the real
  finding.

## Why this is worth your time

You'll get hands-on with one of the deepest ideas in machine learning — **inductive
bias** — in a setting small enough to fully understand. You choose what the model
"believes" about language before it sees a single sentence, and you get to watch, in
clean numbers, exactly when that belief helps and when the data outgrows it.
