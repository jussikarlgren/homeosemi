# Student project notes: Curriculum learning and learning-rate restarts in a tiny language model

*yarns-and-loss project — a starting point for a term project*

## The big idea, in one sentence

If you train a small language model and periodically "restart" its learning rate (jump
it back up, then let it decay again), that restart only helps **if it happens at the
same moment the training data gets harder** — otherwise it just knocks the model around
for no reason.

That one sentence hides a surprising amount of structure, and it's the kind of thing you
can actually test yourself on a laptop in an afternoon, no GPU required.

## Background you need

We train a deliberately *tiny* GPT (about 4.6M parameters) on **BabyLM**, a corpus meant
to resemble the language a child actually hears — a lot of simple, repetitive
child-directed speech, plus some harder text. Because the model is small and the data is
sometimes tiny (we often cap training at 100,000 tokens on purpose), it *overfits* hard.
The low-data regime is where all the interesting effects show up, because that's where
what you *assume* about language matters more than what you can *measure* from data.

Two ideas interact in this project:

1. **Curriculum ordering.** Instead of shuffling the training data, sort it from simple
   utterances to complex ones (we score complexity by how rare a sentence's content
   words are). The model sees "baby talk" first and hard sentences last — like a child.

2. **Warm restarts.** Split training into N equal phases and reset the learning rate at
   each phase boundary, cosine-decaying within each phase. The hope: give the model
   fresh "plasticity" right when harder sentences start arriving.

## What we already found (so you don't repeat it)

- **Warm restarts alone are not free.** With lots of data, restarts make things *worse*
  than a plain schedule. The benefit only appears when restarts line up with a curriculum
  (i.e. with the data actually changing). We proved this with a control run — restarts
  with no curriculum lose.
- **Curriculum helps a plain model but hurts a model that already has a grammatical
  prior.** They're *substitutes*: both constrain the model early, so doing both
  double-counts and backfires at low data.
- **Seeds matter enormously here.** An early "best result" evaporated when we re-ran it
  with three random seeds and a fair validation set. At 100k tokens, the noise between
  seeds is as big as the effects we care about. **Always run ≥3 seeds and report the
  spread.** This is the single most important habit to take from the project.

## Project challenges you could take on

Pick one; each is a real open question, not a solved exercise.

1. **Staged unfreeze (the natural next step).** The grammatical prior and the curriculum
   fight because both are "on" the whole time. What if you *freeze* the structural part
   of the model during the simple phase, then *unfreeze* it exactly at the complexity
   boundary — so the prior scaffolds early and the model adapts late? Does that let the
   two mechanisms finally cooperate instead of cancel?

2. **Make "distribution shift" measurable.** We aligned restarts to curriculum phases by
   hand. Can you *detect* the moment the data distribution shifts (e.g. a running measure
   of vocabulary novelty or perplexity spike) and trigger a restart automatically? Does a
   data-driven restart schedule beat fixed phases?

3. **How many phases, and where?** We used 3 equal phases. Are equal phases optimal? Try
   uneven phase lengths that match the actual complexity distribution of the corpus. Plot
   BPC vs. phase count and phase placement.

4. **Does the effect survive scale?** Everything here is at ~4.6M parameters and 100k
   tokens. Re-run at 2× or 4× model size, or 300k tokens, and find the point where
   curriculum + restarts stop helping. (We have evidence from a related experiment that
   more capacity does *not* automatically rescue these effects — a great thing to test
   yourself.)

5. **A different complexity metric.** We rank sentences by mean log-rank of content
   words. Try syntactic depth, sentence length, or a dependency-based measure. Does the
   curriculum's effect depend on *how* you define "simple"?

## What you'd build and measure

- Reuse the existing harness (`experiments/train_experiment.py`) — it already supports
  `--n_phases`, `--max_train_tokens`, curriculum data dirs, and per-category freezing.
- Your deliverable is a **seed-banded plot**: BPC (bits per character) vs. your variable,
  mean ± std over at least 3 seeds, with a clearly labelled baseline.
- A good writeup states a hypothesis *before* the run, shows the control that could
  falsify it, and is honest when the result is "no effect" — which, in the low-data
  regime, is often the most trustworthy finding.

## Why this is worth your time

You will learn, concretely, that a training trick is not good or bad in the abstract —
it's good or bad *relative to what else the model already knows*. That's a lesson that
scales all the way up to frontier models, and you can discover it on a laptop.
