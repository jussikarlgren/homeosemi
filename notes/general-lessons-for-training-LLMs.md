# General lessons for training language models

*Drawn from the yarns-and-loss encoding experiments, with the child-language-acquisition
framing deliberately removed. These are architecture-agnostic heuristics the setup
illustrates cleanly — see the scope caveat at the end.*

## Headline: a hand-built inductive bias behaves like data — so schedule it, don't fix it

Structure you inject into the representation (typed embeddings, category/frame priors,
frozen scaffolds) is a **regularizer that substitutes for data**. It helps below a data
crossover and turns into a handicap above it: you trade variance reduction for
approximation error, and once there is enough data the error you baked in costs more than
the variance it saved. In these experiments a structured input prior bought ~7–10% at the
lowest data budget, shrank to nothing by ~0.5–1M tokens, and became a small net handicap
beyond.

The non-obvious part: **the crossover is not destiny.** Freezing the structure early and
*releasing* it partway through training kept essentially all of the low-data gain while
shedding most of the high-data cost — nearly no-regret. So the general move is to
**warm-start with a prior/constraint and decay its influence as data accumulates**, rather
than commit to it permanently.

**Caveat with teeth:** even after fully releasing the constraint, a small high-data
residual remained. The *initialization biased the final basin even at fixed compute* —
where you start still matters asymptotically, not just early. A structured init is not free
even when it is fully trainable.

## Four supporting learnings

1. **Regularizers that fix the same under-determination don't stack — they substitute and
   can interfere.** A structured init and a curriculum each helped alone but hurt together;
   two different "structure" priors gave the *same* gain, not the sum. Don't assume ablation
   wins add up: when several tricks all just constrain a data-starved model, you are
   double-counting, and stacking can go negative. Budget your inductive bias.

2. **Perplexity/loss gains ≠ capability gains.** The prior reliably lowered LM loss but did
   not move targeted evaluation at all — the improvement was pure distributional smoothing,
   not competence. A very general trap: a change that improves your headline loss can leave
   the thing you actually care about untouched. Gate decisions on targeted capability evals,
   not loss deltas.

3. **Schedule discontinuities only pay off when aligned to a real distribution shift in the
   data.** Learning-rate restarts / phase resets helped *only* when the reset coincided with
   genuinely new data arriving; on a fixed clock they were a cost, not free re-exploration.
   Relevant to multi-domain / data-mixture pretraining: align schedule changes to mixture
   transitions, don't sprinkle them on a timer.

4. **Two methodology lessons (each cost a real "finding" here).**
   - *Seed variance manufactures results in noisy regimes.* Small models, low data, and
     near-floor evals all inflate run-to-run variance; several headline effects evaporated
     under three seeds. Report bands, and treat effect sizes near the run-to-run variance as
     nothing.
   - *Eval-infra bugs are silent and metric-selective.* A checkpoint loader that mis-tied
     weights corrupted a whole class of results — and it hit *absolute-likelihood* metrics
     hard while *ranking/accuracy* metrics survived almost intact. Always verify a reloaded
     checkpoint reproduces its training-time metric before trusting any downstream eval, and
     know that ranking and likelihood metrics can fail independently.

## Scope caveat

This is a tiny, word-level, CPU-scale setup (a ~4.6M-parameter nanoGPT on ≤~9M tokens).
Treat the above as clean *illustrations* of general heuristics and as hypotheses worth
testing at real scale — not as laws established at scale.

The single most worth-testing claim is the headline one: **does "anneal the prior" stay
no-regret when the crossover is pushed out by model size** — i.e., does a structured
initialization that is frozen early and released later become a free early-training
accelerator rather than a low-data-only trick? That, plus a crossover-vs-model-size scaling
law, would decide whether any of this belongs in a real training regime.
