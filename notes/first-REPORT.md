# Comparing input-encoding schemes for a minimal character-level language model

*yarns-and-loss experiment report — 2026-08-19*

## 1. Objective

We hold a small language model and its training regime fixed and vary **only the
token input embedding** — the matrix that maps each character id to its initial
vector. The goal is to measure how much, and how quickly, a model's learning
depends on *how the input is encoded*, using **train/validation loss curves** as
the response variable. A specific interest is whether **homeosemi** random-indexing
(RI) encodings, which carry distributional co-occurrence structure, give the model
a measurable advantage over structureless or learned embeddings.

## 2. Experimental setup

**Model.** Decoder-only Transformer (nanoGPT): 4 layers, 4 heads, embedding width
`n_embd = 128`, context length 128, no bias terms, dropout 0. Total ≈ 0.80 M
parameters.

**Data.** Character-level *tiny-shakespeare* (1,115,394 characters; 65 distinct
characters). 90/10 train/val split (≈1.00 M / 0.11 M characters). Distributional
encodings are built from the first 200,000 training characters with a co-occurrence
window of ±5.

**Training.** AdamW (lr 1e-3, β = (0.9, 0.99), weight decay 0.1), batch size 32,
1,500 iterations, seed 1337. Loss is next-character cross-entropy in nats
(perplexity = eᴸ). Train/val loss estimated every 150 iterations over 20 minibatches.

**Two controls that make the comparison valid:**

- **Untied output head.** nanoGPT ties the input embedding to the output projection
  (`wte.weight == lm_head.weight`). We untie them so that freezing the input
  embedding does not also freeze the output layer.
- **Scale matching.** Every injected matrix is L2-normalised per row, then rescaled
  so its per-element std (~0.02) matches nanoGPT's default init. Differences between
  runs therefore reflect the *structure* of the encoding, not its magnitude.

With the input embedding frozen (all runs except `learned`), `vocab × n_embd =
65 × 128 = 8,320` parameters are held out of training. Trainable-parameter counts:
**0.821 M** (learned) vs **0.812 M** (all frozen runs).

## 3. Conditions

| Run | Frozen | Description |
|---|---|---|
| `learned` | no | Baseline. Embedding initialised `N(0, 0.02)` and trained jointly with the model. The only run whose input embedding receives gradients. |
| `random` | yes | Dense Gaussian matrix, row-normalised and scaled. Near-orthogonal, carries **no** linguistic information. Isolates how well the model copes with an arbitrary fixed input basis. |
| `onehot` | yes | Each character is a scaled standard basis vector `eᵢ` (`n_embd ≥ vocab`). Exactly orthogonal, equal-norm — maximally distinguishable, zero shared structure. |
| `homeosemi_index` | yes | RI *index* vectors from homeosemi: sparse **ternary** vectors (10 non-zeros per 128 dims, 5×+1 / 5×−1). Structureless like `random`, but sparse rather than dense. |
| `homeosemi_context` | yes | RI *context* vectors: `c(x) = Σ_positions Σ_{y in ±5 window} index(y)`. Characters in similar neighbourhoods get similar vectors — classic Random Indexing distributional semantics at char level. **Carries corpus structure.** |
| `homeosemi_context_ordered` | yes | As above, but each neighbour's index vector is passed through a fixed random permutation depending on whether it occurs *before* or *after* the target — an order-aware / directional RI encoding. |

*Implementation note:* homeosemi's `sparsevectors.permute()` aliases and mutates its
input in place, so the ordered encoding permutes a **dense copy** of each index
vector rather than calling that function.

## 4. Results

Validation loss (nats) by iteration:

| iter | learned | random | onehot | h_index | h_context | h_ctx_ordered |
|---:|---:|---:|---:|---:|---:|---:|
| 0    | 4.180 | 4.209 | 4.187 | 4.171 | 4.130 | 4.211 |
| 150  | 2.454 | 2.451 | 2.440 | 2.470 | 3.056 | 3.360 |
| 300  | 2.242 | 2.274 | 2.169 | 2.238 | 2.752 | 2.922 |
| 600  | 2.010 | 2.008 | 1.947 | 1.969 | 2.389 | 2.598 |
| 900  | 1.863 | 1.853 | 1.814 | 1.847 | 2.196 | 2.418 |
| 1200 | 1.766 | 1.759 | 1.731 | 1.762 | 2.069 | 2.260 |
| 1500 | 1.702 | 1.705 | **1.690** | 1.708 | 1.978 | 2.144 |

Final results:

| Encoding | Final train | Final val | Val perplexity |
|---|---:|---:|---:|
| **onehot** | 1.498 | **1.690** | 5.42 |
| learned | 1.518 | 1.702 | 5.48 |
| random | 1.513 | 1.705 | 5.50 |
| homeosemi_index | 1.514 | 1.708 | 5.52 |
| homeosemi_context | 1.859 | 1.978 | 7.23 |
| homeosemi_context_ordered | 2.092 | 2.144 | 8.53 |

Figure: `experiments/runs/comparison.png` (solid = val, dashed = train).

## 5. Discussion

**Two clean groups emerge.**

1. **The four near-orthogonal encodings cluster tightly** (val 1.69–1.71). Frozen
   `onehot` is the best of all six — beating the trainable `learned` baseline — and
   frozen `random` ties it. At char level the input embedding is **not a
   bottleneck**: the model learns its own layer-1 projection regardless, so a fixed
   full-rank basis it cannot adapt performs as well as a trainable one.

2. **Both distributional encodings are markedly worse** (val 1.978 and 2.144), with
   the ordered variant worst. They also *start* far slower (iter 150: 3.06 and 3.36
   vs ~2.45 for the rest) and never catch up — a cold start, not a warm one.

**Why distributional structure backfires here.** The context encodings deliberately
make co-occurring characters *similar* (small pairwise angle), which collapses the
effective rank of the input space — many characters point in nearly the same
direction. The model must spend capacity *un-collapsing* symbols before it can
distinguish them. For next-character prediction over only 65 symbols, what matters
is **distinguishability**, not semantic similarity; orthogonal/random bases maximise
it. The ordered variant spreads each neighbour across permuted subspaces, adding
correlation/noise — hence worst.

## 6. Conclusion and next steps

At the character level the input encoding barely matters, and distributional
encodings can actively hurt. This is not evidence against homeosemi in general — it
is the expected consequence of a tiny symbol set where distinguishability dominates.

homeosemi's distributional encodings should only pay off where (a) the vocabulary is
large enough that an orthogonal basis is expensive or infeasible at small `n_embd`,
and (b) similarity between tokens is genuinely predictive — i.e. a **word-level**
model. That is the natural next experiment; the harness (freeze / untie / curve
logging) carries over directly, needing only word-level data prep and word-level
homeosemi encoders.

## Appendix: reproduction

```bash
VP="C:/Users/jkarlgre/venvs/yarns-and-loss/Scripts/python.exe"
CFG="--n_layer 4 --n_head 4 --n_embd 128 --block_size 128 --batch_size 32 \
     --max_iters 1500 --eval_interval 150 --eval_iters 20"
"$VP" experiments/train_experiment.py --encoding learned --tag learned $CFG
for ENC in random onehot homeosemi_index homeosemi_context homeosemi_context_ordered; do
  "$VP" experiments/train_experiment.py --encoding "$ENC" --tag "$ENC" --freeze $CFG
done
"$VP" experiments/plot_curves.py
```

See the `run-encoding-experiment` skill for environment details (venv location, SSL
workaround, adding new encodings).
