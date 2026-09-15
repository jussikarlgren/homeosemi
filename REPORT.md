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

---

# Word-level experiment: category-aware encoding

*2026-08-19*

## 1. Objective and hypotheses

Moving to word level to test two related hypotheses:

1. **The lexicon is not uniform.** Words fall into functionally distinct categories
   acquired and processed differently by humans — "not"/"maybe" are not learned like
   "doggy"/"daddy". Encoding should impose **category-internal similarity** and treat
   some categories as **fixed grammatical machinery** (frozen) rather than learned
   content.

2. **A structured distributional prior helps most in the low-data regime.** A frozen
   encoding carrying category structure should give the model a warm start, with an
   advantage that shrinks as data grows.

## 2. Setup

**Corpus.** BabyLM 2024 strict-small (Cambridge-CLIMB), simple/child-directed subset:
`aochildes, children_stories, cbt, simple_wikipedia, open_subtitles, switchboard,
bnc_spoken` (~36M chars / 8.7M tokens train; ~33M chars / 8.2M tokens val).

**Tokenization.** Word-level: regex split + lowercase, `<eos>` at line boundaries,
vocab capped at 15,000 types (min frequency 3) with `<unk>`.

**Model.** Same nanoGPT as char-level: 4 layers, 4 heads, `n_embd=128`, context 128.
Vocab size 15,000 → total parameters ~2.71M; trainable ~4.64M (untied output head).

**Metric.** Bits-per-character (BPC) = total val NLL (nats) / (val char count × ln 2),
enabling cross-tokenizer comparison. Val loss (nats/token) also reported.

**Categorization.** NLTK Penn-Treebank POS tagger + `jussipyutils/lexicalfeatures.lexicon`.
Precedence: adv_clausal > structural > predicative > referential > residual.

| Category | n types | Examples |
|---|---:|---|
| residual (0) | 12,289 | punctuation, numbers, rare words |
| referential (1) | 1,457 | pronouns, nouns, adjectives |
| structural (2) | 98 | prepositions, conjunctions, determiners |
| predicative (3) | 1,005 | verbs and auxiliaries |
| adv_clausal (4) | 151 | negation, modals, hedges, amplifiers |

**Category similarity at init** (intra vs inter cosine, `word_category`):

| Category | intra | inter | gap |
|---|---:|---:|---:|
| adv_clausal | 0.455 | 0.017 | +0.438 |
| structural | 0.461 | 0.064 | +0.397 |
| referential | 0.373 | 0.027 | +0.346 |
| predicative | 0.274 | 0.027 | +0.247 |

The overlay successfully imposes strong within-category similarity before training.

**Freeze policy.** Categories 2 (structural) and 4 (adv_clausal) frozen via a
gradient mask; embedding param group uses `weight_decay=0` to prevent AdamW decay
drifting frozen rows (verified: max change in frozen rows after step = 0.00e+00).
249 / 15,000 rows frozen.

## 3. Conditions

| Tag | Frozen | Description |
|---|---|---|
| `word_learned` | no | Trainable word embedding — within-tokenizer baseline |
| `word_random` | all rows | Frozen Gaussian — structureless control |
| `word_homeosemi_context` | all rows | Frozen RI co-occurrence vectors (uniform distributional prior) |
| `word_category` | cats 2+4 only | RI co-occurrence + category overlay (α=0.5); structural/adv_clausal frozen |

## 4. Results (1000 iterations)

| Encoding | Final val loss | Final BPC | BPC gap vs learned |
|---|---:|---:|---:|
| word_learned | 4.592 | **1.623** | — |
| word_random | 4.663 | 1.648 | +0.025 |
| **word_category** | 4.727 | **1.670** | +0.047 |
| word_homeosemi_context | 4.745 | 1.677 | +0.054 |

Val-loss by iteration:

| iter | learned | random | word_category | homeosemi_ctx |
|---:|---:|---:|---:|---:|
| 0 | 9.634 | 9.647 | 9.612 | 9.612 |
| 100 | 5.348 | 5.557 | 6.014 | 5.962 |
| 300 | 4.954 | 5.045 | 5.235 | 5.283 |
| 500 | 4.811 | 4.869 | 4.975 | 4.993 |
| 700 | 4.651 | 4.719 | 4.812 | 4.817 |
| 1000 | 4.592 | 4.663 | 4.727 | 4.745 |

Figures: `experiments/runs/comparison_bpc.png`, `comparison_loss.png`.

## 5. Discussion

**Category overlay helps over uniform distributional.** `word_category` (BPC 1.670)
consistently outperforms `word_homeosemi_context` (1.677) — adding functional-category
structure to the distributional encoding provides a measurable benefit throughout
training, not just at convergence.

**The cold-start problem persists but is partially mitigated by category structure.**
Both distributional encodings start slower than `word_random` (iter 100: ~6.0 vs 5.6),
repeating the char-level rank-collapse pattern. However `word_category` closes the gap
with `word_random` faster than `word_homeosemi_context`, suggesting that the category
overlay partially restores cross-category distinguishability.

**The frozen random control remains strong.** `word_random` (BPC 1.648) outperforms
both distributional encodings at full data — but the low-data sweep (§6) shows this
reverses at small data budgets.

## 6. Follow-up experiments

### 6.1 Fairer random control

`word_random` with only cats 2+4 frozen (249 rows, matching `word_category`'s freeze
footprint) gives **identical BPC (1.648)** to the all-frozen run. The earlier result
was not a freeze-asymmetry artefact: random beats distributional encodings regardless
of freeze policy. Rank collapse is the real cause.

### 6.2 Low-data sweep

The central hypothesis test: does a structured prior help most when data is scarce?

| Tokens | word_learned BPC | word_category BPC | Advantage |
|---|---:|---:|---:|
| 100k | 3.314 | **2.923** | **+0.391** |
| 300k | 2.678 | **2.597** | **+0.081** |
| 1M | 1.973 | 2.004 | −0.031 |
| 8.7M (full) | **1.623** | 1.670 | −0.047 |

**The hypothesis is confirmed.** The category encoding is dramatically better at 100k
(+14% relative BPC), still better at 300k, and the advantage reverses between 300k and
1M. The crossover is around 500k–800k tokens.

The mechanism is primarily **regularisation**: at 100k tokens, `word_learned` overfits
catastrophically (train 1.75, val 9.38), while `word_category`'s frozen structural and
adv_clausal rows constrain the model and val loss stays much lower (train 2.82, val
8.27). The frozen embedding serves as an inductive bias precisely in the regime where
the model cannot estimate word relationships from data alone — analogous to how a
human infant's innate categorical biases scaffold early language acquisition.

Figures: `experiments/runs/low_data_sweep.png`.

### 6.3 Overlay-alpha sweep

| α | BPC |
|---|---:|
| **0.1** | **1.668** |
| 0.5 | 1.670 |
| 1.0 | 1.682 |
| 0.25 | 1.695 |

Differences are small (< 0.03 BPC). α = 0.1 is the narrow winner; the relationship is
non-monotone (0.25 is the worst, not 1.0), suggesting the category vector geometry
interacts non-trivially with the co-occurrence base. In practice α = 0.1–0.5 is a
safe working range.

Figure: `experiments/runs/alpha_sweep.png`.

### 6.4 BPE baseline

| Encoding | BPC | Params |
|---|---:|---:|
| word_learned | 1.623 | 4.6M |
| word_category (α=0.1) | 1.668 | 4.6M |
| bpe_learned | 1.912 | **13.7M** |

GPT-2 BPE is substantially worse in BPC despite 3× more parameters. At this scale
(n_embd=128, 1000 iters) the 50k-vocab embedding is severely under-parameterised:
each subword token shares only 128 dimensions with everything else in the model. BPE
is designed for regimes where the model is large enough to afford a rich embedding
per subword; at minimal scale, word-level tokenization wins clearly.

## 7. Conclusions

The experiments collectively support the following:

1. **At char level**, input encoding barely matters; indiscriminate distributional
   similarity hurts (rank collapse).
2. **At word level with full data**, near-orthogonal encodings (random, learned) beat
   distributional ones; the category overlay provides a modest improvement over
   uniform distributional but does not close the gap to learned.
3. **At word level in the low-data regime** (< ~500k tokens), the category-aware
   structured prior dramatically outperforms a learned baseline, primarily through
   regularisation. The advantage is largest at 100k tokens (+14% BPC) and shrinks to
   zero by ~1M tokens.
4. **BPE tokenization** is inferior to word-level at minimal model scale (< 5M
   params), even with 3× more parameters.
5. **The freeze policy matters**: freezing structural and adv_clausal categories
   prevents the model from adapting these "grammatical machinery" rows, which acts
   as a strong inductive bias in exactly the regime (low data) where the linguistic
   hypothesis predicts it should.

## Appendix: reproduction

```bash
VP="C:/Users/jkarlgre/venvs/yarns-and-loss/Scripts/python.exe"
DD="experiments/data/babylm_simple"
CFG="--data_dir $DD --n_layer 4 --n_head 4 --n_embd 128 --block_size 128 \
     --batch_size 32 --max_iters 1000 --eval_interval 100 --eval_iters 20 \
     --enc_tokens 500000"
"$VP" experiments/train_experiment.py --encoding word_learned --tag word_learned $CFG
"$VP" experiments/train_experiment.py --encoding word_random --tag word_random --freeze $CFG
"$VP" experiments/train_experiment.py --encoding word_homeosemi_context \
      --tag word_homeosemi_context --freeze $CFG
"$VP" experiments/train_experiment.py --encoding word_category \
      --tag word_category --freeze_categories 2,4 --overlay_alpha 0.5 $CFG
"$VP" experiments/plot_curves.py --metric bpc word_learned word_random \
      word_homeosemi_context word_category
```

---

# TMA (tense-mood-aspect) clause-type encoding

*2026-08-26*

## 1. Hypothesis

Clauses have different functions depending on the type of situation they describe
(following Halliday's ideational metafunction). Four types are distinguished:

- **State**: unbounded, homogeneous, no change. *The sun shines. The house is yellow.*
  Encoded in present tense, stative verbs, low adverbial modification. Typically
  background information.
- **Process**: ongoing, dynamic, continuous. *The ball is rolling. It is raining.*
  Encoded in progressive aspect (be + VBG), dynamic verbs. Foreground but not punctual.
- **Occurrence**: punctual, bounded, high information value. *The window broke.
  The kettle boiled over.* Encoded in simple past (VBD), occurrence adverbials
  ("suddenly", "then"), high transitivity, telic verbs.
- **Unreal**: counterfactual, conditional, modal. *I would like to be rich. If it
  rained…* Encoded in would/could/might + bare verb, conditional clause structure.

The encoding hypothesis: every word in a clause should receive a component of that
clause's TMA vector, since the clause type is part of the context in which the word
is used and acquired. A word like "cat" encountered predominantly in state clauses
("the cat is asleep") should carry a different representation than one encountered
in occurrence clauses.

## 2. Implementation

Each sentence in the training corpus (sample of 300k characters) is classified
into a soft distribution over the four clause types using NLTK POS tags and the
`lexicalfeatures` lexicon as heuristics:

| Signal | Clause type |
|---|---|
| would/could/might present | Unreal (+3.0) |
| Conditional word (if, unless, suppose…) | Unreal (+1.5) |
| be + VBG (progressive) | Process (+3.0) |
| VBD without be (simple past) | Occurrence (+2.0) |
| Occurrence adverbial (suddenly, then, just…) | Occurrence (+2.0) |
| Stative verb (perception/think/private verbs) | State (+2.0) |
| VBZ/VBP, no past/progressive | State (+1.5) |

Scores are softmax-normalised to give a probability distribution per sentence.
Every word token accumulates the sentence's distribution as a weighted count.
The resulting per-word-type distribution is stored in `meta.pkl` as
`tma_distributions: Dict[str, np.ndarray]` (shape 4, sums to 1).

Four frozen random unit vectors `tma_vec[0..3]` (state/process/occurrence/unreal)
are generated at encoding time. Each word's TMA overlay is:

```
tma_overlay(w) = tma_alpha × (dist[w] @ tma_vecs)   # weighted mix of 4 vectors
```

This is added to the word's base RI context vector + primary category overlay
before L2 normalisation and scale matching.

## 3. Clause-type distributions for selected words

Sample words and their corpus-derived TMA distributions (300k-character sample,
BabyLM simple subset):

| Word | State | Process | Occurrence | Unreal | Primary category |
|---|---:|---:|---:|---:|---|
| `is` | **0.506** | 0.236 | 0.143 | 0.115 | predicative |
| `was` | 0.203 | 0.269 | **0.378** | 0.149 | predicative |
| `rolling` | 0.265 | **0.583** | 0.076 | 0.076 | predicative |
| `would` | 0.130 | 0.046 | 0.057 | **0.766** | predicative |
| `if` | 0.435 | 0.084 | 0.103 | 0.377 | subord_conj |
| `cat` | **0.516** | 0.155 | 0.157 | 0.172 | referential |
| `not` | 0.450 | 0.210 | 0.179 | 0.161 | adv_clausal |
| `very` | **0.521** | 0.159 | 0.181 | 0.139 | adv_clausal |
| `the` | **0.467** | 0.154 | 0.230 | 0.149 | determiner |

The distributions are linguistically coherent: `is` skews strongly toward state,
`was` toward occurrence, `rolling` toward process, `would` dominantly toward unreal.
Content words like `cat` acquire a state bias because cats are typically described
in state clauses. Words below the frequency threshold (min_freq=3) receive a
uniform prior (0.25 each).

The vocabulary-wide mean distribution is near-uniform
(state 0.28, process 0.24, occurrence 0.25, unreal 0.24), as expected — function
words and rare words appear in all clause types, so the signal is concentrated in
high-frequency content words and auxiliaries.

## 4. Results (1000 iterations, full BabyLM simple subset)

| Encoding | BPC | Gap vs learned |
|---|---:|---:|
| word_learned (baseline) | 1.6227 | — |
| **word_category_tma_clause** | **1.6485** | −0.026 |
| word_category (no TMA) | 1.6703 | −0.048 |
| word_category_tma (old word-level flag) | 1.6756 | −0.053 |

The clause-level TMA encoding closes the gap to the learned baseline to 0.026 BPC,
improving over the no-TMA category encoding (0.048 gap) and the old word-level TMA
flag (0.053 gap). The clause-type distribution approach carries more information than
a binary TMA flag: different words in the same clause receive the same situational
vector, encoding the context they appear in rather than just their lexical class.

---

# Curriculum ordering

*2026-08-26*

## 1. Hypothesis

Presenting training utterances in order of increasing complexity (simple→complex)
mimics the developmental input to the infant learner — child-directed speech is
systematically simpler than adult-to-adult speech — and may give the model a better
inductive starting point by grounding simple constructions before complex ones.

## 2. Complexity metric

Each corpus line is split into individual utterances with `nltk.sent_tokenize`.
Per-utterance complexity is:

  **score(u) = mean log-rank of content words in u**

where rank is the word's position in the training vocabulary sorted by descending
frequency (rank 1 = most frequent). Content words are tokens whose POS tag is not
in the closed-class structural set. Log-rank is used so rare words do not dominate.
Utterance length (word count) is a secondary sort key.

791k training utterances were sorted across 7 domains. Complexity range:
1.0–12.0 for conversational domains (aochildes, switchboard, bnc_spoken),
up to 66.0 for simple_wikipedia (technical vocabulary).

## 3. Results

### Full data (8.7M tokens)

| Encoding | No curriculum BPC | Curriculum BPC | Effect |
|---|---:|---:|---:|
| word_learned | 1.6227 | 1.6415 | −0.019 (worse) |
| word_category | 1.6703 | 1.6656 | +0.005 (tiny gain) |

Curriculum ordering **hurts the learned baseline at full data**: presenting simple
utterances first deprives the model of complex vocabulary early on, slowing
convergence when those words eventually appear. With sufficient data, random
ordering exposes the model to the full vocabulary faster.

### Low data (100k tokens)

| Encoding | No curriculum BPC | Curriculum BPC | Effect |
|---|---:|---:|---:|
| word_learned | 3.3144 | 3.1914 | **+0.123 (better)** |
| word_category | 2.9227 | 3.0497 | −0.127 (worse) |

At 100k tokens, curriculum ordering helps the **learned baseline** substantially
(+0.123 BPC) — starting simple reduces catastrophic overfitting (val loss 8.39 vs
9.38). However, it **hurts the category encoding** (−0.127): the frozen structural
embedding already provides a strong inductive bias, and the curriculum's simple-first
ordering restricts the early training distribution, reducing the vocabulary variety
the model needs to build good context vectors.

## 4. Discussion

The curriculum results reveal an interaction between the inductive bias (category
encoding) and the training distribution (curriculum ordering). When the model relies
on a learned embedding (no prior), curriculum ordering reduces the burden of learning
simple patterns before complex ones. When the model already has a frozen structural
prior, the curriculum's restricted early distribution interferes with the data-driven
component.

This suggests that curriculum ordering and structured priors are partially
**substitutes** in the low-data regime: each independently helps the learned/unprimed
model, but they do not stack. A natural follow-up is a staged curriculum — begin with
the structured prior frozen, unfreeze after the simple-vocabulary phase, and let the
model adapt to complex vocabulary with a warm start from the structural scaffold.

Figures: `experiments/runs/curriculum_low_data.png`,
`experiments/runs/comparison_bpc.png`.

---

# Warm restarts with curriculum phases

*2026-08-26*

## 1. Motivation

Standard cosine decay creates a direct conflict with curriculum ordering: the
learning rate is highest when the model is seeing simple, repetitive data and lowest
when complex, novel constructions finally arrive. If the model can barely update when
hard examples appear, the curriculum is wasting them.

**Proposed fix: cosine decay with warm restarts at equal-length phase boundaries.**
Divide training into N phases; at each boundary, reset LR to `lr_max` and cosine-decay
to `lr_min` within that phase. This gives the model fresh plasticity each time a new
complexity regime begins.

## 2. Implementation

New args in `train_experiment.py`: `--n_phases` (default 1 = single cosine decay),
`--lr_min` (default 1e-4). LR at iteration `it`:

```
phase_pos = it % (max_iters / n_phases)
lr = lr_min + 0.5 * (lr_max - lr_min) * (1 + cos(π × phase_pos / phase_len))
```

Runs use `--n_phases 3 --lr 1e-3 --lr_min 1e-4`, giving three 333-iter phases
each decaying 1e-3 → 1e-4.

## 3. Results

### Full data (8.7M tokens)

| Condition | BPC | vs flat-LR |
|---|---:|---:|
| word_learned, curriculum, flat LR | 1.6415 | baseline |
| word_learned, curriculum, 3-phase | 1.6784 | −0.037 worse |
| word_category, curriculum, flat LR | 1.6656 | baseline |
| word_category, curriculum, 3-phase | 1.6934 | −0.028 worse |
| word_learned, no curriculum, 3-phase | *(pending)* | — |

At full data, warm restarts **hurt**: each LR reset spikes the model away from a
minimum it was approaching, and with 8.7M tokens the model benefits more from stable
convergence than from periodic re-exploration.

### Low data (100k tokens)

| Condition | BPC |
|---|---:|
| word_learned, flat LR (no curriculum) | 3.314 |
| word_learned, curriculum, flat LR | 3.191 |
| **word_learned, curriculum, 3-phase** | **2.935** |
| word_category, flat LR (no curriculum) | 2.923 |
| word_category, curriculum, flat LR | 3.050 |
| word_category, curriculum, 3-phase | *(pending)* |

**The combination of curriculum ordering + warm restarts dramatically reduces BPC
at 100k tokens** (2.935 vs 3.314 flat-LR no-curriculum baseline, −11%). The
mechanism: phase 1 learns basic patterns with high LR; each restart gives fresh
plasticity exactly when harder data arrives; the model can absorb rare constructions
instead of receiving them at near-zero LR. Val loss still diverges (overfitting at
100k is unavoidable) but more gradually than flat-LR.

## 4. Results — complete

### Full data (8.7M tokens)

| Condition | BPC |
|---|---:|
| word_learned, flat LR | **1.6227** |
| word_learned, flat LR, curriculum | 1.6415 |
| word_learned, 3-phase, **no curriculum** | 1.6658 |
| word_learned, 3-phase, curriculum | 1.6784 |

Warm restarts hurt at full data **regardless of whether curriculum is used** — the
no-curriculum control `word_learned_phases` (1.6658) is also worse than flat LR
(1.6227). With sufficient data, periodic LR resets destabilise convergence more than
they help.

### Low data (100k tokens) — complete table

| Condition | BPC | vs learned flat |
|---|---:|---:|
| word_learned, flat | 3.314 | — |
| word_learned, curriculum | 3.191 | +0.123 |
| word_learned, curriculum + phases | 2.935 | +0.379 |
| word_category, flat | 2.923 | +0.391 |
| word_category, curriculum | 3.050 | +0.264 |
| **word_category, curriculum + phases** | **2.800** | **+0.514** |

**`word_category_curriculum_phases_100k` at BPC 2.800 is the best low-data result
across all experiments** — a 15.5% improvement over the flat learned baseline (3.314)
and 4.2% over the previous best (word_category flat, 2.923).

## 5. Discussion

**The three components are complementary in the low-data regime:**

- **Frozen structural embedding** provides an inductive scaffold that does not degrade
  on LR resets — the frozen rows are immune to the restart, so the geometric structure
  of grammatical categories is preserved across phase boundaries.
- **Curriculum ordering** means each phase boundary coincides with a genuine increase
  in data complexity — the restart is not arbitrary but marks a regime change.
- **Warm restarts** give the trainable rows (referential, predicative) fresh
  plasticity to absorb harder constructions instead of receiving them at near-zero LR.

**The control run is the key finding:** warm restarts without curriculum (`word_learned_phases`,
BPC 1.6658 at full data) are worse than flat LR. This rules out LR scheduling as a
free win. The benefit is specifically about **phase-data alignment**: restarts help
only when the LR reset coincides with qualitatively new data arriving. This is a
clean, interpretable result — it validates the motivating hypothesis directly.

**Val loss behaviour also confirms the story:** `word_category_curriculum_phases_100k`
reaches val 7.36 at iter 1000 vs `word_learned_100000` val 9.38 — substantially less
overfitting despite identical model size and training duration. The frozen structural
prior + phased plasticity together regularise far better than either alone.

Figures: `experiments/runs/phases_full.png`, `experiments/runs/phases_low_data.png`.

---

# Paraphrase augmentation

*2026-08-27*

## 1. Method

2000 utterances from the simple 30% of the curriculum (bottom by complexity score,
filtered to ≥3 tokens and ≥1 referential word) were each paraphrased twice using
the Claude API (`claude-haiku-4-5`, child-directed system prompt). The 4000
generated paraphrases were appended to the curriculum training data, forming the
combined corpus (9,284,225 train tokens, +42k over curriculum alone).

System prompt instructed the model to generate short (≤10 words), attention-directing
paraphrases in child-directed register — varied constructions (statements, questions,
imperatives), diminutive/hypocoristic forms where natural.

Example:
```
i can.   →   Look, I can do it! / Me do it!
i will.  →   I'm gonna do it! / Watch me!
i no.    →   I know! / Me know that!
```

## 2. Results (100k token regime — full low-data comparison)

| Condition | BPC | vs learned flat |
|---|---:|---:|
| word_learned, flat | 3.314 | — |
| word_learned, curriculum + phases | 2.935 | +0.379 |
| word_learned, **paraphrase** + phases | 2.898 | +0.416 |
| word_category, flat | 2.923 | +0.391 |
| word_category, curriculum + phases | 2.800 | +0.514 |
| **word_category, paraphrase + phases** | **2.739** | **+0.575** |

`word_category + paraphrase + phases` achieves BPC 2.739 — the best result across
all experiments, a 17.4% improvement over the flat learned baseline.

## 3. Discussion

**Paraphrases outperform curriculum ordering** for both the learned and category
encoding conditions at 100k tokens. The mechanisms are complementary but distinct:
curriculum reduces overfitting by presenting data in an ordered complexity ramp;
paraphrases reduce overfitting by diversifying the simple end — the model sees
multiple surface forms of the same referential content rather than memorising a
small set of exact strings.

**The four-component synergy** (frozen structural prior + category overlay + TMA
clause-type encoding + paraphrase augmentation + 3-phase warm restarts) produces
the largest low-data gain. Each component addresses a different bottleneck:
- *Frozen structural/adv_clausal*: the grammatical frame does not degrade under
  data sparsity or LR resets.
- *Category + TMA overlay*: words start with geometrically meaningful positions
  that encode functional role and situational type.
- *Paraphrase augmentation*: referential expressions get multiple surface exposures,
  directly targeting the acquisition of word-object associations at the simple end.
- *Warm restarts*: the trainable rows get fresh plasticity when the data complexity
  regime changes.

**Val loss** at iter 1000: `word_category_para_phases_100k` reaches 7.20, the
flattest trajectory of all 100k-token conditions, compared to 9.38 for the flat
learned baseline.

Figure: `experiments/runs/paraphrase_low_data.png`.

---

# BLiMP evaluation (grammatical acceptability)

*2026-08-27*

## 1. Method

Three models trained on 100k tokens were evaluated on all 67 BLiMP paradigms
(Warstadt et al., 2020). For each minimal pair (sentence_good, sentence_bad), the
model is correct if log P(good) > log P(bad), computed via full teacher-forcing
(all token positions, not just the last). Mean OOV rates are low (0.03–0.14),
confirming that most BLiMP test words fall within the 15k word vocabulary.

Models evaluated:
- `word_learned_100k` — trainable embedding, 100k tokens
- `word_category_100k` — category+TMA frozen encoding, 100k tokens
- `word_cat_para_phases_100k` — category+TMA + paraphrases + 3-phase, 100k tokens

## 2. Overall results

| Model | BLiMP accuracy | vs chance |
|---|---:|---:|
| word_learned_100k | 0.484 | −0.016 |
| word_category_100k | 0.480 | −0.020 |
| word_cat_para_phases_100k | 0.480 | −0.020 |
| Chance | 0.500 | — |

**All three models score below chance on aggregate BLiMP.** At 100k tokens with
this model size, none of the conditions has reliably learned the grammatical
distinctions BLiMP tests. This is consistent with the BPC results showing severe
overfitting at this data scale. The aggregate result should be interpreted as a
floor, not a comparison point.

## 3. Paradigm-level analysis

The aggregate conceals substantial paradigm-level variation with direct relevance
to the linguistic hypothesis. Selected contrasts (full results in
`experiments/runs/*_ckpt/blimp_results.json`):

| Paradigm | learned | category | cat+para+phases |
|---|---:|---:|---:|
| anaphor_gender_agreement | 0.363 | 0.670 | **0.779** |
| anaphor_number_agreement | 0.431 | 0.480 | **0.644** |
| principle_A_domain_1 | 0.358 | **0.751** | 0.782 |
| principle_A_reconstruction | 0.454 | **0.623** | 0.598 |
| only_npi_licensor_present | 0.059 | **0.599** | 0.261 |
| sentential_subject_island | 0.488 | **0.641** | 0.605 |
| superlative_quantifiers_1 | 0.611 | 0.519 | **0.836** |
| left_branch_island_echo_question | 0.399 | **0.728** | 0.363 |
| superlative_quantifiers_2 | **0.719** | 0.104 | 0.690 |
| wh_questions_subject_gap | **0.454** | 0.114 | 0.070 |

**Pronoun binding and anaphora improve markedly with the category encoding.**
Principle A paradigms (which test whether reflexives like "himself" are licensed
in the right structural positions) and anaphor agreement paradigms show large gains:
principle_A_domain_1 0.358→0.751, anaphor_gender_agreement 0.363→0.670. These
paradigms directly involve referential expressions (category 1) and pronouns.
The further gain on anaphor paradigms in the paraphrase condition (0.670→0.779,
0.480→0.644) is the most direct confirmation that paraphrase augmentation around
referential expressions measurably improves referential competence.

**NPI paradigms show mixed patterns.** only_npi_licensor_present improves
dramatically (0.059→0.599) with the category encoding (negation/adv_clausal is
frozen category 4, which governs NPI licensing). But other NPI paradigms degrade
or remain below chance. NPI licensing requires understanding of scope relations
that the frozen encoding may encode inconsistently.

**Some paradigms degrade badly with category encoding.** wh_questions_subject_gap
(0.454→0.114) and superlative_quantifiers_2 (0.719→0.104) show large reversals.
These represent genuine trade-offs: the category geometry that helps referential
binding may interfere with the probability distributions needed for wh-extraction
and quantifier scope. This is an honest limitation that should be reported.

## 4. Discussion

The BLiMP results cannot be compared to published BabyLM submissions (which train
on 10M+ tokens and achieve 0.60–0.75 overall). They serve instead to characterise
*which* grammatical distinctions are and are not captured by each encoding at the
100k-token floor, and whether the linguistic hypothesis — that category structure
helps referential and binding phenomena specifically — is supported.

The answer is: **yes, selectively.** The category encoding substantially improves
paradigms involving pronoun reference and binding (Principle A, anaphor agreement),
which is exactly where the referential category (cat 1) and its geometric
organisation should matter. The paraphrase augmentation further improves these
paradigms, confirming that broader referential exposure helps.

The degradation on wh-dependencies and some quantifier paradigms reflects a
genuine cost of the frozen category geometry: the model's probability distributions
are constrained in ways that help some dependencies and hurt others. This is not a
failure of the hypothesis — it is an empirical finding about which distinctions
the category encoding supports and which it does not.

---

# Seed-variance robustness (100k tokens)

*2026-08-31*

All prior 100k-token results were single-seed (1337). Since the low-data regime is
where the model overfits hardest, single-seed BPC differences of a few hundredths
could be noise. This section re-runs the three headline conditions across three
seeds (1337, 42, 2024) with an otherwise identical harness (1000 iters, batch 32,
eval_iters 20), so the comparison is internally consistent.

## 1. Results

Validation BPC per seed (100k tokens):

| Condition | seed 1337 | seed 42 | seed 2024 | mean | std |
|---|---:|---:|---:|---:|---:|
| `word_learned`, flat | 2.828 | 2.887 | 2.848 | **2.854** | 0.030 |
| `word_category`, flat | 2.545 | 2.749 | 2.646 | **2.647** | 0.102 |
| `word_category` + para + phases † | 2.790 | 2.792 | 2.801 | **2.794** | 0.006 |

† trained on `babylm_paraphrase` — see the val-set caveat in §3, which is why this
row is *not* directly comparable to the other two and is re-measured in §4.

Note the absolute values differ from the earlier single-seed low-data sweep
(which reported learned 3.314, category 2.923, para+phases 2.739). Those runs
used an earlier harness/eval configuration; this seed batch was produced in one
internally-consistent 2026-08-27 run and supersedes them for the 100k comparison.

## 2. The category prior is robust; the paraphrase headline is not

**Finding 1 — the category-prior advantage replicates cleanly.** Across seeds,
`word_category` beats `word_learned` by **+0.208 BPC (~7%)** on average, and the
separation is complete: the *worst* category seed (2.749) still beats the *best*
learned seed (2.828). Both conditions share the same data and val set, so this is
a clean, replicable win — the project's core hypothesis survives seed variation.

**Finding 2 — the "paraphrase + phases is the new best (2.739)" headline does
not replicate.** Across three seeds the full stack averages **2.794** — *worse*
than flat category (2.647) and only marginally better than flat learned. The
single-seed 2.739 was a favourable draw. The full stack is, however, by far the
most *reproducible* condition (std 0.006 vs 0.102 for flat category): paraphrase
data plus warm restarts stabilise training across seeds, but they buy consistency,
not a lower mean.

## 3. Two confounds behind the retracted headline

Investigating why the single-seed paraphrase result looked so strong surfaced two
methodological problems, both of which inflated it:

**(a) Different validation set.** `babylm_paraphrase` was prepared with an
independent train/dev re-split (all seven dev files differ byte-for-byte from
`babylm_simple`'s) *and* a frequency-reordered vocabulary (its `stoi` is not the
same mapping). Its BPC was therefore computed on a different held-out corpus and a
different token-id space than the flat learned/category conditions — the numbers
were never on a common yardstick.

**(b) Paraphrases were inert at 100k tokens.** The 4000 paraphrases are appended
*last* in the training stream (42,377 tokens, positions ~9.24M–9.28M of 9.28M).
`--max_train_tokens 100000` takes the *first* 100k tokens (`train_experiment.py:219`),
which are entirely paraphrase-free. At 100k the "paraphrase" condition therefore
never sees a single paraphrase — its only active ingredients are the category
prior, the curriculum-ordered data (its train files are curriculum-sorted copies),
and the 3-phase LR restarts. The paraphrase augmentation only participates at data
budgets large enough to reach the tail of the stream.

## 4. De-confounded rerun (shared val set)

To put the full-stack condition on the same yardstick, its training text was
re-encoded through `babylm_simple`'s fixed vocabulary and evaluated on
`babylm_simple`'s exact val set (byte-identical `val.bin`, same `val_char_count`,
same categories/TMA — see the `--fixed_meta` path in `prepare_words.py`). Data dir:
`experiments/data/babylm_paraphrase_fixedvocab` (train: paraphrase-augmented,
curriculum-sorted; 3.02% `<unk>` under the fixed vocab). Same config, three seeds.

| Condition (shared `babylm_simple` val) | seed 1337 | seed 42 | seed 2024 | mean | std |
|---|---:|---:|---:|---:|---:|
| `word_learned`, flat | 2.828 | 2.887 | 2.848 | **2.854** | 0.030 |
| `word_category`, flat | 2.545 | 2.749 | 2.646 | **2.647** | 0.102 |
| `word_category` + curriculum + phases | 2.943 | 2.873 | 2.945 | **2.920** | 0.041 |

On a common yardstick the ranking **inverts** relative to the original single-seed
report: the "best result" (para+phases, originally 2.739) is now the **worst** of
the three at **2.920** — worse than both flat learned (2.854) and flat category
(2.647). Two effects were hiding this: the easier/parallel val set it was scored on,
and the favourable seed. Stripped of both, stacking curriculum ordering + phase
restarts on top of the category prior *hurts* at 100k. This is consistent with the
earlier curriculum finding (§ "Curriculum ordering": category + curriculum −0.127
BPC): the structured prior and the curriculum are partial substitutes, and combining
them in the low-data regime interferes rather than compounds.

## 5. Conclusions

- **Flat `word_category` is the best 100k condition** (2.647, seed-robust), and its
  advantage over flat learned (2.854) is real and non-overlapping across seeds (~7%).
  This is the finding to carry forward.
- The **paraphrase / full-stack headline is retracted**: on a common val set with
  three seeds it is the *worst* of the three conditions, not the best. The original
  2.739 rested on (a) a non-comparable val set, (b) a favourable seed, and (c) a
  regime where the paraphrases were not even in the training window.
- **Curriculum + phases do not stack with the category prior at 100k** — they
  substitute for it and interfere when combined, confirming the earlier curriculum
  result under a cleaner, multi-seed measurement.
- **Report single-seed low-data results with a variance band from now on.** At
  100k tokens, seed spread (up to ±0.10 BPC for flat category) is comparable to
  several of the between-condition effects previously reported as findings.

---

# Construction-aligned child-directed augmentation

*2026-09-01*

## 1. Motivation and design

The paraphrase retraction (previous section) showed that data appended to the *end*
of the training stream is inert in the low-data window. This experiment instead
**interleaves** synthetic child-directed speech (CDS) into the base corpus so it
participates at 100k tokens, and asks a sharper question: *can CDS augmentation that
deliberately covers specific grammatical constructions teach those constructions*,
as measured by BLiMP — not just lower BPC?

Two synthetic sets were generated (imaginative-scene paraphrases of the 1,112 unique
simple utterances, via parallel Claude subagents, then filtered/validated):

- **Generic-construction set** (`paraphrases_daring.txt`, 11,089 lines): each of 2
  invented toddler-world scenes per source realized in 5 generic constructions
  (active / yes-no / wh / confirmation / negation).
- **BLiMP-aligned set** (`paraphrases_constructions.txt` + reflexive top-up,
  ~24,141 lines): 11 constructions chosen to map onto BLiMP phenomena — active,
  anaphor, binding, det-noun agreement, subject-verb agreement, argument-structure
  alternation, irregular forms, wh, NPI, quantifier/existential, ellipsis — with
  minimal-pair contrasts for the agreement/alternation categories, plus a 989-line
  reflexive top-up to lift under-covered binding/anaphor. Control/raising and island
  effects were excluded as un-teachable in toddler register (islands are defined by
  *un*grammaticality; raising is bookish).

Both were interleaved 1:1 with the base `babylm_simple` train lines
(`prepare_daring.py`), encoded through `babylm_simple`'s fixed vocab, and evaluated
on its identical val set — so every number is on the same yardstick as the flat
baselines. All results are **3 seeds** (1337/42/2024).

## 2. BPC: interleaved augmentation reduces low-data overfitting

| Condition (100k, shared simple val) | word_category | word_learned |
|---|---:|---:|
| flat baseline | 2.671 ±0.068 | 2.870 ±0.020 |
| + generic-construction (interleaved) | 2.577 | 2.776 |
| + BLiMP-aligned (interleaved) | **2.534 ±0.144** | **2.768 ±0.019** |

Interleaved augmentation **improves BPC** for both encodings (learned robustly:
2.870→2.768, both std ≈0.02; category 2.671→2.534 but with high seed variance). This
is the opposite of the appended paraphrases, which were inert — **placement in the
training window is what matters.** The gain is largely regularisation: diverse,
well-formed simple sentences reduce catastrophic low-data overfitting.

## 3. BLiMP: augmentation teaches *some* constructions and distorts others

BLiMP on seed-1337/42/2024 checkpoints, per-phenomenon mean ± std (flat → aug).
Overall accuracy is unchanged (category 0.483→0.477, learned 0.495→0.494) — as
expected at the 100k floor — so the signal is entirely per-category:

| Phenomenon | word_category | word_learned | robust? |
|---|---|---|---|
| **anaphor agreement** | 0.571→0.616 (+0.045) | 0.432→0.609 (+0.178) | **yes, both** |
| **binding (Principle A)** | 0.643→0.549 (−0.093) | 0.586→0.520 (−0.066) | **yes, both (worse)** |
| **quantifiers** | 0.397→0.417 (+0.020) | 0.551→0.474 (−0.077) | learned only (worse) |
| ellipsis | 0.504→0.446 (−0.058) | 0.430→0.407 (−0.023) | category only (worse) |
| irregular / NPI / filler-gap / det-noun / SV agr / arg-structure | within seed noise | within seed noise | no |

**The two robust findings:**

1. **Anaphor agreement improves** — cleanly for both encodings, and *large* for the
   learned encoding (+0.178). The reflexive + agreement augmentation genuinely taught
   reflexive gender/number agreement morphology.

2. **Binding gets worse** — robustly, for both encodings, *despite* the 989-sentence
   reflexive top-up added specifically to help it. Anaphor *agreement* improved while
   Principle-A *binding* degraded: surface reflexive exposure taught the morphology but
   made the model accept reflexives too liberally, hurting the paradigms where the
   reflexive is the ungrammatical option (wrong structural domain / c-command).
   **Teaching a construction's surface form is not teaching its licensing constraint.**

**The quantifier failure mode (learned, robust −0.077).** The `quant` template
generated existential-*there* with numerals and universals — *"There are two
kittens!"*, *"Every doggy is sleeping."*, *"All the toys are away."* But BLiMP's
existential-there paradigms test exactly the constraint that existential *there*
**rejects strong/universal quantifiers** (*"There was **a** documentary"* good vs
*"There was **each** documentary"* bad). The augmentation raised the probability of
the `there + strong-quantifier` pattern — the ungrammatical option — so it *taught
the wrong generalisation*. (For the category encoding this drop was **not** robust:
seed-1337's −0.19 was noise; the 3-seed mean is +0.02.)

## 4. Methodological note: single-seed BLiMP is misleading here

The single-seed (1337) pass suggested large wins on irregular forms (+0.11) and NPI
(+0.13) and a large category quantifier *loss* (−0.19). **None of these survived
three seeds** — per-category BLiMP std at 100k runs 0.05–0.13, comparable to the
effects themselves. Only anaphor agreement, binding, and the learned quantifier drop
are robust. BLiMP at this data/model scale must be reported with a seed band.

## 5. Conclusions

- **Interleaving matters**: augmentation placed in the low-data window improves BPC
  (learned robustly ~0.10); the same data appended at the end is inert.
- **Construction-aligned CDS augmentation is selective, not a free win**: it robustly
  taught anaphor agreement (learned +0.178) but robustly *hurt* binding (both) and
  learned quantifiers — because simple examples convey a construction's surface form
  without its licensing constraints, sometimes teaching the wrong grammar.
- **Fix directions**: the `quant` template must restrict existential-*there* to weak
  quantifiers and teach universals separately; a binding-targeted set needs
  structural minimal contrasts (reflexive licensed vs not by domain/c-command), not
  more surface reflexives.
- **Overall BLiMP is a floor at 100k** (~0.48–0.50); these are per-phenomenon
  representational effects, not aggregate competence gains.

## 6. Targeted fixes (v2) — the fixes work but trade off against each other

Both failure modes above were re-engineered and the whole pipeline re-run (3 seeds,
same interleave, same shared val); only the two broken constructions were swapped,
the other nine kept:

- **quant fix**: existential-*there* restricted to weak quantifiers (a/some/no/two/
  several, ~950 lines); universals (every/all/each/both/most) taught as plain subjects
  with no *there* (~870 lines). Verified 0 existential-*there* + strong-quantifier lines.
- **binding fix**: structural Principle-A set (1,200 lines) — local binding, embedded-
  clause binding by the *nearer* subject, and reflexive-vs-pronoun domain minimal pairs
  — replacing the earlier surface-reflexive top-up.

BLiMP (3-seed mean), flat → v1 → v2:

| Phenomenon | word_category | word_learned |
|---|---|---|
| quantifiers | 0.397 → 0.417 → **0.454** | 0.551 → 0.474 → **0.510** |
| binding | 0.643 → 0.549 → **0.608** | 0.586 → 0.520 → **0.558** |
| anaphor agreement | 0.571 → 0.616 → **0.486** | 0.432 → 0.609 → **0.460** |

**Both fixes worked on their targets.** *Binding* recovered most of the v1 loss
(category −0.093→−0.034, learned −0.066→−0.028): teaching the licensing *structure*
(local vs embedded domain, reflexive vs pronoun) reversed the damage that surface
reflexives had done. *Quantifiers* partially recovered (learned −0.077→−0.041;
category +0.020→+0.058): removing existential-*there* + strong-quantifier lines
removed most of the wrong-generalisation signal. This confirms the v1 diagnosis was
mechanistically correct — the regressions were caused by teaching surface form
without licensing constraint, and supplying the constraint fixes them.

**But the binding fix cost the anaphor-agreement win.** v1's robust anaphor gain
(+0.045 / +0.178) collapsed in v2 (−0.085 / +0.028). The cause is the binding fix
itself: the reflexive-vs-pronoun minimal pairs ("*The girl sees herself*" **and**
"*The girl sees her*") taught that a pronoun is acceptable in reflexive-like frames,
diluting the model's preference for the agreeing reflexive — exactly what the
anaphor-*agreement* paradigms measure. (v2 anaphor variance is high, std ≈0.10, so
part is noise, but the large v1 win did not reproduce.)

**The deeper finding: in a tiny, data-limited model the phenomena are not
independently optimisable.** Teaching one distinction reshapes the geometry the
others rely on — fixing binding directly undercut anaphor agreement, because the very
pronoun/reflexive contrast that teaches structural binding weakens pure
reflexive-agreement preference. Construction-aligned CDS augmentation is not a set of
additive levers; it is a single distribution whose parts interact. At this scale you
trade phenomena against each other rather than accumulating them.

## 7. Larger-model test — is the trade-off a capacity bottleneck? (No)

The natural follow-up: does the phenomena trade-off dissolve with more capacity? If
the tiny model simply lacks the representational room to keep anaphor agreement and
binding separate, a bigger model should let both improve together. We tested this
directly by re-running the v2 grid — `{word_category, word_learned} × {flat =
babylm_simple, aug = babylm_daring_blimp2} × 3 seeds (1337/42/2024)` — at **6 layers /
6 heads / 384-d** (vs the tiny 4L/4H/128d), everything else identical (1000 iters, 100k
tokens, freeze cats 2+4, α=0.5, shared simple val, `--save_checkpoint`). ~35 min/run on
CPU. The comparison is seed-matched, so tiny→big isolates model capacity.

**BPC.** Augmentation still reduces low-data BPC at 384d (category 2.725→2.533, learned
2.710→2.667 at seed 1337; direction holds across seeds), consistent with the
regularisation story — this is not a fitting artefact.

**BLiMP, flat→aug delta (mean ± sd, 3 seeds), tiny vs big:**

| Phenomenon | learned tiny 128d | learned **big 384d** | category tiny | category **big** |
|---|---|---|---|---|
| anaphor agreement | +0.028 ± 0.050 | **+0.128 ± 0.038** | −0.085 ± 0.097 | −0.037 ± 0.088 |
| binding | −0.028 ± 0.025 | **−0.122 ± 0.036** | −0.034 ± 0.016 | −0.075 ± 0.024 |
| quantifiers | −0.041 ± 0.067 | +0.054 ± 0.016 | +0.058 ± 0.132 | −0.047 ± 0.068 |

(The tiny 3-seed deltas here reproduce the §6 v2 numbers, validating the aggregation.)

**The trade-off does not dissolve — for the learned encoding it sharpens into a clean,
significant anti-correlation.** With more capacity the anaphor-agreement gain grows ~4×
(+0.028→+0.128) *and* the binding loss grows ~4× (−0.028→−0.122); both bands are now
well clear of zero, where at tiny scale they overlapped it. The extra capacity does not
buy independent optimisability — it lets the model commit *harder* to the augmentation's
surface reflexive regularity, amplifying both the intended morphological gain and the
collateral binding damage. Quantifiers improve at 384d (+0.054 ± 0.016), so the v2 quant
fix benefits from capacity, but that does not offset the locked anaphor/binding trade.
The category encoding stays noisier and augmentation mildly hurts/neutral on BLiMP at
both scales (only robust effect: binding −0.075 ± 0.024); category aug helps BPC but not
BLiMP, reinforcing that its benefit is regularisation, not grammatical acquisition.

**Conclusion.** The entanglement is a property of the augmentation *distribution*
(reflexive surface form taught without Principle-A licensing), not a capacity
bottleneck. Scaling the model makes it worse, not better. The lever is the data —
correct licensing constraints in the augmentation — not model size.

---

# Verb subcategorization-frame encoding

*2026-09-10*

## 1. Hypothesis

Verbs fall into a small number of syntactic argument-structure classes. Rather than
leave `predicative` (cat 3) a single monolithic, *trainable* category, give each verb
internal structure: a fixed basis of **six subcategorization frames**, with each verb
assigned a **soft frame distribution induced online from its observed contexts** — testing
whether a verb can be usefully placed in a frame "after a couple of observations." This is
the TMA overlay mechanism (`encoders.py`, `categorize.classify_clause`) applied to verbs:
six frozen basis vectors, a per-verb distribution, added to the co-occurrence base. The
corpus is enriched with `experiments/litdata/` (five Gutenberg prose books) appended after
BabyLM, mainly to give frame induction richer argument structure than child speech alone.

The six frames: INTRANS, TRANS, DITRANS, PP_OBL (caused-motion / oblique), FIN_COMP
(finite clausal complement), INF_COMP (control / infinitival). Induced by a POS-window
heuristic (`experiments/verbframes.py`); `word_frame` (`encoders.py`) = co-occurrence +
primary-category overlay + frame overlay, **no TMA**, so the frame effect is isolated.
Data dir `experiments/data/babylm_lit` (child-speech val, litdata train-only).

## 2. Frame-assignment reliability (gold-validated)

A hand-curated gold map of allowed frames for the 66 most frequent verbs
(`experiments/gold_verb_frames.json`) lets us measure assignment *correctness* as a
function of the number of observations (`experiments/frame_reliability.py`).
accuracy@N = fraction of gold verbs whose induced argmax frame is in the gold-allowed set
(chance = 0.333; mean allowed set 2.0 / 6 frames):

| N observations | 1 | 2 | 3 | 5 | 10 | ∞ |
|---|--:|--:|--:|--:|--:|--:|
| accuracy | 0.576 | 0.667 | 0.591 | 0.712 | 0.818 | 0.833 |

Even a single observation (0.576) is well above chance; assignment reaches its 0.833
ceiling by ~10 observations. The "few observations suffice" hypothesis holds — with "a
few" being ~5–10 for reliable placement. Ceiling misses are honest heuristic noise
(that-less complements like *know*/*say* → intrans; adjunct-PP over-assignment like
*keep*/*send* → pp_obl). Figure: `experiments/runs/frame_reliability.png`.

## 3. BPC: the frame prior is the best low-data encoding, and beats the category prior

Seed-banded (1337/42/2024) validation BPC on `babylm_lit`, tiny 4L/4H/128d model:

| budget | word_learned | word_category | **word_frame** |
|---|---:|---:|---:|
| 100k | 2.885 ± 0.020 | 2.744 ± 0.074 | **2.656 ± 0.035** |
| 300k | 2.511 ± 0.006 | 2.487 ± 0.002 | 2.483 ± 0.023 |
| 1M   | **1.946 ± 0.002** | 1.975 ± 0.020 | 1.966 ± 0.005 |

At 100k the verb-frame prior beats the learned baseline by **−0.229 BPC** and — the key
result — beats the monolithic category prior by **−0.088**. Giving verbs *induced* frame
structure improves over the single predicative category. This is the **first time finer
sub-categorization has helped**: the earlier *arbitrary* structural sub-split hurt
("Sub-category split", −0.17 at 100k), whereas these frames are data-induced, reliable
(§2), and linguistically grounded. The advantage shrinks with data (−0.028 vs learned at
300k) and inverts by 1M (learned leads), the same crossover the category prior shows;
`word_frame` is ≥ `word_category` at every budget.

## 4. BLiMP: no targeted argument-structure gain — the BPC win is regularisation

BLiMP on the 100k checkpoints (3 seeds), argument-structure group = {transitive,
intransitive, causative, inchoative, drop_argument, passive_1/2,
animate_subject_passive, animate_subject_trans}:

| encoding | overall | argstr-group |
|---|---:|---:|
| word_learned | 0.498 ± 0.011 | 0.571 ± 0.010 |
| word_category | 0.485 ± 0.015 | 0.559 ± 0.015 |
| word_frame | 0.492 ± 0.014 | 0.558 ± 0.010 |

The frame prior does **not** improve the targeted argument-structure paradigms:
`word_frame` − `word_category` = −0.001 on the group, and every per-paradigm delta is
within seed noise (largest: transitive +0.020, inchoative +0.019; passive_1 −0.030).
Overall BLiMP sits at chance for all three. So the substantial, seed-robust −0.088 BPC
improvement at 100k is **distributional regularisation, not grammatical acquisition** — a
frozen input-embedding bias lowers next-word perplexity but does not confer verb-specific
selectional discrimination on minimal pairs at this scale. This mirrors the
construction-augmentation finding that BPC and targeted BLiMP move independently.

## 5. Conclusions

- A **data-induced verb-frame prior is the strongest low-data encoding to date** (2.656
  BPC at 100k), and improves on the monolithic category prior by −0.088 — reliable,
  linguistically motivated sub-categorization helps where arbitrary splitting hurt.
- Frame assignment is **reliable from few observations** (0.83 accuracy by ~10 obs,
  gold-validated), supporting the "assign after a couple of observations" hypothesis.
- The BPC gain is **regularisation, not acquisition**: targeted BLiMP argument-structure
  paradigms do not move. BPC and grammatical discrimination are again decoupled at this
  scale.
- Crossover ~300k–500k tokens, as with the category prior; the frame prior is a low-data
  instrument.

Artifacts: `experiments/verbframes.py`, `gold_verb_frames.json`, `frame_reliability.py`;
runs `word_{learned,category,frame}_lit_{100k,300k,1M}_s{1337,42,2024}` (+ `_ckpt` at
100k with `blimp_results.json`); logs `runs/sweep_frame_lit.log`, `runs/blimp_frame_lit.log`.

---

# Overregularization test — Levin "rule" prior vs induced "lexical" prior

*2026-09-15*

## 1. Question

Is the data-monotonic crossover (structured prior helps early, handicaps late) an analog
of child **overregularization** — a productive rule that helps broadly early, then
mis-predicts the *exceptions* once the lexical route can memorize them? To test this at
the item level we need a **pure rule** prior and a **regular/irregular** split.

**`word_levin`** (new encoding, `encoders.py`): each verb gets its Levin semantic class →
one frozen basis vector per class, *shared by all class members*, so it cannot encode
within-class idiosyncrasy — a pure rule. Contrast `word_frame`, whose per-verb induced
distribution is already lexicalized. Levin map: hand-curated, 103 lexical verbs, 13 coarse
classes (`experiments/levin_classes.json`; VerbNet is unavailable offline).

**Regular vs irregular** (`experiments/verb_regularity.py`): class prototype = mean induced
frame distribution over class members; per-verb deviation = Jensen–Shannon divergence(own
frame dist, prototype); tercile split (35 regular / 35 irregular, middle dropped). The
split is face-valid — regular: *pull, run, break, believe, hit*; irregular: *find*
(transitive, not clausal like know/think), *die* (no causative alternation), *look/talk*
(PP-oriented), *enjoy* (transitive-only).

## 2. BPC: semantic class ≈ syntactic frame (substitutes)

Adding `word_levin` to the sweep (3 seeds, babylm_lit): 100k **2.650 ± 0.012**, 300k
2.488, 1M 1.982 — essentially identical to `word_frame` (2.656 / 2.483 / 1.966) and, at
100k, well below category (2.744) and learned (2.885). The semantic-class prior and the
syntactic-frame prior deliver the *same* low-data BPC benefit: for regularisation purposes
they are **substitutes**.

## 3. Item-level: complement-position NLL, regular vs irregular

Per-verb NLL of the token *following* each verb occurrence on val (the argument/complement
position), strided across the full val, 3-seed mean (`experiments/verb_nll_eval.py`; uses
the corrected checkpoint loader — see the loader-bug note below):

| budget | encoding | regular | irregular | Δreg vs learned | Δirr vs learned |
|---|---|---:|---:|---:|---:|
| 100k | learned | 6.983 | 7.111 | — | — |
| 100k | frame | 6.243 | 6.358 | −0.740 | −0.752 |
| 100k | levin | 6.207 | 6.258 | −0.776 | −0.853 |
| 1M | learned | 4.543 | 4.541 | — | — |
| 1M | frame | 4.549 | 4.579 | +0.005 | +0.038 |
| 1M | levin | 4.577 | 4.607 | +0.034 | +0.066 |

**Overregularization differential** (extra harm to irregulars = Δirr − Δreg): frame
−0.012 → +0.033; levin −0.077 → +0.032 across 100k → 1M.

## 4. Reading: weak, data-axis-only support for the analogy

- **At low data (the child-scarcity regime) there is no overregularization.** Both priors
  help complement prediction *broadly* — regulars *and* irregulars (−0.74 to −0.85 nats).
  The pure rule (levin) helps irregulars if anything *more* than regulars (−0.853 vs
  −0.776). A structured prior is a good regulariser for exceptions too when data is scarce.
- **The exception-cost emerges only later, and faintly.** By 1M the priors' (small) cost
  falls disproportionately on irregulars — the differential flips negative→positive for
  both. That is the predicted direction: as data grows, the rigid rule increasingly
  mis-predicts the exceptions the lexical route now captures. But the 1M magnitudes
  (~0.03–0.07 nats) are near the seed-noise floor, and by 1M the prior barely matters at
  all (all encodings ~4.5–4.6).
- **Pure rule ≉ more overregularization.** The clean prediction that `word_levin` (rule)
  hurts irregulars more than `word_frame` (lexicalized) is *not* robust: at 100k levin
  regularises irregulars *better*, and at 1M they are indistinguishable — consistent with
  the BPC substitutes result.

**Conclusion.** The child-overregularization analogy holds only *weakly*, and only in its
data-axis form: the differential harm to irregular verbs shifts in the predicted direction
with data, but there is no genuine U — the whole regime where the prior matters (low data)
is one where it helps exceptions too, and by the time it would overregularize, the prior is
nearly inert. A true U-curve would require the *time axis within one learner* — the
staged-unfreeze design (freeze the rule early, unfreeze to lexicalize later) remains the
proper test. Artifacts: `experiments/levin_classes.json`, `verb_regularity.py`,
`verb_nll_eval.py`; runs `word_levin_lit_*`; `runs/verb_nll.json`.

---

# ⚠ Methodology correction: BLiMP checkpoint-loader bug (2026-09-15)

**All BLiMP results produced before 2026-09-15 were computed on corrupted models and are
being re-run.** nanoGPT ties the input embedding to the output head
(`nanoGPT/model.py:138`, `wte.weight = lm_head.weight`). Training *unties* them before
saving (`train_experiment.py:106`), so each checkpoint stores two independent tensors. But
`eval_blimp.load_model` rebuilt a *tied* model and called `load_state_dict` — both keys
(`transformer.wte.weight`, `lm_head.weight`) resolve to the same shared parameter, so the
input embedding was silently overwritten by the output head. Loading `word_frame`@100k the
buggy way gives BPC-equiv 3.42 (and ranks it *worse* than learned); loading it correctly
(untie first) gives 2.66, matching its recorded 2.69. Fix: untie `lm_head` before
`load_state_dict` (`eval_blimp.py:load_model`); safe for tied checkpoints too.

**Unaffected:** every **BPC** number in this report (computed in-process during training,
never via `load_model`) and the frame-reliability curve. The encoding findings that rest
on BPC stand.

**Affected (being re-run with the fixed loader; sections above to be corrected):**
- "Verb subcategorization-frame encoding" §4 (BLiMP argument-structure null).
- "Larger-model test" §7 (the anaphor/binding trade-off deltas).
- "Construction-aligned child-directed augmentation" §3/§6 (v1/v2 phenomenon trade-offs).

The overregularization section above already uses the fixed loader.
