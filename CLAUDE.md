# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project goal

Train a minimal word-level GPT (nanoGPT) on a small, developmentally plausible corpus (BabyLM) and compare different input-encoding schemes — including homeosemi random-indexing (RI) encodings and a linguistically motivated category-aware prior — by their train/val loss curves and bits-per-character (BPC). The central hypothesis is that the lexicon is non-uniform and that a structured encoding encoding that distinguishes functional categories (referential / structural / predicative / clause-adverbial) acts as a useful inductive bias, especially in the low-data regime.

## Environment

- **Python**: `C:/Users/jkarlgre/venvs/yarns-and-loss/Scripts/python.exe` (venv lives OUTSIDE OneDrive — Windows long-path limit + sync issues)
- **GPU**: none — CPU-only PyTorch (`torch 2.13.0+cpu`)
- **Downloads**: Python `requests` fails corporate TLS; use `curl.exe` instead
- **sys.path**: experiment scripts wire up all imports themselves; no `PYTHONPATH` needed

## Repository layout

```
nanoGPT/                    upstream Karpathy nanoGPT (own .git)
  data/shakespeare_char/    char-level tiny-shakespeare data (prepared)
jussipyutils/               homeosemi utility modules (own .git)
  sparsevectors.py          RI vector operations
  languagemodel.py          n-gram language model
  lexicalfeatures.py        lexicon dict: 62 word-list categories
  squintinglinguist.py      NLTK POS tagger + lexicon lookup
hyperdimensionalsemanticspace.py  main homeosemi SemanticSpace class
experiments/
  encoders.py               all encoding schemes (char + word level)
  train_experiment.py       training harness (BPC, per-category freeze)
  categorize.py             word→category mapper (NLTK + lexicon)
  prepare_words.py          word-level data prep → train/val .bin + meta.pkl
  prepare_bpe.py            GPT-2 BPE data prep
  category_similarity.py   intra/inter-category cosine diagnostic
  plot_curves.py            loss/BPC curve comparison plots
  data/babylm_simple/       BabyLM word-level data (prepared)
  data/babylm_bpe/          BabyLM BPE data (prepared)
  runs/                     JSON logs + PNG plots per run
REPORT.md                   full experiment report (char + word level results)
```

## Running experiments

```bash
VP="C:/Users/jkarlgre/venvs/yarns-and-loss/Scripts/python.exe"

# word-level baseline
"$VP" experiments/train_experiment.py --encoding word_learned \
    --data_dir experiments/data/babylm_simple --max_iters 1000

# word-level category encoding (structural + adv_clausal frozen)
"$VP" experiments/train_experiment.py --encoding word_category \
    --data_dir experiments/data/babylm_simple \
    --freeze_categories 2,4 --overlay_alpha 0.1 --max_iters 1000

# low-data sweep
"$VP" experiments/train_experiment.py --encoding word_category \
    --data_dir experiments/data/babylm_simple \
    --freeze_categories 2,4 --max_train_tokens 100000

# compare + plot (BPC metric)
"$VP" experiments/plot_curves.py --metric bpc word_learned word_category
```

Invoke the `run-encoding-experiment` skill for the full workflow reference.

## Key design decisions (do not change without understanding why)

- **Output head untied** from input embedding by default — freezing input must not freeze output
- **Per-category freezing** uses a gradient mask hook + `weight_decay=0` on the embedding param group; verified frozen rows change by 0.00e+00 after an optimizer step
- **BPC** = total val NLL (nats) / (val char count × ln 2) — required for cross-tokenizer comparison; needs `val_char_count` in `meta.pkl`
- **`sparsevectors.permute()` aliases its input** — always permute a dense copy; do not call it on shared index vectors

## Category system (word-level)

| id | name | examples | frozen by default? |
|---|---|---|---|
| 0 | residual | punctuation, OOV | no |
| 1 | referential | pronouns, nouns, adjectives | no |
| 2 | structural | prepositions, conjunctions, determiners | **yes** |
| 3 | predicative | verbs, auxiliaries | no |
| 4 | adv_clausal | negation, modals, hedges, amplifiers | **yes** |

Assigned by `experiments/categorize.py`. Precedence: adv_clausal > structural > predicative > referential > residual.

## Current experimental results (summary)

See `REPORT.md` for full details.

**Best low-data result (100k tokens):** `word_category + paraphrase + 3-phase` BPC **2.739**
(+17.4% vs flat learned baseline 3.314). Four-component synergy:
1. Frozen structural/adv_clausal embedding (grammatical scaffold, immune to LR resets)
2. Category + TMA clause-type overlay (referential/predicative/structural geometry)
3. Paraphrase augmentation (2× child-directed paraphrases of simple referential utterances)
4. 3-phase cosine warm restarts (fresh plasticity at each complexity phase boundary)

**Key interaction:** warm restarts only help when combined with curriculum ordering
(phase-data alignment). They hurt at full data. Structured prior and curriculum are
partial *substitutes* — each helps without the other, but combining them hurts at 100k.

**Full data:** learned baseline (BPC 1.623) still leads; best structured encoding is
`word_category_tma_clause` (BPC 1.648, TMA clause-level distributions).

Crossover ~500k–800k tokens: category prior helps below, learned baseline above.

## Planned next experiments

- Staged unfreeze: freeze structural during simple phase, unfreeze at complexity boundary
- TMA with separate vectors per primary category (vs single shared tma_vec)
- Larger model to separate regularisation from representation effects
- More paraphrases / different utterance selection strategies
