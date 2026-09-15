# How to resume this project in Claude Code

There is no session-resume command. Each new Claude Code session starts fresh,
but CLAUDE.md and the memory system mean it arrives with full context automatically.

## Start a new session

**CLI** (from this directory):
```
cd "C:\Users\jkarlgre\OneDrive - Advanced Micro Devices Inc\monrepos\yarns-and-loss"
claude
```

**Desktop app / IDE extension:** open this folder as the project directory.

## What loads automatically

- `CLAUDE.md` — project layout, run commands, current results, planned next steps
- Memory files — project history, experimental findings, your profile
- `run-encoding-experiment` skill — full workflow, commands, gotchas

## Then just say

> "Continue from where we left off"

or name the next experiment directly, e.g.:

> "Run the curriculum ordering experiment"

---

## Moving to another machine

The prepared data (`experiments/data/`, ~0.5 GB) and results/checkpoints
(`experiments/runs/`, ~1.8 GB) are **regenerable** — do not copy them. Copy the minimal
set below; regenerate the rest.

**Copy (small — code, docs, provenance):**
- Repo code + docs: `experiments/*.py`, `experiments/*.json`, `CLAUDE.md`, `REPORT.md`,
  `HOW_TO_RESUME.md`, `notes/`, root `*.py`, `tma.txt`. (`git clone` covers these once
  they're committed — see the git note below.)
- `nanoGPT/` and `jussipyutils/` — model + utils the scripts import (separate nested git
  repos; clone/copy them alongside).
- `experiments/litdata/` (2.5 MB) — hand-added prose; **nothing regenerates it**.
- Memory files (OUTSIDE the repo): the 4 `.md` in
  `~/.claude/projects/<encoded-project-path>/memory/` (`MEMORY.md` +
  `finding_charlevel_encodings.md`, `project_overview.md`, `user_profile.md`).

**Re-download or copy (external, ~100 MB):** BabyLM raw text
(`experiments/data/babylm_simple/train|dev/*.txt`) and `experiments/blimp/*.jsonl` — both
re-`curl`able from HuggingFace, or just copy them if the corporate-TLS download is painful.

**Recreate (do NOT copy):**
- venv — `python -m venv` at a path OUTSIDE OneDrive, then
  `pip install torch numpy tiktoken tqdm nltk matplotlib` (CPU torch index).
- NLTK data — `nltk.download('punkt')` and `nltk.download('averaged_perceptron_tagger')`.
- Data `.bin`/`meta.pkl` via `prepare_words.py`; `runs/` by re-running the sweeps.

**Three gotchas:**
1. The memory folder name **encodes the old absolute path**; on the new machine the
   encoded name differs. Launch Claude once in the new project dir to create the folder,
   then drop the 4 `.md` files into that new `.../memory/`.
2. `CLAUDE.md` hardcodes the **venv path** and OneDrive project path — update both, or the
   `$VP` run commands won't resolve.
3. Verify NLTK POS/tokenizer data is present (`categorize.py`, `verbframes.py`,
   `eval_blimp.py` all depend on it) before running.

---

## Current status — 2026-09-08 (larger-model test — trade-off is NOT capacity-bound)

Full write-up in `REPORT.md` §7 "Larger-model test" and memory
`finding_charlevel_encodings.md`. Short version:

Re-ran the v2 grid ({category,learned}×{flat,aug=blimp2}×3 seeds) at **6L/6H/384d**
(vs tiny 4L/4H/128d), seed-matched so tiny→big isolates capacity. **Question:** does
capacity dissolve the anaphor-vs-binding trade-off? **Answer: no — it sharpens it.**
For the learned encoding, flat→aug anaphor agreement grows +0.028→**+0.128±0.038** while
binding grows −0.028→**−0.122±0.036** (both ~4×, now clear of zero). More capacity lets
the model commit harder to the augmentation's surface reflexive regularity, amplifying
both the gain and the collateral damage. The entanglement is in the augmentation
*distribution* (surface reflexives without Principle-A licensing), not model size.
Run tags: `word_{cat,learned}_{flat,blimp2}_big384_s{1337,42,2024}`.

**Open next step:** fix the augmentation *licensing* so binding improves WITHOUT the
reflexive-vs-pronoun pairs that killed anaphor agreement — decouple the two phenomena
instead of trading them. That's the real lever the larger-model test points to.

To resume: > "Continue — design the decoupled-licensing augmentation (v3)"

---

## Prior status — 2026-09-07 (construction-aligned CDS augmentation)

Full write-up in `REPORT.md` (section "Construction-aligned child-directed
augmentation", §1–6) and memory `finding_charlevel_encodings.md`. Short version:

**Where we landed.** Built interleaved child-directed augmentation that teaches
grammatical constructions mapped to BLiMP phenomena, generated via parallel Claude
subagents. Two rounds:

- **v1** (11 constructions): robustly *taught* anaphor agreement (learned **+0.178**),
  but robustly *hurt* binding (both) and learned quantifiers — teaching a construction's
  surface form without its licensing constraint taught the wrong grammar.
- **v2** (fixed quant = weak-Q existential-*there* only; fixed binding = structural
  Principle-A contrasts): both fixes **worked** — binding recovered (cat −0.093→−0.034,
  learned −0.066→−0.028), quantifiers partially recovered (learned −0.077→−0.041). BUT
  the binding fix's reflexive-vs-pronoun pairs **cost the anaphor-agreement win**
  (collapsed to −0.085 / +0.028).

**Headline finding:** at this tiny/data-limited scale the BLiMP phenomena are **not
independently optimisable** — construction augmentation is one interacting distribution,
not additive levers. You trade phenomena against each other. BPC held (cat ~2.55,
learned ~2.77, both beat flat 2.671/2.870). All results are 3-seed (1337/42/2024).

**Open next step (the question I paused on):** does the trade-off dissolve with a
**larger model** (more capacity to separate the distinctions)? That's the natural test.
Also offered: clean up `experiments/data/_daring_work/` scratch + the several
`babylm_daring*` data dirs.

**Key artifacts created this cycle:**
- `experiments/prepare_words.py` `--fixed_meta` flag; `experiments/prepare_daring.py`
  (interleaves augmentation into base, fixed vocab, shared simple val)
- data dirs: `babylm_daring`, `babylm_daring_blimp` (v1), `babylm_daring_blimp2` (v2),
  `babylm_paraphrase_fixedvocab`
- generated sets under `experiments/data/babylm_paraphrase/`:
  `paraphrases_daring.*` (generic 5-construction), `paraphrases_constructions.*` (v1
  BLiMP-aligned 11-construction), `paraphrases_blimp_all_v2.txt` (v2 combined)
- runs: `word_{category,learned}_{flatck,blimp,blimp2}_100k_s{1337,42,2024}` (+`_ckpt`),
  BLiMP results in each `*_ckpt/blimp_results.json`
- scratch (safe to delete): `experiments/data/_daring_work/`

To resume: > "Continue the construction-augmentation work — run the larger-model test"
