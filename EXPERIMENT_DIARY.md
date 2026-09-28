# Experiment diary

Running log of research intent, design decisions, and open questions. Results live in
`REPORT.md`; this file records the *thinking* — why we ran things and where we're going.

---

## 2026-09-24 — Frozen categories as a thawable prior; a developmental theory of category acquisition

Recorded from discussion while designing the acquisition-vs-regularisation dissociation
experiment (Exp A scramble control / Exp B category-gating; see
`experiments/overlay_variants.py`, `paradigm_categories.py`, `analyze_dissociation.py`).

### The core idea

A **frozen category is a useful shortcut to learning** — essentially a *prepackaged large
dataset*. Freezing a category's rows injects, for free, the statistics a learner would
otherwise have to accumulate from many observations. So a frozen structured prior stands
in for data.

But (per our earlier experiments) as real observation statistics accrue, the **pure-learner
curve crosses the frozen curve** — beyond some data budget the learner that estimated the
structure from data overtakes the one handed a fixed prior. At that crossover the freeze has
outlived its usefulness and starts to hurt.

**Therefore the freeze should be thawed at the crossover** — learning unleashed onto even
those categories. The thaw point is a **parameter, and it may vary across categories**:
different categories cross over at different data budgets, so each should thaw on its own
schedule.

### The developmental model this points to

A learner **begins from simple assumptions about how language is put together** — a very
small set of categories (the ones we have postulated so far: referential, structural,
predicative, adv-clausal, determiners, etc.). As the data becomes more sophisticated:

1. **new observed features are identified** (candidate categories/distinctions),
2. **frozen** at some (per-feature, variable) learning rate — used as a prior/shortcut,
3. **thawed** when *anomalies contra expectations* occur — i.e. when the data stops fitting
   the frozen assumption and the feature needs to be re-estimated from observation.

Planned next categories to fold in: **constructional configurations** (construction-level
features beyond the current word-class categories).

### Open question (not yet resolved)

**When should the thaw points be assumed to occur?** Both the *per-category crossover* (data
budget at which a learned estimate overtakes the frozen prior) and the *anomaly trigger*
(what counts as enough contra-expectation evidence to thaw) are undetermined. Candidate
handles to explore:
- per-category crossover measured empirically (freeze curve vs learner curve per category),
- a data-driven anomaly signal (rising loss / gradient pressure on frozen rows) as an
  automatic thaw trigger, rather than a fixed `--unfreeze_at_frac`.

### Bearing on the current experiment

The immediate dissociation runs are at **100k tokens = pre-crossover**, the regime where the
frozen prior is *beneficial*. So for testing "does the frozen overlay's linguistic *content*
matter (vs magnitude/regularisation)", freezing the overlaid categories (no thaw yet) is the
right regime. Thawing schedules are a separate, later axis — the subject of the open question
above. Design note that prompted this: `gate_categories` rewrites `word_categories` before
the freeze mask is built, so the frozen set must be made to match the overlaid (ON)
categories per condition to keep Exp B symmetric.

---

## 2026-09-27 — Dissociation result: the category overlay is regularisation, not acquisition

Ran the acquisition-vs-regularisation dissociation (5 conditions × 3 seeds, 100k, primary
overlay only, overlaid categories frozen). Full write-up in REPORT.md "Acquisition vs
regularisation".

**Outcome.** Exp A: `full` ≈ `scrambled` at every category bucket (all diffs within seed
noise) — destroying the overlay's linguistic content while matching magnitude/histogram
does not hurt. Exp B: overlaying a single category does not reliably lift its own paradigms
(on7→cat7 +0.022±0.020, on3→cat3 +0.007±0.009; both marginal/null). So at 100k the overlay
acts as a **magnitude-matched regulariser, not encoded linguistic structure** — a clean,
purpose-built confirmation of the recurring "BPC up, BLiMP flat" theme.

**Bearing on the frozen-prior theory (2026-09-24 entry).** The frozen category prior helps
*because* it regularises (stands in for data / caps variance), not because the model reads
its category geometry as grammar. That is consistent with "frozen category = prepackaged
dataset": it is the *statistical mass/scaffold* that helps in the low-data regime, and the
specific linguistic labelling is (at this scale, on these paradigms) inert. Whether a richer
category set (e.g. constructional configurations) or a post-crossover thaw would let the
*content* start to matter is the open question — this null is the pre-crossover, frozen-prior
baseline against which to test that.

**Only non-null hint:** determiners (cat7), `full` > `scrambled` +0.046 but ±0.052 — not
significant; more seeds on just full/scrambled@cat7 would confirm or kill it.

**Infra lesson (recorded so it isn't repeated):** background jobs on this box persist
invisibly to tasklist/ps; timer-relaunching piled up ~10 concurrent trainers (10675 s vs
~1000 s uncontended). Fix: one job, wait for its completion signal, never timer-relaunch a
job you cannot observe.

---

## 2026-09-28 — Determiner hint killed at 9 seeds

Followed up the one non-null hint from the dissociation (cat7 `full` > `scrambled`, +0.046
at 3 seeds) with 6 more seeds. Over all 9: **+0.027 ± 0.059** (SE 0.020; 95% CI
[−0.011, +0.066], includes zero; 7/9 seeds positive, one −0.104). Not significant — the
+0.046 was a favourable-seed artefact. So the "overlay = regularisation, not linguistic
content" conclusion is clean **including for determiners**. One more instance of the
seed-variance rule that keeps recurring: a 3-seed hint evaporating at 9. Nothing left to
chase here.
