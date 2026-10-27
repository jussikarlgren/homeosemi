# Experimental Plan — Staged Priors at Scale: Curriculum × Construction × Stepped Learning Rate

**Premise.** A pilot on a minimal word-level GPT (BabyLM, nanoGPT, CPU) established that
developmentally-motivated priors — a frozen linguistic scaffold, curriculum ordering, and
warm-restart LR — each lower bits-per-character (BPC) in the low-data regime, but that the
effects are *entangled, substitutive, and confined below a crossover* (~500k–800k tokens;
*Low-data sweep, 2026-08-20*). This project asks whether, and how, those levers survive
scaling, and whether any of them buys **grammatical competence** rather than mere
regularisation.

**Central hypotheses.**
- **H1 (scaling).** The prior-helps / prior-hurts crossover budget moves with model capacity;
  the low-data win is in-regime for larger models at proportionally larger budgets.
- **H2 (complementarity by staging).** Curriculum and structured priors — *substitutes* when
  applied flat (category+curriculum −0.127 BPC vs category alone; *Curriculum ordering,
  2026-08-26*) — become *complementary* when the prior is a time-released scaffold and LR
  restarts are aligned to curriculum phase boundaries.
- **H3 (competence vs regularisation).** A *configurational, induced* construction prior can
  move minimal-pair competence (BLiMP), where label-injection priors only moved perplexity.

---

## Experiment 1 — Crossover scaling law *(gates the whole premise)*
**Q:** Does the crossover-token-budget scale with model size?
**Design:** For model sizes {tiny 4L/128d → 6L/384d → larger}, sweep train budgets
{100k…8.7M} × {learned, flat category} × ≥3 seeds; locate the per-size crossover.
**Grounded in:** *Low-data sweep (2026-08-20)*, +14% BPC at 100k, gone by 1M, −0.047 at 8.7M;
*Larger-model test (2026-09-08, corrected 2026-09-16)*, the anaphor/binding trade-off is not a
capacity bottleneck (grows only mildly with capacity).
**Success:** a crossover(model-size) curve. Flat/pinned ⇒ honest "small-model regulariser";
rising ⇒ in-regime at scale.

## Experiment 2 — Staged scaffold + phase-aligned restarts *(the headline design)*
**Q:** Does freeze-then-release + curriculum + boundary-aligned warm restarts convert
substitution into complementarity?
**Design:** 2×2×2 over {flat vs curriculum-ordered data} × {always-freeze vs staged-unfreeze
at the complexity boundary} × {flat LR vs restarts at the same boundaries}, ≥3 seeds, at ≥2
model sizes.
**Grounded in:** staged unfreeze *dominated* always/never-freeze at every budget
(*Staged unfreeze, 2026-09-17*); warm restarts help *only* under phase-data alignment — the
no-curriculum control hurts (*Warm restarts + curriculum, 2026-08-26*); note part of the
high-data penalty is the structured init *basin*, not just frozen rows (never-freeze worst at
1M).
**Success:** staged + aligned-restart beats the best single-lever condition on BPC at the
in-regime budget from Exp 1 — with a seed band, on a shared val set. (Guard against the
retracted "stack everything" headline: the flat full stack was the *worst* of three
conditions on a common yardstick — *Seed-variance, 2026-08-31*.)

## Experiment 3 — Configurational construction prior, post-crossover *(competence test)*
**Q:** Does a per-occurrence, configuration-aware construction signal convert its certified
perplexity win into BLiMP competence at scale?
**Design:** Positional construction injection vs position-matched scramble vs none; evaluate
BPC *and* BLiMP-construction, swept past the Exp-1 crossover and at larger capacity; add a
*frequency-matched* scramble for clean BPC certification.
**Grounded in:** positional encoding beat its scramble on BPC (−0.036, 5/5 seeds) but was
BLiMP-null (*Configuration-aware construction signal, 2026-10-12*); static per-word overlay
was null (*Construction-prior certification, 2026-09-30*); content lives in *induced, graded,
per-item* reps — verb frames −0.088 BPC (*Verb frame, 2026-09-10*).
**Success:** full − scramble > 0 on BLiMP-construction post-crossover ⇒ first competence
conversion; persistent null ⇒ competence is data-bound, redirect to augmentation-data
licensing (the only lever that ever moved BLiMP — *Construction-aligned CDS augmentation,
2026-09-01*).

*Optional probe — developmental signature:* freeze **verb** rows, release at boundary, measure
irregular-specific release gain (the one genuine time-axis overregularization result:
class-rule only, data-gated; *Verb U-curve, 2026-09-18*).

---

## Controls & methodology (banked from the pilot)
- **Seed bands mandatory**; single-seed BLiMP is misleading (false positives died under 9
  seeds — *Seed-variance, 2026-08-31*; *Determiner follow-up, 2026-09-28*).
- **Scramble / category-gating** is the content-vs-regularisation instrument
  (*Acquisition-vs-regularisation dissociation, 2026-09-27*); BPC certification needs a
  **frequency-matched** scramble, not just magnitude-matched.
- **Untie `lm_head` before `load_state_dict`** — the tied-embedding loader bug corrupted all
  pre-2026-09-15 BLiMP (*BLiMP loader bug, 2026-09-15*); prefer in-process eval.
- Shared val set / fixed vocab across conditions (the paraphrase confound — different val set +
  appended-last placement, *2026-08-31*); one job at a time, wait on completion — no
  timer-relaunch (invisible background-job contention inflated wall-clock ~10×, *2026-09-27*).

## Primary risk
Everything is low-data: if Exp 1 shows the crossover does not scale, the defensible
contribution narrows to "staged developmental priors as a small-model regulariser + a reusable
content-vs-regularisation certification method" — still publishable, but not an at-scale
training claim. **Exp 1 runs first and gates Exp 2–3.**

---

See also: `notes/research-programme_data-efficient-priors.md`, `REPORT.md`, `EXPERIMENT_DIARY.md`.
Section/date references above map to `REPORT.md` section headings and `EXPERIMENT_DIARY.md`
dated entries.
