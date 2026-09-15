# When does resetting the learning rate actually help? Only at a real change in the data.

*Field notes from a minimal-language-model study — yarns-and-loss project*

There is a folk practice in deep learning of "warm restarts": every so often you yank
the learning rate back up and let it decay again, on the theory that periodic
re-exploration shakes the model out of bad minima. It sometimes helps and sometimes
doesn't, and the literature mostly treats *when* as a matter of taste. We ran a small,
controlled study that gives a sharper answer — and the answer is a principle, not a
hyperparameter.

**The claim: a learning-rate restart is only worth it when it coincides with the
training data genuinely changing.** Reset the LR at an arbitrary point and you pay for
it — you spike the model away from a minimum it was approaching and get nothing back.
Reset it exactly when qualitatively harder data starts arriving, and the restart buys
you fresh plasticity precisely when the model needs to absorb something new. Same
schedule, opposite outcomes, depending only on alignment with the data stream.

We tested this on a deliberately tiny word-level GPT (~4.6M parameters) trained on
BabyLM, a developmentally plausible child-language corpus. Two knobs: **curriculum
ordering** (sort utterances simple→complex, so complexity rises monotonically through
training) and **cosine warm restarts** at phase boundaries. The evidence for the
alignment principle is a control run: warm restarts *without* a curriculum are simply
worse than a plain flat schedule (1.666 vs 1.623 bits-per-character at full data). The
restarts only start paying rent once each reset lines up with a real regime change in
the input. The LR schedule isn't a free win; the *alignment* is the win.

**A second, subtler result surprised us: a curriculum and a structured inductive prior
are substitutes, not complements.** Ordering the data simple-first substantially helps a
plain learned model in the low-data regime (it curbs catastrophic overfitting). But bolt
the same curriculum onto a model that *already* carries a structured grammatical prior,
and it *hurts* — the two mechanisms are solving the same problem (constraining the model
before it has enough data to constrain itself), so stacking them interferes instead of
compounding. Two good ideas that quietly cancel.

**The part I'm proudest of is the part that says "we were wrong."** An early single-seed
result had the full stack — prior + curriculum + restarts — as our best-ever low-data
number. It didn't survive. Re-run across three seeds on a common yardstick, that "best"
condition is actually the *worst* of the three, and a plain structured prior wins alone.
Two confounds (a non-identical validation set and a favourable random seed) had inflated
the headline. At this scale, seed-to-seed spread (±0.10 BPC) is as large as several
effects people would happily publish. The correction is the contribution: **in the
low-data regime, report a seed band or don't report.**

Why this stakes out territory worth holding:

- It reframes cyclic/warm-restart schedules from "a trick that sometimes works" to a
  **testable statement about data-schedule alignment** — with a clean control design to
  match.
- It gives a concrete, mechanistic account of curriculum learning's mixed reputation:
  curricula help models that lack other regularisation and interfere with models that
  have it. "Does curriculum learning help?" is the wrong question; "help *relative to
  what other prior?*" is the right one.
- It's a case study in how fragile low-data claims are, and how cheap rigor is if you
  build seed variance into the protocol from the start.

None of this needs a GPU cluster. The entire study runs CPU-only on a laptop, which is
exactly why the methodological discipline matters: small models are where interesting
inductive-bias questions live, and also where noise most easily masquerades as insight.

*Minimal harness, full logs, and the seed-banded tables are in the project's REPORT.md
(sections "Curriculum ordering", "Warm restarts with curriculum phases", and
"Seed-variance robustness").*
