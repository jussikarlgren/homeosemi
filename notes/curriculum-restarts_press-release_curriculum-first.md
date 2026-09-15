# Curriculum learning helps small models — until it doesn't, and knowing why tells you when to reset the learning rate

*Field notes from a minimal-language-model study — yarns-and-loss project*

Teach a child language and you don't start with legal contracts — you start with "doggy,"
"more," "all gone." Curriculum learning takes that intuition literally: order a model's
training data from simple to complex and let difficulty rise over time. It's an old,
appealing idea with a mixed track record, and we ran a small controlled study that
explains the mixed record — and, in the process, turns a second folk trick (learning-rate
"warm restarts") into a testable principle.

**Start with the curriculum. It helps — conditionally.** On a deliberately tiny word-level
GPT (~4.6M parameters) trained on BabyLM child-language data at just 100,000 tokens,
ordering utterances simple→complex substantially improves a plain learned model: seeing
"baby talk" first curbs the catastrophic overfitting that wrecks small models in the
low-data regime. So far, so intuitive.

**Now the finding that stakes out the territory: a curriculum and a structured inductive
prior are substitutes, not complements.** When we gave the model a category-aware
grammatical prior — a frozen encoding that already knows determiners and negation are
different kinds of things — the *same* curriculum that helped the plain model now *hurt*
it. Why? Both mechanisms do the same job: they constrain the model before it has enough
data to constrain itself. Stack them and you double-count the same medicine; they
interfere rather than compound. "Does curriculum learning help?" turns out to be the
wrong question. The right one is "help *relative to what other prior?*" — and once you ask
it that way, the mixed literature stops looking mixed.

**That reframing is what makes warm restarts legible.** A curriculum creates something
valuable: genuine *regime changes* in the training stream — moments where the data
suddenly gets harder. Standard cosine decay fights those moments, because it drives the
learning rate to its lowest exactly when the hardest, newest data finally arrives. So we
reset the learning rate at each curriculum phase boundary — fresh plasticity precisely
when the model needs to absorb something new.

The clean result is the control: **warm restarts only help when the reset coincides with
the data actually changing.** Restarts *without* a curriculum are simply worse than a flat
schedule (1.666 vs 1.623 bits-per-character at full data) — the resets just knock the
model off a minimum it was approaching, for nothing. Same schedule, opposite outcomes,
depending entirely on alignment with the data stream. A learning-rate restart isn't a
free trick; it's a bet that the data has changed, and it pays off exactly when that bet is
true.

**The part I'm proudest of is the part that says "we were wrong."** An early single-seed
result had the full stack — prior + curriculum + restarts — as our best-ever low-data
number. It didn't survive. Re-run across three seeds on a common yardstick, that "best"
condition is actually the *worst* of the three, and a plain structured prior wins alone —
consistent with the substitutes finding above. Two confounds (a non-identical validation
set and a favourable random seed) had inflated the headline. At this scale, seed-to-seed
spread (±0.10 BPC) is as large as several effects people would happily publish. The
correction is the contribution: **in the low-data regime, report a seed band or don't
report.**

Why this is worth holding onto:

- It gives curriculum learning a **mechanistic account** instead of a reputation:
  curricula regularise, so they help models that lack other regularisation and interfere
  with models that already have it.
- It reframes cyclic/warm-restart schedules from "a trick that sometimes works" into a
  **statement about data-schedule alignment** — with a control design that can falsify it.
- It's a case study in how fragile low-data claims are, and how cheap rigor is when you
  build seed variance into the protocol from the start.

And none of it needs a GPU cluster. The whole study runs CPU-only on a laptop — which is
the point: small models are where inductive-bias questions live, and also where noise
most easily masquerades as insight.

*Minimal harness, full logs, and the seed-banded tables are in the project's REPORT.md
(sections "Curriculum ordering", "Warm restarts with curriculum phases", and
"Seed-variance robustness").*
