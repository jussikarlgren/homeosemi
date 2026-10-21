# Notes

## PINNED (2026-10-01): when should a ball split? — size vs coherence

Parked for a clearer head. Short version: we agreed to make ball-splitting *coherence-driven*
instead of *size-driven*, I tried to build the coherence test, and the routing space won't support
it with any linear metric. The better **cut** is implemented; the **trigger** is still size-based and
the real decision is pinned here. This note is the distinction to revisit.

### The two ways to decide a split — this is the fork

A ball has grown large. Do we cut it?

- **Size-triggered** (what runs now). "Larger than `ball_max` (40) → cut it in half, recurse." Simple,
  always terminates, predictable ball sizes. **Problem:** it cuts *coherent* balls too. A ball of 80
  genuine synonyms gets chopped into two arbitrary halves of 40, and because cross-ball similarity is
  null, we then *lose* the ability to compare the two halves. Size is a proxy for "too broad" and it's
  a blunt one.

- **Coherence-triggered** (what we wanted). "Cut a ball *only if it is actually holding two different
  neighbourhoods*; leave a big-but-coherent ball whole." This is the right behaviour. It needs a number
  that says how *heterogeneous* a ball is. That number is what we could not get.

The whole pin is: **we want coherence-triggered, but we can't measure coherence in the routing space.**

### Why every coherence metric I tried failed — and how they differ

All four are attempts to score "is this one neighbourhood or several?" They fail for the same underlying
reason, which is worth understanding:

1. **Tightness = mean cosine of members to the ball centroid.** Intuition: a tight ball is coherent.
   **Fails** because it is *size-confounded in the wrong direction*: the 2047-member blob scored 0.698,
   a clean 40-member ball scored 0.551. Bigger looks tighter. Reason → next paragraph.

2. **Half-separation = cosine between the two half-centroids after a trial cut.** Intuition: a coherent
   cloud splits into two near-identical halves (high cosine); a two-neighbourhood ball splits into two
   that point different ways (low cosine). **Fails** the same way — the 2047 blob's halves were 0.945
   similar. Both halves are dominated by the shared background, so they always look alike.

3. **Mean-centred residual half-separation.** Subtract the ball centroid first, *then* compare halves,
   to remove the shared component. **Fails** to bucket cleanly (0.4–0.85, no gap) because our old cut
   axis was the *farthest pair* — two outliers — not the real axis of spread.

4. **PC1 variance-ratio / bimodality** (principal component of the mean-centred residuals). Intuition:
   a two-neighbourhood ball has a dominant axis separating the two clumps; a coherent ball is isotropic
   noise. **Fails hardest and most informatively:** the numbers are *flat* — every ball, blob or clean,
   has PC1 variance ~0.19–0.245 and bimodality ~0.68. There is no dominant axis at all.

**Root cause (the thing to really absorb):** the **routing space** — the coarse 1000-D random index,
window-2 co-occurrence — *does not linearly encode the semantic category distinctions we want to split
on.* "Animate vs. abstract" is not a direction in that space; it's smeared across hundreds of dimensions,
each carrying a whisper. Two effects stack: (a) every routing vector is dominated by a shared common-
context component, so all centroids point roughly the same way, which is what sinks metrics 1–2; and
(b) what's left after removing it has no low-dimensional structure, which is what sinks 3–4. The coherence
we *see* in the neighbour lists comes from the **local ball spaces** (dim ~20, built from tighter
co-occurrence) — and those only exist *after* the balls are formed. Chicken and egg.

### What is implemented now

- **Principal-direction cut** (`_bisect`): mean-centre, power-iterate for the top eigenvector, project,
  median-split. This is a strictly better *cut* than the old farthest-pair axis (follows the ball's real
  geometry, not two outliers). Worth keeping regardless of how we resolve the trigger.
- **Trigger: size only** (`_split`), `ball_max=40`. The coherence gate is deliberately absent, not
  forgotten.

**Post-implementation check (2026-10-01, `topospace_all.pkl`, 253 balls / 92.8% coverage):** the
principal-direction cut did *not* improve noun-ball coherence vs. the old farthest-pair cut — tightly-
clustered classes stay clean (`said → continued, called, added, took`; `hand → face, cheeks, feet,
arm, fingers`) while size-forced REF splits stay arbitrary (`holmes → virginia, alice, merit, fervour`;
`king → pole, patient, doctor, wolf, dodo, pistol`). This is *expected* and reinforces the pin: cleaner
cut geometry cannot create a semantic boundary the routing space doesn't contain. The fix has to be in
the trigger (B/C above), not the cut.

### Candidate real fixes for the gate — the menu to revisit, and how they differ

- **(A) Richer routing representation.** Bigger window, PMI/idf weighting, or higher dim so categories
  become linearly separable. *Cost:* directly fights the economy thesis (the whole point is small/cheap).
- **(B) EM-style refinement (most promising + most in-keeping).** Form balls → build the local low-D
  spaces → re-measure coherence and re-split *using the local spaces*, where the distinction is actually
  visible. Iterate. Turns the chicken-and-egg into a loop. *Cost:* more machinery, convergence questions.
- **(C) Non-linear coherence.** Stop looking for a linear axis. Build a kNN graph inside the ball and
  look for modular structure — a two-neighbourhood ball is two dense subgraphs joined by a sparse cut,
  which graph clustering (modularity / min-cut) finds even when *no linear axis* separates them. This
  directly answers why 1–4 failed. *Cost:* graph build per ball.
- **(D) Decide granularity at join time, not split time.** Raise/By-class-tune `theta_spawn` so balls
  never over-merge in the first place, and drop post-hoc splitting to pure size-control. *Cost:* the
  single-theta-can't-serve-all-classes problem that started this whole thread.

My lean when we come back: **(C) or (B)** — they're the two that respect *why* the linear metrics failed,
rather than papering over it. (A) and (D) are retreats.

### How to reproduce the evidence

The three calibration sweeps (tightness, half-sep, PC1) were run against `topospace_all.pkl`'s routing
vectors via `b._cluster(...)` per class. Re-run by loading the pickle and recomputing per natural cluster;
the flat PC1 columns are the decisive result.
