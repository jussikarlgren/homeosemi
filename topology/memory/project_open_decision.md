---
name: project-open-decision
description: RESOLVED 2026-09-08 — sub-hub reclassification + ball-splitting decision for the topology model; kept for the design rationale
metadata:
  type: project
---

**Resolved 2026-09-08.** Jussi took the recommendation on both open questions for the topology
model. Context: [[project-sentence-space]], structuring by lexical class [[feedback-basket-structuring]].

**What was decided & implemented in `ballspace.py`:**

1. **Sub-hubs → structural.** Pronouns (`PRP`/`WP`) and modals (`MD`) moved from REF/ARG to STRUCT
   in the Penn→coarse map, plus a `_STRUCTURAL_WORDS` lexical override that forces closed-class
   auxiliary/copula lemmas (be/is/was/have/had/do/did/will/would/can/could/…) to STRUCT even though
   the tagger labels them as verbs. This pulled the `the/him/them/my` hub out of the content classes.

2. **Ball granularity → divisive bisection, moderate theta.** `theta_spawn` raised 0.25→0.35;
   over-large balls (`> ball_max`, default 40) are recursively **2-means bisected** (`_bisect`/`_split`,
   farthest-pair seed + `split_iters` refinement). A split that would strand a part below `n_min` is
   refused. NOTE the first attempt — re-clustering a blob at a higher theta — was tried and **failed**:
   a percolated blob just re-absorbs into one leader (shaves a fringe, won't bisect). 2-means was the fix.

**Result on `data/alice.txt`:** the old single 128-member REF blob (animates+abstract+objects mixed)
became a 32-member mostly-animate ball (queen/king/hatter/gryphon/rabbit/duchess/dormouse/…). Coverage
improved hugely (buffered 960→~237). Residual limitation: size-based bisection cuts coherent groups too
(cat/mouse split off from queen/king), and a few non-animate stragglers remain — animacy isn't fully
separable at window=2 on one short book.

**Multi-file training (2026-09-08→10-01):** `traintopospace.py` now takes multiple `.txt` files
(`python traintopospace.py data/*.txt out.pkl`) and strips Project Gutenberg header/footer. `data/` has
alice, christie, dracula, frankenstein, sherlock (~457k words). Full-corpus model = `topospace_all.pkl`.

**PINNED (2026-10-01): coherence-gated splitting — see `NOTES.md` in the project root.** Jussi chose to
make splitting coherence-driven not size-driven, then pinned the decision ("can't get my head around it
right now") and asked for the distinction written up. Outcome: implemented the **principal-direction cut**
(`_bisect` = mean-centre + power-iteration PC1 + median split — better than farthest-pair), but the
**coherence GATE is unresolved and pinned**, because calibration proved *no linear metric on routing
vectors* distinguishes a big coherent ball from a big heterogeneous one (tightness is size-confounded
backwards; PC1 variance/bimodality are flat ~0.2/0.68 across all balls). Root cause: the coarse routing
space doesn't linearly encode the semantic distinctions; they live in the local ball spaces (built after
balls form) → chicken-and-egg. Splitting is therefore **size-triggered** (`ball_max=40`) for now.
Candidate fixes in NOTES.md: (A) richer routing, (B) EM refinement via local spaces, (C) non-linear /
kNN-graph modularity, (D) decide granularity at join time. My lean: (C) or (B). **Why:** these respect
*why* the linear metrics failed. **How to apply:** when Jussi returns to this, read NOTES.md first.

**Still not started:** benchmark validation (SimLex-999 / WordSim-353, the learns-faster claim).
