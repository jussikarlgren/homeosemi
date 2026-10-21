---
name: feedback-basket-structuring
description: Structure the topology basket by lexical class (5 POS categories), NOT by idf
metadata:
  type: feedback
---

The topology basket-of-balls model ([[project-sentence-space]]) structures the basket by **lexical
class**, not by frequency/idf. Pre-group words (via POS tagging) into **five coarse categories** and
cluster into balls *within* each category:
1. referential terms (~nouns), 2. attribute terms (~adjectives), 3. structural items (preps, conjs,
determiners), 4. argument-structure terms (~verbs), 5. adverbials. A successful grouping yields balls
for animates, foods, etc. within referential; transitive/stative within verbs; adverbial types, etc.

**Why:** the user (2026-09-01) explicitly rejected idf as the criterion in favour of lexical
information. It also solves the hub/function-word blob *structurally* — structural items are their own
class and can't bridge content-word clusters — rather than via brittle frequency weighting. Superseded
the earlier idf-weighting decision (idf was a stopgap for the same blob problem; POS class replaces it).

**How to apply:** POS-tag with nltk (`averaged_perceptron_tagger_eng`, already downloaded); map Penn tags
to the 5 coarse classes; assign each word its dominant class; run within-class clustering. The
`routing_weighting` toggle still exists in `ballspace.py` but defaults OFF now. Ball genesis itself uses
greedy leader/centroid clustering (single-link percolated into one giant ball; see plan). Revisit
candidates: per-occurrence multi-class membership (vs dominant-class), and whether structural items
should cluster at all or stay null.
