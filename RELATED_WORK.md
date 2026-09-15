# Related Work and Publication Notes

*yarns-and-loss — 2026-08-27*

This file covers prior work directly relevant to the experiments in this project,
the key criticisms that need to be addressed before publication, and what a
publishable framing of the contributions should look like.

---

## 1. The BabyLM Challenge — natural venue and baseline

The **BabyLM Challenge** (Warstadt et al., 2023; 2024) is the central community
effort on developmentally plausible, sample-efficient language model training. The
strict-small 10M track uses exactly our dataset (the Cambridge-CLIMB corpus subset),
making results directly comparable to published submissions.

Key resources:
- [BabyLM Challenge 2024 — ACL Anthology](https://aclanthology.org/events/babylm-2024/)
- [Findings of the Second BabyLM Challenge (arXiv:2412.05149)](https://arxiv.org/pdf/2412.05149)
- [Findings of the First BabyLM Challenge (arXiv:2504.08165)](https://arxiv.org/html/2504.08165v1)

**Implication for this work:** BabyLM evaluations use BLiMP (grammatical
acceptability judgements) and standard downstream tasks (GLUE subsets), not BPC.
A reviewer will ask for BLiMP scores to situate the results in the literature.
This is the most important methodological addition before writing up.

---

## 2. Paraphrase augmentation is already a known strong baseline — the critical prior work

**This is the most important finding from the literature search.**

The BabyLM 2024 findings report:

> "The most impactful data ingredient was sentence paraphrases, with the two best
> models being trained on (1) a mix of paraphrase data and BabyLM pretraining data,
> and (2) exclusively paraphrase data."

Haga et al. (2024) did essentially what this project does: generated synthetic
variation sets / paraphrases from CHILDES utterances using GPT-4 and found it
effective. Their hypothesis is that child-directed speech benefits LMs specifically
because of **variation sets** — consecutive rephrasings of the same content that
occur naturally in caregiver speech.

- [Haga et al. — Synthetic CDS variation sets](https://www.emergentmind.com/topics/synthetic-child-directed-data-augmentation)

**Implication:** paraphrase augmentation is not novel in itself. What would be
novel is the combination with a structured linguistic prior and the analysis of
*why* it works: specifically, paraphrases regularise *referential expressions*
(category 1 words) by providing multiple surface forms of the same referential
content — a mechanism distinct from "more data" and grounded in the acquisition
hypothesis about how referential meaning is learned.

---

## 3. Curriculum learning in BabyLM — prior results are mixed and ours resolves why

Multiple BabyLM 2023 submissions tried complexity-ordered curriculum learning and
found inconsistent results:

- [Modeling Easiness for Transformers with CL (RANLP 2023)](https://aclanthology.org/2023.ranlp-1.101/)
  — CL-LRC (length, rarity, comprehensibility) outperforms random ordering on BERT.
- [Curriculum Learning and Human Reading Behaviour (arXiv:2311.18761)](https://arxiv.org/pdf/2311.18761)
  — models underperform on fine-grained grammar but outperform on world-knowledge tasks.
- BabyLM 2023 submission on dependency-based ranking — dependency-based ranking
  helps, but most curriculum models underperform random baselines.
- [Strategic Data Ordering for LLMs (arXiv:2405.07490)](https://arxiv.org/abs/2405.07490)
  — attention-criteria ordering slightly helps for Mistral-7B/Gemma-7B.
- [Curriculum Learning for LLM Pretraining (arXiv:2601.21698)](https://arxiv.org/html/2601.21698v2)
  — analysis of learning dynamics under curriculum vs. random ordering.

**Implication and novel contribution:** our interaction finding — curriculum helps
learned baselines in the low-data regime but hurts structured-prior models because
they are partial *substitutes* — is a genuine new contribution to this debate.
Prior inconsistencies in the curriculum literature may be explained by whether or
not other inductive biases are present. This is a testable claim that unifies
previously contradictory results.

---

## 4. Effectiveness of child-directed speech is contested

[Feng, Goodman & Frank (EMNLP 2024)](https://aclanthology.org/2024.emnlp-main.1231.pdf)
directly asks "Is Child-Directed Speech Effective Training Data for Language Models?"
and finds mixed results depending on evaluation type. A 2025 preprint goes further:

- [Child-Directed Language Does Not Consistently Boost Syntax Learning (arXiv:2505.23689)](https://arxiv.org/html/2505.23689)
- [Child-directed speech facilitates production, not comprehension, in BabyLMs (2026)](https://arxiv.org/html/2606.01045)

**Implication:** we are not claiming CDS is better overall, but that it is the
right regime for testing a low-data structured prior — the regime where the prior's
advantage is largest and where referential expression acquisition is most constrained.
This is a narrower and more defensible claim than "CDS is good training data."
The BabyLM 2024 finding that CDS-heavy corpora improved world-knowledge tasks
(but not fine-grained grammar) is consistent with our hypothesis that referential
learning benefits most.

---

## 5. Random indexing / hyperdimensional computing

Random indexing (RI) and hyperdimensional computing (HDC) have been explicitly noted
as "largely overshadowed after Word2Vec/GloVe" in modern NLP (Kleyko et al., ACM
Computing Surveys 2023), though they retain interest for edge/low-power settings due
to single-pass learning and binary hypervectors.

- [HDC/RI Survey — ACM Computing Surveys 2023 (arXiv:2112.15424)](https://arxiv.org/pdf/2112.15424)
- [Word2HyperVec: Word Embeddings → Hypervectors (GLSVLSI 2024)](https://dl.acm.org/doi/10.1145/3649476.3658795)
- [Language Recognition using Random Indexing](https://www.researchgate.net/publication/269933115_Language_Recognition_using_Random_Indexing)

**Implication for framing:** position homeosemi not as competing with neural
embeddings on their own terms but as a tool for constructing **structured,
theory-motivated priors** that are computationally cheap, single-pass, and
linguistically interpretable. The relevant claim is not "RI beats transformers"
but "RI provides a principled way to inject linguistic category structure as a
frozen prior before training begins."

---

## 6. Frozen embedding initialisation in low-resource settings

[Welch et al. (arXiv:2009.14109)](https://arxiv.org/abs/2009.14109) show that
freezing in-domain embeddings improves low-compute LM performance, especially for
rare words. [Baziotis et al. (EMNLP 2020)](https://aclanthology.org/2020.emnlp-main.615/)
show that a language model prior as frozen regulariser improves low-resource NMT.
The FOCUS re-initialisation method (Minixhofer et al.) shows structured embedding
initialisation outperforms random initialisation especially when transformer blocks
are frozen.

- [Improving Low Compute LM with In-Domain Embedding Init (arXiv:2009.14109)](https://arxiv.org/abs/2009.14109)
- [Language Model Prior for Low-Resource NMT (EMNLP 2020)](https://aclanthology.org/2020.emnlp-main.615/)
- [Embedding structure matters: Adapting multilingual vocabularies (arXiv:2309.04679)](https://arxiv.org/pdf/2309.04679)

**Implication:** situate our frozen category embedding within this line of work.
We extend "in-domain statistical initialisation" to "linguistically motivated
categorical initialisation" — the distinction being that our prior encodes a
theoretical hypothesis about functional category structure, not just corpus
statistics.

---

## 7. Warm restarts / cosine annealing with restarts

Cosine annealing with warm restarts (SGDR; Loshchilov & Hutter, ICLR 2017) is
well-established for image classification. Its application to language model
pretraining with curriculum ordering has not been specifically studied; our finding
that phase-data alignment is required for the benefit to materialise (warm restarts
without curriculum hurt at full data) appears to be a novel empirical observation.

---

## 8. What is novel in this work — a proposed framing

The following are the contributions that go beyond prior work:

### 8.1 The non-uniform lexicon hypothesis, operationalised
Prior work on frozen or structured embeddings uses in-domain statistics or
cross-lingual transfer as the source of initialisation. This work proposes that
functional category membership — the distinction between referential expressions,
structural elements, predicative verbs, clause adverbials, and the TMA complex —
should drive the geometric structure of the embedding space, and that some
categories (structural, adv_clausal) should be treated as fixed grammatical
machinery rather than learned content. This is a linguistic hypothesis, not a
data-engineering choice.

### 8.2 TMA clause-type encoding
Propagating situational vectors (state / process / occurrence / unreal, following
Halliday's ideational metafunction) to every word in a clause — so that a word's
embedding reflects the situational contexts it typically appears in, not just its
co-occurrence statistics — is not represented in the prior literature found.

### 8.3 The interaction between structured priors, curriculum, and warm restarts
The finding that a structured prior and curriculum ordering are partial substitutes
(each helps individually, but combining them in the low-data regime hurts),
and that warm restarts only benefit when phase-aligned with curriculum complexity
boundaries, explains prior inconsistencies in the curriculum learning literature
and motivates a principled approach to combining these techniques.

### 8.4 Mechanism analysis
The claim that the benefit of the structured prior in the low-data regime is
primarily regularisation (frozen rows constrain the model against catastrophic
overfitting) rather than representation quality is supported by the val-loss
divergence curves and is a more precise claim than prior work's general attribution
of benefit to "inductive bias."

---

## 9. Critical methodological issues to address before publication

1. **No variance estimates.** All results are single runs, seed 1337. Run at least
   3–5 seeds for the key conditions (word_learned flat, word_category flat,
   word_category+paraphrase+phases at 100k) and report mean ± std.

2. **BPC is not the BabyLM evaluation metric.** Run the best condition through
   BLiMP and at least one downstream task to situate results in the literature.
   BPC measures compression; grammatical acceptability and downstream performance
   measure what the hypothesis is actually about.

3. **The linguistic-random freeze control is missing for the combined conditions.**
   `word_random_partial + curriculum + phases + paraphrase` would determine whether
   the benefit of the category encoding comes from linguistic structure specifically
   or just from regularisation via partial freezing. This is the single most
   important additional experiment.

4. **The properly regularised learned baseline is missing.** Compare against a
   learned baseline with tuned dropout and weight decay (not just the default
   settings) to avoid inflating the apparent benefit of structured priors against
   an overfit baseline.

5. **The 100k data-distribution confound.** At 100k tokens, curriculum-ordered data
   is not comparable to randomly ordered data — they have different vocabulary
   distributions. Acknowledge this explicitly and frame the 100k results as a
   study of the "early simple-vocabulary phase" rather than a controlled quantity
   comparison.

6. **Paraphrase LLM provenance.** Generated paraphrases are from Claude (claude-haiku-4-5),
   a model trained on vastly more data. Acknowledge that some of the benefit may
   reflect Claude's distributional regularities rather than naturalistic CDS.
   A rule-based or smaller-model paraphrase condition would tighten the claim.

7. **TMA classifier validation.** The clause-type heuristics are unvalidated
   against human judgements. Report inter-rater agreement on a sample or compare
   against a dependency-parsed gold standard before claiming the TMA distributions
   are linguistically meaningful.

---

## 10. Suggested publication venue

**BabyLM Challenge (EMNLP/CoNLL workshop)** is the primary target — the dataset,
evaluation, and community overlap are exact. The framing should emphasise the
linguistic hypothesis and the interaction findings rather than the absolute BPC
numbers.

A longer version with the mechanistic analysis and TMA theory could target
**CoNLL** (main conference) or **CogSci** if the cognitive science angle
(non-uniform lexicon, Halliday's clause-type theory, developmental plausibility)
is foregrounded.
