# The lexicon is not uniform, and your input encoding should know it

*Field notes from a minimal-language-model study — yarns-and-loss project*

Most language models start every word as a blank slate — a randomly initialised
embedding the model has to learn from scratch. That's fine when you have billions of
tokens to learn from. But it quietly assumes the vocabulary is *uniform*: that a
determiner, a pronoun, a verb, and a negation particle are all just points to be placed
by data. They aren't. Grammar treats them as different *kinds* of things. We asked a
simple question: **if you build that categorical structure into the input encoding
itself, does the model learn better — and when?**

The answer is clean and it has a crossover.

**At low data, a structured, category-aware encoding wins decisively — and it's not a
lucky seed.** On a tiny word-level GPT trained on just 100,000 tokens of child-language
data, an encoding that distinguishes functional categories (referential, structural,
predicative, clause-adverbial) and *freezes* the "grammatical machinery" rows beats a
standard learned embedding by about **7% in bits-per-character**, averaged over three
random seeds — and the separation is total: our *worst* structured seed still beats our
*best* learned seed. The mechanism is regularisation. With 100k tokens the learned model
overfits catastrophically; the frozen grammatical scaffold constrains it in exactly the
regime where the model can't estimate word relationships from data alone. It's a
computational echo of the idea that innate categorical biases scaffold early language
acquisition in children.

**And then, predictably, the advantage dies.** Push past roughly 500k–800k tokens and
the learned baseline overtakes the structured prior; by 1M tokens and beyond, the prior
is a slight *handicap*. Once there's enough data to learn the geometry directly, a fixed
structure just gets in the way. This crossover is the whole story: **structured priors
are a low-data instrument.** They buy you the most exactly when data is the binding
constraint, and they should be spent there.

Two sharper findings stake out the territory:

- **Distinguishability beats similarity when the alphabet is small.** We also tried
  encodings built from *distributional similarity* — making co-occurring tokens point in
  similar directions. At the character level (65 symbols) this actively *hurts*: it
  collapses the rank of the input space, and the model wastes capacity pulling symbols
  back apart. Similarity-based encodings only earn their keep when the vocabulary is
  large and similarity is genuinely predictive — i.e. at the word level. The right prior
  depends on the size and structure of the symbol set, not on a universal recipe.

- **More parameters are not a substitute for the right prior.** Subword (BPE)
  tokenization with 3× the parameters was clearly *worse* than plain word-level at this
  scale, and in a companion experiment we found that scaling the model up *amplified* a
  representational trade-off rather than dissolving it. Capacity doesn't rescue a
  mismatched encoding; it commits to it harder.

Why this matters beyond a toy model: the low-data regime is where linguistics, cognitive
science, and machine learning actually meet. It's where the assumptions you build in — the
*inductive bias* — dominate, because there isn't enough data to wash them out. A
category-aware encoding is a concrete, measurable way to inject linguistic knowledge into
a model's very first layer, and to ask which distinctions are worth their weight. Our
result says: the *coarse* functional categories are worth it; splitting them finer isn't
(it needs more data than the regime affords). That's a falsifiable map of where
linguistic structure pays off, drawn on a laptop, CPU-only, with seed bands on every
number.

*Full tables — char-level encodings, the word-level low-data sweep, the α-overlay sweep,
the BPE baseline, and the three-seed robustness check — are in the project's REPORT.md.*
