# Research programme: data-efficient language modelling from precompiled linguistic resources

*Forward-looking synthesis of the yarns-and-loss experiments. The through-line: when you
must learn language from little raw data, precompiled linguistic structure — grammars,
lexica, category systems, construction inventories — is leverage. The open questions are
**which** structure, injected **how**, and how to **certify** it is doing real work rather
than merely regularising.*

---

## The positive arc, read as one programme

**1. Structured priors pay off most exactly where you need them — the low-data regime.**
A category/frame-aware encoding beats a from-scratch learned embedding by ~7–10% BPC at
100k tokens, and the edge narrows as data grows (crossover ~0.5–1M tokens). For this
programme the crossover is not a limitation — it is the *operating point*: below it a
precompiled prior is pure gain. This is the foundational evidence that grammars/lexica can
**substitute for data**.

**2. The biggest low-data win came from a lexical-grammatical resource injected as
structure — and it is cheaply bootstrappable.** Verb subcategorisation frames (an
argument-structure resource, the kind a valency lexicon or grammar supplies) were the
strongest low-data encoding (−0.088 BPC over the coarse category prior) and were reliably
assignable from ~10 observations. So a precompiled resource — or one rapidly induced from a
seed — measurably improves data efficiency. And **finer, linguistically grounded structure
helped more than coarse categories** — the encouraging signal for investing in richer
descriptions.

**3. "Freeze then thaw" makes a precompiled prior a no-regret accelerator, not a crutch.**
Staged unfreeze kept essentially the full low-data benefit while shedding the high-data
penalty. The sustainable recipe in one move: **start from the grammar/lexicon as a frozen
scaffold, learn from data, and release the prior as the model earns the right to override
it** — per-feature, at the crossover or on anomaly (thaw-on-contradiction). The verb U-curve
showed the item-level version working: releasing a frozen *rule* let the model lexicalise
its exceptions — "grammar for the schema, data for the idiosyncrasies."

**4. The category-overlay dissociation is a design principle, not a null.** A
magnitude-matched scramble control showed that *coarse category labels act as
regularisation*, while injected linguistic content changes learning only at **finer,
item/construction-level granularity** (where the verb-frame prior lives). This is a map of
where to invest precompiled effort — rich, item-specific structure (valency frames,
argument-structure constructions, idioms), not flat POS-level tags — and it hands you the
instrument to vet any candidate resource (below).

**5. Combine priors expecting substitution, not addition.** Category prior ≈ curriculum;
syntactic frame ≈ semantic (Levin) class — priors that target the same under-determination
substitute rather than stack, and can interfere. Budget the injected structure.

---

## Where construction grammar fits (and the evidence it already has)

Construction grammar is well-placed by these results, not incidentally:

- **The strongest positive result is already constructional.** Argument-structure
  constructions *are* the verb subcat-frames that gave the best low-data win.
- **The dissociation points toward it.** "Fine, item/construction-level structure beats
  coarse labels" is a verdict in favour of construction-level representations and against
  POS categories.
- **The freeze-thaw dynamic maps onto CxG's schema/idiosyncrasy structure** — productive
  constructions as frozen schemas, lexicalised exceptions thawed, with a Tolerance-Principle
  style productivity threshold as a candidate thaw trigger.

Crucially the method is **formalism-agnostic**: construction grammar is one computable path
to the fine structure that helps, and a good bet — but the certification test below scores
any resource, so the programme can follow the evidence if some other source of computable
fine structure scores better.

---

## Methodological value — the durable deliverable

1. **A certification test for any precompiled resource — the magnitude-matched scramble
   control.** Does injecting resource R beat a permuted, magnitude-matched version of R? If
   yes, R carries usable linguistic content; if not, it is regularisation. Run this gate
   *before* building a pipeline on a grammar/lexicon.
2. **A deployment recipe.** Inject as a frozen scaffold → measure the per-feature crossover
   → thaw at/after it (or on anomaly). Turns precompiled structure into a general
   accelerator rather than a low-data-only crutch.
3. **A granularity heuristic.** Spend on item/construction-level structure; coarse tags
   mostly regularise.
4. **A rigour floor.** Seed bands + controls, non-negotiable when the whole point is many
   small-data claims across resources and languages. (The determiner follow-up — a 3-seed
   +0.046 hint dissolving to +0.027 ± 0.059 at 9 seeds — is this floor working: it caught a
   false positive before it entered the record.)
5. **Eval/infra hygiene.** Verify a reloaded checkpoint reproduces its training-time metric
   (a silent tied/untied loader bug once corrupted a whole class of results); ranking and
   likelihood metrics can fail independently; run one job you can observe rather than
   timer-relaunching processes you cannot see.

---

## The concrete next experiment

Apply the acquisition test to *constructions*: inject a small inventory of argument-structure
constructions (frozen), and check whether it **beats its magnitude-matched scramble** on the
construction-sensitive BLiMP paradigms. If constructional content clears the scramble bar
where flat categories did not, that is direct evidence for construction grammar as a
computable, data-efficient resource — and the freeze-thaw schedule tells you how to deploy
it. Pair it with a crossover-vs-model-size measurement to establish the data budget over
which the constructional prior remains in-regime.
