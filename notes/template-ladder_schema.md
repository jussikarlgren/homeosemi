# Template Ladder — Two-Tier Intention Schema (committed experimental pathway)

Companion to `notes/refined-hypothesis.txt`, `notes/experimental-plan_staged-priors-at-scale.md`.
Operationalises the "intentionality / communicative need" half of the usage-based hypothesis
as a **two-objective model**: the usual next-token LM **plus** an auxiliary **IE-template
slot-filling** objective that recovers each utterance's content. The template is the proxy for
"what the utterance is *for*." In Radical Construction Grammar the template is the *meaning pole*
of the construction; its slots are the roles (agents/patients) from `refined-hypothesis.txt`
line 22 — so roles become a **target**, not just an input overlay.

The template **escalates in lockstep with the curriculum complexity bands**, so each band
boundary coincides with three aligned events: new data complexity + new template sophistication
+ the warm-restart LR step-up. The ladder's order is itself a testable **acquisition-order**
claim ("processing the world AND the agent's self").

## Provisional defaults (not yet hard-confirmed)
- **Direction:** text → template (comprehension) first, not production. Schema works either way.
- **Supervision:** synthesise data from templates (gold slots free; paraphrase = one template
  realised in many constructions = the H3b generalisation signal; interleave 1:1,
  licensing-safe — honour the pilot's retraction + wrong-licensing lessons).

## Template format
Every utterance gets one template:

```
FORCE · PREDICATE(role=filler, …)
```

Two heads over the shared model: a per-utterance **force classifier** + a per-token
**role/slot tagger** (a filler is a token span). At band 4 a content slot may itself hold a
**nested template** (the theory-of-mind / embedding jump).

## Encoding substrate — Random Indexing, for BOTH sides (decided 2026-10-16, for parsimony)
One RI codebook end-to-end (base/item vectors for fillers, roles, predicates, force labels):
- **Input side:** RI signal biases the token embedding (structured — positional/per-occurrence
  or induced-graded; NOT a flat per-word membership tag, which the pilot certified null).
- **Output side:** the template target is a hypervector
  `T = force ⊕ bundleₖ(roleₖ ⊗ fillerₖ)` built from the SAME RI vectors
  (`⊗` = `sparsevectors.permute`, `⊕` = superposition); slots are read out by unbinding +
  **cleanup memory**. Novel filler = fresh RI vector in a known slot → productivity (H3b) by
  construction. Nested band-4 template = bind a sub-template hypervector into `content`.
- **Why:** comprehension and production share one vector code (representational closure).

**Guards.** (a) INFORMATION FIREWALL — the input RI prior must be base co-occurrence
(homeosemi) / induced participation only; it must NOT already contain the gold template the
output head predicts, or slot-filling is trivially solved (copy, not learn). (b) VSA capacity
— bundling has a crosstalk floor; band-4 recursion compounds it; set a dimensionality budget +
cleanup memory before committing band 4. Primitives in `jussipyutils/sparsevectors.py`
(alias gotcha: permute a dense copy — see CLAUDE.md).

## Band 1 — assert; 1 predicate, 1–2 participants; self as experiencer
Force: `{ASSERT}` · Predicates: `SEE(exp,stim)`, `WANT(agt,thm)`, `HAVE(poss,thm)`

| Utterance | Template |
|---|---|
| I see the dog | `ASSERT · SEE(exp=I, stim=dog)` |
| I want milk | `ASSERT · WANT(agt=I, thm=milk)` |
| you have the ball | `ASSERT · HAVE(poss=you, thm=ball)` |

## Band 2 — + request; transfer/location, 2 participants; other agents appear
Force: `{ASSERT, REQUEST}` · add `GIVE(agt,rec,thm)`, `PUT(agt,thm,loc)`, `BE_LOC(thm,loc)`

| Utterance | Template |
|---|---|
| give me the cup | `REQUEST · GIVE(agt=you, rec=me, thm=cup)` |
| she gives you milk | `ASSERT · GIVE(agt=she, rec=you, thm=milk)` |
| the cup is on the table | `ASSERT · BE_LOC(thm=cup, loc=table)` |

## Band 3 — + query, refuse, negation/modality; self's goals vs world
Force: `{ASSERT, REQUEST, QUERY, REFUSE}` · predicates gain `polarity` / `modal` features

| Utterance | Template |
|---|---|
| do you want milk? | `QUERY · WANT(agt=you, thm=milk)` |
| I can't find the ball | `ASSERT[modal=¬able] · FIND(agt=I, thm=ball)` |
| no, I won't | `REFUSE · (ellipsis → prior template)` |

## Band 4 — + mental-state attribution; embedding (theory-of-mind)
Force gains `{mental-state}`; **content slot accepts a nested template**

| Utterance | Template |
|---|---|
| I think you want milk | `ASSERT · THINK(exp=I, content=⟨WANT(agt=you, thm=milk)⟩)` |
| you know where it is | `ASSERT · KNOW(exp=you, content=⟨BE_LOC(thm=it, loc=?)⟩)` |

## Evaluation — how generalisation is measured
- **H3a (behavioural competence):** slot-filling accuracy — correct FORCE + correct roles
  filled — on held-out utterances.
- **H3b (productive generalisation):** two held-out splits per band —
  (i) **novel fillers** — a known predicate with a noun/verb never seen in that frame
      (productivity / wug test);
  (ii) **held-out predicates / forces** introduced in a later band — does the escalation
      transfer.
  Synthesise-from-templates yields gold for both for free.

## Open items (track against the pinned choice point)
- Confirm the per-band **force inventory** matches the intended speech-act acquisition order.
- Confirm **band-4 recursion** (nested template) is the right encoding for ToM/embedding, not
  too strong a jump.
- Concrete per-band grammar/force criteria for the curriculum sort.
- VSA dimensionality / cleanup-memory capacity check before committing band-4 recursion.

Smallest testable version: bolt a slot-filling head onto the existing tiny word-level model,
author band-1/2 templates, measure held-out-template slot accuracy + BPC vs a no-template
control — on the current CPU setup, no scaling needed.
