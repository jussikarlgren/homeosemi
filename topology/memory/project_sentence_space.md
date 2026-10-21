---
name: project-sentence-space
description: Topology project — basket-of-balls-of-yarn local RI semantic space (supersedes the sentence-space framing)
metadata:
  type: project
---

Goal: build the **topology / basket-of-balls-of-yarn** semantic-space prototype in the `topology` working dir, reusing the user's own code: `homeosemi` (core RI `SemanticSpace`) + `jussipyutils` (`sparsevectors`/`languagemodel`/`logger`). Both cloned from github.com/jussikarlgren.

**The real project (per `topospacehypothesis.txt`, JiK Sept 2026):** a single global high-D RI space is *overmodelled*. Replace it with a **basket** (coarse global random index) over ~1000 small **balls of yarn** (local neighbourhoods, each its own low-D RI space, ~10-20D). Items (words first, sentences/utterances later) route by local context into one or more balls. Cross-ball similarity returns **null**, not a low cosine. Claim to test later: learns faster / needs less data than a full space (SimLex/WordSim benchmarks). The old `homeosemi` `cspace` IS the monolithic space he rejects — do NOT reproduce `makebgwordspace.py`; only reuse its primitives.

**Design forks he chose (2026-09-01):** routing = the item's own accumulating context vector (buffer until mature, f_min~3); ball genesis = grow from buffer (spawn a ball when a cluster of similar buffered items reaches n_min~5). Out of scope at outset: ball split/merge, constructional features, sentence items.

Approved plan lives at `C:\Users\jkarlgre\.claude\plans\stateless-watching-bonbon.md`. Files being built: `ballspace.py` (`Ball`, `Basket`), `traintopospace.py`, `studytopospace.py`, `_paths.py`.

**Why:** research prototype for his semantic-topology hypothesis (dir is named `topology`).
**How to apply:** vectors are dicts (sparse ternary). jussipyutils + homeosemi must be on sys.path (flat imports). Each ball = a `SemanticSpace` at small dim. Corpus: user will provide a local text file (still needed to wire the end-to-end driver). See [[user-identity]].
