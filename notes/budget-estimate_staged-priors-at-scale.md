# Compute Budget Estimate — Staged Priors at Scale (9B / 18B / 30B)

Companion to `notes/experimental-plan_staged-priors-at-scale.md`. Costs the three-experiment
plan at three model sizes (medium 9B → huge 30B).

**Headline:** under the plan as written, ~**40–50k H100-hours (≈ $110k–180k), ~3–5 weeks
calendar** on a ~128-GPU cluster — roughly one frontier pretraining run's worth of compute.
A premise-level tension (small developmental corpus vs. 9–30B params) could move this ~10× in
either direction; see Caveats.

## Method & assumptions
- **Training FLOPs** ≈ `6 · N · D` (params × tokens), the standard estimate. Calibration:
  reproduces LLaMA-2-7B's ~180k A100-hr for 2T tokens, so the FLOP model is sound.
- **Hardware:** H100 at ~40% MFU → **1.44e18 FLOP per H100-hour** (eff. 4e14 FLOP/s). Eval
  (BLiMP, BPC on shared val) is <5% of training cost and is ignored.
- **Crossover anchor (load-bearing assumption):** the pilot's crossover sat at ~0.1–0.17
  tokens/param (≈500–800k tokens for a 4.6M model; *Low-data sweep, 2026-08-20*). Assume H1's
  ratio roughly holds: sweep Exp 1 to **0.5 tok/param** (well past expected crossover); train
  Exp 2/3 to **~0.3 tok/param**. **If crossover instead scales toward Chinchilla (20
  tok/param), every number below rises ~40–100×** — the single biggest budget risk, and
  exactly what Exp 1 exists to resolve.
- **Efficiency baked in:** Exp 1's crossover sweep is *one training trajectory per cell* with
  val-BPC logged at log-spaced checkpoints, not a separate run per budget (~5× saving).

## Per-run unit cost
| Model N | H100-hr per 1B tokens |
|---|---|
| 9B  | 37.5 |
| 18B | 75 |
| 30B | 125 |

## Per-experiment totals
| Experiment | Sizes | Runs | Tokens/run | H100-hr |
|---|---|---|---|---|
| **1 — Crossover** (2 enc × 3 seed = 6/size) | 9/18/30B | 18 | 0.5 tok/param (trajectory) | ~16,300 |
| **2 — Staged scaffold + restarts** (8 cond × 3 seed = 24/size) | 9+18B | 48 | 0.3 tok/param | ~12,150 |
| **3 — Configurational construction** (3 cond × 3 seed = 9/size) | 18+30B | 18 | 0.3 tok/param | ~13,770 |
| *optional verb-freeze U-curve probe* | 18+30B | ~6 | 0.3 tok/param | ~4,500 |
| **Total** | | **~84 runs** | | **≈ 42k (≈ 47k w/ probe)** |

Central estimate **~45k H100-hr**; plausible range **30–70k** (seeds, MFU, actual budgets).
Cost at $2.5–4/H100-hr (reserved → on-demand): **~$110k–180k**.

### Worked sub-totals
- **Exp 1:** 9B D=4.5B → 169/run ×6 = 1,013 ; 18B D=9B → 675/run ×6 = 4,050 ;
  30B D=15B → 1,875/run ×6 = 11,250. Subtotal ≈ 16,300.
- **Exp 2** (9B+18B): 9B D=2.7B → 101/run ×24 = 2,430 ; 18B D=5.4B → 405/run ×24 = 9,720.
  Subtotal ≈ 12,150.
- **Exp 3** (18B+30B): 18B D=5.4B → 405/run ×9 = 3,645 ; 30B D=9B → 1,125/run ×9 = 10,125.
  Subtotal ≈ 13,770.

## Wall-clock
Longest single run is 30B → 9B tokens (~1,125 H100-hr); on ~32 H100 (realistic TP+DP for 30B)
that's ~1.5 days. Calendar time is set by parallel packing:

| Standing allocation | Packed compute | Realistic calendar (×1.5–2 for failures/queue) |
|---|---|---|
| 64× H100  | ~29 days | ~6–8 weeks |
| 128× H100 | ~15 days | **~3–5 weeks** |
| 256× H100 | ~7 days  | ~2–3 weeks |

## Cost levers (if you need to cut)
1. **Drop 30B seeds to 2, or confirm 30B only for the winning condition** — Exp 2/3's 30B cells
   are ~2/3 of their cost.
2. **Tighten D_max** — halving the sweep ceiling ~halves Exp 1.
3. **18B is optional** — a 9B/30B bracket still yields a two-point scaling curve.
Lean variant (9B/30B only, 2 seeds, capped budgets): **~20–25k H100-hr / ~2–3 weeks**.

## Caveats that could reshape the budget
- **Premise tension (resolve before costing).** The pilot's framing is a *small,
  developmentally-plausible* corpus (BabyLM ≈ 10M words). At 9–30B params the plan's budgets
  are either (a) kept small → models are wildly overparameterized, sit permanently in the
  overfit regime where priors always help, so a crossover may not exist on that corpus; or
  (b) scaled to billions of tokens to reach crossover → the developmental-plausibility premise
  is gone and you need a corpus 100–1000× larger than BabyLM. You can't have both. This is a
  design fork, not a compute issue — it may argue for much smaller models, or for reframing
  "scale" as *data-scale at fixed small model size*.
- **Infrastructure port.** The pilot is CPU nanoGPT. None of the apparatus (per-category
  frozen-row gradient masks, frozen overlays, construction-recognizer forward-hook, curriculum
  loader, untie-before-load eval fix) exists in a distributed trainer. Porting to
  FSDP/Megatron at 9–30B is real engineering (weeks), not included in the FLOP totals above.

---
All section/date references map to `REPORT.md` headings and `EXPERIMENT_DIARY.md` dated
entries. FLOP/MFU assumptions stated inline; revise `cost_per_1B = 6·N·1e9 / (MFU·peak·3600)`
if hardware or MFU changes.
