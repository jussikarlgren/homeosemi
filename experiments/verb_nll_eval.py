"""Per-verb complement-position NLL on the val set, split regular vs irregular.

For each occurrence of a mapped lexical verb at position i in val, record the model's
NLL of the NEXT token (the argument/complement position) — teacher-forced loss at
position i. Average per verb type, then bucket verbs into regular / irregular (relative
to their Levin class, see verb_regularity) and report mean NLL per bucket per encoding.

The overregularization prediction: a class-based ("rule") prior (word_levin) lowers
complement-NLL for regular (class-conforming) verbs but RAISES it for irregular
(class-deviant) verbs, and the irregular penalty widens from 100k -> 1M. Because every
number is a delta vs word_learned on the same val with the same blocking, block-boundary
effects cancel.

Usage:
  python experiments/verb_nll_eval.py --checkpoints experiments/runs/word_*_lit_{100k,1M}_s*_ckpt
"""
import argparse
import glob
import json
import os
import re
import sys
import collections

import numpy as np
import torch
import torch.nn.functional as F

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "nanoGPT"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from eval_blimp import load_model  # noqa: E402  (reuse checkpoint loader)
from verb_regularity import get_regularity  # noqa: E402
import pickle  # noqa: E402


@torch.no_grad()
def per_verb_nll(model, val, verb_ids, block_size, device,
                 max_val_tokens, batch_blocks=64):
    """Mean NLL of the token following each mapped-verb occurrence, per verb id."""
    sum_nll = collections.defaultdict(float)
    cnt = collections.defaultdict(int)
    T = block_size
    # Spread blocks across the WHOLE val (strided), not the first N contiguous tokens —
    # the val is domain-heterogeneous and a contiguous prefix is unrepresentative.
    n_blocks = max(1, min(max_val_tokens // T, (len(val) - T - 1) // T))
    stride = max(T, (len(val) - T - 1) // n_blocks)
    starts = [k * stride for k in range(n_blocks) if k * stride + T + 1 <= len(val)]
    for b0 in range(0, len(starts), batch_blocks):
        batch = starts[b0:b0 + batch_blocks]
        x = torch.stack([torch.from_numpy(np.asarray(val[s:s + T], dtype=np.int64)) for s in batch]).to(device)
        y = torch.stack([torch.from_numpy(np.asarray(val[s + 1:s + 1 + T], dtype=np.int64)) for s in batch]).to(device)
        logits, _ = model(x, y)                       # (B, T, V)
        logp = F.log_softmax(logits, dim=-1)
        nll = -logp.gather(-1, y.unsqueeze(-1)).squeeze(-1)   # (B, T) NLL of token[i+1]
        xnp = x.cpu().numpy(); nnp = nll.cpu().numpy()
        for bi in range(len(batch)):
            row = xnp[bi]
            for i in range(T):
                vid = int(row[i])
                if vid in verb_ids:
                    sum_nll[vid] += float(nnp[bi, i])
                    cnt[vid] += 1
    return {vid: sum_nll[vid] / cnt[vid] for vid in sum_nll if cnt[vid] > 0}, dict(cnt)


def _parse_tag(ckpt):
    m = re.search(r"word_(\w+?)_lit_(\d+k|1M)_s(\d+)_ckpt", os.path.basename(ckpt))
    return (m.group(1), m.group(2), int(m.group(3))) if m else (None, None, None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoints", nargs="+", required=True)
    ap.add_argument("--data_dir", default=os.path.join(_ROOT, "experiments", "data", "babylm_lit"))
    ap.add_argument("--max_val_tokens", type=int, default=1_000_000)
    ap.add_argument("--min_occ", type=int, default=30, help="min val occurrences to score a verb")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default=os.path.join(_ROOT, "experiments", "runs", "verb_nll.json"))
    args = ap.parse_args()

    ckpts = []
    for c in args.checkpoints:
        ckpts.extend(sorted(glob.glob(c)) if any(ch in c for ch in "*?[") else [c])

    meta = pickle.load(open(os.path.join(args.data_dir, "meta.pkl"), "rb"))
    stoi = meta["stoi"]
    reg, _ = get_regularity(meta)               # {verb: {label, class_name, deviation}}
    verb_ids = {stoi[w]: w for w in reg if w in stoi}
    val = np.memmap(os.path.join(args.data_dir, "val.bin"), dtype=np.uint16, mode="r")

    # per-checkpoint per-verb NLL (cached)
    per_ckpt = {}
    for ck in ckpts:
        enc, bud, seed = _parse_tag(ck)
        model, _ = load_model(ck, args.device)
        nll, cnt = per_verb_nll(model, val, set(verb_ids), meta.get("block_size", 128) or 128,
                                args.device, args.max_val_tokens)
        # keep only well-sampled verbs
        vals = {verb_ids[vid]: nll[vid] for vid in nll if cnt.get(vid, 0) >= args.min_occ}
        per_ckpt[os.path.basename(ck)] = {"enc": enc, "budget": bud, "seed": seed, "nll": vals}
        print(f"{os.path.basename(ck):42s} enc={enc:8s} bud={bud:4s} s={seed}  "
              f"scored {len(vals)} verbs")

    # aggregate: for each (enc,budget) mean over seeds of per-bucket mean NLL,
    # and delta vs word_learned at the same budget
    def bucket_means(enc, bud):
        # average across seeds of (mean NLL over regular / irregular verbs)
        regs, irrs = [], []
        for k, v in per_ckpt.items():
            if v["enc"] == enc and v["budget"] == bud:
                r = [nll for w, nll in v["nll"].items() if reg[w]["label"] == "regular"]
                ir = [nll for w, nll in v["nll"].items() if reg[w]["label"] == "irregular"]
                if r:  regs.append(np.mean(r))
                if ir: irrs.append(np.mean(ir))
        return (np.mean(regs) if regs else np.nan, np.std(regs) if regs else np.nan,
                np.mean(irrs) if irrs else np.nan, np.std(irrs) if irrs else np.nan)

    encs = sorted({v["enc"] for v in per_ckpt.values()})
    buds = [b for b in ("100k", "300k", "1M") if any(v["budget"] == b for v in per_ckpt.values())]
    print("\n=== complement-position NLL (mean+/-sd over seeds), regular | irregular ===")
    table = {}
    for bud in buds:
        print(f"\n[{bud}]  {'encoding':>12s}   {'regular':>14s}   {'irregular':>14s}   {'irr-reg':>8s}")
        base = bucket_means("learned", bud)
        for enc in encs:
            rm, rs, im, isd = bucket_means(enc, bud)
            table[(enc, bud)] = {"reg": rm, "irr": im}
            print(f"        {enc:>12s}   {rm:6.3f}+/-{rs:.3f}   {im:6.3f}+/-{isd:.3f}   {im-rm:+8.3f}")
        # overregularization deltas vs learned
        if not np.isnan(base[0]):
            print(f"        {'-> delta vs learned (reg / irr):':>44s}")
            for enc in encs:
                if enc == "learned":
                    continue
                dr = table[(enc, bud)]["reg"] - base[0]
                di = table[(enc, bud)]["irr"] - base[2]
                print(f"        {enc:>12s}   reg {dr:+.3f}   irr {di:+.3f}")

    json.dump({k: v for k, v in per_ckpt.items()}, open(args.out, "w"), indent=2, default=float)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
