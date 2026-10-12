"""BLiMP eval WITH positional construction injection (Phase 2 of the config-aware test).

A checkpoint trained with --construction_pos carries a frozen con_table.pt; at eval the
model needs the same signal. For each sentence we recognise its constructions (same
recogniser as training), look up the frozen construction vectors, and add them at the
tagged positions via a forward pre-hook — then score the minimal pair as usual.

The recognised construction ids per sentence are cached once (BLiMP sentences are identical
across checkpoints) and reused for all checkpoints. We inject the REAL recognised
constructions for both `full` and `scrambled` models: `full` learned real associations and
should benefit; `scrambled` learned noise and should not.

Usage:
  python experiments/eval_blimp_construction.py --checkpoint <ckpt_dir> ... --paradigms_group construction
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "nanoGPT"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from eval_blimp import load_model, load_vocab, _TOK  # noqa: E402
from recognize_constructions import recognise_tokens  # noqa: E402
from analyze_construction import CONSTRUCTION_PARADIGMS, CONTROL_PARADIGMS  # noqa: E402


def attach_construction(model, ckpt_dir, device):
    """Load con_table.pt and register the injection hook; return con_table or None."""
    p = os.path.join(ckpt_dir, "con_table.pt")
    if not os.path.exists(p):
        return None
    con_table = torch.load(p, map_location=device)
    model._extra_emb = None
    def _hook(module, inp):
        if getattr(model, "_extra_emb", None) is not None:
            return (inp[0] + model._extra_emb,)
    model.transformer.drop.register_forward_pre_hook(_hook)
    return con_table


def prep_sentence(sentence, stoi):
    unk = stoi.get("<unk>", 0)
    toks = _TOK.findall(sentence.lower())
    ids = [stoi.get(t, unk) for t in toks]
    con = recognise_tokens(toks)            # aligned construction ids
    return ids, con


@torch.no_grad()
def logprob(model, ids, con, con_table, device):
    if len(ids) < 2:
        return 0.0
    x = torch.tensor(ids[:-1], dtype=torch.long, device=device).unsqueeze(0)
    y = torch.tensor(ids[1:], dtype=torch.long, device=device)
    if con_table is not None:
        cc = torch.tensor(con[:-1], dtype=torch.long, device=device)
        model._extra_emb = con_table[cc].unsqueeze(0)       # (1,T,C)
    logits, _ = model(x, y.unsqueeze(0))
    lp = F.log_softmax(logits.squeeze(0), dim=-1)
    return lp[range(len(y)), y].sum().item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", nargs="+", required=True)
    ap.add_argument("--blimp_dir", default=os.path.join(_ROOT, "experiments", "blimp"))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--paradigms", nargs="+",
                    default=sorted(set(CONSTRUCTION_PARADIGMS + CONTROL_PARADIGMS)))
    args = ap.parse_args()

    # cache (ids, con) per sentence per paradigm once, using the first ckpt's vocab
    stoi0, _ = load_vocab(json.load(open(os.path.join(args.checkpoint[0], "config.json")))["data_dir"])
    cache = {}
    for para in args.paradigms:
        path = os.path.join(args.blimp_dir, f"{para}.jsonl")
        if not os.path.exists(path):
            continue
        pairs = []
        for line in open(path, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            it = json.loads(line)
            pairs.append((prep_sentence(it["sentence_good"], stoi0),
                          prep_sentence(it["sentence_bad"], stoi0)))
        cache[para] = pairs
    print(f"cached {len(cache)} paradigms")

    for ck in args.checkpoint:
        cfg = json.load(open(os.path.join(ck, "config.json")))
        model, _ = load_model(ck, args.device)
        con_table = attach_construction(model, ck, args.device)
        tag = os.path.basename(ck).replace("_ckpt", "")
        results = {}
        for para, pairs in cache.items():
            correct = 0
            for (gi, gc), (bi, bc) in pairs:
                if logprob(model, gi, gc, con_table, args.device) > \
                   logprob(model, bi, bc, con_table, args.device):
                    correct += 1
            results[para] = {"acc": correct / len(pairs), "n": len(pairs), "oov_rate": 0.0}
        out = {"paradigms": results, "config": cfg, "construction_injected": con_table is not None}
        json.dump(out, open(os.path.join(ck, "blimp_results.json"), "w"), indent=2)
        inj = "con-injected" if con_table is not None else "plain"
        print(f"  {tag:34s} {inj:12s} "
              f"constr {np.mean([results[p]['acc'] for p in CONSTRUCTION_PARADIGMS if p in results]):.3f}")


if __name__ == "__main__":
    main()
