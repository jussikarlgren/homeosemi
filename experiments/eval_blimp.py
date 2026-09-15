"""Evaluate a trained word-level GPT on BLiMP (Benchmark of Linguistic Minimal Pairs).

For each minimal pair (sentence_good, sentence_bad), the model assigns a log-
probability to each sentence; it is correct if log P(good) > log P(bad).
Accuracy is reported per paradigm and aggregated overall.

The word-level vocabulary will have OOV tokens; these are scored as <unk> and
their impact is tracked per paradigm (high OOV rate → less reliable score).

Usage:
  # After training with --save_checkpoint:
  python experiments/eval_blimp.py \\
      --checkpoint experiments/runs/word_category_para_phases_100k_s1337_ckpt \\
      --blimp_dir  experiments/blimp

  # Evaluate multiple checkpoints:
  python experiments/eval_blimp.py \\
      --checkpoint experiments/runs/word_learned_100k_s1337_ckpt \\
                   experiments/runs/word_category_100k_s1337_ckpt \\
      --blimp_dir experiments/blimp
"""
import argparse
import json
import math
import os
import re
import sys

import numpy as np
import torch
import torch.nn.functional as F

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "nanoGPT"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model import GPT, GPTConfig  # noqa: E402

_TOK = re.compile(r"[a-z0-9]+|[''']s|[''']t|[''']ve|[''']re|[''']ll|[''']d|[^\w\s]")

# BLiMP paradigm → linguistic category mapping (for grouped reporting)
PARADIGM_GROUPS = {
    "anaphor_gender_agreement":          "morphology",
    "anaphor_number_agreement":          "morphology",
    "determiner_noun_agreement_1":       "morphology",
    "determiner_noun_agreement_2":       "morphology",
    "determiner_noun_agreement_irregular_1": "morphology",
    "determiner_noun_agreement_irregular_2": "morphology",
    "determiner_noun_agreement_with_adj_2":  "morphology",
    "determiner_noun_agreement_with_adj_irregular_1": "morphology",
    "determiner_noun_agreement_with_adj_irregular_2": "morphology",
    "determiner_noun_agreement_with_adjective_1":     "morphology",
    "regular_plural_subject_verb_agreement_1": "morphology",
    "regular_plural_subject_verb_agreement_2": "morphology",
    "irregular_plural_subject_verb_agreement_1": "morphology",
    "irregular_plural_subject_verb_agreement_2": "morphology",
}

def load_model(ckpt_dir: str, device: str = "cpu"):
    config_path = os.path.join(ckpt_dir, "config.json")
    model_path  = os.path.join(ckpt_dir, "model.pt")
    with open(config_path) as f:
        cfg = json.load(f)

    gpt_cfg = GPTConfig(
        vocab_size = cfg["vocab_size"],
        n_layer    = cfg["n_layer"],
        n_head     = cfg["n_head"],
        n_embd     = cfg["n_embd"],
        block_size = cfg["block_size"],
        bias       = cfg.get("bias", False),
        dropout    = 0.0,
    )
    model = GPT(gpt_cfg)
    # nanoGPT ties wte.weight to lm_head.weight (model.py); training UNTIES before
    # saving (train_experiment.build_model), so the checkpoint has two independent
    # tensors. Untie here BEFORE load_state_dict, otherwise both state-dict keys land
    # on the same shared parameter and the input embedding is clobbered by the output
    # head — corrupting every eval. Safe for tied checkpoints too (identical values).
    model.lm_head.weight = torch.nn.Parameter(model.lm_head.weight.detach().clone())
    state = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model.to(device), cfg


def load_vocab(data_dir: str):
    import pickle
    with open(os.path.join(data_dir, "meta.pkl"), "rb") as f:
        meta = pickle.load(f)
    return meta["stoi"], meta["vocab_size"]


def tokenize(sentence: str, stoi: dict) -> list[int]:
    unk = stoi.get("<unk>", 0)
    tokens = _TOK.findall(sentence.lower())
    return [stoi.get(t, unk) for t in tokens]


@torch.no_grad()
def sentence_log_prob(model, ids: list[int], device: str) -> float:
    """Sum of log P(token_t | context) for t=1..n using teacher-forcing.

    nanoGPT only computes logits for the last token when targets=None.
    We pass targets to force full-sequence logits (B, T, vocab).
    """
    if len(ids) < 2:
        return 0.0
    x = torch.tensor(ids[:-1], dtype=torch.long, device=device).unsqueeze(0)
    y = torch.tensor(ids[1:],  dtype=torch.long, device=device)
    # passing y as targets forces nanoGPT to return logits for all positions
    logits, _ = model(x, y.unsqueeze(0))  # logits: (1, T, vocab)
    logits = logits.squeeze(0)            # (T, vocab)
    log_probs = F.log_softmax(logits, dim=-1)
    return log_probs[range(len(y)), y].sum().item()


def oov_rate(ids: list[int], unk_id: int) -> float:
    if not ids:
        return 0.0
    return sum(1 for i in ids if i == unk_id) / len(ids)


def evaluate_paradigm(path: str, model, stoi: dict, device: str):
    unk_id = stoi.get("<unk>", 0)
    correct, total, total_oov = 0, 0, 0.0
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            good = tokenize(item["sentence_good"], stoi)
            bad  = tokenize(item["sentence_bad"],  stoi)
            lp_good = sentence_log_prob(model, good, device)
            lp_bad  = sentence_log_prob(model, bad,  device)
            if lp_good > lp_bad:
                correct += 1
            total += 1
            total_oov += (oov_rate(good, unk_id) + oov_rate(bad, unk_id)) / 2
    acc     = correct / total if total else float("nan")
    avg_oov = total_oov / total if total else float("nan")
    return acc, avg_oov, total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", nargs="+", required=True,
                    help="checkpoint directory(s) created by --save_checkpoint")
    ap.add_argument("--blimp_dir",
                    default=os.path.join(_ROOT, "experiments", "blimp"))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default=None,
                    help="write results JSON (default: <ckpt_dir>/blimp_results.json)")
    args = ap.parse_args()

    paradigm_files = sorted(
        p for p in [os.path.join(args.blimp_dir, f)
                    for f in os.listdir(args.blimp_dir) if f.endswith(".jsonl")]
    )
    if not paradigm_files:
        print(f"No .jsonl files found in {args.blimp_dir}")
        sys.exit(1)
    print(f"Found {len(paradigm_files)} BLiMP paradigms")

    all_results = {}
    for ckpt_dir in args.checkpoint:
        tag = os.path.basename(ckpt_dir).replace("_ckpt", "")
        print(f"\n{'='*60}")
        print(f"Model: {tag}")
        model, cfg = load_model(ckpt_dir, args.device)
        stoi, _ = load_vocab(cfg["data_dir"])
        print(f"  vocab {cfg['vocab_size']} | "
              f"{cfg['n_layer']}L {cfg['n_head']}H {cfg['n_embd']}d")

        paradigm_results = {}
        accs = []
        print(f"  {'paradigm':50s} {'acc':>6s}  {'oov':>6s}  {'n':>5s}")
        for path in paradigm_files:
            name = os.path.splitext(os.path.basename(path))[0]
            acc, oov, n = evaluate_paradigm(path, model, stoi, args.device)
            paradigm_results[name] = {"acc": acc, "oov_rate": oov, "n": n}
            accs.append(acc)
            flag = " !" if oov > 0.3 else ""
            print(f"  {name:50s} {acc:6.3f}  {oov:6.3f}  {n:5d}{flag}")

        overall = float(np.mean(accs))
        print(f"\n  Overall BLiMP accuracy: {overall:.4f}  "
              f"(chance = 0.500)")
        all_results[tag] = {
            "overall_acc": overall,
            "paradigms":   paradigm_results,
            "config":      cfg,
        }

        # save per-checkpoint
        out = args.out or os.path.join(ckpt_dir, "blimp_results.json")
        with open(out, "w") as f:
            json.dump(all_results[tag], f, indent=2)
        print(f"  wrote {out}")

    # summary table if multiple checkpoints
    if len(args.checkpoint) > 1:
        print(f"\n{'='*60}")
        print("Summary:")
        print(f"  {'model':45s}  {'BLiMP acc':>10s}")
        for tag, res in all_results.items():
            print(f"  {tag:45s}  {res['overall_acc']:10.4f}")


if __name__ == "__main__":
    main()
