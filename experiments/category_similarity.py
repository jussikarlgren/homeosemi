"""Measure mean intra- vs inter-category cosine similarity in an embedding matrix.

Used to verify that the category RI overlay pulls category-mates together,
and to track whether training preserves or destroys category geometry.

Usage:
  # at init (from a run JSON + meta):
  python experiments/category_similarity.py --meta experiments/data/babylm_simple/meta.pkl \\
      --encoding word_category --overlay_alpha 0.5

  # or compare two run JSONs (before/after training):
  python experiments/category_similarity.py \\
      --meta experiments/data/babylm_simple/meta.pkl \\
      --runs experiments/runs/word_category.json
"""
import argparse
import os
import pickle
import sys

import numpy as np
import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "nanoGPT"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import encoders  # noqa: E402
from categorize import CATEGORY_NAMES  # noqa: E402


def cosine_matrix(mat):
    """Return (N, N) cosine similarity matrix for rows of mat."""
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    normed = mat / norms
    return normed @ normed.T


def category_stats(emb_matrix, word_categories, vocab):
    """Return per-category mean intra- and inter-category cosine."""
    mat = emb_matrix.numpy() if hasattr(emb_matrix, "numpy") else emb_matrix
    C = cosine_matrix(mat)
    cat_ids = np.array([word_categories.get(w, 0) for w in vocab])
    results = {}
    for cid, cname in CATEGORY_NAMES.items():
        members = np.where(cat_ids == cid)[0]
        if len(members) < 2:
            continue
        intra = C[np.ix_(members, members)]
        np.fill_diagonal(intra, np.nan)
        intra_mean = np.nanmean(intra)
        others = np.where(cat_ids != cid)[0]
        inter_mean = C[np.ix_(members, others)].mean() if len(others) else float("nan")
        results[cname] = {"intra": float(intra_mean), "inter": float(inter_mean),
                          "n": len(members)}
    return results


def print_stats(stats, label=""):
    print(f"\n{'Category similarity':=<55} {label}")
    print(f"  {'category':18s} {'n':>6s} {'intra':>8s} {'inter':>8s} {'gap':>8s}")
    for cname, s in stats.items():
        gap = s["intra"] - s["inter"]
        print(f"  {cname:18s} {s['n']:6d} {s['intra']:8.4f} {s['inter']:8.4f} {gap:8.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", required=True,
                    help="meta.pkl from prepare_words.py")
    ap.add_argument("--encoding", default="word_category",
                    choices=encoders.WORD_ENCODINGS)
    ap.add_argument("--n_embd", type=int, default=128)
    ap.add_argument("--denseness", type=int, default=10)
    ap.add_argument("--window", type=int, default=5)
    ap.add_argument("--enc_tokens", type=int, default=500_000)
    ap.add_argument("--overlay_alpha", type=float, default=0.5)
    ap.add_argument("--emb_scale", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    with open(args.meta, "rb") as f:
        meta = pickle.load(f)
    vocab = [meta["itos"][i] for i in range(meta["vocab_size"])]
    word_categories = meta["word_categories"]

    data_dir = os.path.dirname(args.meta)
    import numpy as np
    token_ids = np.memmap(os.path.join(data_dir, "train.bin"),
                          dtype=np.uint16, mode="r")

    print(f"Building '{args.encoding}' embedding ...")
    mat = encoders.build_word(
        args.encoding, vocab, args.n_embd,
        token_ids=np.array(token_ids[:args.enc_tokens], dtype=np.int32),
        word_categories=word_categories,
        denseness=args.denseness, window=args.window,
        overlay_alpha=args.overlay_alpha,
        enc_tokens=args.enc_tokens,
        emb_scale=args.emb_scale, seed=args.seed)

    if mat is None:
        print("Encoding is 'learned' — no matrix to analyse at init.")
        return

    stats = category_stats(mat, word_categories, vocab)
    print_stats(stats, label=args.encoding)


if __name__ == "__main__":
    main()
