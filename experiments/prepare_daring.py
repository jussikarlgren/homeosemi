"""Build a training dir that interleaves the daring construction-spread paraphrases
into the base simple corpus, so the augmented data lands INSIDE the low-data
(first-N-token) window rather than being appended at the end (where it is inert).

Reuses an existing prepared dir's fixed vocab + categories + val set (via meta.pkl)
so the resulting BPC is directly comparable to the flat baselines.

Usage:
  python experiments/prepare_daring.py \
      --daring_txt experiments/data/babylm_paraphrase/paraphrases_daring.txt \
      --base_dir   experiments/data/babylm_simple \
      --fixed_meta experiments/data/babylm_simple \
      --out_dir    experiments/data/babylm_daring
"""
import argparse
import os
import pickle
import shutil
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # experiments/
from prepare_words import tokenize_line, _collect_txt  # noqa: E402


def read_lines(path):
    """Yield token-lists (with <eos>) for non-empty, non-header lines."""
    out = []
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):   # skip blanks + #/## headers
                continue
            toks = tokenize_line(line)
            toks.append("<eos>")
            out.append((toks, len(line)))
    return out


def read_base(base_dir):
    out = []
    for path in _collect_txt(os.path.join(base_dir, "train")):
        with open(path, encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                toks = tokenize_line(line)
                toks.append("<eos>")
                out.append((toks, len(line)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--daring_txt", required=True)
    ap.add_argument("--base_dir", required=True)
    ap.add_argument("--fixed_meta", required=True,
                    help="dir whose meta.pkl (vocab/categories/val) is reused")
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    with open(os.path.join(args.fixed_meta, "meta.pkl"), "rb") as f:
        meta = pickle.load(f)
    stoi = meta["stoi"]
    unk = stoi["<unk>"]

    daring = read_lines(args.daring_txt)
    base = read_base(args.base_dir)
    print(f"daring lines: {len(daring):,}  base lines: {len(base):,}")

    # 1:1 interleave (base, daring) until daring is exhausted, then remaining base.
    merged = []
    bi = di = 0
    while di < len(daring) and bi < len(base):
        merged.append(base[bi]); bi += 1
        merged.append(daring[di]); di += 1
    merged.extend(base[bi:])
    merged.extend(daring[di:])  # in case daring outnumbers base (won't here)

    ids, char_count = [], 0
    for toks, nchars in merged:
        ids.extend(stoi.get(t, unk) for t in toks)
        char_count += nchars
    ids = np.array(ids, dtype=np.uint16)
    n_unk = int((ids == unk).sum())
    print(f"train tokens: {len(ids):,}  ({char_count:,} chars)  "
          f"{n_unk:,} <unk> ({100*n_unk/len(ids):.2f}%)")

    ids.tofile(os.path.join(args.out_dir, "train.bin"))
    shutil.copyfile(os.path.join(args.fixed_meta, "val.bin"),
                    os.path.join(args.out_dir, "val.bin"))
    meta = dict(meta)
    meta["train_char_count"] = char_count
    with open(os.path.join(args.out_dir, "meta.pkl"), "wb") as f:
        pickle.dump(meta, f)
    print(f"Wrote train.bin/val.bin/meta.pkl to {args.out_dir}")


if __name__ == "__main__":
    main()
