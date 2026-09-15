"""Prepare the BabyLM simple subset with GPT-2 BPE tokenization (tiktoken).

Writes experiments/data/babylm_bpe/train.bin, val.bin, meta.pkl.
Vocab size: 50,257 (GPT-2 BPE). Tokens stored as uint16 (50257 < 65535).
meta.pkl includes train_char_count and val_char_count for BPC computation.

Usage:
  python experiments/prepare_bpe.py
"""
import os
import pickle
import sys

import numpy as np
import tiktoken

DOMAINS = [
    "aochildes", "children_stories", "cbt",
    "simple_wikipedia", "open_subtitles", "switchboard", "bnc_spoken",
]

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(_ROOT, "experiments", "data", "babylm_simple")
OUT_DIR  = os.path.join(_ROOT, "experiments", "data", "babylm_bpe")


def encode_split(split_dir, enc):
    ids, char_count = [], 0
    for domain in DOMAINS:
        path = os.path.join(split_dir, f"{domain}.txt")
        if not os.path.exists(path):
            print(f"  skip (missing): {path}")
            continue
        with open(path, encoding="utf-8", errors="replace") as f:
            text = f.read()
        char_count += len(text)
        ids.extend(enc.encode_ordinary(text))
    return np.array(ids, dtype=np.uint16), char_count


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    enc = tiktoken.get_encoding("gpt2")
    print(f"GPT-2 BPE vocab size: {enc.n_vocab}")

    print("Encoding train ...")
    train_ids, train_chars = encode_split(os.path.join(DATA_DIR, "train"), enc)
    print(f"  {len(train_ids):,} tokens  ({train_chars:,} chars)")

    print("Encoding val ...")
    val_ids, val_chars = encode_split(os.path.join(DATA_DIR, "dev"), enc)
    print(f"  {len(val_ids):,} tokens  ({val_chars:,} chars)")

    train_ids.tofile(os.path.join(OUT_DIR, "train.bin"))
    val_ids.tofile(os.path.join(OUT_DIR, "val.bin"))

    # itos/stoi as string token-id keys (harness only needs vocab_size + char counts)
    meta = {
        "vocab_size":       enc.n_vocab,
        "itos":             {i: f"<{i}>" for i in range(enc.n_vocab)},
        "stoi":             {f"<{i}>": i for i in range(enc.n_vocab)},
        "train_char_count": train_chars,
        "val_char_count":   val_chars,
        "word_categories":  None,
        "tokenizer":        "gpt2-bpe",
    }
    with open(os.path.join(OUT_DIR, "meta.pkl"), "wb") as f:
        pickle.dump(meta, f)
    print(f"Wrote to {OUT_DIR}")


if __name__ == "__main__":
    main()
