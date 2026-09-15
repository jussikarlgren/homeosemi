"""Prepare a word-level dataset from the BabyLM simple/child-directed subset.

Writes to --out_dir:
  train.bin       uint16 token ids, train split
  val.bin         uint16 token ids, val split
  meta.pkl        vocab_size, itos, stoi, train_char_count, val_char_count,
                  word_categories {word: category_id}

Usage:
  python experiments/prepare_words.py
  python experiments/prepare_words.py --vocab_size 12000 --min_freq 3
"""
import argparse
import os
import pickle
import re
import sys
import collections

import numpy as np
import nltk

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # experiments/

from categorize import assign_categories  # noqa: E402
import verbframes  # noqa: E402

TRAIN_DOMAINS = [
    "aochildes", "children_stories", "cbt",
    "simple_wikipedia", "open_subtitles", "switchboard", "bnc_spoken",
]

# simple regex tokenizer: split off punctuation as separate tokens
_TOK = re.compile(r"[a-z0-9]+|[''']s|[''']t|[''']ve|[''']re|[''']ll|[''']d|[^\w\s]")


def tokenize_line(line: str):
    """Lowercase + split a line into word tokens, return list of strings."""
    return _TOK.findall(line.lower())


def read_corpus(paths, add_eos=True):
    """Yield (tokens, char_count) for each line across all paths."""
    for path in paths:
        with open(path, encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                toks = tokenize_line(line)
                if add_eos:
                    toks.append("<eos>")
                yield toks, len(line)


def read_litdata(paths, add_eos=True):
    """Yield (tokens, char_count) per line from Gutenberg prose, boilerplate stripped."""
    for path in paths:
        with open(path, encoding="utf-8", errors="replace") as f:
            text = verbframes.strip_gutenberg(f.read())
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            toks = tokenize_line(line)
            if add_eos:
                toks.append("<eos>")
            yield toks, len(line)


def train_stream(train_paths, extra_paths, add_eos=True):
    """BabyLM domains first, then litdata prose appended after."""
    yield from read_corpus(train_paths, add_eos=add_eos)
    if extra_paths:
        yield from read_litdata(extra_paths, add_eos=add_eos)


LEVIN_PATH = os.path.join(_ROOT, "experiments", "levin_classes.json")


def build_levin_map(vocab, path=LEVIN_PATH):
    """Return (levin_classes {word: class_id}, n_levin, class_names).

    class_id 0 = no Levin class; 1..K = the classes in listed order.
    """
    import json
    spec = json.load(open(path, encoding="utf-8"))
    classes = spec["classes"]
    cid = {name: i + 1 for i, name in enumerate(classes)}
    vmap = spec["verbs"]
    levin = {w: cid.get(vmap.get(w), 0) for w in vocab}
    return levin, len(classes) + 1, ["none"] + classes


def build_vocab(train_paths, max_vocab: int, min_freq: int, extra_paths=None):
    freq = collections.Counter()
    for toks, _ in train_stream(train_paths, extra_paths, add_eos=False):
        freq.update(toks)
    # specials first
    specials = ["<unk>", "<eos>"]
    candidates = [w for w, c in freq.most_common() if c >= min_freq]
    candidates = candidates[:max_vocab - len(specials)]
    vocab = specials + candidates
    stoi = {w: i for i, w in enumerate(vocab)}
    itos = {i: w for i, w in enumerate(vocab)}
    return vocab, stoi, itos, freq


def encode(token_lists, stoi):
    unk_id = stoi["<unk>"]
    ids, char_count = [], 0
    for toks, nchars in token_lists:
        ids.extend(stoi.get(t, unk_id) for t in toks)
        char_count += nchars
    return np.array(ids, dtype=np.uint16), char_count


def _prepare_fixed_vocab(args, out_dir, train_dir):
    """Encode train/ through an existing dir's vocab; reuse its val.bin + meta.

    Lets augmented training text be evaluated on the SAME held-out val set as a
    reference dataset (identical vocab, categories, tma, val_char_count).
    """
    import shutil

    src = args.fixed_meta
    with open(os.path.join(src, "meta.pkl"), "rb") as f:
        meta = pickle.load(f)
    stoi = meta["stoi"]
    print(f"Reusing vocab ({meta['vocab_size']:,}) + val from {src}")

    train_paths = _collect_txt(train_dir)
    print("Encoding train (fixed vocab) ...")
    train_ids, train_char_count = encode(read_corpus(train_paths), stoi)
    unk_id = stoi["<unk>"]
    n_unk = int((train_ids == unk_id).sum())
    print(f"  {len(train_ids):,} tokens  ({train_char_count:,} chars)  "
          f"{n_unk:,} <unk> ({100*n_unk/max(len(train_ids),1):.2f}%)")

    train_ids.tofile(os.path.join(out_dir, "train.bin"))
    shutil.copyfile(os.path.join(src, "val.bin"),
                    os.path.join(out_dir, "val.bin"))

    meta = dict(meta)
    meta["train_char_count"] = train_char_count
    with open(os.path.join(out_dir, "meta.pkl"), "wb") as f:
        pickle.dump(meta, f)
    print(f"Wrote train.bin ({len(train_ids):,} tokens), copied val.bin, "
          f"meta.pkl to {out_dir}")


def _collect_txt(d):
    known = [os.path.join(d, f"{dom}.txt") for dom in TRAIN_DOMAINS
             if os.path.exists(os.path.join(d, f"{dom}.txt"))]
    extra = sorted(p for p in
                   [os.path.join(d, f) for f in os.listdir(d) if f.endswith(".txt")]
                   if p not in known)
    if extra:
        print(f"  extra files: {[os.path.basename(p) for p in extra]}")
    return known + extra


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir",
                    default=os.path.join(_ROOT, "experiments", "data", "babylm_simple"))
    ap.add_argument("--out_dir", default=None,
                    help="defaults to --data_dir")
    ap.add_argument("--vocab_size", type=int, default=15_000,
                    help="max vocabulary including specials")
    ap.add_argument("--min_freq", type=int, default=3,
                    help="minimum token frequency to enter vocab")
    ap.add_argument("--cat_sample", type=int, default=300_000,
                    help="characters to sample for POS-based categorization")
    ap.add_argument("--extra_dir", default=None,
                    help="directory of .txt prose (e.g. experiments/litdata) appended "
                         "AFTER the BabyLM train stream; also informs frame induction")
    ap.add_argument("--frame_obs_cap", type=int, default=None,
                    help="cap on observations per verb for frame induction "
                         "(default: unlimited). Reliability script overrides this.")
    ap.add_argument("--fixed_meta", default=None,
                    help="path to an existing prepared data dir; reuse its vocab "
                         "(stoi/itos), categories, tma and val.bin/val_char_count. "
                         "Only train.bin is re-encoded from this --data_dir's train/. "
                         "Use to encode augmented training text through a fixed vocab "
                         "so it can be evaluated on the same held-out val set.")
    args = ap.parse_args()

    out_dir = args.out_dir or args.data_dir
    os.makedirs(out_dir, exist_ok=True)

    train_dir = os.path.join(args.data_dir, "train")
    dev_dir   = os.path.join(args.data_dir, "dev")

    if args.fixed_meta:
        _prepare_fixed_vocab(args, out_dir, train_dir)
        return
    # Use all .txt files present; TRAIN_DOMAINS order first, then any extras
    train_paths = _collect_txt(train_dir)
    dev_paths   = _collect_txt(dev_dir)
    extra_paths = None
    if args.extra_dir:
        extra_paths = sorted(os.path.join(args.extra_dir, f)
                             for f in os.listdir(args.extra_dir) if f.endswith(".txt"))
        print(f"  appending {len(extra_paths)} extra prose files after BabyLM: "
              f"{[os.path.basename(p) for p in extra_paths]}")

    # ── vocab ────────────────────────────────────────────────────────────────
    print(f"Building vocab (max {args.vocab_size}, min_freq {args.min_freq}) ...")
    vocab, stoi, itos, freq = build_vocab(train_paths, args.vocab_size, args.min_freq,
                                          extra_paths=extra_paths)
    print(f"  Vocab size: {len(vocab):,}  (total types seen: {len(freq):,})")

    # ── encode ───────────────────────────────────────────────────────────────
    print("Encoding train ...")
    train_ids, train_char_count = encode(train_stream(train_paths, extra_paths), stoi)
    print(f"  {len(train_ids):,} tokens  ({train_char_count:,} chars)")

    print("Encoding val ...")
    val_ids, val_char_count = encode(read_corpus(dev_paths), stoi)
    print(f"  {len(val_ids):,} tokens  ({val_char_count:,} chars)")

    # ── categorize ───────────────────────────────────────────────────────────
    print("Categorizing vocabulary ...")
    word_categories, tma_categories = assign_categories(
        vocab,
        sample_size=args.cat_sample,
        corpus_files=train_paths,
        verbose=True,
    )

    # ── induce verb subcategorization frames ─────────────────────────────────
    print("Inducing verb frames ...")
    frame_distributions = verbframes.induce_frame_distributions(
        vocab, babylm_files=train_paths, lit_files=extra_paths,
        obs_cap=args.frame_obs_cap, verbose=True,
    )

    # ── Levin semantic verb classes (hand-curated map) ───────────────────────
    levin_classes, n_levin, levin_class_names = build_levin_map(vocab)
    n_levin_verbs = sum(1 for v in levin_classes.values() if v != 0)
    print(f"Levin classes: {n_levin - 1} classes, {n_levin_verbs} verbs mapped")

    # ── write ────────────────────────────────────────────────────────────────
    train_ids.tofile(os.path.join(out_dir, "train.bin"))
    val_ids.tofile(os.path.join(out_dir, "val.bin"))

    meta = {
        "vocab_size":       len(vocab),
        "itos":             itos,
        "stoi":             stoi,
        "train_char_count": train_char_count,
        "val_char_count":   val_char_count,
        "word_categories":    word_categories,
        "tma_distributions":  tma_categories,  # now a {word: np.ndarray(4,)} dict
        "frame_distributions": frame_distributions,  # {word: np.ndarray(6,)} for verbs
        "n_frames":           verbframes.N_FRAMES,
        "levin_classes":      levin_classes,   # {word: class_id}, 0 = none
        "n_levin":            n_levin,
        "levin_class_names":  levin_class_names,
    }
    with open(os.path.join(out_dir, "meta.pkl"), "wb") as f:
        pickle.dump(meta, f)

    print(f"Wrote train.bin ({len(train_ids):,} tokens), "
          f"val.bin ({len(val_ids):,} tokens), meta.pkl to {out_dir}")


if __name__ == "__main__":
    main()
