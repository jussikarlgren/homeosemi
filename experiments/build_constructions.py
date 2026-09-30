"""Build a per-word construction distribution from constructions.json and (optionally)
patch it into an existing prepared dataset's meta.pkl.

Each word that appears in the inventory (as a participating verb or a characteristic
marker of one or more constructions) gets a normalised multi-hot vector over the K
constructions; all other words get a zero vector (no overlay). This mirrors the
frame/levin overlays and is injected by the `word_construction` encoding as a frozen
prior; a magnitude-matched scramble (overlay_variants.scramble_maps) is the control.

Usage:
  # inspect
  python experiments/build_constructions.py --data_dir experiments/data/babylm_lit
  # patch meta in place (adds construction_map / n_constructions / construction_names)
  python experiments/build_constructions.py --data_dir experiments/data/babylm_lit --patch
"""
import argparse
import json
import os
import pickle
import sys
import collections

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CON_PATH = os.path.join(_ROOT, "experiments", "constructions.json")


def build_construction_map(vocab, path=CON_PATH):
    """Return (construction_map {word: np.ndarray(K)}, n_constructions, names).

    construction_map[w] is a normalised multi-hot over the K constructions w participates
    in (verb or marker); words not in the inventory get an all-zero vector.
    """
    spec = json.load(open(path, encoding="utf-8"))["constructions"]
    names = list(spec.keys())
    cid = {name: i for i, name in enumerate(names)}
    K = len(names)

    word_cons = collections.defaultdict(set)
    for name, d in spec.items():
        for w in d.get("verbs", []) + d.get("markers", []):
            word_cons[w].add(cid[name])

    zero = np.zeros(K, dtype=np.float32)
    cmap = {}
    for w in vocab:
        cs = word_cons.get(w)
        if cs:
            v = np.zeros(K, dtype=np.float32)
            for c in cs:
                v[c] = 1.0
            cmap[w] = v / v.sum()
        else:
            cmap[w] = zero.copy()
    return cmap, K, names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default=os.path.join(_ROOT, "experiments", "data", "babylm_lit"))
    ap.add_argument("--patch", action="store_true", help="write construction_map into meta.pkl")
    args = ap.parse_args()

    meta_path = os.path.join(args.data_dir, "meta.pkl")
    meta = pickle.load(open(meta_path, "rb"))
    vocab = [meta["itos"][i] for i in range(meta["vocab_size"])]
    cmap, K, names = build_construction_map(vocab)

    n_assigned = sum(1 for w in vocab if cmap[w].sum() > 0)
    print(f"constructions: {K}  {names}")
    print(f"words with a construction overlay: {n_assigned}")
    # per-construction word counts
    counts = collections.Counter()
    for w in vocab:
        for i in np.nonzero(cmap[w])[0]:
            counts[names[i]] += 1
    for name in names:
        print(f"  {name:22s} {counts[name]:3d} words")

    if args.patch:
        meta["construction_map"] = cmap
        meta["n_constructions"] = K
        meta["construction_names"] = names
        pickle.dump(meta, open(meta_path, "wb"))
        print(f"patched {meta_path}")


if __name__ == "__main__":
    main()
