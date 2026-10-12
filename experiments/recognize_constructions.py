"""Per-token argument-structure construction recogniser + aligned construction streams.

Unlike the static per-word `word_construction` overlay, this tags each token POSITION with
the construction its span instantiates, derived from local context — so the same verb gets
different ids in different sentences (ditransitive "gave him a book" vs dative "gave a book
to him"). A frozen construction embedding is then added at that position (see
train_experiment `--construction_pos`); a magnitude/histogram-matched scramble is the control.

Reuses verbframes.classify_verb_frame (already a per-occurrence, context-derived classifier)
and adds passive (be + VBN [+ by]) and existential-there (there + be).

The stream is built by replicating prepare_words' tokenisation ORDER exactly so the ids
align 1:1 with train.bin / val.bin (one construction id per token).

Usage:
  python experiments/recognize_constructions.py --build            # write babylm_lit streams
  python experiments/recognize_constructions.py --spot             # configurational spot-check
"""
import os
import sys
import numpy as np
import nltk

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import verbframes as vf  # noqa: E402

# construction ids (0 = none)
CONSTRUCTIONS = ["none", "intransitive", "transitive", "ditransitive",
                 "caused_motion", "clausal", "passive", "existential"]
CID = {c: i for i, c in enumerate(CONSTRUCTIONS)}
N_CON = len(CONSTRUCTIONS) - 1  # excluding 'none'

# verbframes frame id -> construction name
_FRAME2CON = {
    vf.INTRANS:  "intransitive",
    vf.TRANS:    "transitive",
    vf.DITRANS:  "ditransitive",
    vf.PP_OBL:   "caused_motion",
    vf.FIN_COMP: "clausal",
    vf.INF_COMP: "clausal",
}
_BE = {"be", "am", "is", "are", "was", "were", "been", "being"}


def recognise_tokens(toks):
    """Return a list of construction ids aligned 1:1 with `toks` (which may end in <eos>).

    Tags verb positions with their per-occurrence construction; adds existential at a
    `there`+be, and passive at a be+VBN verb (and a following `by`).
    """
    ids = [0] * len(toks)
    # POS-tag the real tokens (exclude a trailing <eos>)
    real_n = len(toks) - 1 if toks and toks[-1] == "<eos>" else len(toks)
    real = toks[:real_n]
    if not real:
        return ids
    tags = [t for _, t in nltk.pos_tag(real)]
    for i in range(real_n):
        tok, tag = real[i], tags[i]
        # existential there + be
        if tok == "there" and i + 1 < real_n and real[i + 1] in _BE:
            ids[i] = CID["existential"]
            continue
        if tag in vf._VERB_TAGS:
            # passive: be + past participle
            prev = set(real[max(0, i - 3):i])
            if tag == "VBN" and (prev & _BE):
                ids[i] = CID["passive"]
                continue
            frame = int(vf.classify_verb_frame(i, real, tags).argmax())
            ids[i] = CID[_FRAME2CON[frame]]
        elif tok == "by" and i > 0 and ids[i - 1] == CID["passive"]:
            ids[i] = CID["passive"]
    return ids


# ── stream builder (replicates prepare_words tokenisation order) ──────────────
def _streams(data_dir, extra_dir):
    import prepare_words as pw
    train_paths = pw._collect_txt(os.path.join(data_dir, "train"))
    dev_paths = pw._collect_txt(os.path.join(data_dir, "dev"))
    extra_paths = (sorted(os.path.join(extra_dir, f) for f in os.listdir(extra_dir)
                          if f.endswith(".txt")) if extra_dir else None)
    train_gen = pw.train_stream(train_paths, extra_paths)   # (toks, nchars) per line
    val_gen = pw.read_corpus(dev_paths)
    return train_gen, val_gen


def build_stream(gen):
    out = []
    for toks, _ in gen:
        out.extend(recognise_tokens(toks))
    return np.array(out, dtype=np.uint8)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default=os.path.join(_ROOT, "experiments", "data", "babylm_simple"))
    ap.add_argument("--extra_dir", default=os.path.join(_ROOT, "experiments", "litdata"))
    ap.add_argument("--out_dir", default=os.path.join(_ROOT, "experiments", "data", "babylm_lit"))
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--spot", action="store_true")
    args = ap.parse_args()

    if args.spot:
        for s in ["she gave him a book .", "she gave a book to him .",
                  "the vase was broken by the boy .", "there is a dog .",
                  "he put the cup on the table .", "the ice melted ."]:
            toks = s.split()
            ids = recognise_tokens(toks)
            print("  " + " ".join(f"{t}:{CONSTRUCTIONS[c]}" if c else t
                                  for t, c in zip(toks, ids)))
        return

    if args.build:
        import collections
        train_gen, val_gen = _streams(args.data_dir, args.extra_dir)
        print("building train construction stream ...")
        con_train = build_stream(train_gen)
        print("building val construction stream ...")
        con_val = build_stream(val_gen)
        # alignment assertions
        tb = np.memmap(os.path.join(args.out_dir, "train.bin"), dtype=np.uint16, mode="r")
        vb = np.memmap(os.path.join(args.out_dir, "val.bin"), dtype=np.uint16, mode="r")
        assert len(con_train) == len(tb), f"train mismatch {len(con_train)} vs {len(tb)}"
        assert len(con_val) == len(vb), f"val mismatch {len(con_val)} vs {len(vb)}"
        con_train.tofile(os.path.join(args.out_dir, "train_construction.bin"))
        con_val.tofile(os.path.join(args.out_dir, "val_construction.bin"))
        for split, arr in [("train", con_train), ("val", con_val)]:
            nz = int((arr > 0).sum())
            cc = collections.Counter(arr.tolist())
            print(f"{split}: {len(arr):,} tokens, {nz:,} tagged ({100*nz/len(arr):.1f}%)")
            for c in range(1, len(CONSTRUCTIONS)):
                print(f"    {CONSTRUCTIONS[c]:14s} {cc.get(c,0):,}")
        print(f"wrote train/val_construction.bin to {args.out_dir}")


if __name__ == "__main__":
    main()
