"""Frame-assignment reliability curve: accuracy@N vs a gold verb-frame map.

Answers the "after a couple of observations" question: how many corpus
observations of a verb are needed before its induced argmax frame matches gold?

Single corpus pass records, per gold verb, its ordered list of per-occurrence frame
votes (classify_verb_frame). accuracy@N = fraction of scored gold verbs whose
argmax over the first N votes falls in the verb's gold-allowed frame set.

Usage:
  python experiments/frame_reliability.py \
      --data_dir experiments/data/babylm_lit --lit_dir experiments/litdata
"""
import argparse
import json
import os
import sys
import pickle
import collections

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import verbframes as vf  # noqa: E402

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default=os.path.join(_ROOT, "experiments", "data", "babylm_lit"))
    ap.add_argument("--lit_dir",  default=os.path.join(_ROOT, "experiments", "litdata"))
    ap.add_argument("--gold",     default=os.path.join(_ROOT, "experiments", "gold_verb_frames.json"))
    ap.add_argument("--babylm_sample_chars", type=int, default=2_000_000)
    ap.add_argument("--caps", default="1,2,3,5,10")
    ap.add_argument("--out", default=os.path.join(_ROOT, "experiments", "runs", "frame_reliability.json"))
    args = ap.parse_args()

    gold = json.load(open(args.gold))["verbs"]
    gold_verbs = set(gold)

    # ── which babylm train files + litdata ────────────────────────────────────
    train_dir = os.path.join(args.data_dir, "train")
    if not os.path.isdir(train_dir):  # babylm_lit has no train/; fall back to simple
        train_dir = os.path.join(_ROOT, "experiments", "data", "babylm_simple", "train")
    babylm_files = [os.path.join(train_dir, f) for f in os.listdir(train_dir) if f.endswith(".txt")]
    lit_files = [os.path.join(args.lit_dir, f) for f in os.listdir(args.lit_dir) if f.endswith(".txt")]

    # ── single pass: record ordered frame votes per gold verb ─────────────────
    votes = collections.defaultdict(list)  # verb -> [frame_vec, ...] in corpus order
    sources = [(babylm_files, args.babylm_sample_chars, False), (lit_files, None, True)]
    for paths, sample, strip_gut in sources:
        for toks, tags in vf._iter_sentences_from_files(paths, sample, strip_gut):
            for i, (tok, tag) in enumerate(zip(toks, tags)):
                if tag in vf._VERB_TAGS and tok in gold_verbs:
                    votes[tok].append(vf.classify_verb_frame(i, toks, tags))

    caps = [int(x) for x in args.caps.split(",")] + [None]  # None = all (inf)

    def accuracy_at(cap):
        hit = scored = 0
        for w, g in gold.items():
            vs = votes.get(w)
            if not vs:
                continue
            use = vs if cap is None else vs[:cap]
            if not use:
                continue
            am = int(np.stack(use).sum(axis=0).argmax())
            scored += 1
            hit += int(am in g["allowed"])
        return hit, scored

    print(f"gold verbs: {len(gold)}   observed: {sum(1 for w in gold if votes.get(w))}")
    print(f"{'N':>5s}  {'accuracy':>8s}  {'hit/scored':>12s}")
    curve = {}
    for cap in caps:
        hit, scored = accuracy_at(cap)
        acc = hit / scored if scored else float("nan")
        label = "inf" if cap is None else str(cap)
        curve[label] = {"accuracy": acc, "hit": hit, "scored": scored}
        print(f"{label:>5s}  {acc:8.3f}  {hit:>5d}/{scored:<5d}")

    # median observations per gold verb (context for the curve)
    obs_counts = sorted(len(votes[w]) for w in gold if votes.get(w))
    median_obs = obs_counts[len(obs_counts) // 2] if obs_counts else 0

    result = {"curve": curve, "n_gold": len(gold), "median_obs_per_verb": median_obs,
              "babylm_sample_chars": args.babylm_sample_chars}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(result, open(args.out, "w"), indent=2)
    print(f"median obs/gold-verb: {median_obs}")
    print(f"wrote {args.out}")

    # ── plot ──────────────────────────────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        xs = [1, 2, 3, 5, 10, max(12, median_obs)]  # place inf at the right edge
        ys = [curve[k]["accuracy"] for k in ["1", "2", "3", "5", "10", "inf"]]
        plt.figure(figsize=(6, 4))
        plt.plot(xs[:-1], ys[:-1], "o-", label="accuracy@N")
        plt.axhline(ys[-1], ls="--", color="gray", label=f"all obs ({ys[-1]:.3f})")
        plt.xlabel("observations per verb (N)")
        plt.ylabel("frame accuracy vs gold")
        plt.title("Verb-frame assignment reliability")
        plt.ylim(0, 1); plt.legend(); plt.tight_layout()
        png = args.out.replace(".json", ".png")
        plt.savefig(png, dpi=120)
        print(f"wrote {png}")
    except Exception as e:  # noqa: BLE001
        print(f"(plot skipped: {e})")


if __name__ == "__main__":
    main()
