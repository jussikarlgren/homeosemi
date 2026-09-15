"""Compute mean ± std of final BPC and val loss across multi-seed runs.

Groups JSON files by a condition prefix (tag up to the last _s<seed> suffix)
and reports statistics per condition.

Usage:
  python experiments/summarize.py                        # all runs/
  python experiments/summarize.py --prefix word_learned  # matching prefix
  python experiments/summarize.py --metric val_bpc       # default
"""
import argparse
import glob
import json
import os
import re

import numpy as np

RUNS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs")
SEED_RE  = re.compile(r"_s(\d+)$")


def load_runs(runs_dir, prefix=None):
    paths = sorted(glob.glob(os.path.join(runs_dir, "*.json")))
    runs  = []
    for p in paths:
        name = os.path.splitext(os.path.basename(p))[0]
        if name.startswith("_"):           # smoke runs
            continue
        if prefix and not name.startswith(prefix):
            continue
        with open(p, encoding="utf-8") as f:
            runs.append(json.load(f))
    return runs


def group_by_condition(runs):
    """Strip trailing _s<seed> to get the condition name."""
    groups: dict[str, list] = {}
    for r in runs:
        tag  = r["tag"]
        cond = SEED_RE.sub("", tag)   # remove _s1337, _s42, etc.
        groups.setdefault(cond, []).append(r)
    return groups


def final_metric(run, key):
    h = run.get("history", [])
    if not h:
        return float("nan")
    return h[-1].get(key, float("nan"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default=None)
    ap.add_argument("--metric", default="val_bpc",
                    choices=["val_bpc", "val_loss", "train_loss"])
    ap.add_argument("--runs_dir", default=RUNS_DIR)
    args = ap.parse_args()

    runs   = load_runs(args.runs_dir, args.prefix)
    groups = group_by_condition(runs)

    print(f"\n{'Condition':45s}  {'n':>3s}  {'mean':>8s}  {'std':>8s}  {'min':>8s}  {'max':>8s}  {'seeds'}")
    print("-" * 110)

    for cond in sorted(groups):
        vals  = [final_metric(r, args.metric) for r in groups[cond]]
        seeds = [r["config"].get("seed", "?") for r in groups[cond]]
        vals  = [v for v in vals if not np.isnan(v)]
        if not vals:
            continue
        arr = np.array(vals)
        print(f"{cond:45s}  {len(arr):3d}  {arr.mean():8.4f}  {arr.std():8.4f}  "
              f"{arr.min():8.4f}  {arr.max():8.4f}  {seeds}")


if __name__ == "__main__":
    main()
