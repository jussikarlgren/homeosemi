"""Compare logged encoding runs by plotting their loss curves.

  python experiments/plot_curves.py                         # all runs
  python experiments/plot_curves.py learned homeosemi_context
  python experiments/plot_curves.py --metric bpc word_learned word_category
"""
import argparse
import glob
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RUNS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs")


def load_runs(names):
    if names:
        paths = [os.path.join(RUNS_DIR, n if n.endswith(".json") else f"{n}.json")
                 for n in names]
    else:
        paths = sorted(glob.glob(os.path.join(RUNS_DIR, "*.json")))
        paths = [p for p in paths if not os.path.basename(p).startswith("_")]
    runs = []
    for p in paths:
        if not os.path.exists(p):
            print(f"skip (missing): {p}")
            continue
        with open(p, "r", encoding="utf-8") as f:
            runs.append(json.load(f))
    return runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="*", help="run tags or .json paths")
    ap.add_argument("--metric", default="val_loss",
                    choices=["val_loss", "bpc"],
                    help="metric to plot on y-axis")
    ap.add_argument("--out", default=None, help="output image path")
    args = ap.parse_args()

    runs = load_runs(args.runs)
    if not runs:
        print("no runs found in", RUNS_DIR)
        return

    metric = args.metric
    metric_key = "val_bpc" if metric == "bpc" else "val_loss"

    fig, ax = plt.subplots(figsize=(9, 5))
    has_bpc = any(metric_key in h for r in runs for h in r["history"])

    hdr = f"{'run':32s} {'final train':>12s} {'final val':>12s}"
    if has_bpc and metric == "bpc":
        hdr += f" {'final bpc':>10s}"
    hdr += f" {'params':>10s}"
    print(hdr)

    for r in runs:
        iters = [h["iter"] for h in r["history"]]
        tr    = [h["train_loss"] for h in r["history"]]
        va    = [h["val_loss"] for h in r["history"]]
        yvals = [h.get(metric_key, h["val_loss"]) for h in r["history"]]

        line, = ax.plot(iters, yvals, label=f"{r['tag']}")
        ax.plot(iters, tr, "--", color=line.get_color(), alpha=0.4)

        row = f"{r['tag']:32s} {tr[-1]:12.4f} {va[-1]:12.4f}"
        if has_bpc and metric == "bpc":
            bpc = r["history"][-1].get("val_bpc", float("nan"))
            row += f" {bpc:10.4f}"
        row += f" {r.get('n_trainable', 0)/1e6:9.3f}M"
        print(row)

    ylabel = "bits per character (BPC)" if metric == "bpc" else "loss (nats)"
    ax.set_xlabel("iteration")
    ax.set_ylabel(ylabel)
    title = f"Encoding comparison — {metric} (solid=val/{metric}, dashed=train loss)"
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    out = args.out or os.path.join(RUNS_DIR,
          f"comparison_{'bpc' if metric == 'bpc' else 'loss'}.png")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    print("wrote", out)


if __name__ == "__main__":
    main()
