"""Read the acquisition-vs-regularisation runs and print the dissociation.

Regularisation is global and undifferentiated; acquisition is targeted. So the
signal is the INTERACTION: overlaying category X should lift X's paradigms more
than other categories' paradigms. This script loads each run's blimp_results.json,
buckets paradigm accuracies by overlay category (via paradigm_categories.py),
drops structural paradigms and (optionally) high-OOV ones, and prints a table:
condition x per-category mean accuracy.

Reading:
  * Exp A  full vs scrambled: if full ~ scrambled the effect is regularisation.
  * Exp B  ON=7 (determiners) should lift the DET(7) row above the PRED(3) row,
           and ON=3 the reverse. Both rows moving together regardless of ON = null.

Usage (on the compute box, after training + eval_blimp on each checkpoint):
  python experiments/analyze_dissociation.py \
      --runs_dir experiments/runs \
      --tags word_learned_100k word_category_100k word_category_100k_scrambled \
             word_category_100k_on7 word_category_100k_on3 \
      --seeds s1337 s42 s2024 \
      --categories 7 3 1
  # (a tag is expanded to <tag>_<seed>; results read from <that>_ckpt/blimp_results.json)
"""
import argparse
import json
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paradigm_categories import PARADIGM_CATEGORIES, DROPPED  # noqa: E402


def load_paradigm_accs(path, max_oov=0.3):
    with open(path) as f:
        res = json.load(f)
    out = {}
    for name, d in res.get("paradigms", {}).items():
        if name in DROPPED:
            continue
        if d.get("oov_rate", 0.0) > max_oov:
            continue
        out[name] = d["acc"]
    return out


def category_mean(accs, cat_id):
    """Mean accuracy over paradigms whose primary category is exactly cat_id
    (single-category paradigms only, so the bucket is clean)."""
    vals = [accs[p] for p, cats in PARADIGM_CATEGORIES.items()
            if cats == (cat_id,) and p in accs]
    return statistics.mean(vals) if vals else float("nan")


def main():
    import re
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_dir", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "runs"))
    ap.add_argument("--tags", nargs="+", required=True,
                    help="FULL run tags (each a complete <condition>_s<seed>[_variant]); "
                         "seeds are grouped by stripping the _s<digits> token. Reads "
                         "<runs_dir>/<tag>_ckpt/blimp_results.json.")
    ap.add_argument("--categories", nargs="+", type=int, default=[7, 3, 1])
    ap.add_argument("--max_oov", type=float, default=0.3)
    args = ap.parse_args()

    cats = args.categories
    groups, order = {}, []
    for tag in args.tags:
        base = re.sub(r"_s\d+", "", tag)              # group seeds: drop _s1337 etc.
        if base not in groups:
            groups[base] = {"n": 0, **{c: [] for c in cats}}
            order.append(base)
        path = os.path.join(args.runs_dir, f"{tag}_ckpt", "blimp_results.json")
        if not os.path.exists(path):
            continue
        accs = load_paradigm_accs(path, args.max_oov)
        groups[base]["n"] += 1
        for c in cats:
            m = category_mean(accs, c)
            if m == m:  # not nan
                groups[base][c].append(m)

    hdr = f"  {'condition':40s} " + " ".join(f"cat{c:>5}" for c in cats)
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for base in order:
        cells = [f"{statistics.mean(groups[base][c]):6.3f}" if groups[base][c]
                 else "   -  " for c in cats]
        print(f"  {base:40s} " + " ".join(cells) + f"   (seeds={groups[base]['n']})")

    print("\n  cat7=DETERMINER  cat3=PREDICATIVE  cat1=REFERENTIAL")
    print("  Exp B signal: ON=<c> should lift cat<c> above the other rows;")
    print("  Exp A signal: full ~ scrambled across all rows => regularisation.")


def _selftest():
    # synthetic results: ON=7 lifts only DET paradigms
    import tempfile
    det = [p for p, c in PARADIGM_CATEGORIES.items() if c == (7,)]
    pred = [p for p, c in PARADIGM_CATEGORIES.items() if c == (3,)]
    base = {p: 0.55 for p in det + pred}
    lifted = dict(base);
    for p in det: lifted[p] = 0.75          # only DET improved
    accs = load_paradigm_accs  # noqa
    m_det = statistics.mean(lifted[p] for p in det)
    m_pred = statistics.mean(lifted[p] for p in pred)
    assert abs(m_det - 0.75) < 1e-9 and abs(m_pred - 0.55) < 1e-9
    # category_mean over a dict
    assert abs(category_mean(lifted, 7) - 0.75) < 1e-9
    assert abs(category_mean(lifted, 3) - 0.55) < 1e-9
    print("analyze_dissociation self-test OK: dissociation is readable "
          f"(DET {m_det:.2f} vs PRED {m_pred:.2f} when ON=7)")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        _selftest()
    else:
        main()
