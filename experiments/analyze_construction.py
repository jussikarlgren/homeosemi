"""Certification read for the construction-prior experiment.

full = word_construction (real inventory, frozen); scrambled = magnitude-matched
permutation; none = word_learned floor. The test: does `full` beat `scrambled` on the
construction-sensitive BLiMP paradigms, beyond seed noise — and does the gain concentrate
on argument-structure paradigms rather than spreading uniformly?

Usage:
  python experiments/analyze_construction.py --seeds 1337 42 2024 7 21
"""
import argparse
import json
import os
import statistics as st

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = os.path.join(_ROOT, "experiments", "runs")

# construction-sensitive paradigms (argument structure, passive, existential, raising)
CONSTRUCTION_PARADIGMS = [
    "transitive", "intransitive", "causative", "inchoative", "drop_argument",
    "passive_1", "passive_2", "animate_subject_passive", "animate_subject_trans",
    "existential_there_quantifiers_1", "existential_there_quantifiers_2",
    "existential_there_object_raising", "existential_there_subject_raising",
    "tough_vs_raising_1", "tough_vs_raising_2",
]
# a control group that the construction prior should NOT specifically help
CONTROL_PARADIGMS = [
    "anaphor_gender_agreement", "anaphor_number_agreement",
    "determiner_noun_agreement_1", "determiner_noun_agreement_2",
    "wh_vs_that_no_gap", "wh_vs_that_with_gap",
]


def accs(tag):
    p = os.path.join(RUNS, f"{tag}_ckpt", "blimp_results.json")
    if not os.path.exists(p):
        return None
    return {k: v["acc"] for k, v in json.load(open(p))["paradigms"].items()}


def group_mean(a, names, max_oov=0.3):
    p = json.load(open(os.path.join(RUNS, a, "blimp_results.json")))["paradigms"] if isinstance(a, str) else None
    vals = []
    for n in names:
        if p is not None:
            if n in p and p[n].get("oov_rate", 0) <= max_oov:
                vals.append(p[n]["acc"])
    return st.mean(vals) if vals else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=[1337, 42, 2024, 7, 21])
    args = ap.parse_args()
    S = args.seeds

    def gm(tagfmt, names):
        return [group_mean(f"{tagfmt.format(s=s)}_ckpt", names) for s in S]

    print(f"seeds: {S}\n")
    for label, names in [("CONSTRUCTION group", CONSTRUCTION_PARADIGMS),
                         ("CONTROL group", CONTROL_PARADIGMS)]:
        none = gm("con_none_100k_s{s}", names)
        full = gm("con_full_100k_s{s}", names)
        scr = gm("con_full_100k_s{s}_scrambled", names)
        print(f"=== {label} ({len(names)} paradigms) ===")
        print(f"  none      {st.mean(none):.3f} +/- {st.pstdev(none):.3f}")
        print(f"  full      {st.mean(full):.3f} +/- {st.pstdev(full):.3f}")
        print(f"  scrambled {st.mean(scr):.3f} +/- {st.pstdev(scr):.3f}")
        dfs = [f - c for f, c in zip(full, scr)]
        dfn = [f - n for f, n in zip(full, none)]
        se = st.pstdev(dfs) / (len(dfs) ** 0.5)
        print(f"  full-scrambled: per-seed {['%+.3f'%d for d in dfs]}")
        print(f"                  mean {st.mean(dfs):+.3f} +/- {st.pstdev(dfs):.3f} (SE {se:.3f}); "
              f"{sum(d>0 for d in dfs)}/{len(dfs)} positive")
        print(f"  full-none:      mean {st.mean(dfn):+.3f} +/- {st.pstdev(dfn):.3f}\n")

    # per-paradigm full-scrambled on the construction group (concentration check)
    print("=== per-paradigm full - scrambled (construction group), mean over seeds ===")
    rows = []
    for para in CONSTRUCTION_PARADIGMS:
        ds = []
        for s in S:
            f = accs(f"con_full_100k_s{s}"); c = accs(f"con_full_100k_s{s}_scrambled")
            if f and c and para in f and para in c:
                ds.append(f[para] - c[para])
        if ds:
            rows.append((para, st.mean(ds), st.pstdev(ds)))
    for para, m, sd in sorted(rows, key=lambda r: -r[1]):
        print(f"  {para:36s} {m:+.3f} +/- {sd:.3f}")


if __name__ == "__main__":
    main()
