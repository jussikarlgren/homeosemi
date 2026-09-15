"""Regular vs irregular verbs relative to their Levin class.

A verb is "regular" if its induced syntactic-frame distribution matches its Levin
class's canonical profile (the class prototype = mean induced frame distribution over
class members), and "irregular" if it deviates. Deviation = Jensen-Shannon divergence
(base 2, in [0,1]). This defines the item split for the overregularization test: a
class-based ("rule") prior should help regulars and hurt irregulars.

Reusable: get_regularity(meta) -> {verb: {class_id, class_name, deviation, label}}.

Usage:
  python experiments/verb_regularity.py --data_dir experiments/data/babylm_lit
"""
import argparse
import os
import pickle
import sys
import collections

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _js_divergence(p, q, eps=1e-12):
    p = np.asarray(p, dtype=np.float64) + eps
    q = np.asarray(q, dtype=np.float64) + eps
    p /= p.sum(); q /= q.sum()
    m = 0.5 * (p + q)
    def _kl(a, b):
        return float(np.sum(a * np.log2(a / b)))
    return 0.5 * _kl(p, m) + 0.5 * _kl(q, m)


def get_regularity(meta, low_q=1/3, high_q=2/3):
    """Return per-verb regularity info + prototype dict.

    Only verbs with a Levin class (id != 0) AND a nonzero induced frame distribution
    are considered. Labels: 'regular' (deviation <= low tercile), 'irregular'
    (deviation >= high tercile), else 'mid'.
    """
    fd = meta["frame_distributions"]
    levin = meta["levin_classes"]
    names = meta.get("levin_class_names")

    members = collections.defaultdict(list)
    for w, cid in levin.items():
        if cid == 0:
            continue
        d = fd.get(w)
        if d is not None and np.sum(d) > 0:
            members[cid].append(w)

    # class prototype = mean induced frame distribution over class members
    prototype = {}
    for cid, ws in members.items():
        proto = np.mean([np.asarray(fd[w], dtype=np.float64) for w in ws], axis=0)
        prototype[cid] = proto / proto.sum()

    info = {}
    for cid, ws in members.items():
        for w in ws:
            dev = _js_divergence(fd[w], prototype[cid])
            info[w] = {"class_id": cid,
                       "class_name": names[cid] if names else str(cid),
                       "deviation": dev}

    devs = sorted(v["deviation"] for v in info.values())
    lo = devs[int(len(devs) * low_q)]
    hi = devs[int(len(devs) * high_q)]
    for w, v in info.items():
        v["label"] = ("regular" if v["deviation"] <= lo
                      else "irregular" if v["deviation"] >= hi else "mid")
    return info, prototype


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default=os.path.join(_ROOT, "experiments", "data", "babylm_lit"))
    args = ap.parse_args()
    meta = pickle.load(open(os.path.join(args.data_dir, "meta.pkl"), "rb"))
    info, proto = get_regularity(meta)

    labels = collections.Counter(v["label"] for v in info.values())
    print(f"mapped verbs: {len(info)}   " +
          "  ".join(f"{k}={labels[k]}" for k in ("regular", "mid", "irregular")))

    print("\nMost REGULAR (closest to class prototype):")
    for w, v in sorted(info.items(), key=lambda kv: kv[1]["deviation"])[:12]:
        print(f"  {w:12s} {v['class_name']:16s} JS={v['deviation']:.3f}")
    print("\nMost IRREGULAR (farthest from class prototype):")
    for w, v in sorted(info.items(), key=lambda kv: -kv[1]["deviation"])[:12]:
        print(f"  {w:12s} {v['class_name']:16s} JS={v['deviation']:.3f}")


if __name__ == "__main__":
    main()
