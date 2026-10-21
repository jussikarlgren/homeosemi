"""Evaluate a basket-of-balls model on word-similarity benchmarks (SimLex-999, WordSim-353).

Usage:
    python evaltopospace.py <model.pkl>

The model answers a pair's similarity only when both words share a ball; otherwise it returns
*null* (that is the design -- cross-neighbourhood comparisons are out of scope, not low). So a single
Spearman number is not the whole story. We report, per benchmark:

  - coverage: how many pairs the model actually answers (non-null), and why the rest drop out
              (OOV = a word never made it into any ball; null = both placed but no shared ball)
  - Spearman rho on the answered pairs (the standard metric, computed only where the model speaks)
  - a global-routing-cosine baseline (cosine in the single coarse index space) over the same pairs,
    i.e. what "one big space" would score -- the thing the basket hypothesis argues against.

This is a snapshot of quality+coverage on one corpus, NOT the learns-faster claim (that needs
learning curves over corpus size).
"""
import _paths  # noqa: F401
import sys

import sparsevectors
from ballspace import Basket, _sparse_dot

SIMLEX = "data/benchmarks/simlex/SimLex-999/SimLex-999.txt"
WS353 = "data/benchmarks/ws353/EN-WS353.txt"


def load_simlex(path):
    pairs = []
    with open(path, encoding="utf-8") as fh:
        next(fh)  # header
        for line in fh:
            f = line.rstrip("\n").split("\t")
            if len(f) >= 4:
                pairs.append((f[0].lower(), f[1].lower(), float(f[3])))
    return pairs


def load_ws353(path):
    pairs = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            f = line.rstrip("\r\n").split("\t")
            if len(f) >= 3:
                pairs.append((f[0].lower(), f[1].lower(), float(f[2])))
    return pairs


def _ranks(xs):
    """Average (fractional) ranks, so ties share a rank -- needed for a correct Spearman."""
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def spearman(xs, ys):
    if len(xs) < 3:
        return None
    rx, ry = _ranks(xs), _ranks(ys)
    n = len(xs)
    mx, my = sum(rx) / n, sum(ry) / n
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    vx = sum((a - mx) ** 2 for a in rx)
    vy = sum((b - my) ** 2 for b in ry)
    if vx == 0 or vy == 0:
        return None
    return cov / (vx * vy) ** 0.5


def global_cosine(basket, a, b):
    """Cosine in the single coarse routing space -- the 'one big space' comparator."""
    if a not in basket.routing or b not in basket.routing:
        return None
    va, vb = basket.routing[a], basket.routing[b]
    na, nb = sparsevectors.sparselength(va), sparsevectors.sparselength(vb)
    if na == 0 or nb == 0:
        return None
    return _sparse_dot(va, vb) / (na * nb)


def evaluate(basket, name, pairs):
    placed = lambda w: bool(basket.membership.get(w))
    ball_h, ball_s = [], []        # human, model (ball) on answered pairs
    g_h, g_s = [], []              # human, model (global cosine) on in-vocab pairs
    n_oov = n_null = 0
    for w1, w2, human in pairs:
        gc = global_cosine(basket, w1, w2)
        if gc is not None:
            g_h.append(human); g_s.append(gc)
        s = basket.similarity(w1, w2)
        if s is not None:
            ball_h.append(human); ball_s.append(s)
        elif not (placed(w1) and placed(w2)):
            n_oov += 1
        else:
            n_null += 1
    total = len(pairs)
    answered = len(ball_s)
    print("=== {} ===  ({} pairs)".format(name, total))
    print("  answered (shared ball) : {:4d}  ({:4.1f}%)".format(answered, 100 * answered / total))
    print("  null (no shared ball)  : {:4d}  ({:4.1f}%)".format(n_null, 100 * n_null / total))
    print("  OOV (word never placed): {:4d}  ({:4.1f}%)".format(n_oov, 100 * n_oov / total))
    rho = spearman(ball_h, ball_s)
    grho = spearman(g_h, g_s)
    print("  Spearman rho, BALL model (on {} answered)   : {}".format(
        answered, "n/a" if rho is None else "{:.3f}".format(rho)))
    print("  Spearman rho, GLOBAL cosine (on {} in-vocab): {}".format(
        len(g_s), "n/a" if grho is None else "{:.3f}".format(grho)))
    # apples-to-apples: global cosine restricted to the pairs the ball model answered
    if answered >= 3:
        sub_h, sub_s = [], []
        answered_set = set()
        for w1, w2, human in pairs:
            if basket.similarity(w1, w2) is not None:
                gc = global_cosine(basket, w1, w2)
                if gc is not None:
                    sub_h.append(human); sub_s.append(gc)
        srho = spearman(sub_h, sub_s)
        print("  Spearman rho, GLOBAL cosine (same answered subset): {}".format(
            "n/a" if srho is None else "{:.3f}".format(srho)))


def main(argv):
    if len(argv) < 2:
        print(__doc__)
        return 1
    basket = Basket.load(argv[1])
    print("model: {}  (vocab={}, balls={})".format(
        argv[1], len(basket.index), len(basket.balls)))
    evaluate(basket, "SimLex-999", load_simlex(SIMLEX))
    print()
    evaluate(basket, "WordSim-353", load_ws353(WS353))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
