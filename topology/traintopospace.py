"""Train a basket-of-balls semantic space from one or more plain-text files and save it.

Usage:
    python traintopospace.py <input.txt> [more.txt ...] [output.pkl]
    python traintopospace.py data/*.txt

Every .txt argument is read and concatenated into one corpus; an optional .pkl argument names
the output (default topospace.pkl). Tunables are the Basket constructor kwargs below.
"""
import sys

from ballspace import Basket

CONFIG = dict(
    dim_index=1000,     # basket (routing) dimensionality -- coarse addressing
    denseness_index=10,
    dim_ball=20,        # local ball dimensionality -- the economy knob (hypothesis wants ~10)
    denseness_ball=6,
    window=2,
    f_min=3,            # occurrences before an item is mature enough to place
    theta_spawn=0.35,   # cosine to a ball centroid required to join it (moderate)
    n_min=5,            # smallest cluster that may become a ball
    ball_max=40,        # balls larger than this are divisively bisected (median cut)
    split_max_depth=6,
    cluster_order="freq_desc",
)


def strip_gutenberg(text):
    """Drop Project Gutenberg header/footer so licence boilerplate does not form junk balls."""
    lo = text.upper()
    start = lo.find("*** START OF")
    if start != -1:
        start = text.find("\n", start)
        text = text[start + 1:] if start != -1 else text
    lo = text.upper()
    end = lo.find("*** END OF")
    if end != -1:
        text = text[:end]
    return text


def main(argv):
    infiles = [a for a in argv[1:] if a.lower().endswith(".txt")]
    outs = [a for a in argv[1:] if a.lower().endswith(".pkl")]
    if not infiles:
        print(__doc__)
        return 1
    outfile = outs[0] if outs else "topospace.pkl"

    parts = []
    for infile in infiles:
        with open(infile, "r", encoding="utf-8", errors="replace") as fh:
            parts.append(strip_gutenberg(fh.read()))
    text = "\n".join(parts)
    print("corpus: {} file(s), {} chars".format(len(infiles), len(text)))

    basket = Basket(**CONFIG).fit(text)
    basket.save(outfile)

    s = basket.stats()
    print("saved:", outfile)
    print("vocab={vocab} mature={mature} balls={n_balls} buffered={buffered}".format(**s))
    print("balls by class:", s["balls_by_class"])
    print("mature by class:", s["mature_by_class"])
    print("ball sizes (desc):", s["ball_sizes"][:30])

    # sample within-ball neighbours from the largest balls
    for ball in sorted(basket.balls, key=lambda b: len(b.members), reverse=True)[:8]:
        item = max(ball.members, key=lambda m: basket.frequency[m])
        nbrs = [w for w, _ in basket.neighbours(item, 8)]
        print("  ball {:>3} [{:>6}] ({:>3}) {} -> {}".format(
            ball.id, ball.wordclass, len(ball.members), item, ", ".join(nbrs)))

    # demonstrate cross-ball null
    if len(basket.balls) >= 2 and basket.balls[0].members and basket.balls[1].members:
        a = sorted(basket.balls[0].members)[0]
        b = sorted(basket.balls[1].members)[0]
        print("cross-ball similarity({}, {}) = {}".format(a, b, basket.similarity(a, b)))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
