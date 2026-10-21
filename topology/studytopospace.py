"""Load a saved basket-of-balls space and query it.

Usage:
    python studytopospace.py <model.pkl>                 # print stats + ball overview
    python studytopospace.py <model.pkl> <word>          # within-ball neighbours of <word>
    python studytopospace.py <model.pkl> <word> <word2>  # similarity (None if no shared ball)
"""
import sys

from ballspace import Basket


def main(argv):
    if len(argv) < 2:
        print(__doc__)
        return 1
    basket = Basket.load(argv[1])

    if len(argv) == 2:
        print("stats:", basket.stats())
        for ball in sorted(basket.balls, key=lambda b: len(b.members), reverse=True):
            members = sorted(ball.members, key=lambda m: -basket.frequency[m])
            print("  ball {:>3} [{:>6}] ({:>3}): {}".format(
                ball.id, ball.wordclass, len(ball.members),
                ", ".join(members[:14]) + (" ..." if len(ball.members) > 14 else "")))
        return 0

    if len(argv) == 3:
        word = argv[2]
        print("balls of {}: {}".format(word, sorted(basket.membership.get(word, set()))))
        print("neighbours:", [(w, round(s, 3)) for w, s in basket.neighbours(word, 15)])
        return 0

    a, b = argv[2], argv[3]
    print("similarity({}, {}) = {}".format(a, b, basket.similarity(a, b)))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
