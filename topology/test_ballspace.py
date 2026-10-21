"""Sanity check for the basket-of-balls model: two obvious topical clusters must separate.

Animals and cooking sentences share no vocabulary, so they must land in different balls;
within a cluster similarity is defined and positive, across clusters it is null (None).
"""
import random

from ballspace import Basket

ANIMALS = ["horse", "donkey", "cow", "goat", "sheep"]
COOKING = ["fried", "boiled", "chopped", "grilled", "roasted"]


def build_text():
    lines = []
    for a in ANIMALS:
        for a2 in ANIMALS:
            if a != a2:
                lines.append("one {} grazed while another {} rested".format(a, a2))
    for c in COOKING:
        for c2 in COOKING:
            if c != c2:
                lines.append("chefs {} then {} everything quickly".format(c, c2))
    random.shuffle(lines)
    return ". ".join(lines) + "."


def main():
    random.seed(42)
    text = build_text()
    basket = Basket(dim_index=1000, dim_ball=20, window=2,
                    f_min=3, theta_spawn=0.12, n_min=3)
    basket.fit(text)

    print("stats:", basket.stats())
    for a in ANIMALS[:2]:
        print("  neighbours({}):".format(a),
              [(w, round(s, 3)) for w, s in basket.neighbours(a, 4)])

    # membership: an animal and a cooking word must not share a ball
    assert basket.stats()["n_balls"] >= 2, "expected at least two balls"

    within = basket.similarity("horse", "donkey")
    print("similarity(horse, donkey) =", within)
    assert within is not None and within > 0, "within-cluster similarity should be positive"

    across = basket.similarity("horse", "fried")
    print("similarity(horse, fried)  =", across)
    assert across is None, "cross-cluster similarity must be null (None)"

    print("OK")


if __name__ == "__main__":
    main()
