"""Pure-python overlay variants for the acquisition-vs-regularisation controls.

No numpy/torch: it only rearranges the assignment maps, so it runs anywhere and is
unit-testable without the ML stack. Wire into train_experiment.py:main() AFTER the
maps load from meta.pkl and BEFORE build_word().

Experiment A - scrambled overlay: a seeded permutation gives each word another
word's whole bundle (category + tma + frame + levin permuted together). The value
multiset per map is preserved, so the category histogram and (after
_rows_to_tensor's L2-normalise + rescale) the per-row magnitude are matched to the
real overlay; only the linguistic content is destroyed. real ~ scrambled => the
effect was regularisation, not linguistic structure.

Experiment B - targeted dissociation: keep the overlay only for a chosen set of
categories; every other word is set to RESIDUAL (0) so encoders.py's `cat_id != 0`
guard withholds its overlay. Run the primary-category overlay only (tma/frame
alphas 0), since those are soft distributions that cannot be subset-gated cleanly.
"""
import random

RESIDUAL = 0


def scramble_maps(seed, vocab, *maps):
    """Return copies of each map with one shared seeded permutation applied:
    word ``vocab[i]`` receives ``vocab[perm[i]]``'s entry in EVERY map. A word whose
    donor is absent from a given map is simply dropped from that map (same as the
    real map, which also omits uncategorised words)."""
    rng = random.Random(seed)
    perm = list(range(len(vocab)))
    rng.shuffle(perm)
    donor = {vocab[i]: vocab[perm[i]] for i in range(len(vocab))}
    return tuple(
        None if m is None else {w: m[donor[w]] for w in vocab if donor[w] in m}
        for m in maps
    )


def gate_categories(word_categories, on_ids, residual=RESIDUAL):
    """Return a copy of ``word_categories`` with any word whose category is not in
    ``on_ids`` set to ``residual`` (0). RI context is untouched (handled elsewhere)."""
    on = set(on_ids)
    return {w: (c if c in on else residual) for w, c in word_categories.items()}


if __name__ == "__main__":
    # self-test - pure python, no deps
    vocab = [f"w{i}" for i in range(1000)]
    cats = {w: (i % 8) for i, w in enumerate(vocab)}          # categories 0..7
    tma  = {w: [i] for i, w in enumerate(vocab)}              # stand-in for np arrays

    import collections
    (sc_cats, sc_tma) = scramble_maps(1337, vocab, cats, tma)
    assert collections.Counter(sc_cats.values()) == collections.Counter(cats.values()), \
        "scramble must preserve the category histogram"
    assert sc_cats != cats, "scramble must actually move something"
    # same permutation across maps: the word that got w_k's category also got w_k's tma
    donor = {w: sc_tma[w][0] for w in vocab}                  # tma value == donor index
    assert all(sc_cats[w] == cats[vocab[donor[w]]] for w in vocab), \
        "category and tma must be permuted together (consistent bundle)"
    # determinism
    assert scramble_maps(1337, vocab, cats)[0] == sc_cats, "same seed => same permutation"
    assert scramble_maps(42, vocab, cats)[0] != sc_cats, "different seed => different permutation"

    gated = gate_categories(cats, [3, 7])
    assert all((gated[w] in (0, 3, 7)) for w in vocab), "gate leaves only ON ids or 0"
    assert all((gated[w] == cats[w]) for w in vocab if cats[w] in (3, 7)), "ON categories kept"
    assert all((gated[w] == 0) for w in vocab if cats[w] not in (3, 7)), "OFF categories -> 0"
    kept = sum(1 for w in vocab if gated[w] != 0)
    print(f"scramble: histogram preserved, bundle consistent, deterministic - OK")
    print(f"gate ON={{3,7}}: {kept}/{len(vocab)} words keep an overlay, rest -> residual - OK")
    print("all self-tests passed")
