"""Input-encoding schemes for char-level and word-level LM experiments.

Each encoder returns a torch.FloatTensor of shape (vocab_size, n_embd) to be
loaded as a token-embedding matrix, or None for the 'learned' baseline
(meaning: keep nanoGPT's default trainable embedding).

Char-level encoders: random indexing (RI) over characters.
Word-level encoders: RI over words, with an optional category overlay that
  pulls words in the same functional category (referential / structural /
  predicative / adv_clausal) closer together in embedding space.
"""
import os
import sys
import random as _random

import numpy as np
import torch

# Make homeosemi (repo root) and its utils (jussipyutils) importable.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_ROOT, os.path.join(_ROOT, "jussipyutils")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import sparsevectors  # noqa: E402
from hyperdimensionalsemanticspace import SemanticSpace  # noqa: E402

# char-level
CHAR_ENCODINGS = ["learned", "random", "onehot", "homeosemi_index",
                  "homeosemi_context", "homeosemi_context_ordered"]
# word-level
WORD_ENCODINGS = ["word_learned", "word_random",
                  "word_homeosemi_context", "word_category", "word_frame",
                  "word_levin", "bpe_learned"]

AVAILABLE = CHAR_ENCODINGS + WORD_ENCODINGS


def _rows_to_tensor(rows, dim, emb_scale):
    """Stack dense rows, L2-normalise each, then rescale.

    emb_scale > 0 sets the target per-element std so a frozen matrix starts at
    roughly the same scale as nanoGPT's default init (std ~0.02); emb_scale == 0
    leaves rows at unit L2 norm.
    """
    mat = np.asarray(rows, dtype=np.float32)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    mat = mat / norms
    if emb_scale > 0:
        mat = mat * (emb_scale * np.sqrt(dim))
    return torch.from_numpy(mat)


def _dense_index_vectors(vocab, dim, denseness, seed):
    """One sparse random index vector per character, returned dense (vocab, dim).

    Built via a homeosemi SemanticSpace so the index vectors are genuine RI
    vectors; converted to dense arrays with sparsevectors.listify.
    """
    _random.seed(seed)
    space = SemanticSpace(dimensionality=dim, denseness=denseness)
    for ch in vocab:
        space.observe(ch, update=False)
    idx = {ch: np.asarray(sparsevectors.listify(space.indexspace[ch], dim),
                          dtype=np.float32)
           for ch in vocab}
    return idx


def _filtered_chars(text, vocabset):
    return [c for c in text if c in vocabset]


def build(name, vocab, dim, text=None, denseness=10, window=5,
          enc_chars=200_000, emb_scale=0.02, seed=1337):
    """Return a (vocab_size, dim) FloatTensor for the given scheme, or None
    for 'learned'. `vocab` is the ordered list of characters (index == token id).
    """
    if name == "learned":
        return None

    if name == "onehot":
        if dim < len(vocab):
            raise ValueError(f"onehot needs dim >= vocab ({len(vocab)}), got {dim}")
        rows = [[1.0 if j == i else 0.0 for j in range(dim)] for i in range(len(vocab))]
        return _rows_to_tensor(rows, dim, emb_scale)

    if name == "random":
        rng = np.random.default_rng(seed)
        rows = rng.standard_normal((len(vocab), dim))
        return _rows_to_tensor(rows, dim, emb_scale)

    idx = _dense_index_vectors(vocab, dim, denseness, seed)

    if name == "homeosemi_index":
        rows = [idx[ch] for ch in vocab]
        return _rows_to_tensor(rows, dim, emb_scale)

    if name in ("homeosemi_context", "homeosemi_context_ordered"):
        if text is None:
            raise ValueError(f"{name} requires the training text")
        ordered = name.endswith("_ordered")
        # Correct, non-mutating permutations for before/after (homeosemi's own
        # sparsevectors.permute aliases its input, so we do it here in numpy).
        rng = np.random.default_rng(seed + 1)
        perm_before = rng.permutation(dim) if ordered else None
        perm_after = rng.permutation(dim) if ordered else None

        vocabset = set(vocab)
        chars = _filtered_chars(text[:enc_chars], vocabset)
        ctx = {ch: np.zeros(dim, dtype=np.float32) for ch in vocab}
        n = len(chars)
        for i in range(n):
            center = chars[i]
            lo, hi = max(0, i - window), min(n, i + window + 1)
            for j in range(lo, hi):
                if j == i:
                    continue
                v = idx[chars[j]]
                if ordered:
                    v = v[perm_before] if j < i else v[perm_after]
                ctx[center] += v
        rows = [ctx[ch] for ch in vocab]
        return _rows_to_tensor(rows, dim, emb_scale)

    raise ValueError(f"unknown encoding '{name}'; choose from {AVAILABLE}")


# ── word-level encoders ──────────────────────────────────────────────────────

def _word_index_vectors(vocab, dim, denseness, seed):
    """One sparse RI index vector per word type, returned as dense numpy arrays."""
    _random.seed(seed)
    space = SemanticSpace(dimensionality=dim, denseness=denseness)
    for w in vocab:
        space.observe(w, update=False)
    return {w: np.asarray(sparsevectors.listify(space.indexspace[w], dim),
                           dtype=np.float32)
            for w in vocab}


def _word_context_vectors(vocab, token_ids, dim, denseness, window, seed,
                          max_tokens=None):
    """RI co-occurrence context vectors over a word-token stream.

    token_ids: 1-D array of integer token ids (uint16 from .bin file).
    vocab: ordered list of word strings (index == token id).
    Returns dict {word: dense_context_vector}.
    """
    idx = _word_index_vectors(vocab, dim, denseness, seed)
    ctx = {w: np.zeros(dim, dtype=np.float32) for w in vocab}
    n = len(token_ids) if max_tokens is None else min(len(token_ids), max_tokens)
    for i in range(n):
        center = vocab[token_ids[i]]
        lo = max(0, i - window)
        hi = min(n, i + window + 1)
        for j in range(lo, hi):
            if j == i:
                continue
            ctx[center] += idx[vocab[token_ids[j]]]
    return ctx


def build_word(name, vocab, dim, token_ids=None, word_categories=None,
               tma_categories=None, frame_categories=None, levin_categories=None,
               denseness=10, window=5,
               overlay_alpha=0.5, tma_alpha=0.3, frame_alpha=0.3, levin_alpha=0.3,
               enc_tokens=500_000, emb_scale=0.02, seed=1337):
    """Return a (vocab_size, dim) FloatTensor for a word-level encoding, or None
    for 'word_learned' / 'bpe_learned' (both handled as trainable by the harness).

    Parameters
    ----------
    vocab         : ordered list of word strings (index i == token id i)
    dim           : embedding dimension (n_embd)
    token_ids     : 1-D uint16 numpy array of training token ids
                    (required for word_homeosemi_context and word_category)
    word_categories : dict {word: category_id} from categorize.py
                    (required for word_category)
    overlay_alpha : strength of category RI overlay (0 = no overlay)
    enc_tokens    : number of training tokens to use for co-occurrence stats
    """
    if name in ("word_learned", "bpe_learned"):
        return None

    if name == "word_random":
        rng = np.random.default_rng(seed)
        rows = rng.standard_normal((len(vocab), dim))
        return _rows_to_tensor(rows, dim, emb_scale)

    if name in ("word_homeosemi_context", "word_category", "word_frame", "word_levin"):
        if token_ids is None:
            raise ValueError(f"{name} requires token_ids (training .bin data)")
        print(f"  building word co-occurrence vectors "
              f"(window={window}, tokens={min(enc_tokens, len(token_ids)):,}) ...")
        ctx = _word_context_vectors(vocab, token_ids, dim, denseness,
                                    window, seed, max_tokens=enc_tokens)
        rows = np.stack([ctx[w] for w in vocab])

        if name in ("word_category", "word_frame", "word_levin"):
            if word_categories is None:
                raise ValueError(f"{name} requires word_categories dict")
            from categorize import N_PRIMARY_CATS, N_TMA_TYPES
            n_cats = N_PRIMARY_CATS  # covers primary category ids 0-7
            rng = np.random.default_rng(seed + 42)
            # one frozen random unit vector per primary category
            cat_vecs = rng.standard_normal((n_cats, dim)).astype(np.float32)
            cat_vecs /= np.linalg.norm(cat_vecs, axis=1, keepdims=True)
            for i, w in enumerate(vocab):
                # primary category overlay (both word_category and word_frame)
                cat_id = word_categories.get(w, 0)
                if cat_id != 0:
                    rows[i] += overlay_alpha * cat_vecs[cat_id]

            if name == "word_category":
                # TMA overlay: weighted mix of 4 clause-type vectors (frame excludes this)
                tma_vecs = rng.standard_normal((N_TMA_TYPES, dim)).astype(np.float32)
                tma_vecs /= np.linalg.norm(tma_vecs, axis=1, keepdims=True)
                if tma_categories:
                    for i, w in enumerate(vocab):
                        dist = tma_categories.get(w)
                        if dist is not None and hasattr(dist, '__len__'):
                            rows[i] += tma_alpha * (np.asarray(dist, dtype=np.float32) @ tma_vecs)

            if name == "word_frame":
                # verb subcategorization-frame overlay: mix of N_FRAMES frozen basis vecs
                if not frame_categories:
                    raise ValueError("word_frame requires frame_categories dict")
                from verbframes import N_FRAMES
                frame_vecs = rng.standard_normal((N_FRAMES, dim)).astype(np.float32)
                frame_vecs /= np.linalg.norm(frame_vecs, axis=1, keepdims=True)
                for i, w in enumerate(vocab):
                    dist = frame_categories.get(w)
                    if dist is not None and hasattr(dist, '__len__') and np.sum(dist) > 0:
                        rows[i] += frame_alpha * (np.asarray(dist, dtype=np.float32) @ frame_vecs)

            if name == "word_levin":
                # Levin semantic-class overlay: one frozen vector per class, shared by all
                # class members (a pure "rule" prior — no within-class idiosyncrasy).
                if not levin_categories:
                    raise ValueError("word_levin requires levin_categories dict")
                n_levin = max(levin_categories.values()) + 1
                levin_vecs = rng.standard_normal((n_levin, dim)).astype(np.float32)
                levin_vecs /= np.linalg.norm(levin_vecs, axis=1, keepdims=True)
                for i, w in enumerate(vocab):
                    cid = levin_categories.get(w, 0)
                    if cid != 0:
                        rows[i] += levin_alpha * levin_vecs[cid]

        return _rows_to_tensor(rows, dim, emb_scale)

    raise ValueError(f"unknown word encoding '{name}'; choose from {WORD_ENCODINGS}")
