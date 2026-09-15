"""Syntactic subcategorization-frame induction for verbs.

Each verb type is assigned a soft distribution over six syntactic frames, induced
from its observed local contexts in the corpus. This mirrors the TMA clause-type
overlay (see categorize.classify_clause + accumulation), but the distribution is
per-verb-occurrence and accumulated only onto the verb token.

Six frames
----------
  0 INTRANS   no complement                sleep, cry, laugh
  1 TRANS     + direct-object NP           eat X, see X, want X
  2 DITRANS   + two NPs                     give me X, tell him X
  3 PP_OBL    + preposition / particle      put X on, go to, look at
  4 FIN_COMP  + finite clause (that ...)     think X, know X, say X
  5 INF_COMP  + to-infinitive               want to go, try to X

In the encoder each verb's frame overlay is
    frame_overlay(w) = frame_alpha * sum_f( dist[w][f] * frame_vec[f] )
with six frozen random unit basis vectors frame_vec[0..5].

Usage (spot-check):
  python experiments/verbframes.py \
      --data_dir experiments/data/babylm_simple --lit_dir experiments/litdata
"""
import os
import re
import sys
import collections
from typing import Dict, List, Optional

import numpy as np
import nltk

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # experiments/

# ── frame ids ────────────────────────────────────────────────────────────────
INTRANS  = 0
TRANS    = 1
DITRANS  = 2
PP_OBL   = 3
FIN_COMP = 4
INF_COMP = 5
N_FRAMES = 6

FRAME_NAMES = {
    INTRANS:  "intrans",
    TRANS:    "trans",
    DITRANS:  "ditrans",
    PP_OBL:   "pp_obl",
    FIN_COMP: "fin_comp",
    INF_COMP: "inf_comp",
}

# ── POS tag sets ─────────────────────────────────────────────────────────────
_VERB_TAGS = {"VB", "VBZ", "VBP", "VBD", "VBN", "VBG"}   # MD (modals) excluded
_TENSED_VERB_TAGS = {"VBZ", "VBP", "VBD"}                # finite verb forms
_NOMINAL_TAGS = {"NN", "NNS", "NNP", "NNPS", "PRP", "CD"}  # heads of an object NP
_NP_INTERNAL_TAGS = {"DT", "PDT", "JJ", "JJR", "JJS", "PRP$", "POS", "CD"}
_ADV_TAGS = {"RB", "RBR", "RBS"}
_CLAUSE_BOUNDARY = {".", ",", ";", ":", "!", "?", "<eos>"}

# Complementiser / subordinator cues for a finite clausal complement
_FIN_COMP_MARKERS = {"that", "whether", "if", "what", "who", "how", "why", "where", "when"}


def _find_object_np(tags: List[str], toks: List[str], start: int, stop: int):
    """Return (kind, next_index) scanning an NP starting at `start`.

    kind in {"np", None}. Skips NP-internal material (det/adj/poss) to reach a
    nominal head. next_index is the position just after the NP (for DITRANS check).
    """
    j = start
    saw_internal = False
    while j < stop:
        t = tags[j]
        if t in _NOMINAL_TAGS:
            return "np", j + 1
        if t in _NP_INTERNAL_TAGS:
            saw_internal = True
            j += 1
            continue
        break
    # a determiner/adjective run with no explicit head noun still implies an NP
    if saw_internal:
        return "np", j
    return None, start


def classify_verb_frame(i: int, toks: List[str], tags: List[str]) -> np.ndarray:
    """Soft distribution over the six frames for the verb at position `i`.

    Inspects the in-clause window after the verb (up to the next clause boundary,
    max 6 tokens). Cue precedence: INF_COMP > FIN_COMP > DITRANS > TRANS > PP_OBL
    > INTRANS. Returns a near-one-hot vector with light smoothing.
    """
    n = len(toks)
    stop = min(n, i + 7)
    # clip the window at the next clause boundary
    for k in range(i + 1, stop):
        if toks[k] in _CLAUSE_BOUNDARY or tags[k] in _CLAUSE_BOUNDARY:
            stop = k
            break

    frame = INTRANS
    j = i + 1
    # skip immediate adverbs ("run quickly to ...", "see clearly that ...")
    while j < stop and tags[j] in _ADV_TAGS:
        j += 1

    if j < stop:
        tok, tag = toks[j], tags[j]
        # INF_COMP: to + base verb
        if tok == "to" and j + 1 < min(n, stop + 2) and tags[j + 1] == "VB":
            frame = INF_COMP
        elif tok == "to" and tag == "TO":
            # "to" + verb just past the window edge still counts as control
            if j + 1 < n and tags[j + 1] == "VB":
                frame = INF_COMP
            else:
                frame = PP_OBL  # infinitival marker unclear → treat as oblique "to"
        # FIN_COMP: explicit complementiser, or NP + finite verb (that-less)
        elif tok in _FIN_COMP_MARKERS:
            frame = FIN_COMP
        else:
            kind, after = _find_object_np(tags, toks, j, stop)
            if kind == "np":
                # DITRANS: a second NP right after the first
                kind2, _ = _find_object_np(tags, toks, after, stop)
                if kind2 == "np":
                    frame = DITRANS
                # that-less finite complement: NP + tensed verb ("think it works")
                elif after < stop and tags[after] in _TENSED_VERB_TAGS:
                    frame = FIN_COMP
                # caused-motion / oblique: NP + governing PP ("put X on", "take X to")
                elif after < stop and tags[after] in {"IN", "RP"}:
                    frame = PP_OBL
                else:
                    frame = TRANS
            elif tag in {"IN", "RP"} or tag == "TO":
                frame = PP_OBL
            else:
                frame = INTRANS

    out = np.full(N_FRAMES, 0.02, dtype=np.float32)  # light smoothing
    out[frame] += 1.0
    return out / out.sum()


# ── Gutenberg boilerplate stripper ───────────────────────────────────────────
_GUT_START = re.compile(r"\*\*\*\s*START OF TH(E|IS) PROJECT GUTENBERG.*?\*\*\*",
                        re.IGNORECASE | re.DOTALL)
_GUT_END = re.compile(r"\*\*\*\s*END OF TH(E|IS) PROJECT GUTENBERG.*?\*\*\*",
                      re.IGNORECASE | re.DOTALL)


def strip_gutenberg(text: str) -> str:
    """Return only the body between the START and END Gutenberg markers."""
    m1 = _GUT_START.search(text)
    if m1:
        text = text[m1.end():]
    m2 = _GUT_END.search(text)
    if m2:
        text = text[:m2.start()]
    return text.strip()


def _iter_sentences_from_files(paths, sample_chars: Optional[int], strip_gut: bool):
    """Yield tokenised+tagged sentences from files, up to sample_chars per call."""
    buf, remaining = [], (sample_chars if sample_chars else None)
    for path in paths:
        if remaining is not None and remaining <= 0:
            break
        with open(path, encoding="utf-8", errors="replace") as f:
            text = f.read()
        if strip_gut:
            text = strip_gutenberg(text)
        buf.append(text)
        if remaining is not None:
            remaining -= len(text)
    for sent in nltk.sent_tokenize("\n".join(buf)):
        toks = nltk.word_tokenize(sent.lower())
        if not toks:
            continue
        tags = [t for _, t in nltk.pos_tag(toks)]
        yield toks, tags


def induce_frame_distributions(vocab: List[str],
                               babylm_files: List[str],
                               lit_files: List[str] = None,
                               obs_cap: Optional[int] = None,
                               babylm_sample_chars: int = 2_000_000,
                               verbose: bool = True) -> Dict[str, np.ndarray]:
    """Induce a per-verb frame distribution over `vocab`.

    Accumulates classify_verb_frame() onto each verb token (POS in VB*), capping at
    the first `obs_cap` occurrences per verb type (None = unlimited). Reads ALL of
    lit_files (small, clean prose) plus a `babylm_sample_chars` sample of BabyLM.

    Returns {word: np.ndarray(N_FRAMES,)}; verbs never observed as a verb get an
    all-zero vector (→ no overlay in the encoder).
    """
    vocab_set = set(vocab)
    accum: Dict[str, np.ndarray] = collections.defaultdict(
        lambda: np.zeros(N_FRAMES, dtype=np.float32))
    counts: Dict[str, int] = collections.defaultdict(int)

    sources = [(babylm_files, babylm_sample_chars, False)]
    if lit_files:
        sources.append((lit_files, None, True))  # all litdata, strip Gutenberg

    for paths, sample, strip_gut in sources:
        for toks, tags in _iter_sentences_from_files(paths, sample, strip_gut):
            for i, (tok, tag) in enumerate(zip(toks, tags)):
                if tag not in _VERB_TAGS or tok not in vocab_set:
                    continue
                if obs_cap is not None and counts[tok] >= obs_cap:
                    continue
                accum[tok] += classify_verb_frame(i, toks, tags)
                counts[tok] += 1

    frame_distributions: Dict[str, np.ndarray] = {}
    zero = np.zeros(N_FRAMES, dtype=np.float32)
    n_verbs = 0
    for w in vocab:
        a = accum.get(w)
        if a is not None and a.sum() > 0:
            frame_distributions[w] = a / a.sum()
            n_verbs += 1
        else:
            frame_distributions[w] = zero.copy()

    if verbose:
        print(f"  induced frames for {n_verbs:,} verb types "
              f"(obs_cap={obs_cap}); mean frame mass:")
        stacked = np.stack([frame_distributions[w] for w in vocab
                            if frame_distributions[w].sum() > 0])
        if len(stacked):
            mean = stacked.mean(axis=0)
            for f in range(N_FRAMES):
                print(f"    {FRAME_NAMES[f]:9s}: {mean[f]:.3f}")
    return frame_distributions


# ── CLI spot-check ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    import pickle

    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default=os.path.join(_ROOT, "experiments", "data", "babylm_simple"))
    ap.add_argument("--lit_dir", default=os.path.join(_ROOT, "experiments", "litdata"))
    ap.add_argument("--obs_cap", type=int, default=None)
    ap.add_argument("--spot", nargs="+",
                    default=["give", "want", "put", "think", "sleep", "go", "see",
                             "tell", "eat", "know", "try", "look", "run", "say"])
    args = ap.parse_args()

    meta_path = os.path.join(args.data_dir, "meta.pkl")
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)
    vocab = [meta["itos"][i] for i in range(meta["vocab_size"])]

    train_dir = os.path.join(args.data_dir, "train")
    babylm_files = ([os.path.join(train_dir, fn) for fn in os.listdir(train_dir)
                     if fn.endswith(".txt")] if os.path.isdir(train_dir) else [])
    lit_files = ([os.path.join(args.lit_dir, fn) for fn in os.listdir(args.lit_dir)
                  if fn.endswith(".txt")] if os.path.isdir(args.lit_dir) else [])

    dists = induce_frame_distributions(vocab, babylm_files, lit_files,
                                       obs_cap=args.obs_cap, verbose=True)

    print("\nSpot-check (argmax frame + distribution):")
    hdr = "  {:12s} {:9s}  ".format("verb", "argmax") + "  ".join(
        f"{FRAME_NAMES[f]:>8s}" for f in range(N_FRAMES))
    print(hdr)
    for w in args.spot:
        d = dists.get(w, np.zeros(N_FRAMES))
        if d.sum() == 0:
            print(f"  {w:12s} {'(unseen)':9s}")
            continue
        am = FRAME_NAMES[int(d.argmax())]
        print(f"  {w:12s} {am:9s}  " + "  ".join(f"{d[f]:8.3f}" for f in range(N_FRAMES)))
