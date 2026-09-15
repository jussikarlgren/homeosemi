"""Assign each word type to one of seven functional categories (plus residual),
and compute per-word TMA (tense-mood-aspect) clause-type distributions.

Primary categories
------------------
  0  RESIDUAL      punctuation, numbers, OOV, uncategorised
  1  REFERENTIAL   nouns, proper nouns, pronouns, adjectives
  2  PREPOSITION   prepositions, particles
  3  PREDICATIVE   verbs and bare auxiliaries
  4  ADV_CLAUSAL   negation, hedges, amplifiers (epistemic/evaluative)
  5  COORD_CONJ    coordinating conjunctions (and, but, or, nor, yet, so, for)
  6  SUBORD_CONJ   subordinating conjunctions / complementisers
  7  DETERMINER    articles, demonstratives, quantifiers

TMA clause-type distributions (clause-level, not word-level)
------------------------------------------------------------
  Four clause types after Halliday (see tma.txt):
    0  STATE      The sun shines. The house is yellow.  (present, stative)
    1  PROCESS    The ball is rolling. It is raining.   (progressive, dynamic)
    2  OCCURRENCE The window broke. The kettle boiled.  (simple past, punctual)
    3  UNREAL     I would like to be rich. If it rained.(modal/conditional)

  Each sentence is classified into a soft distribution over these four types.
  Every word token in that sentence accumulates weighted counts.
  The resulting per-word-type distribution is stored as a 4-d float array
  (sums to 1.0) in tma_distributions: Dict[str, np.ndarray].

  In the encoder, each word's TMA overlay is:
    tma_overlay(w) = tma_alpha * sum_t( dist[w][t] * tma_vec[t] )
  where tma_vec[0..3] are four frozen random unit vectors.

Usage:
  python experiments/categorize.py --vocab_file experiments/data/babylm_simple/meta.pkl
"""
import os
import sys
import pickle
import collections
from typing import Dict, List

import numpy as np
import nltk

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "jussipyutils"))

from lexicalfeatures import lexicon as LEX  # noqa: E402

# ── primary category ids ─────────────────────────────────────────────────────
RESIDUAL    = 0
REFERENTIAL = 1
PREPOSITION = 2
PREDICATIVE = 3
ADV_CLAUSAL = 4
COORD_CONJ  = 5
SUBORD_CONJ = 6
DETERMINER  = 7

CATEGORY_NAMES = {
    RESIDUAL:    "residual",
    REFERENTIAL: "referential",
    PREPOSITION: "preposition",
    PREDICATIVE: "predicative",
    ADV_CLAUSAL: "adv_clausal",
    COORD_CONJ:  "coord_conj",
    SUBORD_CONJ: "subord_conj",
    DETERMINER:  "determiner",
}

N_PRIMARY_CATS = 8  # used by encoders.py to size cat_vec array

# ── lexicon helpers ──────────────────────────────────────────────────────────
def _lset(key):
    return set(w.lower() for w in LEX.get(key, []))

# ── ADV_CLAUSAL (cat 4) — epistemic/evaluative only ──────────────────────────
_NEGATION  = _lset("negation")
_HEDGE     = _lset("hedgelist")
_AMPLIFIER = _lset("amplifier")
ADV_CLAUSAL_WORDS = _NEGATION | _HEDGE | _AMPLIFIER

# ── TMA clause-type indices ──────────────────────────────────────────────────
TMA_STATE      = 0  # The sun shines. The house is yellow.
TMA_PROCESS    = 1  # The ball is rolling. It is raining.
TMA_OCCURRENCE = 2  # The window broke. The kettle boiled.
TMA_UNREAL     = 3  # I would like to be rich. If it rained...
N_TMA_TYPES    = 4

TMA_TYPE_NAMES = {
    TMA_STATE:      "state",
    TMA_PROCESS:    "process",
    TMA_OCCURRENCE: "occurrence",
    TMA_UNREAL:     "unreal",
}

# ── Auxiliary and modal sets (used for clause classification) ─────────────────
AUX_BE   = {"be","am","is","are","was","were","been","being"}
AUX_HAVE = {"have","has","had","having"}
AUX_DO   = {"do","does","did","done","doing"}
AUX_WORDS = AUX_BE | AUX_HAVE | AUX_DO

_NEC_MODAL = _lset("necessitymodals")
_POS_MODAL = _lset("possibilitymodals")
_PRD_MODAL = _lset("predictivemodals")
MODAL_WORDS = _NEC_MODAL | _POS_MODAL | _PRD_MODAL

# Stative verbs → STATE signal
_STATIVE_WORDS = (
    _lset("perceptionverbs") |   # see, hear, feel, know, smell, taste
    _lset("thinkverbs") |        # believe, think, understand, realize
    _lset("privateverbs")        # want, need, like, love, hate, prefer
)

# Occurrence adverbials → OCCURRENCE signal
OCCURRENCE_ADVS = {
    "suddenly","immediately","abruptly","instantly","unexpectedly",
    "all","at","once","just","then","finally","eventually","already",
    "quickly","suddenly","sharply","swiftly","briskly",
}

# Unreal/conditional markers
CONDITIONAL_WORDS = {"if","unless","whether","suppose","supposing",
                     "provided","assuming","imagine","wish","hoped"}


# ── clause-type classifier ────────────────────────────────────────────────────
def classify_clause(tokens: list, pos_tags: list) -> np.ndarray:
    """Return a soft distribution [state, process, occurrence, unreal] for
    a sentence represented as (tokens, pos_tags) both lowercase.

    Heuristics (additive scores, then softmax):
      UNREAL:     modal present + conditional structure
      PROCESS:    progressive (be + VBG)
      OCCURRENCE: simple past (VBD) + occurrence adverbials
      STATE:      present tense stative verb, no progressive/past
    """
    scores = np.zeros(N_TMA_TYPES, dtype=np.float32)

    tag_set   = set(pos_tags)
    token_set = set(tokens)

    has_vbg  = "VBG" in tag_set
    has_vbd  = "VBD" in tag_set
    has_vbz  = any(t in {"VBZ","VBP"} for t in pos_tags)
    has_be   = bool(token_set & AUX_BE)
    has_modal = bool(token_set & MODAL_WORDS)
    has_cond  = bool(token_set & CONDITIONAL_WORDS)
    has_occ_adv = bool(token_set & OCCURRENCE_ADVS)
    has_stative = bool(token_set & _STATIVE_WORDS)
    has_would   = bool(token_set & {"would","could","might"})

    # UNREAL: modal auxiliaries (would/could/might) are the strongest signal;
    # reinforced by conditional clause structure
    if has_would:
        scores[TMA_UNREAL] += 3.0
    if has_modal and not has_would:
        scores[TMA_UNREAL] += 1.0
    if has_cond:
        scores[TMA_UNREAL] += 1.5

    # PROCESS: be + progressive participle
    if has_vbg and has_be:
        scores[TMA_PROCESS] += 3.0
    elif has_vbg:
        scores[TMA_PROCESS] += 1.0

    # OCCURRENCE: simple past is the primary marker; occurrence adverbials add
    if has_vbd and not has_be:   # pure simple past, not perfect
        scores[TMA_OCCURRENCE] += 2.0
    elif has_vbd:
        scores[TMA_OCCURRENCE] += 1.0
    if has_occ_adv:
        scores[TMA_OCCURRENCE] += 2.0

    # STATE: present tense + stative verb, no past/progressive
    if has_stative:
        scores[TMA_STATE] += 2.0
    if has_vbz and not has_vbd and not (has_vbg and has_be):
        scores[TMA_STATE] += 1.5

    # Default: if no strong signal, treat as state (background information)
    if scores.sum() == 0:
        scores[TMA_STATE] = 1.0

    # Softmax normalise
    scores = np.exp(scores - scores.max())
    return scores / scores.sum()

# ── PREDICATIVE (cat 3): verb-class lexicon ──────────────────────────────────
_VERB_CLASSES = ["motionverbs","sayverbs","thinkverbs","perceptionverbs",
                 "privateverbs","publicverbs","suasiveverbs","implicativeverbs"]
PREDICATIVE_WORDS = set().union(*[_lset(k) for k in _VERB_CLASSES])
PREDICATIVE_WORDS |= AUX_WORDS  # auxiliaries are predicative (primary cat)

# ── REFERENTIAL (cat 1) ───────────────────────────────────────────────────────
REFERENTIAL_WORDS = _lset("personalpronouns") | _lset("canonicals")

# ── Structural sub-categories ────────────────────────────────────────────────
COORD_CONJ_WORDS = {"for","and","nor","but","or","yet","so"}

SUBORD_CONJ_WORDS = {
    "although","because","before","if","since","though","unless","until",
    "when","where","whereas","whether","while","after","as","that","once",
    "provided","insofar","lest","albeit","supposing","except",
}

DETERMINER_WORDS = {
    "a","an","the","this","that","these","those","each","every","any",
    "some","all","both","either","neither","no","such","another","few",
    "many","more","most","much","other","several","what","which","whose",
    "less","least","enough","plenty",
}

PREPOSITION_WORDS = {
    "about","above","across","after","against","along","among","around",
    "at","before","behind","below","beneath","beside","between","beyond",
    "by","despite","during","except","from","in","inside","into","near",
    "of","off","on","onto","out","outside","over","past","through",
    "throughout","to","toward","towards","under","until","up","upon",
    "via","with","within","without","s","'s",
}
# Note: "before/after/since/until/as/that" → SUBORD_CONJ (takes precedence)
# "for" → COORD_CONJ (takes precedence over PREPOSITION)

# ── POS tag sets ─────────────────────────────────────────────────────────────
_PREDICATIVE_TAGS = {"VB","VBZ","VBP","VBD","VBN","VBG","MD"}
_REFERENTIAL_TAGS = {"NN","NNS","NNP","NNPS","PRP","PRP$","JJ","JJR","JJS","CD","EX"}
_TMA_INFLECTED_TAGS = {"VBD","VBG","VBZ"}  # past, progressive, 3sg-present


# ── core categorisation function ─────────────────────────────────────────────
def _word_category_from_lexicon_and_tag(word: str, tag) -> int:
    """Precedence: ADV_CLAUSAL > COORD > SUBORD > DET > PREP > PRED > REF > RESIDUAL.

    Modals and time adverbials are NOT in ADV_CLAUSAL here — they live in the TMA
    overlay. Their primary category is assigned as PREDICATIVE (modals) or RESIDUAL
    (time adverbials not otherwise classified).
    """
    if word in ADV_CLAUSAL_WORDS:
        return ADV_CLAUSAL
    if word in COORD_CONJ_WORDS or tag == "CC":
        return COORD_CONJ
    if word in SUBORD_CONJ_WORDS:
        return SUBORD_CONJ
    if word in DETERMINER_WORDS or tag in {"DT", "PDT", "WDT", "WP$"}:
        return DETERMINER
    if word in PREPOSITION_WORDS or tag in {"IN", "TO", "RP"}:
        return PREPOSITION
    if word in PREDICATIVE_WORDS or tag in _PREDICATIVE_TAGS:
        return PREDICATIVE
    if word in REFERENTIAL_WORDS or tag in _REFERENTIAL_TAGS:
        return REFERENTIAL
    return RESIDUAL


# ── main assignment function ─────────────────────────────────────────────────
def assign_categories(vocab: List[str],
                      sample_size: int = 200_000,
                      corpus_files: List[str] = None,
                      verbose: bool = True):
    """Return (category_dict, tma_distributions) for every word in vocab.

    category_dict     : {word: primary_category_id}
    tma_distributions : {word: np.ndarray shape (4,)} — clause-type distribution
                        [state, process, occurrence, unreal], sums to 1.
                        Words not seen in corpus get a uniform prior.
    """
    specials = {"<unk>", "<eos>"}
    cat_votes: Dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    # accumulated weighted clause-type counts per word type
    tma_accum: Dict[str, np.ndarray] = collections.defaultdict(
        lambda: np.zeros(N_TMA_TYPES, dtype=np.float32))

    if corpus_files:
        buf, remaining = [], sample_size
        for path in corpus_files:
            if remaining <= 0:
                break
            with open(path, encoding="utf-8", errors="replace") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    buf.append(line)
                    remaining -= len(line)
                    if remaining <= 0:
                        break
        sentences = nltk.sent_tokenize(" ".join(buf))
        for sent in sentences:
            tokens = nltk.word_tokenize(sent.lower())
            tagged = nltk.pos_tag(tokens)
            tags = [t for _, t in tagged]
            # primary category votes
            for word, tag in tagged:
                cat_votes[word][_word_category_from_lexicon_and_tag(word, tag)] += 1
            # clause-level TMA distribution — same distribution for every token
            dist = classify_clause(tokens, tags)
            for word in tokens:
                tma_accum[word] += dist

    # Build output dicts
    category: Dict[str, int] = {}
    tma_distributions: Dict[str, np.ndarray] = {}
    uniform = np.ones(N_TMA_TYPES, dtype=np.float32) / N_TMA_TYPES

    for word in vocab:
        if word in specials:
            category[word] = RESIDUAL
            tma_distributions[word] = uniform.copy()
            continue
        w = word.lower()
        category[word] = (cat_votes[w].most_common(1)[0][0]
                          if w in cat_votes
                          else _word_category_from_lexicon_and_tag(w, None))
        if w in tma_accum and tma_accum[w].sum() > 0:
            d = tma_accum[w]
            tma_distributions[word] = d / d.sum()
        else:
            tma_distributions[word] = uniform.copy()

    if verbose:
        counts = collections.Counter(category.values())
        print("\nCategory assignment summary:")
        for cid in sorted(CATEGORY_NAMES):
            sample = [w for w, c in category.items() if c == cid][:6]
            print(f"  {CATEGORY_NAMES[cid]:15s} ({cid})  "
                  f"{counts[cid]:5d} types   e.g. {sample}")
        # TMA summary: mean distribution over all vocab
        all_dists = np.stack(list(tma_distributions.values()))
        mean_dist = all_dists.mean(axis=0)
        print(f"\n  TMA mean clause-type distribution over vocab:")
        for t, name in TMA_TYPE_NAMES.items():
            print(f"    {name:12s}: {mean_dist[t]:.3f}")
        print()

    return category, tma_distributions


# ── CLI spot-check ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab_file", help="meta.pkl from prepare_words.py")
    ap.add_argument("--spot", nargs="+",
                    default=["not","maybe","the","doggy","in","run","and",
                             "said","quickly","was","because","had","yesterday",
                             "never","should","very","when","a","but"])
    args = ap.parse_args()

    if args.vocab_file:
        with open(args.vocab_file, "rb") as f:
            meta = pickle.load(f)
        vocab = [meta["itos"][i] for i in range(meta["vocab_size"])]
        data_dir = os.path.dirname(args.vocab_file)
        train_dir = os.path.join(data_dir, "train")
        files = ([os.path.join(train_dir, fn) for fn in os.listdir(train_dir)
                  if fn.endswith(".txt")] if os.path.isdir(train_dir) else [])
        cats, tma_dists = assign_categories(vocab, corpus_files=files, verbose=True)
    else:
        cats, tma_dists = {}, {}

    print("Spot-check (primary category + TMA clause-type distribution):")
    hdr = f"  {'word':20s}  {'category':15s}  {'state':>7s}  {'process':>7s}  {'occur':>7s}  {'unreal':>7s}"
    print(hdr)
    for w in args.spot:
        c = cats.get(w, _word_category_from_lexicon_and_tag(w, None))
        d = tma_dists.get(w, np.ones(N_TMA_TYPES)/N_TMA_TYPES)
        print(f"  {w:20s}  {CATEGORY_NAMES[c]:15s}  "
              f"{d[0]:7.3f}  {d[1]:7.3f}  {d[2]:7.3f}  {d[3]:7.3f}")
