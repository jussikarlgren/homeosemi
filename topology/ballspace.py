"""Basket-of-balls-of-yarn semantic space (topology prototype).

Realises the hypothesis in topospacehypothesis.txt: instead of one global high-dimensional
Random Indexing space, keep a coarse *basket* structured into balls (local neighbourhoods),
each a small RI space. Similarity is answered only within a shared ball and is *null* (None)
across balls.

Basket structure = **lexical class**. Words are POS-tagged and pre-grouped into five coarse
categories; balls are then formed *within* each category by clustering distributionally similar
words. So referential terms (nouns) settle into animates / foods / ..., argument-structure terms
(verbs) into transitive / stative / ..., adverbials by type, and so on. Structural items
(prepositions, conjunctions, determiners) form their own class and therefore cannot bridge
content-word clusters -- the function-word "hub" problem is handled structurally, not by
frequency weighting.

Ball genesis = greedy **leader / centroid** clustering (single-link connected components
percolates into one giant ball on real text). Each word is tested against a running ball
centroid; a cluster that reaches size n_min becomes a ball; the rest stay buffered (null).
Training is two passes: pass 1 settles class + membership, pass 2 fills local ball vectors.
"""
import _paths  # noqa: F401  -- prepends homeosemi + jussipyutils to sys.path
import math
import pickle
from collections import Counter, defaultdict

import sparsevectors
from hyperdimensionalsemanticspace import SemanticSpace
from nltk import sent_tokenize, word_tokenize, pos_tag

# ---- five coarse lexical classes -------------------------------------------------
REF = "ref"        # referential terms  (~nouns)
ATTR = "attr"      # attribute terms    (~adjectives)
ARG = "arg"        # argument-structure (~lexical verbs)
ADV = "adv"        # adverbials         (~adverbs)
STRUCT = "struct"  # structural items   (preps, conjs, determiners, particles, pronouns, auxiliaries)
OTHER = "other"    # numerals, foreign, interjections, symbols
COARSE_ORDER = [REF, ARG, ATTR, ADV, STRUCT, OTHER]

_PENN_TO_COARSE = {
    "NN": REF, "NNS": REF, "NNP": REF, "NNPS": REF,
    "JJ": ATTR, "JJR": ATTR, "JJS": ATTR,
    "VB": ARG, "VBD": ARG, "VBG": ARG, "VBN": ARG, "VBP": ARG, "VBZ": ARG,
    "RB": ADV, "RBR": ADV, "RBS": ADV, "WRB": ADV,
    "IN": STRUCT, "CC": STRUCT, "DT": STRUCT, "TO": STRUCT, "PDT": STRUCT, "WDT": STRUCT,
    "PRP$": STRUCT, "WP$": STRUCT, "POS": STRUCT, "RP": STRUCT, "EX": STRUCT,
    # pronouns are closed-class referents -> treat as structural, not content nouns
    "PRP": STRUCT, "WP": STRUCT, "MD": STRUCT,
}

# Closed-class auxiliary / copula lemmas. POS taggers label these as verbs (VB*), so they leak
# into ARG and anchor a catch-all sub-hub there. Force them structural regardless of tag.
_STRUCTURAL_WORDS = frozenset({
    "be", "is", "am", "are", "was", "were", "been", "being",   # copula / auxiliary BE
    "have", "has", "had", "having",                            # auxiliary HAVE
    "do", "does", "did", "doing",                              # auxiliary DO
    "will", "would", "shall", "should",                        # modals
    "can", "could", "may", "might", "must",
})


def coarse_class(word, tag):
    if word in _STRUCTURAL_WORDS:
        return STRUCT
    return _PENN_TO_COARSE.get(tag, OTHER)


def tokenised_sentences(text):
    """Lowercase, sentence-split, word-tokenise; keep alphabetic tokens only."""
    for sentence in sent_tokenize(text.lower()):
        yield [w for w in word_tokenize(sentence) if w.isalpha()]


def _sparse_dot(a, b):
    """Dot product of two sparse vectors held as {dim: weight} dicts."""
    if len(a) > len(b):
        a, b = b, a
    return sum(v * b.get(k, 0.0) for k, v in a.items())


class Ball:
    """A local neighbourhood within one lexical class: a small RI space + a routing centroid."""

    def __init__(self, ballid, wordclass, dim_ball, denseness_ball):
        self.id = ballid
        self.wordclass = wordclass
        self.space = SemanticSpace(dim_ball, denseness_ball, name="ball-{}".format(ballid))
        self.members = set()
        self.centroid = {}  # lives in basket (dim_index) space, used for routing

    def add_member(self, item):
        self.members.add(item)

    def observe_local(self, item, neighbour):
        """Accumulate item's local (low-D) context vector from a co-occurring neighbour."""
        self.space.additemintoitem(item, neighbour)

    def neighbours(self, item, number=0):
        if item not in self.space.contextspace:
            return []
        return self.space.contextneighbours(item, number, weights=True)

    def local_similarity(self, a, b):
        if a in self.space.contextspace and b in self.space.contextspace:
            return sparsevectors.sparsecosine(self.space.contextspace[a],
                                              self.space.contextspace[b])
        return None


class Basket:
    """The whole model: five lexical-class partitions, each a growing collection of balls."""

    def __init__(self, dim_index=1000, denseness_index=10, dim_ball=20, denseness_ball=6,
                 window=2, f_min=3, theta_spawn=0.35, n_min=5,
                 ball_max=40, split_max_depth=6,
                 cluster_order="freq_desc", routing_weighting=None):
        self.dim_index = dim_index
        self.denseness_index = denseness_index
        self.dim_ball = dim_ball
        self.denseness_ball = denseness_ball
        self.window = window
        self.f_min = f_min
        self.theta_spawn = theta_spawn        # cosine to a ball centroid required to join it
        self.n_min = n_min                    # smallest cluster that may become a ball
        self.ball_max = ball_max              # balls larger than this are divisively bisected
        self.split_max_depth = split_max_depth
        self.cluster_order = cluster_order    # "freq_desc" | "freq_asc"
        self.routing_weighting = routing_weighting  # None or "idf" (superseded; kept as a toggle)

        self.index = {}        # item -> global sparse index vector (basket vocabulary)
        self.routing = {}      # item -> accumulating routing vector (dim_index)
        self.frequency = {}    # item -> int
        self.membership = {}   # item -> set(ball id)
        self.balls = []        # list[Ball]; ball.id == list position
        self.total_tokens = 0
        self.posvote = defaultdict(Counter)  # item -> Counter(coarse class -> count)
        self.wordclass = {}    # item -> dominant coarse class

    # ------------------------------------------------------------------ basket layer
    def _ensure_item(self, item):
        if item not in self.index:
            self.index[item] = sparsevectors.newrandomvector(self.dim_index, self.denseness_index)
            self.routing[item] = sparsevectors.newemptyvector(self.dim_index)
            self.frequency[item] = 0
            self.membership[item] = set()
        return self.index[item]

    def _count(self, text):
        for words in tokenised_sentences(text):
            for word, tag in pos_tag(words):
                self._ensure_item(word)
                self.frequency[word] += 1
                self.total_tokens += 1
                self.posvote[word][coarse_class(word, tag)] += 1
        for word, votes in self.posvote.items():
            self.wordclass[word] = votes.most_common(1)[0][0]

    def _idf(self, item):
        f = self.frequency.get(item, 0)
        if f <= 0:
            return 0.0
        return math.log((1 + self.total_tokens) / (1 + f))

    def _weight(self, neighbour):
        return self._idf(neighbour) if self.routing_weighting == "idf" else 1.0

    def _accumulate_routing(self, text):
        w = self.window
        for words in tokenised_sentences(text):
            n = len(words)
            for i, word in enumerate(words):
                for j in range(max(0, i - w), min(n, i + w + 1)):
                    if j == i:
                        continue
                    nb = words[j]
                    self.routing[word] = sparsevectors.sparseadd(
                        self.routing[word],
                        sparsevectors.normalise(self.index[nb]),
                        self._weight(nb))

    # ------------------------------------------------------------------ ball formation
    def _mature(self):
        return [w for w, f in self.frequency.items() if f >= self.f_min]

    def _new_ball(self, wordclass):
        ball = Ball(len(self.balls), wordclass, self.dim_ball, self.denseness_ball)
        self.balls.append(ball)
        return ball

    def _admit(self, ball, item):
        ball.add_member(item)
        self.membership[item].add(ball.id)

    def _recompute_centroid(self, ball):
        ball.centroid = sparsevectors.centroid([self.routing[m] for m in ball.members])

    def _cluster(self, items, theta=None):
        """Greedy leader/centroid clustering: each item joins the closest ball centroid above
        theta, else seeds a new one. Returns a list of member lists."""
        if theta is None:
            theta = self.theta_spawn
        if self.cluster_order == "freq_asc":
            items = sorted(items, key=lambda w: self.frequency[w])
        else:
            items = sorted(items, key=lambda w: -self.frequency[w])
        centroids, sums, groups = [], [], []
        for w in items:
            nw = sparsevectors.normalise(self.routing[w])
            best, bi = theta, -1
            for i, c in enumerate(centroids):
                s = sparsevectors.sparsecosine(nw, c, rounding=False)
                if s > best:
                    best, bi = s, i
            if bi >= 0:
                groups[bi].append(w)
                sums[bi] = sparsevectors.sparseadd(sums[bi], nw)
                centroids[bi] = sparsevectors.normalise(sums[bi])
            else:
                centroids.append(nw)
                sums.append(dict(nw))
                groups.append([w])
        return groups

    def _principal_axis(self, residuals, iters=5):
        """Top eigenvector of the residual covariance by power iteration -- the axis of greatest
        spread within the ball. residuals is {item: (vector - ball centroid)}."""
        rows = list(residuals.values())
        axis = dict(rows[0])
        nrm = math.sqrt(_sparse_dot(axis, axis)) or 1.0
        axis = {k: v / nrm for k, v in axis.items()}
        for _ in range(iters):
            acc = {}
            for r in rows:
                coef = _sparse_dot(r, axis)
                if coef == 0.0:
                    continue
                for k, x in r.items():
                    acc[k] = acc.get(k, 0.0) + coef * x
            nrm = math.sqrt(_sparse_dot(acc, acc))
            if nrm == 0.0:
                break
            axis = {k: v / nrm for k, v in acc.items()}
        return axis

    def _bisect(self, members):
        """Median cut along the ball's principal axis of variation. Mean-centres the (normalised)
        routing vectors, finds the top eigenvector by power iteration, projects every member onto
        it, and splits at the median. The principal axis is the real direction of spread -- unlike
        a farthest-pair axis, which keys on two outliers -- so the halves are balanced and the cut
        follows the ball's own geometry. Returns (groupA, groupB).

        NB: this cut is principled; the *trigger* for calling it is not. See NOTES.md -- the routing
        space does not linearly encode the semantic distinctions we would like to split on, so
        splitting is currently size-driven (ball_max), not coherence-driven. Open question, pinned."""
        nv = {m: sparsevectors.normalise(self.routing[m]) for m in members}
        centre = sparsevectors.normalise(sparsevectors.centroid(list(nv.values())))
        res = {m: sparsevectors.sparseadd(nv[m], centre, -1.0) for m in members}
        axis = self._principal_axis(res)
        ordered = sorted(members, key=lambda m: _sparse_dot(res[m], axis))
        mid = len(ordered) // 2
        return ordered[:mid], ordered[mid:]

    def _split(self, members, depth):
        """Recursively bisect a ball while it is larger than ball_max. A split that would strand a
        part below n_min is refused and the ball is kept whole. The trigger is size only -- a
        coherence gate is pinned open (NOTES.md): no linear metric on routing vectors distinguishes
        a big coherent ball from a big heterogeneous one."""
        if len(members) <= self.ball_max or depth >= self.split_max_depth:
            return [members]
        groupA, groupB = self._bisect(members)
        if min(len(groupA), len(groupB)) < self.n_min:
            return [members]                  # no clean cut -> keep as-is
        return self._split(groupA, depth + 1) + self._split(groupB, depth + 1)

    def _form_balls(self):
        mature = self._mature()
        for cls in COARSE_ORDER:
            items = [w for w in mature if self.wordclass.get(w) == cls]
            for members in self._cluster(items):
                if len(members) < self.n_min:
                    continue
                for final in self._split(members, 0):
                    if len(final) >= self.n_min:
                        ball = self._new_ball(cls)
                        for item in final:
                            self._admit(ball, item)
                        self._recompute_centroid(ball)

    # ------------------------------------------------------------------ local layer
    def _accumulate_local(self, text):
        w = self.window
        for words in tokenised_sentences(text):
            n = len(words)
            for i, word in enumerate(words):
                balls_of = self.membership.get(word)
                if not balls_of:
                    continue
                for j in range(max(0, i - w), min(n, i + w + 1)):
                    if j == i:
                        continue
                    for bid in balls_of:
                        self.balls[bid].observe_local(word, words[j])

    # ------------------------------------------------------------------ public API
    def fit(self, text):
        """Count + tag, settle class + membership, then fill local ball vectors."""
        self._count(text)
        self._accumulate_routing(text)
        self._form_balls()
        self._accumulate_local(text)
        return self

    def similarity(self, a, b):
        """Within-shared-ball cosine (max over shared balls), or None if they share no ball."""
        shared = self.membership.get(a, set()) & self.membership.get(b, set())
        best = None
        for bid in shared:
            s = self.balls[bid].local_similarity(a, b)
            if s is not None:
                best = s if best is None else max(best, s)
        return best

    def neighbours(self, item, number=10):
        """Nearest items within the item's ball(s) only, merged across balls."""
        scores = {}
        for bid in self.membership.get(item, set()):
            for nb, sc in self.balls[bid].neighbours(item, 0):
                if nb == item:
                    continue
                scores[nb] = max(scores.get(nb, float("-inf")), sc)
        return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:number]

    def stats(self):
        mature = self._mature()
        by_class = defaultdict(int)
        for b in self.balls:
            by_class[b.wordclass] += 1
        return {
            "vocab": len(self.index),
            "mature": len(mature),
            "n_balls": len(self.balls),
            "balls_by_class": dict(by_class),
            "ball_sizes": sorted((len(b.members) for b in self.balls), reverse=True),
            "buffered": sum(1 for w in mature if not self.membership[w]),
            "mature_by_class": dict(Counter(self.wordclass.get(w, OTHER) for w in mature)),
        }

    def save(self, path):
        with open(path, "wb") as fh:
            pickle.dump(self, fh)

    @staticmethod
    def load(path):
        with open(path, "rb") as fh:
            return pickle.load(fh)
