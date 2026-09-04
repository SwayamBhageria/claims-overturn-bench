"""Predictors that need no API key, so the headline result reproduces offline.

Four, in increasing order of how much they know:

`majority`      always answers the training-set majority. It exists to print
                the number every other row has to beat, and to make the point
                that accuracy alone cannot tell the difference between it and
                a working system.

`prior`         predicts the base rate as a probability for every case. The
                calibration reference: perfectly calibrated, zero skill.

`tfidf_lr`      TF-IDF over the case text into logistic regression. What a
                competent team builds in an afternoon before buying anything.

`precedent_knn` nearest published decisions by TF-IDF cosine, predicting the
                weighted outcome of the k most similar. This is the one that
                matters commercially, because it is the only baseline that can
                show its work: every prediction comes with the decisions that
                produced it, which is what a file note or a regulator needs.
                `checker/` is this model with a usable interface on it.

All four are fit inside a fold and scored outside it. The split is grouped by
respondent business so that no insurer appears on both sides: without that, a
model can learn "Aviva travel claims get upheld" from the training half and
score well on the test half without having learned anything about claims. The
ungrouped number is also reported, because the gap between them is itself a
finding about how much of this task is memorising defendants.
"""
from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity


class Baseline:
    name = "base"

    def fit(self, texts: list[str], y: list[bool]) -> "Baseline":
        raise NotImplementedError

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        raise NotImplementedError

    def predict(self, texts: list[str]) -> list[bool]:
        # Python bools, not np.bool_: these end up in JSON and in identity
        # comparisons, and np.bool_ silently fails `is True`.
        return [bool(p >= 0.5) for p in self.predict_proba(texts)]


class Majority(Baseline):
    name = "majority"

    def fit(self, texts, y):
        self.label = sum(y) > len(y) / 2
        return self

    def predict_proba(self, texts):
        # 0.51/0.49 rather than 1/0: a constant predictor has no confidence
        # ordering, and giving it one would let it fake an abstention curve.
        return np.full(len(texts), 0.51 if self.label else 0.49)


class Prior(Baseline):
    name = "prior"

    def fit(self, texts, y):
        self.rate = sum(y) / len(y)
        return self

    def predict_proba(self, texts):
        return np.full(len(texts), self.rate)


class TfidfLR(Baseline):
    name = "tfidf_lr"

    def __init__(self, max_features: int = 40_000, C: float = 1.0):
        self.vec = TfidfVectorizer(
            max_features=max_features, ngram_range=(1, 2),
            min_df=2, sublinear_tf=True, strip_accents="unicode",
        )
        self.clf = LogisticRegression(max_iter=2000, C=C, class_weight="balanced")

    def fit(self, texts, y):
        X = self.vec.fit_transform(texts)
        self.clf.fit(X, np.asarray(y, dtype=int))
        return self

    def predict_proba(self, texts):
        return self.clf.predict_proba(self.vec.transform(texts))[:, 1]

    def top_terms(self, n: int = 20) -> list[tuple[str, float]]:
        """Most upheld-associated terms. Read these before trusting the model:
        if they are insurer names or products rather than conduct, it is
        learning the defendant."""
        names = self.vec.get_feature_names_out()
        coefs = self.clf.coef_[0]
        idx = np.argsort(coefs)[::-1][:n]
        return [(names[i], float(coefs[i])) for i in idx]


class PrecedentKNN(Baseline):
    name = "precedent_knn"

    def __init__(self, k: int = 15, max_features: int = 40_000):
        self.k = k
        self.vec = TfidfVectorizer(
            max_features=max_features, ngram_range=(1, 2),
            min_df=2, sublinear_tf=True, strip_accents="unicode",
        )

    def fit(self, texts, y):
        self.X = self.vec.fit_transform(texts)
        self.y = np.asarray(y, dtype=float)
        self.texts = texts
        return self

    def _neighbours(self, texts: list[str]):
        sims = cosine_similarity(self.vec.transform(texts), self.X)
        idx = np.argsort(sims, axis=1)[:, ::-1][:, :self.k]
        return sims, idx

    def predict_proba(self, texts):
        sims, idx = self._neighbours(texts)
        out = np.empty(len(texts))
        for r in range(len(texts)):
            w = sims[r, idx[r]]
            # A case with no similar precedent should land on the base rate,
            # not on whatever noise the nearest zero-similarity rows carry.
            out[r] = float(self.y[idx[r]] @ w / w.sum()) if w.sum() > 0 else self.y.mean()
        return out

    def neighbours(self, text: str) -> list[tuple[int, float, bool]]:
        """(training index, similarity, outcome) for one case — the citations."""
        sims, idx = self._neighbours([text])
        return [(int(i), float(sims[0, i]), bool(self.y[i])) for i in idx[0]]


class PrecedentLSA(PrecedentKNN):
    """`PrecedentKNN` over a reduced space instead of raw terms.

    Lexical similarity has an obvious failure here: two decisions about the
    same conduct can share almost no vocabulary, because one says "we never
    received the medical report" and the other says "the treating clinician's
    letter was not obtained". Truncated SVD over the same TF-IDF matrix folds
    those onto nearby axes.

    It is a separate model rather than a replacement because the reduction can
    also destroy the distinctions that matter — "did not obtain the evidence"
    and "obtained the evidence" are neighbours in almost any embedding — so
    which one is better is a question for the corpus, not for taste. Both are
    in the results table.
    """
    name = "precedent_lsa"

    def __init__(self, k: int = 15, max_features: int = 40_000,
                 components: int = 200):
        super().__init__(k=k, max_features=max_features)
        self.components = components

    def fit(self, texts, y):
        from sklearn.decomposition import TruncatedSVD
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import Normalizer

        raw = self.vec.fit_transform(texts)
        # Components cannot exceed the rank of the matrix; a small fold would
        # otherwise raise from inside sklearn rather than here.
        n = min(self.components, min(raw.shape) - 1)
        self.svd = make_pipeline(TruncatedSVD(n_components=max(n, 2),
                                              random_state=0),
                                 Normalizer(copy=False))
        self.X = self.svd.fit_transform(raw)
        self.y = np.asarray(y, dtype=float)
        self.texts = texts
        return self

    def _neighbours(self, texts: list[str]):
        q = self.svd.transform(self.vec.transform(texts))
        sims = cosine_similarity(q, self.X)
        # SVD similarities can be negative; a negative weight would subtract a
        # precedent's outcome from the risk, which is meaningless.
        sims = np.clip(sims, 0.0, None)
        idx = np.argsort(sims, axis=1)[:, ::-1][:, :self.k]
        return sims, idx


ALL = [Majority, Prior, TfidfLR, PrecedentKNN, PrecedentLSA]


def grouped_folds(groups: list[str], n_splits: int = 5, seed: int = 0
                  ) -> list[tuple[list[int], list[int]]]:
    """Folds in which a group never straddles the split.

    Groups are assigned to folds largest-first into whichever fold is
    currently smallest, which keeps folds comparable in size even when one
    insurer dominates the corpus.
    """
    from collections import defaultdict
    members = defaultdict(list)
    for i, g in enumerate(groups):
        members[g or f"__none_{i}"].append(i)

    rng = np.random.default_rng(seed)
    order = sorted(members, key=lambda g: (-len(members[g]), g))
    buckets: list[list[int]] = [[] for _ in range(n_splits)]
    for g in order:
        target = min(range(n_splits), key=lambda b: (len(buckets[b]), rng.random()))
        buckets[target].extend(members[g])

    # A bucket can come out empty when there are fewer distinct groups than
    # splits — which happens on any subset of the corpus, not just a toy one.
    # An empty test fold means asking a fitted model to transform a zero-row
    # matrix, which raises deep inside scikit-learn with a message about
    # `ensure_min_samples` and nothing about folds. Drop them here instead.
    buckets = [b for b in buckets if b]
    if len(buckets) < 2:
        raise ValueError(
            f"cannot build folds: {len(members)} distinct group(s) across "
            f"{len(groups)} cases gives {len(buckets)} non-empty fold(s). "
            f"Grouped cross-validation needs at least two.")

    folds = []
    all_idx = set(range(len(groups)))
    for b in buckets:
        test = sorted(b)
        folds.append((sorted(all_idx - set(test)), test))
    return folds


def random_folds(n: int, n_splits: int = 5, seed: int = 0
                 ) -> list[tuple[list[int], list[int]]]:
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    chunks = np.array_split(idx, n_splits)
    return [(sorted(set(range(n)) - set(c.tolist())), sorted(c.tolist()))
            for c in chunks]
