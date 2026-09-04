"""Scoring, written around the decision a claims team actually makes.

Accuracy is close to useless here and the corpus makes it obvious why: a
predictor that answers "not upheld" to every case scores whatever the base
rate happens to be, and looks respectable, while being exactly the system that
lets every wrong decline through. So the headline numbers here are:

- **base rate** — printed next to every accuracy, always, as the floor to beat
- **recall on the upheld class** — of the decisions the ombudsman overturned,
  how many did the model flag? This is the number that maps to money
- **balanced accuracy** — the mean of the two class recalls, which a
  constant predictor cannot inflate
- **an abstention curve** — because nobody is proposing to automate the final
  call. The realistic deployment is triage: the model handles the confident
  cases and escalates the rest. The curve reports, at each level of coverage,
  how accurate the automated portion is. A system that is 95% accurate on the
  60% it is sure about is a product; one that is 71% accurate on everything is
  not.

Confidence intervals are Wilson, and the comparison between two systems is a
paired McNemar test, because both are scored on the same cases. Reporting two
independent intervals and eyeballing the overlap answers a different and
easier question than the one being asked.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict

UPHELD = True
NOT_UPHELD = False


@dataclass
class Scores:
    n: int
    base_rate: float          # share of the corpus the ombudsman upheld
    accuracy: float
    balanced_accuracy: float
    recall_upheld: float      # sensitivity to the overturned decisions
    recall_not_upheld: float
    precision_upheld: float
    predicted_upheld_rate: float
    tp: int
    fp: int
    tn: int
    fn: int

    def as_dict(self) -> dict:
        return asdict(self)


def score(y_true: list[bool], y_pred: list[bool]) -> Scores:
    if len(y_true) != len(y_pred):
        raise ValueError(f"length mismatch: {len(y_true)} vs {len(y_pred)}")
    if not y_true:
        raise ValueError("nothing to score")

    tp = sum(1 for t, p in zip(y_true, y_pred) if t and p)
    fp = sum(1 for t, p in zip(y_true, y_pred) if not t and p)
    tn = sum(1 for t, p in zip(y_true, y_pred) if not t and not p)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t and not p)

    pos, neg = tp + fn, tn + fp
    rec_u = tp / pos if pos else float("nan")
    rec_n = tn / neg if neg else float("nan")
    return Scores(
        n=len(y_true),
        base_rate=pos / len(y_true),
        accuracy=(tp + tn) / len(y_true),
        balanced_accuracy=(rec_u + rec_n) / 2,
        recall_upheld=rec_u,
        recall_not_upheld=rec_n,
        precision_upheld=tp / (tp + fp) if (tp + fp) else float("nan"),
        predicted_upheld_rate=(tp + fp) / len(y_true),
        tp=tp, fp=fp, tn=tn, fn=fn,
    )


def wilson(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval. Behaves at 0 and at n, unlike the normal one."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = successes / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


def mcnemar_exact(y_true: list[bool], a: list[bool], b: list[bool]) -> dict:
    """Exact paired test for two systems scored on the same cases.

    Returns the discordant counts and a two-sided exact binomial p. Only the
    cases where the two systems disagree carry information; the count of cases
    they both get right is not evidence about which is better.
    """
    n01 = sum(1 for t, x, y in zip(y_true, a, b) if (x == t) and (y != t))
    n10 = sum(1 for t, x, y in zip(y_true, a, b) if (x != t) and (y == t))
    n = n01 + n10
    if n == 0:
        return {"n01": 0, "n10": 0, "p": 1.0}

    def comb(k: int) -> float:
        return math.comb(n, k)

    tail = sum(comb(k) for k in range(0, min(n01, n10) + 1)) / (2 ** n)
    return {"n01": n01, "n10": n10, "p": min(1.0, 2 * tail)}


def abstention_curve(y_true: list[bool], y_prob: list[float],
                     steps: int = 21) -> list[dict]:
    """Accuracy of the automated portion as a function of coverage.

    Cases are ranked by confidence — distance from 0.5 — and the least
    confident are handed to a human first. Each row answers: if we automate
    this fraction of the book, how often is the automated answer right, and
    how many overturned decisions are we still missing?
    """
    if len(y_true) != len(y_prob):
        raise ValueError("length mismatch")
    order = sorted(range(len(y_true)), key=lambda i: -abs(y_prob[i] - 0.5))

    rows = []
    total = len(y_true)
    for s in range(1, steps + 1):
        k = max(1, round(total * s / steps))
        idx = order[:k]
        yt = [y_true[i] for i in idx]
        yp = [y_prob[i] >= 0.5 for i in idx]
        sc = score(yt, yp)
        escalated = order[k:]
        rows.append({
            "coverage": k / total,
            "n_automated": k,
            "accuracy": sc.accuracy,
            "balanced_accuracy": sc.balanced_accuracy,
            "recall_upheld": sc.recall_upheld,
            # Overturned decisions the automated portion got wrong. These are
            # the ones that reach the ombudsman: the cost of this row.
            "upheld_missed_in_automated": sc.fn,
            "escalated": len(escalated),
            "upheld_in_escalated": sum(1 for i in escalated if y_true[i]),
        })
    return rows


def calibration(y_true: list[bool], y_prob: list[float], bins: int = 10) -> list[dict]:
    """Predicted probability against observed frequency, per bin."""
    out = []
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        idx = [i for i, p in enumerate(y_prob)
               if (p >= lo and p < hi) or (b == bins - 1 and p == 1.0)]
        if not idx:
            out.append({"bin_lo": lo, "bin_hi": hi, "n": 0,
                        "mean_predicted": None, "observed": None})
            continue
        out.append({
            "bin_lo": lo, "bin_hi": hi, "n": len(idx),
            "mean_predicted": sum(y_prob[i] for i in idx) / len(idx),
            "observed": sum(1 for i in idx if y_true[i]) / len(idx),
        })
    return out
