"""Metric behaviour, with emphasis on the failure this repo is built around:
a constant predictor must not be able to look good."""
import math

import pytest

from bench import metrics


def test_constant_predictor_scores_the_base_rate_but_not_balanced_accuracy():
    y = [True] * 30 + [False] * 70
    always_no = [False] * 100

    s = metrics.score(y, always_no)
    assert s.accuracy == pytest.approx(0.70)      # looks fine
    assert s.base_rate == pytest.approx(0.30)
    assert s.balanced_accuracy == pytest.approx(0.50)   # and is worthless
    assert s.recall_upheld == 0.0                 # misses every overturn


def test_score_counts_the_confusion_matrix():
    y = [True, True, False, False]
    p = [True, False, True, False]
    s = metrics.score(y, p)
    assert (s.tp, s.fn, s.fp, s.tn) == (1, 1, 1, 1)
    assert s.recall_upheld == 0.5
    assert s.precision_upheld == 0.5


def test_score_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        metrics.score([True], [True, False])


def test_wilson_stays_inside_zero_one_at_the_extremes():
    lo, hi = metrics.wilson(0, 20)
    assert lo == 0.0 and 0 < hi < 1
    lo, hi = metrics.wilson(20, 20)
    assert hi == 1.0 and 0 < lo < 1


def test_wilson_matches_a_known_interval():
    lo, hi = metrics.wilson(62, 82)
    assert lo == pytest.approx(0.653, abs=0.002)
    assert hi == pytest.approx(0.836, abs=0.002)


def test_mcnemar_uses_only_discordant_pairs():
    y = [True] * 10
    a = [True] * 10
    b = [True] * 10
    assert metrics.mcnemar_exact(y, a, b)["p"] == 1.0

    # a right, b wrong, on 6 of 6 discordant cases
    y = [True] * 6
    a = [True] * 6
    b = [False] * 6
    r = metrics.mcnemar_exact(y, a, b)
    assert (r["n01"], r["n10"]) == (6, 0)
    assert r["p"] == pytest.approx(2 * (1 / 64))


def test_abstention_curve_covers_more_as_it_goes():
    y = [True, False] * 25
    prob = [0.9 if t else 0.1 for t in y]     # a perfect, confident model
    rows = metrics.abstention_curve(y, prob, steps=5)
    assert [r["coverage"] for r in rows] == sorted(r["coverage"] for r in rows)
    assert rows[-1]["coverage"] == 1.0
    assert rows[-1]["escalated"] == 0
    assert all(r["accuracy"] == 1.0 for r in rows)


def test_abstention_escalates_the_least_confident_first():
    # One case sits at 0.5 and is wrong; it must be the last one automated.
    y = [True, True, False, False]
    prob = [0.99, 0.98, 0.02, 0.5]
    rows = metrics.abstention_curve(y, prob, steps=4)
    assert rows[0]["accuracy"] == 1.0
    assert rows[-1]["n_automated"] == 4
    assert rows[-1]["accuracy"] == 0.75


def test_calibration_reports_empty_bins_rather_than_dropping_them():
    y = [True, False]
    prob = [0.95, 0.05]
    rows = metrics.calibration(y, prob, bins=10)
    assert len(rows) == 10
    assert sum(r["n"] for r in rows) == 2
    assert any(r["n"] == 0 for r in rows)


def test_score_refuses_an_empty_corpus():
    with pytest.raises(ValueError):
        metrics.score([], [])
