"""Baseline and fold behaviour.

The fold test is the one that matters. Splitting at random lets a model learn
"claims against this insurer get upheld" on one side and cash it in on the
other, which scores well and means nothing. Grouping by respondent is what
stops that, so the grouping has to be actually watertight.
"""
import numpy as np
import pytest

from bench import baselines


def test_grouped_folds_never_split_a_respondent():
    groups = ["Aviva"] * 40 + ["Chubb"] * 30 + ["AXA"] * 20 + ["Admiral"] * 10
    for train, test in baselines.grouped_folds(groups, n_splits=5):
        train_g = {groups[i] for i in train}
        test_g = {groups[i] for i in test}
        assert not (train_g & test_g), f"{train_g & test_g} on both sides"


def test_grouped_folds_cover_every_case_exactly_once():
    groups = [f"firm{i % 17}" for i in range(200)]
    folds = baselines.grouped_folds(groups, n_splits=5)
    seen = [i for _train, test in folds for i in test]
    assert sorted(seen) == list(range(200))


def test_random_folds_cover_every_case_exactly_once():
    folds = baselines.random_folds(97, n_splits=5)
    seen = [i for _train, test in folds for i in test]
    assert sorted(seen) == list(range(97))


def test_majority_predicts_one_class_with_no_confidence_ordering():
    y = [True] * 3 + [False] * 7
    m = baselines.Majority().fit(["x"] * 10, y)
    p = m.predict_proba(["a", "b", "c"])
    assert set(m.predict(["a", "b"])) == {False}
    assert len(set(p)) == 1          # a constant model cannot fake a curve


def test_prior_returns_the_base_rate():
    y = [True] * 30 + [False] * 70
    m = baselines.Prior().fit(["x"] * 100, y)
    assert m.predict_proba(["a"])[0] == pytest.approx(0.30)


def test_tfidf_lr_learns_a_separable_signal():
    texts = ["delay evidence not obtained"] * 20 + ["exclusion clearly applied"] * 20
    y = [True] * 20 + [False] * 20
    m = baselines.TfidfLR().fit(texts, y)
    assert m.predict(["delay evidence not obtained"])[0] is True
    assert m.predict(["exclusion clearly applied"])[0] is False


def test_precedent_knn_cites_its_neighbours():
    texts = ["travel medical evidence delay"] * 5 + ["motor exclusion applied"] * 5
    y = [True] * 5 + [False] * 5
    m = baselines.PrecedentKNN(k=3).fit(texts, y)
    neigh = m.neighbours("travel medical evidence delay")
    assert len(neigh) == 3
    assert all(idx < 5 for idx, _sim, _up in neigh)
    assert all(up for _idx, _sim, up in neigh)


def test_precedent_knn_falls_back_to_the_base_rate_with_no_similar_case():
    texts = ["travel medical evidence"] * 4 + ["motor exclusion"] * 6
    y = [True] * 4 + [False] * 6
    m = baselines.PrecedentKNN(k=3).fit(texts, y)
    # Nothing in the corpus shares a term with this.
    p = m.predict_proba(["zzz qqq"])[0]
    assert p == pytest.approx(np.mean(y))


def test_grouped_folds_never_emit_an_empty_fold():
    # Four groups, five requested splits. The fifth bucket would be empty, and
    # an empty test fold raises deep inside scikit-learn with a message that
    # says nothing about folds.
    groups = ["a"] * 10 + ["b"] * 10 + ["c"] * 10 + ["d"] * 10
    folds = baselines.grouped_folds(groups, n_splits=5)
    assert len(folds) == 4
    assert all(test for _train, test in folds)
    assert all(train for train, _test in folds)


def test_grouped_folds_refuse_a_single_group():
    with pytest.raises(ValueError, match="at least two"):
        baselines.grouped_folds(["only"] * 20, n_splits=5)


def test_text_cache_reextracts_when_the_pdf_changes(tmp_path, monkeypatch):
    """A re-fetched decision must not be served from a stale text cache."""
    from bench import dataset

    monkeypatch.setattr(dataset, "TEXT_CACHE", tmp_path / "text")
    pdf = tmp_path / "DRN-1.pdf"
    pdf.write_bytes(b"%PDF-fake")

    calls = []

    def fake_extract(path):
        calls.append(str(path))
        return f"extraction {len(calls)}"

    monkeypatch.setattr(dataset.sections, "extract_text", fake_extract)

    assert dataset._text_for(pdf, "DRN-1") == "extraction 1"
    assert dataset._text_for(pdf, "DRN-1") == "extraction 1"   # cached
    assert len(calls) == 1

    import os
    os.utime(pdf, (0, 0))                                       # "re-fetched"
    assert dataset._text_for(pdf, "DRN-1") == "extraction 2"
    assert len(calls) == 2
    # and the stale entry is gone rather than accumulating
    assert len(list((tmp_path / "text").glob("DRN-1.*.txt"))) == 1


def test_lsa_retriever_never_returns_a_negative_similarity():
    """A negative weight would subtract a precedent's outcome from the risk."""
    texts = ["travel medical evidence delay"] * 10 + ["motor exclusion applied"] * 10
    y = [True] * 10 + [False] * 10
    m = baselines.PrecedentLSA(k=5).fit(texts, y)
    for _idx, sim, _up in m.neighbours("something entirely unrelated"):
        assert sim >= 0.0
    p = m.predict_proba(["travel medical evidence delay"])[0]
    assert 0.0 <= p <= 1.0


def test_lsa_survives_a_corpus_smaller_than_its_component_count():
    texts = ["travel evidence"] * 3 + ["motor exclusion"] * 3
    y = [True] * 3 + [False] * 3
    m = baselines.PrecedentLSA(k=2, components=200).fit(texts, y)
    assert 0.0 <= m.predict_proba(["travel evidence"])[0] <= 1.0


def test_weak_threshold_is_measured_from_the_corpus_not_assumed():
    """A fixed threshold was wrong in both directions on real data."""
    from checker.check import weak_threshold
    from tests.test_run import make_case

    # Genuinely varied vocabulary. Texts differing only by a number collapse
    # to identical vectors, because the vectoriser's min_df drops terms that
    # appear once — every similarity is then 1.0 and the threshold saturates.
    subjects = ["travel", "pet", "motor", "home", "gadget", "warranty", "boat", "bike"]
    faults = ["medical evidence not obtained", "unreasonable delay in settling",
              "exclusion applied to circumstances it did not cover",
              "settlement offer well below market value",
              "policyholder never told why cover was refused"]
    texts = [f"{s} insurance claim declined, {f}" for s in subjects for f in faults]
    cases = [make_case(i, i % 2 == 0, f"F{i % 4}", text=t)
             for i, t in enumerate(texts)]
    m = baselines.PrecedentKNN(k=10).fit([c.text for c in cases],
                                         [c.upheld for c in cases])
    thr = weak_threshold(m, cases, sample=40)
    assert 0.0 < thr < 1.0
