"""The checker, driven on an injected corpus so it needs no fetched data.

What matters here is not the risk number but the things a claims file needs
from it: that the citations are real and traceable, that a case with no
similar precedent says so instead of inventing a case-specific answer, and
that the checklist follows the grounds the neighbours were actually upheld on.
"""
import pytest

from checker import check as chk
from tests.test_run import make_case


@pytest.fixture
def corpus():
    """A small index with varied wording.

    The variation matters: the weak-match threshold is calibrated from how
    similar real cases are to their own nearest neighbour, so a corpus of
    identical strings would put that threshold at 1.0 and flag everything.
    """
    upheld_texts = [
        "travel insurance claim declined medical evidence requested but never "
        "chased, four months open, policyholder had to call six times",
        "travel claim turned down after hospital report was not obtained, "
        "insurer relied on absence of evidence it never asked for",
        "medical certificate delayed, claim refused without chasing the "
        "treating clinician, customer chased repeatedly over several weeks",
        "illness abroad, claim declined for lack of a medical report the "
        "insurer had not requested from the hospital",
    ]
    not_upheld_texts = [
        "motor insurance claim declined because the driver held no licence, "
        "exclusion clearly worded and drawn to attention at sale",
        "car claim refused, policyholder had not disclosed a previous "
        "conviction, exclusion applied on its plain terms",
        "vehicle claim turned down where the policy excluded commercial use "
        "and the van was being used for deliveries",
        "motor claim declined under a clear exclusion for driving otherwise "
        "than in accordance with the licence held",
    ]
    # Each repetition gets a distinguishing clause, so cases are similar to
    # one another without being identical — as real decisions are.
    detail = ["policy taken out online", "claim reported by telephone",
              "second opinion obtained later"]
    cases = []
    variants = [f"{t}, {d}" for d in detail for t in upheld_texts]
    for i, t in enumerate(variants):
        cases.append(make_case(i, True, f"Firm{i % 3}", product="travel",
                               reasoning=("There was an unreasonable delay and it "
                                          "failed to obtain the medical evidence "
                                          "it needed."),
                               outcome="Example must pay the claim.", text=t))
    variants = [f"{t}, {d}" for d in detail for t in not_upheld_texts]
    for i, t in enumerate(variants):
        cases.append(make_case(100 + i, False, f"Firm{i % 3}", product="motor",
                               text=t))
    return cases


def test_a_case_like_the_upheld_ones_scores_high_and_cites_them(corpus):
    rep = chk.check("travel insurance claim declined medical evidence never chased "
                    "for four months", cases=corpus)
    assert rep.overturn_risk > 0.8
    assert rep.upheld_among_precedents >= 7
    # every citation is a real reference with a resolvable link
    for p in rep.precedents:
        assert p.drn.startswith("DRN-")
        assert p.url.endswith(".pdf")


def test_a_case_like_the_defended_ones_scores_low(corpus):
    rep = chk.check("motor insurance claim declined exclusion clearly applied "
                    "drove without a licence", cases=corpus)
    assert rep.overturn_risk < 0.2


def test_a_match_as_close_as_a_real_case_gets_is_not_flagged_weak(corpus):
    # The threshold is relative: "weak" means less similar than nine out of
    # ten real cases manage against their own nearest precedent. A query that
    # is itself a case must therefore clear it.
    rep = chk.check(corpus[0].text, cases=corpus)
    assert not rep.weak_match
    assert rep.mean_similarity > rep.weak_threshold


def test_no_similar_precedent_is_declared_rather_than_answered(corpus):
    rep = chk.check("a dispute about crop yields on a farm in Manitoba",
                    cases=corpus)
    assert rep.weak_match
    assert "weak match" in rep.render()


def test_checklist_follows_the_grounds_the_neighbours_turned_on(corpus):
    rep = chk.check("travel insurance claim declined medical evidence never chased "
                    "for four months", cases=corpus)
    assert "evidence" in rep.grounds_seen
    assert "delay" in rep.grounds_seen
    joined = " ".join(rep.checks)
    assert "requested in writing" in joined       # the evidence check
    assert "how long" in joined.lower()           # the delay check
    assert "ICOBS 8.1" in joined                  # always present


def test_report_serialises_for_a_file_note(corpus):
    rep = chk.check("travel insurance claim declined medical evidence",
                    cases=corpus)
    d = rep.as_dict()
    assert isinstance(d["overturn_risk"], float)
    assert isinstance(d["precedents"], list)
    assert "grounds" in d["precedents"][0]
