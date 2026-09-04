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
    upheld = [
        make_case(i, True, f"Firm{i % 3}", product="travel",
                  reasoning=("There was an unreasonable delay and it failed to "
                             "obtain the medical evidence it needed."),
                  outcome="Example must pay the claim.",
                  text=("travel insurance claim declined medical evidence "
                        "requested but never chased, four months open"))
        for i in range(12)
    ]
    not_upheld = [
        make_case(100 + i, False, f"Firm{i % 3}", product="motor",
                  text="motor insurance claim declined exclusion clearly applied "
                       "policyholder drove without a licence")
        for i in range(12)
    ]
    return upheld + not_upheld


def test_a_case_like_the_upheld_ones_scores_high_and_cites_them(corpus):
    rep = chk.check("travel insurance claim declined medical evidence never chased "
                    "for four months", cases=corpus)
    assert rep.overturn_risk > 0.8
    assert rep.upheld_among_precedents >= 8
    assert not rep.weak_match
    # every citation is a real reference with a resolvable link
    for p in rep.precedents:
        assert p.drn.startswith("DRN-")
        assert p.url.endswith(".pdf")


def test_a_case_like_the_defended_ones_scores_low(corpus):
    rep = chk.check("motor insurance claim declined exclusion clearly applied "
                    "drove without a licence", cases=corpus)
    assert rep.overturn_risk < 0.2


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
