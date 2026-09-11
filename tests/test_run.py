"""The analysis functions, exercised on synthetic cases.

`bench.run` is the expensive step — it needs the whole fetched corpus — so its
pieces are tested here on cases built by hand. The point is that when the real
corpus lands, nothing in the run fails for a reason that had nothing to do with
the data.
"""
import pytest

from bench import grounds, run
from bench.dataset import Case


def make_case(i: int, upheld: bool, business: str, product: str = "travel",
              reasoning: str = "", outcome: str = "", text: str = "",
              investigator: bool = False) -> Case:
    return Case(
        drn=f"DRN-{1000000 + i}",
        date="01 Jan 2025",
        business=business,
        upheld=upheld,
        product=product,
        complaint_type="claim",
        url=f"https://example.invalid/DRN-{1000000 + i}.pdf",
        text=text or f"The complaint\nCase {i} about a {product} insurance claim.",
        text_with_investigator=(text or f"Case {i}") + " Our investigator agreed.",
        has_investigator_view=investigator,
        reasoning=reasoning,
        outcome_text=outcome,
        grounds=grounds.tag(reasoning) if upheld else [],
    )


@pytest.fixture
def cases():
    out = []
    # Five firms against a two-cycle label: coprime, so no firm predicts
    # the outcome and a grouped split still sees both classes.
    firms = ["Aviva", "Chubb", "AXA", "Admiral", "Assurant"]
    for i in range(120):
        upheld = i % 2 == 0
        reasoning = ("There was an unreasonable delay and it failed to obtain "
                     "the medical evidence." if upheld and i % 4 == 0 else
                     "The exclusion does not apply here." if upheld else "")
        outcome = ("Example must pay the claim and £300 compensation for the "
                   "distress and inconvenience." if upheld and i % 4 == 0 else
                   "Example must pay £150 compensation for the distress and "
                   "inconvenience." if upheld else "")
        out.append(make_case(i, upheld, firms[i % len(firms)],
                             product=["travel", "pet", "motor"][i % 3],
                             reasoning=reasoning, outcome=outcome,
                             text=("delay evidence not obtained" if upheld
                                   else "exclusion clearly applied"),
                             investigator=i % 3 != 0))
    return out


def test_composition_counts_non_claim_decisions(cases):
    rows = [{"drn": c.drn, "date": c.date, "complaint_type": "claim"} for c in cases]
    rows += [{"drn": "DRN-9", "date": "01 Jan 2025", "complaint_type": "pricing"}] * 30
    comp = run.composition(rows, cases)
    assert comp["decisions_harvested"] == 150
    assert comp["not_about_a_claim"] == 30
    assert comp["not_about_a_claim_share"] == pytest.approx(0.2)
    assert comp["benchmark_cases"] == 120


def test_leakage_reports_zero_on_clean_inputs(cases):
    lk = run.leakage(cases)
    assert lk["inputs_containing_verdict_wording"] == 0
    assert lk["cases_checked"] == 120


def test_grounds_analysis_splits_disturbed_from_stood(cases):
    g = run.grounds_analysis(cases)
    assert g["upheld_cases"] == 60
    # Half the upheld cases direct payment of the claim; half award only
    # distress. The split must show that rather than calling all 60 the same.
    assert g["claim_decision_disturbed"] == 30
    assert g["claim_decision_stood"] == 30
    assert g["claim_decision_stood_share"] == pytest.approx(0.5)


def test_cost_extracts_awards_and_keeps_the_case_fee(cases):
    c = run.cost(cases)
    assert c["cases_with_award"] == 60
    assert c["median_award_gbp"] == pytest.approx(225.0)   # 30x £300, 30x £150
    assert c["fos_case_fee_gbp_2026_27"] == 680


def test_cost_splits_the_award_by_whether_the_claim_decision_stood(cases):
    """The all-upheld median belongs to neither subset, which is the trap.

    Here it is £225 while no case in the corpus was awarded £225: the cases
    where the declinature stood got £150 and the ones where it was disturbed
    got £300. Quoting the headline against either subset is wrong in a
    direction that is not obvious from the headline alone.
    """
    c = run.cost(cases)
    assert c["claim_stood_upheld"] == 30
    assert c["claim_stood_cases_with_award"] == 30
    assert c["claim_stood_median_award_gbp"] == pytest.approx(150.0)

    assert c["claim_disturbed_upheld"] == 30
    assert c["claim_disturbed_cases_with_award"] == 30
    assert c["claim_disturbed_median_award_gbp"] == pytest.approx(300.0)

    assert c["median_award_gbp"] not in (
        c["claim_stood_median_award_gbp"], c["claim_disturbed_median_award_gbp"])


def test_prediction_runs_both_splits_and_beats_majority(cases):
    p = run.prediction(cases)
    for key in ("grouped_by_respondent", "random_split"):
        scores = p[key]["scores"]
        assert set(scores) == {"majority", "prior", "tfidf_lr",
                               "precedent_knn", "precedent_lsa"}
        # Majority sits near chance on balanced accuracy however good its plain
        # accuracy looks — that is the whole reason it is in the table. It is
        # not pinned to exactly 0.5: pooled across folds whose training
        # majority differs, a constant-per-fold predictor drifts either side.
        assert 0.4 <= scores["majority"]["balanced_accuracy"] <= 0.6
        # The synthetic signal is separable, so a real model must clear it.
        assert scores["tfidf_lr"]["balanced_accuracy"] > 0.9
        assert (scores["tfidf_lr"]["balanced_accuracy"]
                > scores["majority"]["balanced_accuracy"] + 0.2)
    assert "top_upheld_terms" in p


def test_triage_curve_is_monotonic_in_coverage(cases):
    p = run.prediction(cases)
    t = run.triage(p)
    cov = [r["coverage"] for r in t["precedent_knn"]["curve"]]
    assert cov == sorted(cov)
    assert cov[-1] == pytest.approx(1.0)


def test_investigator_ablation_is_paired_on_the_same_cases(cases):
    ab = run.investigator_ablation(cases)
    if "note" in ab:
        pytest.skip(ab["note"])
    assert ab["cases"] == sum(1 for c in cases if c.has_investigator_view)
    assert set(ab["without_investigator_view"]["scores"]) == \
           set(ab["with_investigator_view"]["scores"])


def test_product_mix_is_not_presented_as_a_rate(cases):
    mix = run.product_mix(cases)
    assert mix
    for v in mix.values():
        assert "rate" not in v
        assert v["note"] == "sampled, not a rate"


def test_date_range_is_parsed_not_sorted_as_text():
    """min()/max() over "9 May 2025" strings sorts alphabetically. The real
    corpus reported a 2024-2026 harvest as running Dec 2024 to May 2025."""
    rows = [{"complaint_type": "claim", "date": d} for d in
            ["1 Dec 2024", "9 May 2025", "15 Jan 2024", "3 Aug 2026"]]
    comp = run.composition(rows, [], None)
    assert comp["date_min"] == "2024-01-15"
    assert comp["date_max"] == "2026-08-03"
    assert comp["dates_parsed"] == 4


def test_unparseable_dates_are_skipped_not_crashed_on():
    rows = [{"complaint_type": "claim", "date": d} for d in
            ["1 Dec 2024", None, "not a date"]]
    comp = run.composition(rows, [], None)
    assert comp["dates_parsed"] == 1
    assert comp["date_min"] == "2024-12-01"
