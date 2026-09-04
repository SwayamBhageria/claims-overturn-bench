"""Classifier behaviour, fixed on real opening sentences.

Every string below is the opening of a published decision (references in the
comments), including the ones an earlier version of the rules got wrong. The
failures are kept as tests because they are the whole reason the rules look
the way they do.
"""
from corpus import classify

# --- complaint type ---------------------------------------------------------

CLAIMS = [
    # DRN-6297755
    "Mrs I has complained that Aviva Insurance Limited declined a claim she made "
    "on a travel insurance policy.",
    # DRN-6488150 — verb *before* "claim". An earlier forward-only rule missed it.
    "Mr S complains that National House-Building Council (NHBC) has unfairly "
    "handled a claim made on his Buildmark Home Warranty policy.",
    # DRN-6504989 — likewise.
    "Mr S and Mrs S have complained that Admiral Insurance (Gibraltar) Limited "
    "haven't settled their travel insurance claim in full.",
    # DRN-6503235
    "Miss R complains about the service she received from AXA Insurance UK Plc "
    "when she attempted to make a claim under her home insurance policy.",
    # DRN-6505965
    "Mr K has complained about the settlement offered by Assurant General "
    "Insurance Limited to replace his mobile phone after his claim.",
]

NOT_CLAIMS = [
    # DRN-6260215-style pricing complaint
    "Mr S has complained about the extent of the increase of his pet insurance "
    "premium at the renewal for December 2025 sent by his broker.",
    # policy cancellation, not a claim
    "Mr K complains Advantage Insurance Company Limited unfairly cancelled his "
    "car insurance policy and recorded a marker against him.",
]


def test_claim_complaints_are_recognised():
    for text in CLAIMS:
        assert classify.complaint_type(text) == classify.CLAIM, text[:60]


def test_non_claim_complaints_are_excluded():
    for text in NOT_CLAIMS:
        assert classify.complaint_type(text) != classify.CLAIM, text[:60]


def test_claim_signal_wins_over_a_pricing_mention():
    # Decisions routinely recount how the policy was bought before getting to
    # the claim. The claim is what the complaint is about.
    text = ("Mr A complains that his premium increased at renewal and that "
            "Aviva then declined the claim he made under the policy.")
    assert classify.complaint_type(text) == classify.CLAIM


def test_is_benchmarkable_matches_complaint_type():
    assert classify.is_benchmarkable(CLAIMS[0])
    assert not classify.is_benchmarkable(NOT_CLAIMS[0])


# --- product line -----------------------------------------------------------

PRODUCTS = [
    ("a claim she made on a travel insurance policy", "travel"),
    ("a claim under his pet insurance policy for vet's fees", "pet"),
    ("a claim under his mobile phone insurance policy", "gadget"),
    ("declined a claim under her home insurance policy", "property"),
    ("a claim under his motor insurance policy", "motor"),
    ("a claim made on his Buildmark Home Warranty policy", "warranty"),
]


def test_product_line_reads_the_opening_sentence():
    for text, expected in PRODUCTS:
        assert classify.product_line(text) == expected, text


def test_unknown_product_is_reported_not_guessed():
    assert classify.product_line(
        "Mr M has complained that Chubb failed to settle his claim fairly.") == "unknown"


def test_query_wording_does_not_decide_the_product():
    # The retrieval query said "motor"; the decision says travel. The decision
    # wins — this is the whole reason the classifier exists.
    text = ("Mrs P has complained that her travel insurance claim was declined "
            "after her hire car was damaged abroad.")
    assert classify.product_line(text) == "travel"
