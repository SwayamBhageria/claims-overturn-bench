"""The split has to be airtight: an input that contains its own answer turns
the whole benchmark into a measurement of nothing."""
import pytest

from corpus import sections

DECISION = """DRN-1234567

 The complaint

 Mrs I has complained that Example Insurance Limited declined a claim she made
 on a travel insurance policy.

 What happened

 Mrs I took a trip abroad starting on 25 March 2025. She made a claim for
 medical costs. Example declined the claim on the basis that the circumstances
 are not covered under the policy terms.

 Our investigator thought that Example had acted reasonably in declining the
 claim. Mrs I disagrees and so the complaint has been passed to me.

 What I've decided - and why

 I've considered all the available evidence. Example failed to obtain the
 medical evidence it needed. It follows that I uphold this complaint.

 Putting things right

 Example must pay the claim and add 8% simple interest.

 My final decision

 I uphold this complaint.
"""


def test_split_finds_every_section():
    d = sections.split(DECISION)
    assert d.drn == "DRN-1234567"
    assert "declined a claim she made" in d.complaint
    assert "trip abroad" in d.what_happened
    assert "failed to obtain" in d.reasoning
    assert "uphold this complaint" in d.outcome_text


def test_verdict_never_reaches_the_input():
    d = sections.split(DECISION)
    assert sections.leak_terms(d.prompt_input) == []
    assert "I uphold" not in d.prompt_input
    assert "8% simple interest" not in d.prompt_input


def test_investigator_view_is_removed_but_kept():
    d = sections.split(DECISION)
    assert "Our investigator" not in d.what_happened
    assert "Our investigator" in d.investigator_view
    # and the ablation input puts it back
    assert "Our investigator" in d.prompt_input_with_investigator


def test_leak_terms_catches_each_verdict_form():
    for phrase in ["I uphold the complaint",
                   "I do not uphold this",
                   "I don't uphold it",
                   "My final decision is that",
                   "It follows that I cannot agree"]:
        assert sections.leak_terms(phrase), phrase


def test_leak_terms_is_quiet_on_ordinary_narrative():
    ordinary = ("The policyholder said the claim should be paid. Example said "
                "the exclusion applied and declined it.")
    assert sections.leak_terms(ordinary) == []


def test_missing_skeleton_raises_rather_than_half_parsing():
    with pytest.raises(sections.SectionError):
        sections.split("A document with no headings at all.")


def test_provisional_decision_also_cuts():
    text = DECISION.replace("What I've decided - and why", "My provisional decision")
    d = sections.split(text)
    assert sections.leak_terms(d.prompt_input) == []


# --- prior views that must never reach the input ----------------------------
# Each of these was found leaking on real data, and each one moved a headline
# number. They are regression tests, not hypotheticals.

def _happened(text: str):
    doc = f"""DRN-9
 The complaint
 Mrs I complained that Example declined a claim.
 What happened
 {text}
 What I've decided - and why
 Reasoning.
 My final decision
 I uphold this complaint.
"""
    d = sections.split(doc)
    return d.what_happened, d.investigator_view


def test_the_adjudicators_recommendation_is_removed_across_sentences():
    # A line-based stripper removed "Our investigator looked at it." and left
    # the recommendation, which is the outcome, in 56% of inputs.
    kept, removed = _happened(
        "Mrs I made a claim. Our investigator looked at it. She recommended "
        "that Example settle the claim. She also said Example should pay £100 "
        "compensation. Mrs I disagrees.")
    assert "recommended" not in kept
    assert "£100 compensation" not in kept
    assert "settle the claim" in removed
    assert "Mrs I made a claim." in kept


def test_the_insurers_own_offer_is_kept():
    # Claim history a handler would hold. Stripping it would remove a fact,
    # not a leak.
    kept, _removed = _happened(
        "Mrs I made a claim. They offered Mrs A £250 compensation as a result.")
    assert "£250 compensation" in kept


def test_a_pronoun_returning_to_the_complainant_is_not_swept_up():
    kept, removed = _happened(
        "Our investigator thought Example acted reasonably. She disagreed and "
        "asked for an ombudsman's decision.")
    assert "disagreed" in kept
    assert "acted reasonably" in removed


def test_plural_investigators_are_caught():
    kept, removed = _happened(
        "One of our Investigators recommended this complaint be upheld.")
    assert "upheld" not in kept
    assert "Investigators" in removed


def test_a_recommendation_by_someone_who_is_not_the_adjudicator_is_kept():
    kept, _removed = _happened(
        "The surveyor recommended that the claim be declined.")
    assert "surveyor" in kept


def test_provisional_decision_wording_is_a_leak():
    for phrase in [
        "I issued a provisional decision explaining that I was intending to "
        "uphold Mr S's complaint.",
        "I made a provisional decision that I was minded to not uphold it.",
    ]:
        assert sections.leak_terms(phrase), phrase


def test_provisional_reasoning_is_held_out_of_the_input():
    kept, removed = _happened(
        "Mrs I made a claim. In my provisional decision I set out the evidence.")
    assert "provisional" not in kept
    assert "provisional" in removed
