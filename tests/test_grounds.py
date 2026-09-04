"""Ground tagging, on the ombudsman's actual phrasing.

The wordings below are the recurring formulas in real decisions. The point of
fixing them here is that the coverage/handling split is the repo's headline
finding, so a silent change in what counts as "handling" would move a number
the README states.
"""
from bench import grounds


def test_handling_grounds():
    assert "evidence" in grounds.tag(
        "Aviva failed to obtain the medical evidence it needed before declining.")
    assert "evidence" in grounds.tag(
        "It did not consider the photographs Mr B sent showing the damage.")
    assert "delay" in grounds.tag(
        "There was an unreasonable delay of four months in progressing the claim.")
    assert "communication" in grounds.tag(
        "It failed to explain to her why the claim had been declined.")
    assert "quantum" in grounds.tag(
        "The settlement was too low and the deduction for betterment was unfair.")
    assert "process" in grounds.tag(
        "ICOBS 8.1 requires an insurer to handle claims promptly and fairly.")


def test_coverage_grounds():
    assert "policy_wording" in grounds.tag(
        "In my view the exclusion does not apply to these circumstances.")
    assert "misrepresentation" in grounds.tag(
        "This was not a qualifying misrepresentation under CIDRA.")
    assert "fraud" in grounds.tag(
        "I don't think fraud has been established on the evidence available.")


def test_families_are_assigned_from_the_grounds():
    assert grounds.family(["delay", "evidence"]) == grounds.HANDLING
    assert grounds.family(["policy_wording"]) == grounds.COVERAGE
    assert grounds.family([]) == "untagged"


def test_a_case_faulting_both_is_counted_as_both():
    tags = grounds.tag(
        "The exclusion does not apply, and there was an unreasonable delay "
        "in reaching that conclusion.")
    assert grounds.family(tags) == "both"


def test_explain_returns_the_matched_words():
    why = grounds.explain("There was an unreasonable delay in the claim.")
    assert "delay" in why
    assert any("delay" in phrase.lower() for phrase in why["delay"])


def test_awards_ignore_sums_that_are_not_awards():
    text = ("The repair estimate was £4,000 and the car was valued at £8,500. "
            "Example must pay £300 compensation for the distress caused.")
    assert grounds.awards(text) == [300.0]


def test_awards_read_thousands_and_pence():
    text = "Example must pay £1,250.50 in compensation."
    assert grounds.awards(text) == [1250.50]


# Award extraction covers compensation only. Remedy sections name the
# settlement, the earlier offer, the valuation and the policy limit in
# adjacent sentences, and an earlier version that took the largest sum in an
# award-shaped sentence returned a vehicle's £24,300 valuation as an award.
# Every wording below is from a real decision.

def test_the_claim_settlement_is_not_counted_as_compensation():
    text = ("I require Covea Insurance plc to: Pay £25,924 to settle the "
            "claim. Pay £100 compensation.")
    assert grounds.awards(text) == [100.0]


def test_a_valuation_in_the_remedy_is_not_an_award():
    text = ("Pay Mr K a further £550 in settlement of the value of Mr K's "
            "vehicle, this being the difference between its latest valuation "
            "of £24,300 and its previous valuation of £23,750.")
    assert grounds.awards(text) == []


def test_the_figure_nearest_the_word_wins_over_the_largest():
    text = ("Pay Mr K £250 in compensation for the distress and inconvenience "
            "caused. It can deduct the £150 previously offered.")
    assert grounds.awards(text) == [250.0]


def test_business_interruption_settlement_is_not_compensation():
    text = ("Pay £40,000 in respect of the business interruption claim, "
            "together with interest at 8% simple per annum. Pay Mr G the sum "
            "of £1,250 compensation for the distress and inconvenience caused.")
    assert grounds.awards(text) == [1250.0]


def test_distribution_counts_untagged_cases():
    d = grounds.distribution([["delay"], ["delay", "evidence"], []])
    assert d["n"] == 3
    assert d["grounds"]["delay"] == 2
    assert d["untagged"] == 1


# --- did the ombudsman actually change the claim outcome? -------------------
# "Upheld" covers both "your decision was wrong" and "your decision stood but
# your handling didn't". Conflating them would let the headline finding be an
# artefact of what the word covers.

def test_remedy_that_changes_the_claim_outcome():
    for remedy in [
        "Example must now pay the claim in line with the remaining policy terms.",
        "Example must reconsider the claim.",
        "Example should settle the claim and add 8% simple interest.",
        "Example must remove the decline and reinstate the policy.",
    ]:
        assert grounds.claim_decision_disturbed(remedy), remedy


def test_remedy_that_leaves_the_claim_outcome_alone():
    remedy = ("Example acted fairly in declining the claim. However it must pay "
              "£300 for the distress and inconvenience caused by the delay.")
    assert not grounds.claim_decision_disturbed(remedy)
    assert grounds.compensation_only(remedy)


def test_compensation_only_is_false_when_the_claim_is_also_paid():
    remedy = ("Example must pay the claim and a further £300 for the distress "
              "and inconvenience caused.")
    assert grounds.claim_decision_disturbed(remedy)
    assert not grounds.compensation_only(remedy)


# Remedy wordings taken verbatim from published decisions, including every one
# an earlier version of `claim_decision_disturbed` got wrong. Auditing the
# cases it called "the claim stood" is what produced this list, and the share
# it reported before the fix was wrong by roughly a factor of two.
REAL_REMEDIES = [
    ("Domestic and General Insurance PLC to: Settle Mr S' claims under the "
     "remaining policy terms, less any excess payable.", True),          # DRN-4536448
    ("Amtrust Europe Limited to accept the claim and repair the laptop. Pay "
     "Ms G and Mr G £100 compensation for the trouble and upset caused.", True),  # DRN-4553239
    ("Increase the cash settlement to £355", True),                      # DRN-4544487
    ("Red Sands should pay Mr W's claim for his pet's treatment made during "
     "the policy period, subject to the remaining policy terms.", True),  # DRN-4686053
    ("To pay the outstanding balance of her vet's invoice for her dog's "
     "treatment up to the policy limit of £3,000.", True),               # DRN-4727954
    ("I direct IPA to pay Mrs W £300 compensation for distress and "
     "inconvenience.", False),                                           # DRN-4594392
    ("direct Liverpool Victoria Insurance Company Limited to pay a total of "
     "£450 compensation, if they haven't already done so.", False),      # DRN-4648437
    ("UKI should pay Ms C a total of £550 for distress and inconvenience.", False),  # DRN-4670932
]


def test_remedy_classifier_on_real_wordings():
    for remedy, expected in REAL_REMEDIES:
        assert grounds.claim_decision_disturbed(remedy) is expected, remedy[:70]


def test_a_bare_mention_of_the_claim_is_not_a_direction_about_it():
    # "\\w+'s? claims?" once matched "the claim", which made almost every
    # remedy look claim-affecting and inflated the headline split.
    assert not grounds.claim_decision_disturbed(
        "Example acted fairly in declining the claim. However it must pay "
        "£300 for the distress and inconvenience caused by the delay.")


def test_remedy_text_starts_at_the_operative_directions():
    reasoning = "Long discursive reasoning about the policy wording."
    outcome = "Putting things right Example must accept the claim."
    assert grounds.remedy_text(reasoning, outcome).startswith("Putting things right")


def test_remedy_text_falls_back_to_the_whole_decision():
    # A decision whose directions cannot be located should be read in full,
    # not scored on an empty string.
    assert grounds.remedy_text("reasoning", "no heading here") == \
        "reasoning\nno heading here"
