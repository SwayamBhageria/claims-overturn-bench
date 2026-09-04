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
    assert grounds.distress_only(remedy)


def test_distress_only_is_false_when_the_claim_is_also_paid():
    remedy = ("Example must pay the claim and a further £300 for the distress "
              "and inconvenience caused.")
    assert grounds.claim_decision_disturbed(remedy)
    assert not grounds.distress_only(remedy)
