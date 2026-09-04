"""Why did the ombudsman overturn it?

An upheld complaint is not one kind of event. Some turn on what the policy
covers — a question of underwriting judgement, argued from wording. Others
turn on how the claim was run: evidence that was never obtained, a decision
that took four months, a customer who had to chase six times. The distinction
matters commercially, because only one of the two is inside the operating
control of whoever administers the claim.

So each upheld decision is tagged with the ground(s) its reasoning rests on,
grouped into two families:

  COVERAGE   the decision itself was wrong on the terms
             - policy_wording      exclusion misapplied, ambiguous term
             - misrepresentation   CIDRA qualifying-misrep test misapplied
             - fraud               allegation not made out to the standard

  HANDLING   the decision may have been defensible; the handling was not
             - evidence            not obtained, not considered, wrong expert
             - delay               time taken, chasing, failure to update
             - communication       misleading, unexplained, poor service
             - quantum             settlement or deduction unfair in amount
             - process             claim not considered, wrong policy applied

Tagging is rules over the ombudsman's own reasoning, not a model, so any tag
can be traced to the sentence that produced it — `explain()` returns exactly
that. `bench/validate_rules.py` scores the tagger against a hand-labelled
sample and prints per-ground precision and recall, so the accuracy of this
layer is a measured number in the README rather than an assurance.

Also pulled out here: money. FOS awards appear in "Putting things right" as
sterling amounts, and `awards()` extracts them, which is what turns a
distribution of grounds into a distribution of cost.
"""
from __future__ import annotations

import re
from collections import Counter

COVERAGE = "coverage"
HANDLING = "handling"

# ground -> (family, patterns). Patterns run against the ombudsman's reasoning.
GROUNDS: dict[str, tuple[str, list[str]]] = {
    "policy_wording": (COVERAGE, [
        r"\bexclusion (?:does not|doesn't|did not|didn't) apply\b",
        r"\b(?:not|isn't|wasn't) (?:entitled|fair) to (?:rely|decline)[^.]{0,60}\b(?:term|exclusion|condition)\b",
        r"\bambiguous\b|\bcontra proferentem\b",
        r"\bterm (?:is|was) (?:unclear|not clear|unfair)\b",
        r"\bICOBS 6\b|\bsignificant exclusion\b|\bunusual (?:term|limitation)\b",
        r"\bmisinterpret\w+ (?:the )?polic\w+\b",
        r"\bpolicy (?:does|did) cover\b",
    ]),
    "misrepresentation": (COVERAGE, [
        r"\bCIDRA\b|\bConsumer Insurance \(Disclosure and Representations\) Act\b",
        r"\bqualifying misrepresentation\b",
        r"\b(?:careless|deliberate or reckless) misrepresentation\b",
        r"\bnon[- ]disclosure\b",
    ]),
    "fraud": (COVERAGE, [
        r"\bfraudulent(?:ly)?\b[^.]{0,80}\b(?:not|hasn't|haven't|failed to)\b",
        r"\bfraud (?:has not|hasn't) been (?:established|proven|made out)\b",
        # The negation often sits before the noun: "I don't think fraud has
        # been established", "I'm not persuaded this claim was fraudulent".
        r"\b(?:don't|do not|doesn't|isn't|not) (?:think|satisfied|persuaded|accept)\b[^.]{0,60}\bfraud\w*\b",
        r"\bburden of proof\b[^.]{0,80}\bfraud\b",
        r"\bCIFAS\b[^.]{0,80}\b(?:remove|unfair)\b",
    ]),
    "evidence": (HANDLING, [
        r"\b(?:didn't|did not|failed to|hasn't|haven't) (?:obtain|request|seek|gather|get)\b[^.]{0,60}\b(?:evidence|report|information|medical|records)\b",
        # The noun list is deliberately loose: an earlier version listed
        # "photos" and missed every decision that said "photographs".
        r"\b(?:didn't|did not|failed to) (?:properly )?(?:consider|take into account|address|review|look at)\b[^.]{0,60}\b(?:evidence|report\w*|information|photo\w*|record\w*|statement\w*|invoice\w*|receipt\w*)\b",
        r"\b(?:no|insufficient|inadequate) (?:evidence|basis) (?:to|for)\b[^.]{0,50}\b(?:decline|reject|refus\w+)\b",
        r"\breport (?:was|is) (?:inadequate|insufficient|not sufficient|flawed)\b",
        r"\bshould have (?:obtained|sought|asked for|arranged)\b",
    ]),
    "delay": (HANDLING, [
        r"\b(?:unnecessary|avoidable|unreasonable|significant|considerable) delay\b",
        r"\bdelay(?:s|ed)?\b[^.]{0,60}\b(?:unreasonab\w+|unnecessar\w+|too long|avoidable)\b",
        r"\btook (?:far )?too long\b",
        r"\bshould have (?:been )?(?:dealt with|settled|resolved|progressed)[^.]{0,40}\b(?:sooner|quicker|faster|earlier)\b",
    ]),
    "communication": (HANDLING, [
        r"\b(?:poor|inadequate|unclear|misleading) (?:communication|service|information)\b",
        r"\b(?:didn't|did not|failed to) (?:explain|tell|inform|update|keep)\b[^.]{0,60}\b(?:him|her|them|informed|updated)\b",
        r"\bhad to chase\b|\bchasing\b",
        r"\bgave (?:him|her|them) (?:incorrect|wrong|misleading)\b",
        r"\bdistress and inconvenience\b",
    ]),
    "quantum": (HANDLING, [
        r"\b(?:settlement|offer|payment) (?:was|is) (?:too low|unfair|insufficient|not fair)\b",
        r"\b(?:unfair|incorrect|wrong) (?:deduction|depreciation|betterment|excess)\b",
        r"\bmarket value\b[^.]{0,60}\b(?:too low|understated|unfair)\b",
        r"\bshould (?:pay|settle|increase)\b[^.]{0,60}\b(?:more|the (?:full|balance))\b",
        r"\b8% simple interest\b",
    ]),
    "process": (HANDLING, [
        r"\b(?:didn't|did not|failed to) (?:consider|assess|deal with|progress|handle) the claim\b",
        r"\b(?:wrong|incorrect) (?:policy|section|cover) (?:was )?applied\b",
        r"\bclaim (?:was )?(?:never|not) (?:properly )?(?:considered|assessed|investigated)\b",
        r"\bICOBS 8\.1\b|\bhandle claims promptly and fairly\b|\bunreasonably (?:decline|reject)\b",
    ]),
}

_COMPILED = {g: (fam, [re.compile(p, re.I) for p in pats])
             for g, (fam, pats) in GROUNDS.items()}

# "£1,250.00" / "£300" — the award figures in "Putting things right".
_MONEY = re.compile(r"£\s?([\d,]+(?:\.\d{2})?)")
# Sentences that are actually awarding money, rather than reciting a sum in
# dispute. Without this, every quoted repair estimate becomes an "award".
_AWARD_CONTEXT = re.compile(
    r"(?i)\b(?:pay|award|compensat\w+|in recognition|for the (?:distress|trouble)|"
    r"total of|plus interest)\b")


def tag(reasoning: str) -> list[str]:
    """Grounds present in this reasoning, in a stable order."""
    return [g for g in GROUNDS if any(p.search(reasoning) for p in _COMPILED[g][1])]


def explain(reasoning: str) -> dict[str, list[str]]:
    """ground -> the matched substrings. The audit trail for every tag."""
    out: dict[str, list[str]] = {}
    for g, (_fam, pats) in _COMPILED.items():
        hits = [m.group(0).strip() for p in pats for m in p.finditer(reasoning)]
        if hits:
            out[g] = hits
    return out


def family(grounds: list[str]) -> str:
    """Which family a case belongs to.

    A case tagged with both is counted as `both`, never silently assigned to
    one: about a fifth of upheld decisions genuinely fault the decision *and*
    the handling, and collapsing those into either family would overstate it.
    """
    fams = {GROUNDS[g][0] for g in grounds if g in GROUNDS}
    if fams == {COVERAGE}:
        return COVERAGE
    if fams == {HANDLING}:
        return HANDLING
    if fams == {COVERAGE, HANDLING}:
        return "both"
    return "untagged"


# Did the ombudsman actually disturb the claim outcome, or only the handling?
# "Upheld" covers both, and they are commercially different events: one says
# the decision was wrong, the other says the decision stood and the service
# around it did not.
_CLAIM_DISTURBED = [
    re.compile(r"(?i)\bmust (?:now )?(?:pay|settle|meet|reimburse) the claim\b"),
    re.compile(r"(?i)\b(?:must|should) (?:now )?(?:reconsider|reassess|review) "
               r"(?:the|this|his|her|their) claim\b"),
    re.compile(r"(?i)\bdeal with the claim\b[^.]{0,40}\b(?:policy terms|remaining)\b"),
    re.compile(r"(?i)\b(?:pay|settle)[^.]{0,60}\bthe (?:full |outstanding |remaining )?"
               r"(?:claim|settlement|balance)\b"),
    re.compile(r"(?i)\bcover the (?:cost|claim)\b"),
    re.compile(r"(?i)\bremove the (?:decline|declinature)\b"),
    re.compile(r"(?i)\breinstate the (?:policy|claim)\b"),
]
# Money that is only compensation for the experience, not the claim itself.
_DISTRESS_ONLY = re.compile(
    r"(?i)\b(?:distress and inconvenience|trouble and upset|the impact on)\b")


def claim_decision_disturbed(remedy: str) -> bool:
    """True when the remedy changes what happens to the claim itself.

    False means the ombudsman upheld the complaint without touching the
    outcome — the declinature stood and the insurer lost on how it got there.
    """
    return any(p.search(remedy) for p in _CLAIM_DISTURBED)


def distress_only(remedy: str) -> bool:
    """The remedy is compensation for the experience and nothing else."""
    return bool(_DISTRESS_ONLY.search(remedy)) and not claim_decision_disturbed(remedy)


def awards(putting_right: str) -> list[float]:
    """Sterling sums awarded in the remedy section, in sentence context."""
    out: list[float] = []
    for sentence in re.split(r"(?<=[.;])\s+", putting_right):
        if not _AWARD_CONTEXT.search(sentence):
            continue
        for m in _MONEY.finditer(sentence):
            try:
                out.append(float(m.group(1).replace(",", "")))
            except ValueError:
                continue
    return out


def distribution(tagged: list[list[str]]) -> dict:
    """Ground and family counts over a set of tagged decisions."""
    grounds = Counter(g for tags in tagged for g in tags)
    fams = Counter(family(tags) for tags in tagged)
    return {
        "n": len(tagged),
        "grounds": dict(grounds.most_common()),
        "families": dict(fams.most_common()),
        "untagged": fams.get("untagged", 0),
    }
