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
        r"\b(?:didn't|did not|failed to|hasn't|haven't) (?:obtain|request|seek|gather|get|commission|arrange)\b[^.]{0,70}\b(?:evidence|report|information|medical|records|inspection|survey)\b",
        # The noun list is deliberately loose: an earlier version listed
        # "photos" and missed every decision that said "photographs".
        r"\b(?:didn't|did not|failed to) (?:properly |fully )?(?:consider|take into account|address|review|look at|engage with)\b[^.]{0,70}\b(?:evidence|report\w*|information|photo\w*|record\w*|statement\w*|invoice\w*|receipt\w*|testimony)\b",
        r"\b(?:no|insufficient|inadequate|not enough) (?:evidence|basis|grounds) (?:to|for)\b[^.]{0,60}\b(?:decline|reject|refus\w+|repudiat\w+|void\w*)\b",
        r"\b(?:report|inspection|survey|assessment) (?:was|is|were)\b[^.]{0,60}\b(?:inadequate|insufficient|not sufficient|flawed|limited|incomplete|unreliable)\b",
        r"\bshould have (?:obtained|sought|asked for|arranged|commissioned|investigated|checked)\b",
        r"\b(?:didn't|did not|hasn't|haven't) (?:show|prove|demonstrate|establish)\b[^.]{0,60}\b(?:entitled to|the claim|it was)\b",
        r"\bburden (?:of proof|is on)\b[^.]{0,60}\b(?:insurer|it|them)\b",
    ]),
    "delay": (HANDLING, [
        r"\b(?:unnecessary|avoidable|unreasonable|significant|considerable|lengthy|undue) delay\b",
        r"\bdelay(?:s|ed|ing)?\b[^.]{0,70}\b(?:unreasonab\w+|unnecessar\w+|too long|avoidable|not acceptable)\b",
        r"\btook (?:far |much )?too long\b",
        r"\bshould have (?:been )?(?:dealt with|settled|resolved|progressed|paid|actioned)\b[^.]{0,50}\b(?:sooner|quicker|faster|earlier|promptly|more quickly)\b",
        r"\b(?:months|weeks) (?:went by|passed)\b|\bstill (?:not|hasn't been) (?:resolved|settled|paid)\b",
    ]),
    "communication": (HANDLING, [
        # NOT "distress and inconvenience": that is the remedy attached to most
        # upheld decisions whatever the ground, and including it made this the
        # largest category by measuring the award rather than the fault.
        r"\b(?:poor|inadequate|unclear|misleading|confusing) (?:communication|service|information|explanation)\b",
        r"\b(?:didn't|did not|failed to|should have) (?:explain|tell|inform|update|direct|make clear|set out|warn)\b",
        r"\bhad to chase\b|\bchasing\b|\bchased\b",
        r"\bgave (?:him|her|them|it) (?:incorrect|wrong|misleading|conflicting)\b",
        r"\b(?:could|should) have (?:directed|pointed|signposted|told)\b",
        r"\bwasn't made (?:clear|aware)\b|\bwas not made (?:clear|aware)\b",
        r"\bconflicting information\b",
    ]),
    "quantum": (HANDLING, [
        r"\b(?:settlement|offer|payment|valuation|sum offered) (?:was|is|wasn't|isn't)\b[^.]{0,50}\b(?:too low|unfair|insufficient|not fair|inadequate|understated)\b",
        r"\b(?:didn't|did not|don't|do not) (?:agree|think|accept)\b[^.]{0,60}\b(?:offer|settlement|valuation|amount)\b[^.]{0,40}\b(?:fair|reasonable)\b",
        r"\b(?:unfair|incorrect|wrong|excessive) (?:deduction|depreciation|betterment|excess|reduction)\b",
        r"\bmarket value\b[^.]{0,70}\b(?:too low|understated|unfair|not fair)\b",
        r"\bshould (?:pay|settle|increase|cover)\b[^.]{0,70}\b(?:more|the (?:full|balance|remainder|cost)|in full)\b",
        r"\b8% simple interest\b",
        r"\b(?:fair|reasonable) (?:that|for)\b[^.]{0,60}\bsettle the claim\b",
    ]),
    "process": (HANDLING, [
        r"\b(?:didn't|did not|failed to) (?:consider|assess|deal with|progress|handle|investigate) (?:the|this|his|her|their) claim\b",
        r"\b(?:wrong|incorrect) (?:policy|section|cover|term) (?:was )?(?:applied|relied on)\b",
        r"\bclaim (?:was )?(?:never|not) (?:properly |fairly )?(?:considered|assessed|investigated|handled)\b",
        r"\bICOBS 8\.1\b|\bhandle claims promptly and fairly\b|\bunreasonably (?:decline|reject)\w*\b",
        # The single most common formula in the corpus, and it was missing.
        r"\bunfairly (?:declin\w+|reject\w+|refus\w+|repudiat\w+|void\w+|cancel\w+|settl\w+|handl\w+|turned down)\b",
        r"\b(?:it|they|the insurer) (?:acted|behaved) unfairly\b",
        r"\b(?:unfair|not fair|wasn't fair|isn't fair) (?:to|that|of)\b[^.]{0,60}\b(?:declin\w+|reject\w+|refus\w+|cancel\w+|void\w+)\b",
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


# Did the ombudsman disturb the claim outcome, or only the handling?
#
# "Upheld" covers both and they are commercially different events. Getting this
# split from the *reasoning* does not work — the reasoning is discursive. The
# operative directions do work, because they are a short, formulaic list:
# "I direct X to: accept the claim... pay £150 compensation."
#
# The first version of this looked for "must pay the claim" and called
# everything else untouched. Reading the cases it classified that way found
# "accept the claim and repair the laptop", "increase the cash settlement to
# £355" and "pay Mr W's claim for his pet's treatment" all counted as the claim
# standing. The share it produced was wrong by roughly a factor of two, so the
# verb list below is deliberately broad and the possessive gap is allowed for.
_CLAIM_AFFECTING = re.compile(
    r"\b(?:accept|pay|settle|meet|reimburse|refund|cover|honour|"
    r"reconsider|reassess|re-assess|review|repair|replace|increase|"
    r"reinstate|remove|rectify|process|progress)\b"
    r"[^.;\n]{0,80}?"
    r"\b(?:the claim|this claim|his claim|her claim|their claim|the settlement|"
    r"the cash settlement|the excess|policy benefit|"
    r"the (?:full |outstanding |remaining |repair )?costs?|the decline|"
    r"the declinature|the policy|the benefit|the invoice|the treatment|"
    r"the damage|the loss|the outstanding balance|the balance)\b"
    # or the direction names a sum that is plainly not compensation
    r"|\b(?:reimburse|refund|pay)\b[^.;\n]{0,60}\bfor the (?:cost|costs|"
    r"price|value|repair|replacement|treatment|survey|report)\b"
    r"|\bincrease the (?:cash )?settlement\b"
    r"|\b(?:accept|cover)\b[^.;\n]{0,60}\bunder the\b[^.;\n]{0,40}"
    r"\b(?:policy|section|terms)\b"
    # "Settle Mr S' claims under the remaining policy terms" — a possessive
    # ending in a bare apostrophe, and a plural. Both were missed.
    r"|\b\w+['’]s?\s+claims?\b"
    r"|\b(?:vet|vet's|repairer's|garage's) invoice\b"
    r"|\bbenefit for\b",
    re.I)


# Compensation for the experience: the sum that is not the claim.
_COMPENSATION = re.compile(
    r"£\s?[\d,]+(?:\.\d{2})?\s*(?:in )?compensation\b"
    r"|\bcompensation\b[^.;\n]{0,40}\b(?:distress|inconvenience|trouble|"
    r"upset|impact|worry)\b"
    r"|\b(?:distress and inconvenience|trouble and upset)\b",
    re.I)

# Where the operative directions live.
_REMEDY_SECTION = re.compile(
    r"(?is)(?:putting things right|my final decision|i direct|must now|should now)")


def remedy_text(reasoning: str, outcome: str) -> str:
    """The part of a decision that contains the operative directions.

    Falls back to the whole text: a decision whose directions we cannot locate
    should be read in full rather than scored on nothing.
    """
    whole = f"{reasoning}\n{outcome}"
    m = _REMEDY_SECTION.search(whole)
    return whole[m.start():] if m else whole


def claim_decision_disturbed(remedy: str) -> bool:
    """True when the directions change what happens to the claim itself.

    False means the ombudsman upheld the complaint without touching the
    outcome — the declinature stood and the insurer lost on how it got there.
    """
    return bool(_CLAIM_AFFECTING.search(remedy))


def compensation_only(remedy: str) -> bool:
    """The directions award compensation for the experience and nothing else."""
    return bool(_COMPENSATION.search(remedy)) and not claim_decision_disturbed(remedy)


def awards(remedy: str) -> list[float]:
    """Compensation awarded for the experience, in sterling.

    **Only compensation.** The claim settlement is deliberately not extracted,
    and that is a limitation with a reason rather than an oversight. Remedy
    sections quote the settlement, the insurer's earlier offer, the vehicle's
    latest valuation and the policy limit in adjacent sentences, and taking the
    largest sum in an award-shaped sentence picks the wrong one often enough to
    be useless: in one decision it returned £24,300, the vehicle's valuation,
    where the sum actually awarded was the £550 difference between two
    valuations.

    Compensation is well defined — the ombudsman names it as compensation, for
    distress, inconvenience, trouble or upset — so that is what is measured.
    It is the smaller half of the cost, and the README says so.
    """
    out: list[float] = []
    for sentence in re.split(r"(?<=[.;])\s+", remedy):
        if not _COMPENSATION.search(sentence):
            continue
        # A sentence awarding compensation can still name another sum ("deduct
        # the £150 previously offered"), so take the figure sitting closest to
        # the word itself rather than the largest in the sentence.
        best: tuple[int, float] | None = None
        for word in re.finditer(r"compensation|distress|inconvenience|trouble|upset",
                                sentence, re.I):
            for m in _MONEY.finditer(sentence):
                try:
                    value = float(m.group(1).replace(",", ""))
                except ValueError:
                    continue
                gap = abs(m.start() - word.start())
                if best is None or gap < best[0]:
                    best = (gap, value)
        if best:
            out.append(best[1])
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
