"""What is this decision actually about?

Two labels are assigned from the decision's own words, not from the search
query that retrieved it.

**Product line.** Read off the opening sentence, which names it almost every
time: "a claim she made on a travel insurance policy". This exists because
`Keyword` is a full-text search: a query for "motor insurance" returns travel
decisions that mention a hire car, so a count of query hits is not a count of
decisions about motor.

**Complaint type.** This is the filter that decides what the benchmark is even
measuring. A large share of published insurance decisions are not about a
claim at all — they are about a premium increase, a mis-sale at the point of
sale, or a mid-term cancellation. Scoring a claim-decision model on those
would inflate the sample with cases where the model is answering a question
nobody asked it. Only `CLAIM` cases enter the benchmark; the rest are counted
and reported, because how large that share is turns out to matter.

Both classifiers are rules over the text rather than a model, for the reason
that governs the whole repo: every number here has to be checkable by hand.
`tests/test_classify.py` fixes the behaviour on real openings, and
`bench/validate_rules.py` reports agreement against a hand-labelled sample.
"""
from __future__ import annotations

import re

# --- product line -----------------------------------------------------------
# Ordered: the first pattern to match wins, so the more specific phrasings
# ("travel insurance") are tried before generic ones ("home").
PRODUCT_PATTERNS: list[tuple[str, str]] = [
    ("travel",    r"\btravel (?:insurance|policy|cover)\b|\bsingle[- ]trip\b|\bannual multi[- ]trip\b"),
    ("pet",       r"\bpet (?:insurance|policy|cover)\b|\bveterinary\b|\bvet(?:'s)? (?:fees|bills)\b"),
    ("gadget",    r"\b(?:gadget|mobile phone|mobile telephone|phone) (?:insurance|policy|cover)\b"
                  r"|\b(?:replace|repair|settle\w*)\w*\b[^.]{0,40}\b(?:mobile phone|handset)\b"),
    ("motor",     r"\b(?:motor|car|vehicle|van|motorcycle) (?:insurance|policy|cover)\b|\bcomprehensive motor\b"),
    ("property",  r"\b(?:home|buildings|contents|household|landlord) (?:insurance|policy|cover)\b|\bbuildings and contents\b"),
    ("warranty",  r"\b(?:warranty|guarantee|breakdown) (?:insurance|policy|cover)\b|\bextended warranty\b"
                  r"|\bhome warranty\b|\bBuildmark\b"),
    ("health",    r"\b(?:private medical|health|dental|critical illness|income protection) (?:insurance|policy|cover)\b"),
    ("life",      r"\b(?:life|term|over[- ]50s) (?:insurance|assurance|policy|cover)\b"),
    ("commercial", r"\b(?:commercial|business|public liability|professional indemnity|employers'? liability) (?:insurance|policy|cover)\b"),
]
_PRODUCT = [(name, re.compile(pat, re.I)) for name, pat in PRODUCT_PATTERNS]

# The six lines a claims administrator handling personal and small-commercial
# business actually staffs for. Used to scope the benchmark; not a judgement
# about the others.
CORE_LINES = ("travel", "pet", "gadget", "motor", "property", "warranty")

# --- complaint type ---------------------------------------------------------
# A claim complaint says a claim exists and something was done to it. The verb
# may sit on either side of the word "claim" — "declined the claim" and "the
# claim was unfairly handled" are the same complaint — so both orders are
# matched against one verb list. Getting this asymmetric silently drops real
# claim cases: on the 29-decision pilot, an earlier version that only looked
# forwards missed "unfairly handled a claim made on his ... policy" and
# "haven't settled their travel insurance claim in full".
_CLAIM_VERB = (r"(?:declin\w+|reject\w+|refus\w+|turn\w+ down|settl\w+|repudiat\w+|"
               r"pay\w*|paid|handl\w+|delay\w+|deal\w+ with|assess\w+|valu\w+|"
               r"deduct\w+|void\w+|cancel\w+|reduc\w+|withdraw\w+|dispute\w+)")
_CLAIM = re.compile(
    rf"(?i)\bclaim\w*\b[^.]{{0,120}}?\b{_CLAIM_VERB}\b"
    rf"|\b{_CLAIM_VERB}\b[^.]{{0,120}}?\bclaim\w*\b"
    r"|\bmade a claim\b|\bclaimed (?:on|under)\b|\bmake a claim\b"
    r"|\bclaim (?:she|he|they|it|was)\b"
)
# Things that are complaints about the contract, not about a claim on it.
_PRICING = re.compile(
    r"(?i)\b(?:premium|price|pricing|renewal (?:price|quote|premium)|cost of (?:his|her|their) (?:policy|insurance))\b"
    r"[^.]{0,80}\b(?:increas\w+|rise|rose|went up|high\w*|unfair\w*|loyalty)\b"
    r"|\bincrease of (?:his|her|their) .{0,30}premium\b"
)
_SALES = re.compile(
    r"(?i)\bmis[- ]?s(?:old|ale|elling)\b|\bwhen (?:it|they|he|she) sold\b"
    r"|\bat the (?:time|point) of sale\b|\bdidn't explain\b|\bdid not explain\b"
    r"|\bwasn't made (?:clear|aware)\b|\bwas not made (?:clear|aware)\b"
)
_ADMIN = re.compile(
    r"(?i)\bcancel(?:led|lation) (?:of )?(?:his|her|their|the) polic\w+\b"
    r"|\bmid[- ]term (?:adjustment|cancellation)\b|\brecorded (?:a|the) .{0,20}\bmarker\b"
    r"|\bCIFAS\b|\bcredit file\b"
)

CLAIM = "claim"
PRICING = "pricing"
SALES = "sales"
ADMIN = "admin"
OTHER = "other"


def product_line(text: str) -> str:
    """Best-guess product line from the decision's own wording."""
    head = text[:1500]
    for name, pat in _PRODUCT:
        if pat.search(head):
            return name
    for name, pat in _PRODUCT:      # fall back to the whole text
        if pat.search(text):
            return name
    return "unknown"


def complaint_type(text: str) -> str:
    """Is this about a claim, or about the contract around it?

    A claim signal wins over a pricing or sales signal, because decisions that
    mention both are almost always a claim complaint whose narrative recounts
    how the policy was bought. The ordering is asserted in the tests.
    """
    head = text[:2500]
    if _CLAIM.search(head):
        return CLAIM
    if _PRICING.search(head):
        return PRICING
    if _SALES.search(head):
        return SALES
    if _ADMIN.search(head):
        return ADMIN
    return OTHER


def is_benchmarkable(text: str) -> bool:
    """Does this case belong in a claim-decision benchmark?"""
    return complaint_type(text) == CLAIM
