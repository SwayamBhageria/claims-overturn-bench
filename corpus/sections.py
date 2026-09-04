"""Split a published decision into the part a claims team would have had, and
the part only the ombudsman had.

Every decision follows the same skeleton:

    The complaint                 <- who is complaining, about whom, about what
    What happened                 <- the facts, and what the insurer did
    What I've decided - and why   <- the reasoning
    [Putting things right]        <- remedy
    My final decision             <- the outcome

The cut is made at the first reasoning heading. Everything above it is the
**input**: roughly the file a handler holds when deciding. Everything from it
down is **held out**, because it contains the answer.

Two hazards, both handled here rather than left as caveats.

1. `What happened` routinely ends with the adjudicator's provisional view
   ("Our investigator thought Aviva had acted reasonably"). That is a prior
   adjudication of the same case, not a fact of the claim. It is located and
   returned separately so the benchmark can run with and without it, because
   a model that merely echoes the last opinion it was shown would look
   accurate here and fail in production, where no such opinion exists.

2. The verdict wording itself must never survive the cut. `leak_terms` finds
   any of it, and the corpus builder drops the case rather than repairing it.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, asdict

# The reasoning starts at whichever of these appears first. "My provisional
# decision" is included because a small number of decisions reproduce the
# provisional one before the final findings.
_REASONING = re.compile(
    r"(?mi)^[ \t]*("
    r"What I(?:’|')ve (?:provisionally )?decided[^\n]*"
    r"|My (?:provisional )?findings[^\n]*"
    r"|My provisional decision[^\n]*"
    r")[ \t]*$"
)

_COMPLAINT = re.compile(r"(?mi)^[ \t]*The complaint[ \t]*$")
_WHAT_HAPPENED = re.compile(r"(?mi)^[ \t]*(?:What happened|Background)[^\n]*$")
_FINAL = re.compile(r"(?mi)^[ \t]*My (?:final )?decision[ \t]*$")

# The adjudicator's provisional view. FOS calls the first-stage decision maker
# an "investigator" and, in older decisions, an "adjudicator".
#
# This has to work sentence by sentence, and the first version did not — it
# removed the *line* containing the word, and PDF text wraps, so it deleted
# "Our investigator looked at the complaint." and left the sentence that
# followed: "She recommended that D&G settle the claim in line with the
# remaining policy terms, and pay Mr S £100 compensation." That recommendation
# is the outcome, in the input, in 56% of cases. It is the reason the strongest
# terms predicting "upheld" were `compensation`, `recommended` and `£100`.
#
# The distinction that matters: the adjudicator's recommendation is a prior
# adjudication and must go, but the *insurer's own* offer during the claim
# ("They offered Mrs A £250 compensation as a result") is claim history a
# handler would hold, and removing it would strip a real fact.
_INVESTIGATOR_SENTENCE = re.compile(
    # named, in either FOS vocabulary
    r"\b(?:our|the|an|of our)\s+(?:investigators?|adjudicators?)\b"
    # the same person, carried by pronoun into the following sentences
    r"|\b(?:she|he|they)\s+(?:also\s+)?(?:recommended|didn't think|did not think|"
    r"thought|concluded|wasn't persuaded|was not persuaded)\b"
    # the recommendation itself, whoever is named as making it
    r"|\brecommend\w*\b[^.]{0,80}?\b(?:pay|settle|accept|increase|reconsider|"
    r"compensat\w+|uphold)\b"
    # this service reaching a view, as opposed to the business doing something
    r"|\b(?:this|our) service\b[^.]{0,60}\b(?:thought|considered|recommended|"
    r"asked|told)\b",
    re.I)

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")

# Once the adjudicator has been introduced, the decision refers to them by
# pronoun for a sentence or two: "Our investigator looked at this. She
# recommended... She also said the business should pay £100 compensation."
# A bare pronoun alone is not enough to attribute, because the very next
# sentence is often the complainant — "She disagreed and asked for an
# ombudsman's decision" — so a continuation also has to be *about* what the
# business ought to do.
_PRONOUN_START = re.compile(r"^\s*(?:she|he|they)\b", re.I)

# A decision issued after a provisional one reproduces the provisional
# reasoning in its preamble, and that reasoning states the intended outcome:
# "I issued a provisional decision explaining that I was intending to uphold
# Mr S\'s complaint." That is the answer, above the cut, in about a fifth of
# cases. Sentences mentioning a provisional decision are held out with the
# adjudicator\'s view, and `leak_terms` carries the phrasings as a backstop so
# anything that gets past this drops the case rather than poisoning it.
_PROVISIONAL = re.compile(
    r"\bprovisional(?:ly)?\s+(?:decision|decided|findings|conclusions?|view)\b"
    r"|\bI provisionally\b|\bmy provisional\b", re.I)
_RECOMMENDING = re.compile(
    r"\b(?:should|ought to|must|to pay|pay|settle|accept|increase|reconsider|"
    r"compensat\w+|uphold|fair|reasonable|unfair|unreasonable)\b", re.I)

# Verdict wording. If any of this is above the cut, the input contains its own
# answer and the case is unusable.
_LEAKS = [
    re.compile(r"(?i)\bI (?:do not|don’t|don't) uphold\b"),
    re.compile(r"(?i)\bI uphold\b"),
    re.compile(r"(?i)\bI(?:’|')m (?:not )?upholding\b"),
    re.compile(r"(?i)\bmy final decision is\b"),
    re.compile(r"(?i)\bit follows that I\b"),
    # Provisional-decision wording. A final decision issued after a
    # provisional one quotes its own earlier intention, which is the outcome.
    re.compile(r"(?i)\b(?:intending|intend|minded|proposing) to (?:uphold|not uphold)\b"),
    re.compile(r"(?i)\bI (?:was|am) (?:intending|minded) to\b"),
    re.compile(r"(?i)\bprovisionally (?:uphold|decided to uphold|concluded)\b"),
]

_DRN = re.compile(r"\bDRN-\d+\b")


class SectionError(ValueError):
    """The decision does not have the expected skeleton."""


@dataclass(frozen=True)
class Decision:
    drn: str
    complaint: str          # "The complaint" — one or two sentences
    what_happened: str      # facts, investigator's view already removed
    investigator_view: str  # what was removed, kept for the ablation
    reasoning: str          # held out
    outcome_text: str       # held out

    @property
    def prompt_input(self) -> str:
        """What a model is shown: the case as a handler would hold it."""
        return f"The complaint\n{self.complaint}\n\nWhat happened\n{self.what_happened}".strip()

    @property
    def prompt_input_with_investigator(self) -> str:
        """The same case plus the adjudicator's provisional view."""
        base = self.prompt_input
        if not self.investigator_view:
            return base
        return f"{base}\n\n{self.investigator_view}".strip()

    def as_dict(self) -> dict:
        return asdict(self)


def leak_terms(text: str) -> list[str]:
    """Verdict wording present in `text`. Empty means the cut held."""
    return [m.group(0) for p in _LEAKS for m in p.finditer(text)]


def split(text: str, drn: str = "") -> Decision:
    """Split extracted PDF text into a `Decision`.

    Raises `SectionError` rather than returning a half-parsed object: a
    decision whose skeleton we cannot find is a decision whose input we cannot
    prove is answer-free.
    """
    if not drn:
        m = _DRN.search(text)
        drn = m.group(0) if m else ""

    cut = _REASONING.search(text)
    if not cut:
        raise SectionError(f"{drn or '?'}: no reasoning heading found")
    head, tail = text[:cut.start()], text[cut.start():]

    m_complaint = _COMPLAINT.search(head)
    m_happened = _WHAT_HAPPENED.search(head)
    if not m_complaint:
        raise SectionError(f"{drn or '?'}: no 'The complaint' heading")

    if m_happened and m_happened.start() > m_complaint.end():
        complaint = head[m_complaint.end():m_happened.start()]
        happened = head[m_happened.end():]
    else:
        # A minority of decisions run the facts on without the heading.
        complaint = head[m_complaint.end():]
        happened = ""

    happened, investigator = _strip_investigator(happened)

    m_final = _FINAL.search(tail)
    reasoning = tail[:m_final.start()] if m_final else tail
    outcome = tail[m_final.start():] if m_final else ""

    return Decision(
        drn=drn,
        complaint=_tidy(complaint),
        what_happened=_tidy(happened),
        investigator_view=_tidy(investigator),
        reasoning=_tidy(reasoning),
        outcome_text=_tidy(outcome),
    )


def _strip_investigator(text: str) -> tuple[str, str]:
    """Split the facts from the adjudicator's provisional view.

    Sentence by sentence, on whitespace-normalised text: the section arrives
    hard-wrapped from the PDF, so a line is not a unit of meaning here and
    treating it as one is what let the recommendation survive.
    """
    flat = re.sub(r"\s*\n\s*", " ", text)
    flat = re.sub(r"[ \t]+", " ", flat).strip()
    kept, removed = [], []
    in_view = False
    for sentence in _SENTENCE_SPLIT.split(flat):
        if _INVESTIGATOR_SENTENCE.search(sentence) or _PROVISIONAL.search(sentence):
            in_view = True
        elif in_view and _PRONOUN_START.match(sentence) and _RECOMMENDING.search(sentence):
            pass                      # still the adjudicator, by pronoun
        else:
            in_view = False
        (removed if in_view else kept).append(sentence)
    return " ".join(kept).strip(), " ".join(removed).strip()


def _tidy(s: str) -> str:
    s = s.replace("’", "'").replace("‘", "'")
    s = s.replace("“", '"').replace("”", '"')
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def extract_text(pdf_path) -> str:
    """PDF -> text. pypdf rather than a system binary, so a clone reproduces."""
    import pypdf
    reader = pypdf.PdfReader(str(pdf_path))
    return "\n".join(page.extract_text() or "" for page in reader.pages)
