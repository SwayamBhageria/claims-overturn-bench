"""Load the corpus: metadata from the repo, text from the local cache.

`data/corpus.jsonl` is committed and holds one row per decision — reference,
date, respondent, outcome, product, complaint type, and the SHA-256 of the
source PDF. The decision text is not committed. It is fetched once by
`corpus.build` into `.cache/pdfs/` and re-derived here.

That split is deliberate. The decisions are the ombudsman's publications,
already public at a stable URL; a hash and a fetcher reproduce them byte for
byte, prove nothing was edited on the way through, and keep the repository to
metadata. `verify_hashes()`, exposed as `python -m tools.verify_corpus`, is the check
that this actually holds.

`load_cases()` returns only what the benchmark is entitled to use: the case as
a handler would have held it, with the reasoning and the outcome kept out of
`text`. Anything that fails the leak check is dropped here as well as at build
time, so a stale cache cannot reintroduce an answer into an input.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

from corpus import classify, sections

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / ".cache" / "pdfs"
CORPUS = ROOT / "data" / "corpus.jsonl"


@dataclass
class Case:
    drn: str
    date: str | None
    business: str | None
    upheld: bool
    product: str
    complaint_type: str
    url: str
    text: str                      # complaint + what happened, no verdict
    text_with_investigator: str    # the same, plus the adjudicator's view
    has_investigator_view: bool
    reasoning: str                 # held out; used only to tag grounds
    outcome_text: str              # held out
    grounds: list[str] = field(default_factory=list)

    @property
    def family(self) -> str:
        from bench import grounds as g
        return g.family(self.grounds)


def read_rows(path: Path = CORPUS) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def load_cases(*, claims_only: bool = True, path: Path = CORPUS,
               require_text: bool = True) -> list[Case]:
    """Corpus rows joined to their cached text.

    `claims_only` keeps the benchmark to complaints that are actually about a
    claim decision. Roughly a third of published insurance decisions are about
    a premium rise, a mis-sale or a cancellation; scoring a claims model on
    those measures nothing anyone asked for. The excluded share is reported
    rather than quietly dropped — see `bench.report`.
    """
    from bench import grounds as g

    out: list[Case] = []
    for row in read_rows(path):
        if claims_only and row.get("complaint_type") != classify.CLAIM:
            continue
        pdf = CACHE / f"{row['drn']}.pdf"
        if not pdf.exists():
            if require_text:
                continue
            raise FileNotFoundError(pdf)
        try:
            text = sections.extract_text(pdf)
            dec = sections.split(text, row["drn"])
        except Exception:                      # noqa: BLE001 - a stale cache entry
            continue
        if sections.leak_terms(dec.prompt_input):
            continue

        out.append(Case(
            drn=row["drn"], date=row.get("date"), business=row.get("business"),
            upheld=bool(row["upheld"]), product=row.get("product", "unknown"),
            complaint_type=row.get("complaint_type", "unknown"),
            url=row["pdf_url"],
            text=dec.prompt_input,
            text_with_investigator=dec.prompt_input_with_investigator,
            has_investigator_view=bool(dec.investigator_view),
            reasoning=dec.reasoning,
            outcome_text=dec.outcome_text,
            grounds=g.tag(dec.reasoning) if row["upheld"] else [],
        ))
    return out


def verify_hashes(path: Path = CORPUS) -> dict:
    """Re-hash every cached PDF against the recorded digest."""
    rows = read_rows(path)
    ok = missing = mismatch = 0
    bad: list[str] = []
    for row in rows:
        pdf = CACHE / f"{row['drn']}.pdf"
        if not pdf.exists():
            missing += 1
            continue
        digest = hashlib.sha256(pdf.read_bytes()).hexdigest()
        if digest == row["pdf_sha256"]:
            ok += 1
        else:
            mismatch += 1
            bad.append(row["drn"])
    return {"rows": len(rows), "ok": ok, "missing": missing,
            "mismatch": mismatch, "bad": bad}
