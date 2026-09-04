"""Uphold rates, measured where they are not biased by our own sampling.

The corpus is built by asking for upheld and not-upheld decisions in equal
numbers, which makes it useful for training and scoring and useless for
estimating how often complaints succeed. Its class balance is a choice we
made. Reading a rate off it would be reading back our own sampling.

The rate is therefore taken from a different place: the search's own result
totals. Asking for a phrase with `IsUpheld=1` and again with `IsUpheld=0` over
the same window returns two counts covering the same set of documents, and
their ratio is the uphold rate among published decisions matching that phrase.
Two requests, no scraping, nothing estimated.

Two limits, both stated wherever the numbers appear:

- **a phrase is not a product.** `Keyword` is full-text, so "motor insurance
  claim" also catches travel decisions about a hire car. These are rates for
  decisions *mentioning a phrase*, and are labelled that way.
- **a published decision is not a claim.** Decisions are the small tail of
  complaints that an ombudsman had to decide. This says which disputes are
  hardest to defend once they get that far — not how often claims go wrong.

    python -m bench.rates --date-from 2024-01-01 --date-to 2026-08-31
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import requests

from bench.metrics import wilson
from corpus.search import UA, count

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "uphold_rates.json"

PHRASES = [
    "travel insurance claim",
    "pet insurance claim",
    "gadget insurance claim",
    "mobile phone insurance claim",
    "motor insurance claim",
    "home insurance claim",
    "buildings insurance claim",
    "contents insurance claim",
    "warranty claim",
    "declined the claim",
]


def rates(date_from: str, date_to: str, phrases: list[str] | None = None) -> dict:
    session = requests.Session()
    session.headers["User-Agent"] = UA
    out: dict[str, dict] = {}
    for phrase in (phrases or PHRASES):
        up = count(phrase, upheld=True, date_from=date_from,
                   date_to=date_to, session=session)
        not_up = count(phrase, upheld=False, date_from=date_from,
                       date_to=date_to, session=session)
        total = up + not_up
        lo, hi = wilson(up, total) if total else (None, None)
        out[phrase] = {"upheld": up, "not_upheld": not_up, "total": total,
                       "rate": up / total if total else None, "ci95": [lo, hi]}
        print(f"{phrase!r:32} {up:5} / {total:5} = "
              f"{(up / total * 100) if total else 0:5.1f}%", flush=True)
    return {"date_from": date_from, "date_to": date_to,
            "note": ("rates over published decisions matching a full-text "
                     "phrase; a phrase is not a product and a published "
                     "decision is not a claim"),
            "by_phrase": out}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--date-from", default="2024-01-01")
    ap.add_argument("--date-to", default="2026-08-31")
    a = ap.parse_args()
    r = rates(a.date_from, a.date_to)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(r, indent=2))
    print(f"\nwrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
