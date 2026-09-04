"""Client for the Financial Ombudsman Service published-decision search.

The search is a plain GET form, so every field it exposes is a filter we can
use directly: keyword, industry sector, date range, and — the one that makes a
benchmark possible at all — `IsUpheld`, the outcome the ombudsman reached.

Two things about this endpoint are load-bearing and easy to get wrong.

`Keyword` is full-text, not a product taxonomy. Searching "motor insurance"
returns every decision containing those words, including travel decisions that
mention a hire car. So keyword is used here to *retrieve* candidates and never
to *count* them; the product label comes from the decision text itself, in
`corpus.classify`. The result counts this module returns are counts of
documents matching a phrase, which is a different quantity from the number of
decisions about a product, and nothing downstream may confuse the two.

The result listing carries the outcome, the respondent business and the date
next to each reference, so class balance and per-insurer counts are readable
without fetching a single PDF.
"""
from __future__ import annotations

import html
import re
import time
import urllib.parse
from dataclasses import dataclass, asdict

import requests

BASE = ("https://www.financial-ombudsman.org.uk/businesses/resolving-complaint"
        "/ombudsman-decisions/search")
PDF = "https://www.financial-ombudsman.org.uk/decision/{drn}.pdf"

# A browser UA. robots.txt (fetched 2026-09-04) disallows only four PDF forms
# and permits everything else, but the WAF rejects a default python UA.
UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36")

SECTOR_INSURANCE = 3          # "Insurance (excluding PPI)"
PAGE_SIZE = 10                # fixed by the site
DELAY = 1.0                   # seconds between requests, deliberately polite

_ITEM = re.compile(r"Decision Reference (DRN-\d+)(.*?)(?=Decision Reference DRN-|\Z)", re.S)
_TOTAL = re.compile(r"([\d,]+)\s+results")
_DATE = re.compile(r"\d{1,2} \w{3} \d{4}")
_SCRIPT = re.compile(r"<script.*?</script>", re.S)
_TAG = re.compile(r"<[^>]+>")


class SearchError(RuntimeError):
    """Raised when the search returns something we cannot read.

    Deliberately loud. A scraper that returns an empty list when the page
    shape changes reports a plausible nothing and every downstream number
    silently shrinks, so the failure has to be an exception rather than [].
    """


@dataclass(frozen=True)
class Hit:
    """One row of the result listing. No PDF fetched yet."""
    drn: str
    date: str | None
    upheld: bool | None
    business: str | None
    query: str

    def as_dict(self) -> dict:
        return asdict(self)


def _session() -> requests.Session:
    s = requests.Session()
    s.headers["User-Agent"] = UA
    return s


def _url(keyword: str, start: int, upheld: bool | None,
         date_from: str, date_to: str) -> str:
    q: list[tuple[str, str]] = [
        ("Keyword", keyword),
        (f"IndustrySectorID[{SECTOR_INSURANCE}]", str(SECTOR_INSURANCE)),
        ("DateFrom", date_from),
        ("DateTo", date_to),
        ("Sort", "date"),
        ("action_doSearchDecisions", "Search decisions"),
    ]
    if upheld is not None:
        flag = "1" if upheld else "0"
        q.append((f"IsUpheld[{flag}]", flag))
    if start:
        q.append(("Start", str(start)))
    return BASE + "?" + urllib.parse.urlencode(q)


def parse_results(page: str, query: str = "") -> tuple[int | None, list[Hit]]:
    """Parse one result page into (total_matching, hits).

    `total` is the site's own count for the query and is reported, not trusted
    as a product count — see the module docstring.
    """
    total = None
    m = _TOTAL.search(page)
    if m:
        total = int(m.group(1).replace(",", ""))

    body = _SCRIPT.sub("", page)
    hits: list[Hit] = []
    for item in _ITEM.finditer(body):
        drn, blob = item.group(1), item.group(2)
        flat = html.unescape(re.sub(r"\s+", " ", _TAG.sub("|", blob)))
        parts = [p.strip() for p in flat.split("|") if p.strip()]

        date = next((p for p in parts if _DATE.fullmatch(p)), None)
        upheld: bool | None = None
        for p in parts:
            if p == "Upheld":
                upheld = True
                break
            if p == "Not upheld":
                upheld = False
                break

        business = None
        if date and date in parts:
            i = parts.index(date)
            for p in parts[i + 1:i + 5]:
                if p not in ("Upheld", "Not upheld", "Insurance"):
                    business = p
                    break

        hits.append(Hit(drn=drn, date=date, upheld=upheld,
                        business=business, query=query))

    if total is None and not hits:
        raise SearchError("no result count and no result rows — page shape changed")
    return total, hits


def count(keyword: str, *, upheld: bool | None = None,
          date_from: str, date_to: str, session=None) -> int:
    """How many decisions the site reports for a query. One request."""
    s = session or _session()
    r = s.get(_url(keyword, 0, upheld, date_from, date_to), timeout=60)
    r.raise_for_status()
    total, _ = parse_results(r.text, keyword)
    if total is None:
        raise SearchError(f"no result count for {keyword!r}")
    return total


def search(keyword: str, *, upheld: bool | None = None,
           date_from: str, date_to: str, limit: int = 100,
           session=None, delay: float = DELAY) -> list[Hit]:
    """Page through the listing, newest first, up to `limit` hits.

    Stops on the site's own total, on an empty page, or when a page returns
    only references already seen — the last is the guard that matters, because
    an endpoint that ignores an out-of-range `Start` and re-serves page one
    would otherwise loop forever collecting duplicates.
    """
    s = session or _session()
    out: list[Hit] = []
    seen: set[str] = set()
    start = 0
    total: int | None = None

    while len(out) < limit:
        r = s.get(_url(keyword, start, upheld, date_from, date_to), timeout=60)
        r.raise_for_status()
        page_total, hits = parse_results(r.text, keyword)
        if total is None:
            total = page_total
        if not hits:
            break

        fresh = [h for h in hits if h.drn not in seen]
        if not fresh:
            break
        for h in fresh:
            seen.add(h.drn)
        out.extend(fresh)

        start += PAGE_SIZE
        if total is not None and start >= total:
            break
        time.sleep(delay)

    return out[:limit]
