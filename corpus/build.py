"""Build the corpus: search -> fetch -> split -> classify -> JSONL.

Run:  python -m corpus.build --limit-per-query 220

What is written to the repo is metadata and derived fields — reference, date,
respondent, outcome, product line, complaint type, word counts, and the SHA-256
of the source PDF. The decision texts themselves are cached under `.cache/`
and are not redistributed: they are the ombudsman's publications, they are
already public at a stable URL, and a hash plus a fetcher reproduces them
exactly. `python -m tools.verify_corpus` re-hashes every cached file
against the digest recorded here.

Failures are counted per stage and printed. A stage that silently drops
documents would shrink every rate downstream while still looking like a clean
run, so the build refuses to write a corpus if more than `MAX_LOSS` of the
fetched decisions fail to parse.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import sys
import time
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import requests

from corpus import classify, sections
from corpus.search import DELAY, PDF, UA, Hit, search

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / ".cache" / "pdfs"
DATA = ROOT / "data"
CORPUS = DATA / "corpus.jsonl"
QUERY_COUNTS = DATA / "query_counts.json"

MAX_LOSS = 0.10   # refuse to write a corpus that lost more than this to parsing

# Retrieval queries. Deliberately phrase-level and overlapping: they are nets,
# not categories. What a decision is about is decided later, from its text.
QUERIES = [
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


@dataclass
class Stats:
    hits: int = 0
    unique: int = 0
    fetched: int = 0
    fetch_failed: int = 0
    split_failed: int = 0
    leaked: int = 0
    written: int = 0

    def report(self) -> str:
        return (f"hits={self.hits} unique={self.unique} fetched={self.fetched} "
                f"fetch_failed={self.fetch_failed} split_failed={self.split_failed} "
                f"leaked={self.leaked} written={self.written}")


def fetch_pdf(drn: str, session: requests.Session, delay: float = DELAY,
              retries: int = 3) -> bytes:
    """Fetch one decision, with backoff.

    Retrying matters more than it looks. Over a run of 1,500 fetches a single
    read timeout is close to certain, and without a retry it silently costs a
    decision — the sort of loss that only shows up at batch scale and never in
    a test. This run lost exactly one that way before the retry existed.

    A 200 is also not enough on its own: the site answers some errors with an
    HTML page, so the magic bytes are checked before the file is kept.
    """
    path = CACHE / f"{drn}.pdf"
    if path.exists():
        return path.read_bytes()

    last: Exception | None = None
    for attempt in range(retries):
        try:
            r = session.get(PDF.format(drn=drn), timeout=90)
            r.raise_for_status()
            if not r.content.startswith(b"%PDF"):
                raise ValueError(
                    f"{drn}: response is not a PDF ({len(r.content)} bytes)")
            CACHE.mkdir(parents=True, exist_ok=True)
            path.write_bytes(r.content)
            time.sleep(delay)
            return r.content
        except Exception as exc:                      # noqa: BLE001 - retried
            last = exc
            if attempt < retries - 1:
                time.sleep(delay * (2 ** attempt))
    raise RuntimeError(f"{drn}: {retries} attempts failed: {last}") from last


def month_slices(date_from: str, date_to: str, months: int) -> list[tuple[str, str]]:
    """Split [date_from, date_to] into consecutive windows of `months`."""
    start = date.fromisoformat(date_from)
    end = date.fromisoformat(date_to)
    out: list[tuple[str, str]] = []
    cur = start
    while cur <= end:
        y, m = divmod((cur.year * 12 + cur.month - 1) + months, 12)
        nxt = date(y, m + 1, 1)
        stop = min(nxt - timedelta(days=1), end)
        out.append((cur.isoformat(), stop.isoformat()))
        cur = nxt
    return out


def collect_hits(date_from: str, date_to: str, limit_per_query: int,
                 slice_months: int, session: requests.Session
                 ) -> tuple[list[Hit], dict]:
    """Run every query, both outcome filters, in each time slice.

    Two sampling choices, both deliberate and both stated in the README.

    The outcome filters are run separately and at equal budget. Running
    unfiltered would inherit the site's own relevance ordering and, with it,
    whatever class balance that ordering happens to produce; asking for each
    outcome explicitly makes the class balance a stated choice rather than an
    accident. It also means the corpus balance is *not* an estimate of the real
    uphold rate — that is measured separately, from query counts, in
    `bench.rates`.

    The range is sampled in time slices rather than as one window, because the
    listing is date-sorted: a single query over three years returns only the
    most recent weeks, which would make every result a statement about the last
    month of the range and hide any drift.
    """
    by_drn: dict[str, Hit] = {}
    counts: dict[str, dict] = {}
    windows = month_slices(date_from, date_to, slice_months)
    per_window = max(limit_per_query // (2 * len(windows)), 5)
    print(f"{len(windows)} time slices, {per_window} hits per query/outcome/slice",
          flush=True)

    for w_from, w_to in windows:
        for q in QUERIES:
            key = f"{q}|{w_from}"
            counts[key] = {}
            for upheld in (True, False):
                hits = search(q, upheld=upheld, date_from=w_from, date_to=w_to,
                              limit=per_window, session=session)
                counts[key]["upheld" if upheld else "not_upheld"] = len(hits)
                for h in hits:
                    by_drn.setdefault(h.drn, h)
            print(f"  {w_from} {q!r:30} -> corpus {len(by_drn)}", flush=True)
    return list(by_drn.values()), counts


def build(date_from: str, date_to: str, limit_per_query: int,
          slice_months: int = 6) -> Stats:
    session = requests.Session()
    session.headers["User-Agent"] = UA
    st = Stats()

    print(f"searching {date_from} .. {date_to}", flush=True)
    hits, query_counts = collect_hits(date_from, date_to, limit_per_query,
                                      slice_months, session)
    st.hits = sum(sum(v.values()) for v in query_counts.values())
    st.unique = len(hits)

    DATA.mkdir(parents=True, exist_ok=True)
    QUERY_COUNTS.write_text(json.dumps(
        {"date_from": date_from, "date_to": date_to,
         "slice_months": slice_months, "per_query": query_counts},
        indent=2))

    rows = []
    for i, h in enumerate(hits, 1):
        if i % 50 == 0:
            print(f"  fetched {i}/{len(hits)} ({st.report()})", flush=True)
        try:
            blob = fetch_pdf(h.drn, session)
        except Exception as exc:                      # noqa: BLE001 - reported
            st.fetch_failed += 1
            print(f"  ! fetch {h.drn}: {exc}", file=sys.stderr)
            continue
        st.fetched += 1

        try:
            text = sections.extract_text(CACHE / f"{h.drn}.pdf")
            dec = sections.split(text, h.drn)
        except Exception as exc:                      # noqa: BLE001 - reported
            st.split_failed += 1
            print(f"  ! split {h.drn}: {exc}", file=sys.stderr)
            continue

        leaks = sections.leak_terms(dec.prompt_input)
        if leaks:
            st.leaked += 1
            continue

        rows.append({
            "drn": h.drn,
            "date": h.date,
            "business": h.business,
            "upheld": h.upheld,
            "query": h.query,
            "product": classify.product_line(text),
            "complaint_type": classify.complaint_type(text),
            "has_investigator_view": bool(dec.investigator_view),
            "input_words": len(dec.prompt_input.split()),
            "reasoning_words": len(dec.reasoning.split()),
            "pdf_sha256": hashlib.sha256(blob).hexdigest(),
            "pdf_url": PDF.format(drn=h.drn),
        })

    lost = (st.split_failed + st.fetch_failed) / max(st.unique, 1)
    if lost > MAX_LOSS:
        raise SystemExit(
            f"refusing to write: lost {lost:.1%} of {st.unique} decisions to "
            f"fetch/parse failures (limit {MAX_LOSS:.0%}). {st.report()}")

    with CORPUS.open("w") as fh:
        for row in sorted(rows, key=lambda r: r["drn"]):
            fh.write(json.dumps(row) + "\n")
    st.written = len(rows)

    print("\n" + st.report())
    print("by outcome:", collections.Counter(r["upheld"] for r in rows))
    print("by type:   ", collections.Counter(r["complaint_type"] for r in rows))
    print("by product:", collections.Counter(r["product"] for r in rows))
    return st


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--date-from", default="2024-01-01")
    ap.add_argument("--date-to", default="2026-08-31")
    ap.add_argument("--limit-per-query", type=int, default=200)
    ap.add_argument("--slice-months", type=int, default=6)
    a = ap.parse_args()
    build(a.date_from, a.date_to, a.limit_per_query, a.slice_months)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
