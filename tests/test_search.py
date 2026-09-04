"""Search-client behaviour, against a fragment of a real result page.

The fixture is verbatim markup from the ombudsman's decision search
(`fixtures/search_results.html`), so if the page shape changes these tests
fail rather than the harvest silently returning fewer decisions.
"""
from pathlib import Path

import pytest

from corpus import search

FIXTURE = (Path(__file__).parent / "fixtures" / "search_results.html").read_text()


def test_parses_reference_date_outcome_and_respondent():
    total, hits = search.parse_results(FIXTURE, query="travel insurance")
    assert total == 588
    assert len(hits) >= 2

    first = hits[0]
    assert first.drn == "DRN-6476059"
    assert first.date == "21 Jul 2026"
    assert first.upheld is True
    assert first.business == "Chubb European Group SE"
    assert first.query == "travel insurance"

    second = hits[1]
    assert second.drn == "DRN-6241966"
    assert second.upheld is False
    assert second.business == "Aviva Insurance Limited"


def test_unreadable_page_raises_rather_than_returning_nothing():
    # A scraper that returns [] here reports a plausible nothing and every
    # rate downstream quietly shrinks.
    with pytest.raises(search.SearchError):
        search.parse_results("<html><body>maintenance</body></html>")


def test_url_carries_every_filter():
    url = search._url("travel insurance claim", start=20, upheld=True,
                      date_from="2025-01-01", date_to="2025-06-30")
    assert "Keyword=travel+insurance+claim" in url
    assert "IndustrySectorID%5B3%5D=3" in url
    assert "IsUpheld%5B1%5D=1" in url
    assert "DateFrom=2025-01-01" in url
    assert "DateTo=2025-06-30" in url
    assert "Start=20" in url


def test_outcome_filter_is_omitted_when_not_requested():
    url = search._url("x", 0, None, "2025-01-01", "2025-06-30")
    assert "IsUpheld" not in url


def test_pagination_stops_when_a_page_repeats_itself(monkeypatch):
    """An endpoint that ignores an out-of-range Start and re-serves page one
    must terminate the loop, not spin collecting duplicates."""
    calls = {"n": 0}

    class FakeResponse:
        text = FIXTURE

        def raise_for_status(self):
            return None

    class FakeSession:
        def get(self, url, timeout=0):
            calls["n"] += 1
            if calls["n"] > 20:
                raise AssertionError("pagination did not terminate")
            return FakeResponse()

    hits = search.search("travel", date_from="2025-01-01", date_to="2025-06-30",
                         limit=100, session=FakeSession(), delay=0)
    # The fixture always returns the same two references, so the second page
    # is entirely duplicates and the loop must stop there.
    assert len(hits) == 2
    assert calls["n"] == 2
