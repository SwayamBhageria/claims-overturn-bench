"""README generation and claim verification.

These two are the guard against the failure that matters most once the work
leaves the repository: a figure that was true when it was typed and is not
true now. `bench.report` regenerates every table from the results file;
`tools.verify_claims` checks anything written outside the README against the
same files.
"""
import json

import pytest

from bench import report
from tools import verify_claims


def test_render_replaces_only_the_marked_block():
    readme = ("intro\n<!--AUTO:COST-->\nold table\n<!--/AUTO:COST-->\nouttro\n")
    analysis = {"cost": {"cases_with_award": 10, "share_of_upheld_with_award": 0.5,
                         "median_award_gbp": 300, "mean_award_gbp": 350.0,
                         "p90_award_gbp": 700.0, "max_award_gbp": 2000,
                         "fos_case_fee_gbp_2026_27": 680}}
    out = report.render(readme, analysis)
    assert out.startswith("intro\n")
    assert out.endswith("outtro\n")
    assert "old table" not in out
    assert "£680" in out
    assert "median compensation" in out


def test_render_is_idempotent():
    readme = "<!--AUTO:COST-->\nx\n<!--/AUTO:COST-->\n"
    analysis = {"cost": {"cases_with_award": 1, "share_of_upheld_with_award": 1.0,
                         "median_award_gbp": 100, "mean_award_gbp": 100.0,
                         "p90_award_gbp": 100.0, "max_award_gbp": 100,
                         "fos_case_fee_gbp_2026_27": 680}}
    once = report.render(readme, analysis)
    assert report.render(once, analysis) == once


def test_render_leaves_absent_blocks_alone():
    readme = "no markers here"
    assert report.render(readme, {"cost": {"cases_with_award": 0}}) == readme


def test_dig_walks_keys_that_contain_spaces():
    data = {"by_phrase": {"declined the claim": {"rate": 0.41}}}
    assert verify_claims.dig(data, "by_phrase.declined the claim.rate") == 0.41


def test_dig_reports_the_missing_key_rather_than_crashing():
    with pytest.raises(KeyError, match="no key matching"):
        verify_claims.dig({"a": 1}, "b.c")


def test_share_formatting_offers_the_renderings_prose_would_use():
    assert "41.1%" in verify_claims._fmt(0.411, "share")
    assert "1,232" in verify_claims._fmt(1232, None)
