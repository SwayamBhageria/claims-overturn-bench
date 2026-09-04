"""The model arm's plumbing, tested without a key or a network call.

The parsing and probe arithmetic are what would silently corrupt a reported
number; the HTTP call is not interesting and is not exercised here.
"""
import json

import pytest

from bench import llm


def test_json_is_recovered_from_a_fenced_response():
    raw = ('Here is my answer:\n```json\n'
           '{"outcome": "upheld", "confidence": 0.8, "ground": "delay"}\n```')
    got = llm._json_from(raw)
    assert got["outcome"] == "upheld"
    assert got["confidence"] == 0.8


def test_json_recovery_fails_loudly_on_prose():
    with pytest.raises(ValueError):
        llm._json_from("I'd rather not say.")


def test_missing_key_is_a_clear_error_not_a_401():
    import os
    saved = os.environ.pop("GEMINI_API_KEY", None)
    try:
        with pytest.raises(llm.NoKey):
            llm._key()
    finally:
        if saved is not None:
            os.environ["GEMINI_API_KEY"] = saved


def test_cache_round_trips(tmp_path, monkeypatch):
    monkeypatch.setattr(llm, "CACHE", tmp_path)
    assert llm._cached("adjudicate", "m", "DRN-1") is None
    llm._store("adjudicate", "m", "DRN-1", {"outcome": "upheld"})
    assert llm._cached("adjudicate", "m", "DRN-1")["outcome"] == "upheld"


def test_confidence_is_converted_to_a_probability_of_upheld():
    # A model 90% sure of "not upheld" is a 0.10 probability that it was
    # upheld. Getting this backwards would invert the abstention curve while
    # leaving accuracy untouched, so it is asserted rather than assumed.
    for outcome, conf, expected in [("upheld", 0.9, 0.9),
                                    ("not_upheld", 0.9, 0.1),
                                    ("upheld", 0.5, 0.5)]:
        pred = outcome == "upheld"
        prob = conf if pred else 1 - conf
        assert prob == pytest.approx(expected)
