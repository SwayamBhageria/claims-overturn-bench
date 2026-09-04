"""The optional model arm, and the probe that says whether to believe it.

Everything else in this repository runs with no API key. This does not, which
is why it is separate and why the headline results do not depend on it: a
finding that only exists behind someone's billing account is a weaker finding.

    export GEMINI_API_KEY=...
    python -m bench.llm --limit 400          # adjudicate
    python -m bench.llm --contamination 150  # probe first, ideally

Two things are done here.

**Adjudication.** Each case is presented as a handler would hold it and the
model is asked for an outcome, a confidence, and the ground it turns on. The
confidence is what makes the model comparable to the local baselines on the
abstention curve rather than only on accuracy.

**Contamination.** These decisions are public documents with stable references
and may well be in the model's training data. If so, an accuracy figure here
measures recall, not judgement. Two probes, both cheap:

- *reference-only*: give the model nothing but `DRN-6297755` and ask for the
  outcome. Anything meaningfully above the base rate, on references it cannot
  reason about, is memorisation.
- *continuation*: give the opening of `What happened` and ask for the next
  sentence. High verbatim overlap on a document it claims not to know is the
  same finding from the other side.

The probes are reported next to the accuracy, always. An uncontaminated null
result is worth as much here as a positive one, and neither is assumed.

Every response is cached under `results/llm_cache/`, so a re-run costs nothing
and anyone can audit exactly what the model returned rather than taking the
aggregate on trust.
"""
from __future__ import annotations

import argparse
import difflib
import json
import os
import random
import re
import time
from pathlib import Path

import requests

from bench import metrics
from bench.dataset import load_cases

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "results" / "llm_cache"
OUT = ROOT / "results" / "llm.json"

MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.1-flash-lite")
URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
DELAY = float(os.environ.get("LLM_DELAY", "1.2"))   # free-tier friendly

ADJUDICATE_PROMPT = """You are reviewing a UK general insurance complaint before it reaches the Financial Ombudsman Service.

Below are the facts of the case as the parties presented them. Decide, applying the FCA's ICOBS rules and what is fair and reasonable in the circumstances, whether the Financial Ombudsman would UPHOLD the complaint (the insurer got it wrong, in outcome or in handling) or NOT UPHOLD it.

Answer with JSON only:
{{"outcome": "upheld" | "not_upheld", "confidence": 0.0-1.0, "ground": one of ["policy_wording","misrepresentation","fraud","evidence","delay","communication","quantum","process","none"], "reason": "one sentence"}}

CASE
{case}
"""

REFERENCE_PROBE = """A Financial Ombudsman Service final decision has the reference {drn}.

Without being told anything else about it, state whether that decision upheld the complaint. If you do not know, say so — guessing is worse than admitting it.

Answer with JSON only:
{{"known": true | false, "outcome": "upheld" | "not_upheld" | "unknown"}}
"""

CONTINUATION_PROBE = """Continue the following extract from a published Financial Ombudsman Service decision with the sentence that follows it in the original document. Reproduce it as closely as you can. Output only the sentence.

{opening}
"""


class NoKey(RuntimeError):
    pass


def _key() -> str:
    k = os.environ.get("GEMINI_API_KEY", "").strip()
    if not k:
        raise NoKey(
            "GEMINI_API_KEY is not set. Every other result in this repository "
            "runs without it; only this module needs a key.")
    return k


def _call(prompt: str, session: requests.Session, model: str = MODEL,
          max_retries: int = 4) -> str:
    payload = {"contents": [{"parts": [{"text": prompt}]}],
               "generationConfig": {"temperature": 0.0, "maxOutputTokens": 400}}
    url = URL.format(model=model) + f"?key={_key()}"
    for attempt in range(max_retries):
        r = session.post(url, json=payload, timeout=90)
        if r.status_code == 429:
            wait = 2 ** attempt * 5
            print(f"    rate limited, sleeping {wait}s", flush=True)
            time.sleep(wait)
            continue
        r.raise_for_status()
        body = r.json()
        try:
            return body["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError) as exc:
            raise RuntimeError(f"unexpected response shape: {body}") from exc
    raise RuntimeError(f"gave up after {max_retries} attempts (rate limited)")


def _json_from(text: str) -> dict:
    """Models fence their JSON. Pull the first object out."""
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        raise ValueError(f"no JSON object in response: {text[:200]!r}")
    return json.loads(m.group(0))


def _cached(kind: str, model: str, drn: str) -> dict | None:
    p = CACHE / model / kind / f"{drn}.json"
    return json.loads(p.read_text()) if p.exists() else None


def _store(kind: str, model: str, drn: str, payload: dict) -> None:
    p = CACHE / model / kind / f"{drn}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=2))


def adjudicate(cases, model: str = MODEL, limit: int | None = None) -> dict:
    session = requests.Session()
    sample = cases[:limit] if limit else cases
    y_true: list[bool] = []
    y_pred: list[bool] = []
    y_prob: list[float] = []
    failures = 0

    for i, c in enumerate(sample, 1):
        hit = _cached("adjudicate", model, c.drn)
        if hit is None:
            try:
                raw = _call(ADJUDICATE_PROMPT.format(case=c.text), session, model)
                hit = _json_from(raw)
            except Exception as exc:                   # noqa: BLE001 - counted
                failures += 1
                print(f"  ! {c.drn}: {exc}", flush=True)
                time.sleep(DELAY)
                continue
            _store("adjudicate", model, c.drn, hit)
            time.sleep(DELAY)
        if i % 25 == 0:
            print(f"  {i}/{len(sample)}", flush=True)

        pred = str(hit.get("outcome", "")).lower() == "upheld"
        conf = float(hit.get("confidence", 0.5) or 0.5)
        y_true.append(c.upheld)
        y_pred.append(pred)
        # Confidence is stated for the model's own answer; convert it to a
        # probability that the complaint was upheld, so it is on the same
        # scale as every other predictor here.
        y_prob.append(conf if pred else 1 - conf)

    if not y_true:
        raise SystemExit("no usable model responses")

    sc = metrics.score(y_true, y_pred)
    return {
        "model": model,
        "n": len(y_true),
        "failures": failures,
        "scores": sc.as_dict(),
        "abstention_curve": metrics.abstention_curve(y_true, y_prob, steps=10),
        "calibration": metrics.calibration(y_true, y_prob),
    }


def contamination(cases, model: str = MODEL, n: int = 150, seed: int = 3) -> dict:
    """Can the model produce outcomes it was never shown the facts for?"""
    session = requests.Session()
    rng = random.Random(seed)
    sample = rng.sample(cases, min(n, len(cases)))

    claimed = 0
    correct_when_claimed = 0
    guessed = 0
    correct_when_guessed = 0

    for c in sample:
        hit = _cached("reference", model, c.drn)
        if hit is None:
            try:
                hit = _json_from(_call(REFERENCE_PROBE.format(drn=c.drn), session, model))
            except Exception:                          # noqa: BLE001
                continue
            _store("reference", model, c.drn, hit)
            time.sleep(DELAY)

        stated = str(hit.get("outcome", "unknown")).lower()
        if stated not in ("upheld", "not_upheld"):
            continue
        pred = stated == "upheld"
        if hit.get("known"):
            claimed += 1
            correct_when_claimed += int(pred == c.upheld)
        else:
            guessed += 1
            correct_when_guessed += int(pred == c.upheld)

    # Continuation overlap, on a smaller sample — it costs more tokens.
    overlaps: list[float] = []
    for c in sample[:30]:
        hit = _cached("continuation", model, c.drn)
        opening = " ".join(c.text.split()[:80])
        if hit is None:
            try:
                hit = {"text": _call(CONTINUATION_PROBE.format(opening=opening),
                                     session, model)}
            except Exception:                          # noqa: BLE001
                continue
            _store("continuation", model, c.drn, hit)
            time.sleep(DELAY)
        rest = " ".join(c.text.split()[80:160])
        if rest:
            overlaps.append(difflib.SequenceMatcher(
                None, hit["text"].lower(), rest.lower()).ratio())

    base = sum(1 for c in sample if c.upheld) / len(sample)
    lo, hi = metrics.wilson(correct_when_claimed, claimed) if claimed else (None, None)
    return {
        "model": model,
        "sampled": len(sample),
        "base_rate": base,
        "claimed_to_know": claimed,
        "accuracy_when_claiming_to_know": (correct_when_claimed / claimed) if claimed else None,
        "accuracy_when_claiming_ci95": [lo, hi],
        "admitted_not_knowing_but_answered": guessed,
        "accuracy_when_guessing": (correct_when_guessed / guessed) if guessed else None,
        "continuation_overlap_mean": (sum(overlaps) / len(overlaps)) if overlaps else None,
        "continuation_overlap_max": max(overlaps) if overlaps else None,
        "reading": ("Accuracy from the reference alone that exceeds the base "
                    "rate is memorisation, and any accuracy figure for this "
                    "model on this corpus should be read as an upper bound."),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--contamination", type=int, default=0)
    ap.add_argument("--model", default=MODEL)
    a = ap.parse_args()

    cases = load_cases()
    if not cases:
        raise SystemExit("no corpus — run `python -m corpus.build` first")

    out = json.loads(OUT.read_text()) if OUT.exists() else {}
    if a.contamination:
        out["contamination"] = contamination(cases, a.model, a.contamination)
        print(json.dumps(out["contamination"], indent=2))
    if a.limit is not None or not a.contamination:
        out["adjudication"] = adjudicate(cases, a.model, a.limit)
        print(json.dumps(out["adjudication"]["scores"], indent=2))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2, default=float))
    print(f"wrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
