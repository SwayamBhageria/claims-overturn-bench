"""Check a written claim against `results/`, so prose cannot outrun the data.

    python -m tools.verify_claims claims.json

`claims.json` is a list of assertions, each naming a path into a results file
and the value the prose states:

    [
      {"text": "the insurer loses 41.1% of the time",
       "file": "uphold_rates.json",
       "path": "by_phrase.declined the claim.rate",
       "expect": 0.411, "tolerance": 0.0005, "format": "share"},
      {"text": "1,232 benchmark cases",
       "file": "analysis.json",
       "path": "composition.benchmark_cases",
       "expect": 1232}
    ]

The README's own tables are generated from the same files by `bench.report`,
so this exists for everything written *outside* the README — an email, a post,
a summary — where a figure can be typed once, go stale two commits later, and
never fail anything.

There is a second check worth as much as the first and it is not automatable
here: whether the artefact actually displays what the prose quotes. A number
can be correct in `results/analysis.json` and absent from the page a reader
lands on. `--require-in-readme` covers the mechanical part of that by
requiring each claim's figure to appear in the rendered README.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
README = ROOT / "README.md"


def dig(data, path: str):
    """Walk a dotted path. Segments may contain dots if they name a key."""
    cur = data
    rest = path
    while rest:
        if isinstance(cur, list):
            head, _, rest = rest.partition(".")
            cur = cur[int(head)]
            continue
        if not isinstance(cur, dict):
            raise KeyError(f"cannot descend into {type(cur).__name__} at {rest!r}")
        # Longest matching key first, so "by_phrase.declined the claim.rate"
        # resolves even though the key itself contains spaces and no dots.
        for key in sorted(cur, key=len, reverse=True):
            if rest == key:
                return cur[key]
            if rest.startswith(key + "."):
                cur = cur[key]
                rest = rest[len(key) + 1:]
                break
        else:
            raise KeyError(f"no key matching {rest!r}; have {sorted(cur)[:8]}")
    return cur


def _fmt(value, style: str | None) -> list[str]:
    """Renderings of a value that might plausibly appear in prose."""
    out = []
    if isinstance(value, (int, float)):
        if style == "share":
            out += [f"{value * 100:.1f}%", f"{value * 100:.0f}%"]
        if isinstance(value, int) or float(value).is_integer():
            out += [f"{int(value):,}", str(int(value))]
        else:
            out += [f"{value:.1f}", f"{value:.2f}"]
    else:
        out.append(str(value))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("claims", type=Path)
    ap.add_argument("--require-in-readme", action="store_true")
    a = ap.parse_args()

    claims = json.loads(a.claims.read_text())
    readme = README.read_text() if README.exists() else ""
    cache: dict[str, dict] = {}
    failures = 0

    for c in claims:
        name = c["file"]
        if name not in cache:
            p = RESULTS / name
            if not p.exists():
                print(f"FAIL  {c['text']!r}\n      no results file {name}")
                failures += 1
                continue
            cache[name] = json.loads(p.read_text())

        try:
            actual = dig(cache[name], c["path"])
        except (KeyError, IndexError, ValueError) as exc:
            print(f"FAIL  {c['text']!r}\n      {exc}")
            failures += 1
            continue

        expect = c["expect"]
        tol = c.get("tolerance", 0)
        ok = (abs(float(actual) - float(expect)) <= tol
              if isinstance(expect, (int, float)) and isinstance(actual, (int, float))
              else actual == expect)
        if not ok:
            print(f"FAIL  {c['text']!r}\n      states {expect}, data says {actual}")
            failures += 1
            continue

        if a.require_in_readme:
            renderings = _fmt(actual, c.get("format"))
            if not any(r in readme for r in renderings):
                print(f"FAIL  {c['text']!r}\n      correct, but none of "
                      f"{renderings} appears in the README — the artefact does "
                      f"not show what the prose quotes")
                failures += 1
                continue

        print(f"ok    {c['text']!r}  ({actual})")

    print(f"\n{len(claims) - failures}/{len(claims)} claims verified")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
