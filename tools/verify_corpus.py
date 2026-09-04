"""Re-hash every cached decision against the digest recorded in the corpus.

    python -m tools.verify_corpus

The repository ships metadata, not the ombudsman's PDFs. That is only a safe
trade if the fetched bytes can be proved identical to the ones the published
numbers were computed from, which is what this checks.

Exit code is non-zero on any mismatch, so it can gate a run.
"""
from __future__ import annotations

import json

from bench.dataset import verify_hashes


def main() -> int:
    r = verify_hashes()
    print(json.dumps(r, indent=2))
    if r["mismatch"]:
        print(f"\n{r['mismatch']} cached file(s) do not match the recorded hash: "
              f"{', '.join(r['bad'][:10])}")
        return 1
    if r["missing"]:
        print(f"\n{r['missing']} of {r['rows']} decisions are not cached — "
              f"run `python -m corpus.build` to fetch them.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
