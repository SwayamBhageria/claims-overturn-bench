"""Export a prebuilt precedent index so the checker runs straight after a clone.

    python -m tools.build_index

Without this, using the checker meant running `corpus.build` first: about an
hour of fetching before the tool does anything. That is a fine story for
reproducing the numbers and a bad one for someone who just wants to see what
the thing does, which is most people.

**What ships, and why it is not a redistribution of the decisions.** The index
is a TF-IDF matrix: per document, which vocabulary terms appear and with what
weight. Word order is gone, so the decision text cannot be reconstructed from
it. Alongside it goes the metadata already in `data/corpus.jsonl` — reference,
date, respondent, outcome, product, tagged grounds, and the ombudsman's own URL
for each case. Anyone wanting the text follows the links, which is the same
position the repository has taken throughout.

`checker.check` loads this when present and falls back to building from the
local corpus when it is not, so the two paths give the same answers.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from bench.baselines import PrecedentKNN
from bench.dataset import load_cases

ROOT = Path(__file__).resolve().parent.parent
MATRIX = ROOT / "data" / "index.npz"
META = ROOT / "data" / "index_meta.json"


def build() -> dict:
    cases = load_cases()
    if len(cases) < 100:
        raise SystemExit(f"only {len(cases)} cases — run `python -m corpus.build` first")

    model = PrecedentKNN(k=15).fit([c.text for c in cases], [c.upheld for c in cases])

    # The weak-match threshold is corpus-dependent, so it is computed once here
    # rather than recomputed on every CLI invocation.
    from checker.check import weak_threshold
    thr = weak_threshold(model, cases)

    sp.save_npz(MATRIX, model.X.astype(np.float32), compressed=True)

    vec = model.vec
    META.write_text(json.dumps({
        "note": ("Prebuilt precedent index. The matrix is TF-IDF weights over a "
                 "fixed vocabulary; word order is not retained and the decision "
                 "text cannot be reconstructed from it. Follow `url` for the "
                 "decision itself."),
        "n_cases": len(cases),
        "weak_threshold": thr,
        "vocabulary": {term: int(i) for term, i in vec.vocabulary_.items()},
        "idf": [float(x) for x in vec.idf_],
        "cases": [{"drn": c.drn, "date": c.date, "business": c.business,
                   "upheld": c.upheld, "product": c.product, "url": c.url,
                   "grounds": c.grounds} for c in cases],
    }))
    return {"cases": len(cases),
            "matrix_mb": MATRIX.stat().st_size / 1e6,
            "meta_mb": META.stat().st_size / 1e6,
            "weak_threshold": thr}


def main() -> int:
    r = build()
    print(f"wrote data/index.npz ({r['matrix_mb']:.2f} MB) and "
          f"data/index_meta.json ({r['meta_mb']:.2f} MB) "
          f"for {r['cases']:,} cases; weak threshold {r['weak_threshold']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
