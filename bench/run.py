"""Run every experiment and write `results/analysis.json`.

    python -m bench.run

Nothing in the README is typed by hand. `bench.report` reads this file and
rewrites the marked blocks, so a number in the prose and the number the code
produced cannot drift apart.

The experiments, in the order they answer the question:

1. **composition** — what the corpus is, including the share of published
   insurance decisions that are not about a claim at all.
2. **leakage** — proof that the inputs do not contain their own answers, and
   how many cases were dropped to keep that true.
3. **grounds** — what the ombudsman actually faults, split into the decision
   itself versus the handling of it. This is the finding the rest supports.
4. **cost** — the sterling awards attached to those grounds.
5. **prediction** — can the overturned cases be identified in advance, by
   models that need no API key, under a split where no insurer appears on both
   sides.
6. **investigator ablation** — how much of any accuracy is the model agreeing
   with the adjudicator's view that was already in the file.
7. **triage** — the abstention curve: what share of the book a system like
   this could take, at what accuracy, and how many overturns it still misses.
"""
from __future__ import annotations

import json
import statistics
from collections import Counter
from pathlib import Path

import numpy as np

from bench import baselines, grounds, metrics
from bench.dataset import CORPUS, load_cases, read_rows, verify_hashes
from corpus import classify, sections

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results" / "analysis.json"
N_SPLITS = 5
SEED = 0


def _as_date(value):
    """Parse the listing's "9 May 2025" into a date.

    Worth its own function because the obvious thing is wrong: these are
    strings, and min()/max() over them sorts alphabetically. A corpus running
    from January 2024 to August 2026 reported its range as "1 Dec 2024 to
    9 May 2025", which is what you get when "1" sorts before "9" and "Dec"
    before "May".
    """
    from datetime import datetime
    try:
        return datetime.strptime(value, "%d %b %Y").date()
    except (TypeError, ValueError):
        return None


def composition(rows: list[dict], cases, corpus_path=None) -> dict:
    types = Counter(r.get("complaint_type") for r in rows)
    n = len(rows)
    dates = sorted(d for d in (_as_date(r.get("date")) for r in rows) if d)
    return {
        "decisions_harvested": n,
        "date_min": dates[0].isoformat() if dates else None,
        "date_max": dates[-1].isoformat() if dates else None,
        "dates_parsed": len(dates),
        "by_complaint_type": dict(types.most_common()),
        "not_about_a_claim": n - types.get(classify.CLAIM, 0),
        "not_about_a_claim_share": (n - types.get(classify.CLAIM, 0)) / n if n else 0,
        "benchmark_cases": len(cases),
        "by_product": dict(Counter(c.product for c in cases).most_common()),
        "by_outcome": {"upheld": sum(1 for c in cases if c.upheld),
                       "not_upheld": sum(1 for c in cases if not c.upheld)},
        "by_respondent_top10": dict(Counter(c.business for c in cases).most_common(10)),
        "distinct_respondents": len({c.business for c in cases}),
        "input_words_median": (statistics.median(len(c.text.split()) for c in cases)
                               if cases else None),
        "hash_check": verify_hashes(corpus_path) if corpus_path else verify_hashes(),
    }


def leakage(cases) -> dict:
    """Every input is re-checked here, not just at build time."""
    leaked = [c.drn for c in cases if sections.leak_terms(c.text)]
    with_inv = sum(1 for c in cases if c.has_investigator_view)
    return {
        "cases_checked": len(cases),
        "inputs_containing_verdict_wording": len(leaked),
        "leaked_refs": leaked[:20],
        "carrying_investigator_view": with_inv,
        "carrying_investigator_view_share": with_inv / len(cases) if cases else 0,
    }


def grounds_analysis(cases) -> dict:
    upheld = [c for c in cases if c.upheld]
    tagged = [c.grounds for c in upheld]
    dist = grounds.distribution(tagged)

    fams = Counter(c.family for c in upheld)
    n = len(upheld)
    handling_touched = sum(1 for c in upheld
                           if grounds.HANDLING in
                           {grounds.GROUNDS[g][0] for g in c.grounds if g in grounds.GROUNDS})
    coverage_touched = sum(1 for c in upheld
                           if grounds.COVERAGE in
                           {grounds.GROUNDS[g][0] for g in c.grounds if g in grounds.GROUNDS})

    by_product: dict[str, dict] = {}
    for prod in sorted({c.product for c in upheld}):
        sub = [c.grounds for c in upheld if c.product == prod]
        if len(sub) >= 15:
            by_product[prod] = grounds.distribution(sub)

    # "Upheld" is two different events. Separating them is the point.
    remedies = [grounds.remedy_text(c.reasoning, c.outcome_text) for c in upheld]
    disturbed = sum(1 for r in remedies if grounds.claim_decision_disturbed(r))
    compensation_only = sum(1 for r in remedies if grounds.compensation_only(r))
    stood = n - disturbed

    return {
        "upheld_cases": n,
        "claim_decision_disturbed": disturbed,
        "claim_decision_stood": stood,
        "claim_decision_stood_share": stood / n if n else 0,
        "compensation_only": compensation_only,
        **dist,
        "families": dict(fams.most_common()),
        "handling_touched": handling_touched,
        "handling_touched_share": handling_touched / n if n else 0,
        "coverage_touched": coverage_touched,
        "coverage_touched_share": coverage_touched / n if n else 0,
        "tagged_share": 1 - (dist["untagged"] / n) if n else 0,
        "by_product": by_product,
    }


def cost(cases) -> dict:
    """Sterling awarded, from the remedy section of upheld decisions."""
    per_case: list[float] = []
    for c in cases:
        if not c.upheld:
            continue
        sums = grounds.awards(grounds.remedy_text(c.reasoning, c.outcome_text))
        if sums:
            per_case.append(max(sums))
    if not per_case:
        return {"cases_with_award": 0}
    return {
        "cases_with_award": len(per_case),
        "share_of_upheld_with_award": len(per_case) / sum(1 for c in cases if c.upheld),
        "median_award_gbp": statistics.median(per_case),
        "mean_award_gbp": round(statistics.mean(per_case), 2),
        "p90_award_gbp": round(float(np.percentile(per_case, 90)), 2),
        "max_award_gbp": max(per_case),
        # The award is on top of the claim, and on top of the case fee the
        # respondent pays whatever the outcome. Fee source cited in the README.
        "fos_case_fee_gbp_2026_27": 680,
    }


def fold_health(y: list[bool], folds) -> dict:
    """Per-fold class balance, and whether any fold is degenerate.

    Grouping by respondent can produce a training fold containing only one
    outcome — a small firm that appears three times and lost all three, landing
    whole in one bucket. Every model then predicts that class on the test fold
    and the pooled scores look strange for a reason no metric explains. This
    surfaces it instead of leaving it to be discovered from a confusing table.
    """
    rows = []
    for i, (train, test) in enumerate(folds):
        tr = [y[j] for j in train]
        te = [y[j] for j in test]
        rows.append({
            "fold": i,
            "train_n": len(tr), "train_upheld": sum(tr),
            "test_n": len(te), "test_upheld": sum(te),
            "train_single_class": len(set(tr)) < 2,
            "test_single_class": len(set(te)) < 2,
        })
    return {"folds": rows,
            "degenerate_folds": sum(1 for r in rows
                                    if r["train_single_class"] or r["test_single_class"])}


def _cv(cases, texts: list[str], folds) -> dict:
    y = [c.upheld for c in cases]
    out: dict[str, dict] = {}
    preds: dict[str, list[bool]] = {}
    probs: dict[str, list[float]] = {}
    order: list[int] = []
    health = fold_health(y, folds)

    for cls in baselines.ALL:
        yp_all: list[bool] = []
        pr_all: list[float] = []
        idx_all: list[int] = []
        for train, test in folds:
            m = cls().fit([texts[i] for i in train], [y[i] for i in train])
            p = m.predict_proba([texts[i] for i in test])
            pr_all.extend(float(x) for x in p)
            yp_all.extend(bool(x >= 0.5) for x in p)
            idx_all.extend(test)
        order = idx_all
        yt = [y[i] for i in idx_all]
        sc = metrics.score(yt, yp_all)
        lo, hi = metrics.wilson(sc.tp + sc.tn, sc.n)
        out[cls.name] = {**sc.as_dict(), "accuracy_ci95": [lo, hi]}
        preds[cls.name] = yp_all
        probs[cls.name] = pr_all

    yt = [y[i] for i in order]
    comparisons = {}
    names = [c.name for c in baselines.ALL]
    for a in names:
        for b in names:
            if a < b:
                comparisons[f"{a}_vs_{b}"] = metrics.mcnemar_exact(yt, preds[a], preds[b])
    return {"scores": out, "mcnemar": comparisons, "fold_health": health,
            "_probs": probs, "_y": yt, "_order": order}


def prediction(cases) -> dict:
    texts = [c.text for c in cases]
    groups = [c.business or "" for c in cases]
    grouped = baselines.grouped_folds(groups, N_SPLITS, SEED)
    random = baselines.random_folds(len(cases), N_SPLITS, SEED)

    g = _cv(cases, texts, grouped)
    r = _cv(cases, texts, random)

    # What the linear model keys on. If these are insurer names, it is
    # learning the defendant rather than the conduct.
    lr = baselines.TfidfLR().fit(texts, [c.upheld for c in cases])
    return {
        "grouped_by_respondent": {k: v for k, v in g.items() if not k.startswith("_")},
        "random_split": {k: v for k, v in r.items() if not k.startswith("_")},
        "top_upheld_terms": lr.top_terms(25),
        "_grouped_probs": g["_probs"],
        "_grouped_y": g["_y"],
    }


def investigator_ablation(cases) -> dict:
    """Does the model just agree with the adjudicator already on the file?

    Scored on the same cases both ways, so the comparison is paired.
    """
    sub = [c for c in cases if c.has_investigator_view]
    if len(sub) < 50:
        return {"cases": len(sub), "note": "too few cases to test"}

    groups = [c.business or "" for c in sub]
    folds = baselines.grouped_folds(groups, N_SPLITS, SEED)
    without = _cv(sub, [c.text for c in sub], folds)
    with_ = _cv(sub, [c.text_with_investigator for c in sub], folds)

    name = baselines.TfidfLR.name
    delta = (with_["scores"][name]["balanced_accuracy"]
             - without["scores"][name]["balanced_accuracy"])
    return {
        "cases": len(sub),
        "share_of_corpus": len(sub) / len(cases),
        "without_investigator_view": {k: v for k, v in without.items()
                                      if not k.startswith("_")},
        "with_investigator_view": {k: v for k, v in with_.items()
                                   if not k.startswith("_")},
        "balanced_accuracy_delta_tfidf_lr": delta,
        "mcnemar_with_vs_without_tfidf_lr": metrics.mcnemar_exact(
            without["_y"], without["_probs"][name] and
            [p >= 0.5 for p in without["_probs"][name]],
            [p >= 0.5 for p in with_["_probs"][name]]),
    }


def triage(pred: dict) -> dict:
    y = pred["_grouped_y"]
    out = {}
    for name in ("tfidf_lr", "precedent_knn", "precedent_lsa"):
        probs = pred["_grouped_probs"][name]
        out[name] = {
            "curve": metrics.abstention_curve(y, probs, steps=10),
            "calibration": metrics.calibration(y, probs),
        }
    return out


def product_mix(cases) -> dict:
    """Composition of the benchmark by product line.

    Deliberately not an uphold *rate*. The corpus is sampled to equal numbers
    of upheld and not-upheld decisions, so any rate read off it is a reading of
    our own sampling. The unbiased rates come from search result totals and
    live in `bench.rates` / `results/uphold_rates.json`.
    """
    out = {}
    for prod in sorted({c.product for c in cases}):
        sub = [c for c in cases if c.product == prod]
        if len(sub) < 20:
            continue
        out[prod] = {"n": len(sub),
                     "upheld_in_sample": sum(1 for c in sub if c.upheld),
                     "note": "sampled, not a rate"}
    return out


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", type=Path, default=CORPUS,
                    help="corpus JSONL to analyse (default: data/corpus.jsonl)")
    ap.add_argument("--out", type=Path, default=RESULTS)
    a = ap.parse_args()

    rows = read_rows(a.corpus)
    if not rows:
        raise SystemExit(f"no corpus at {a.corpus} — run `python -m corpus.build`")
    cases = load_cases(path=a.corpus)
    if len(cases) < 100:
        raise SystemExit(f"only {len(cases)} usable cases; expected hundreds. "
                         "Check .cache/pdfs is populated.")

    print(f"{len(rows)} decisions, {len(cases)} benchmark cases", flush=True)
    pred = prediction(cases)

    out = {
        "corpus_file": str(a.corpus),
        "n_splits": N_SPLITS,
        "seed": SEED,
        "composition": composition(rows, cases, a.corpus),
        "leakage": leakage(cases),
        "grounds": grounds_analysis(cases),
        "cost": cost(cases),
        "prediction": {k: v for k, v in pred.items() if not k.startswith("_")},
        "investigator_ablation": investigator_ablation(cases),
        "triage": triage(pred),
        "product_mix": product_mix(cases),
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=2, default=float))
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
