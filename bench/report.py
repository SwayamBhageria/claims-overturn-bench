"""Rewrite the README's marked blocks from `results/analysis.json`.

    python -m bench.report            # rewrite
    python -m bench.report --check    # fail if the README is out of date

Every table and every figure quoted in the README sits between
`<!--AUTO:NAME-->` and `<!--/AUTO:NAME-->` and is generated here. Nothing about
the results is typed by hand, so the prose cannot drift from the run that
produced it. `--check` is the CI gate that enforces it.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
README = ROOT / "README.md"
RESULTS = ROOT / "results" / "analysis.json"


def _pct(x: float | None, dp: int = 1) -> str:
    return "—" if x is None else f"{100 * float(x):.{dp}f}%"


def block_composition(a: dict) -> str:
    c = a["composition"]
    t = c["by_complaint_type"]
    rows = [
        "| | |",
        "|---|---:|",
        f"| decisions harvested | {c['decisions_harvested']:,} |",
        f"| date range | {c['date_min']} – {c['date_max']} |",
        f"| distinct respondent firms | {c['distinct_respondents']:,} |",
        f"| **not about a claim** (premium, mis-sale, cancellation) | "
        f"**{c['not_about_a_claim']:,}  ({_pct(c['not_about_a_claim_share'])})** |",
        f"| benchmark cases (claim decisions only) | **{c['benchmark_cases']:,}** |",
        f"| upheld / not upheld | {c['by_outcome']['upheld']:,} / "
        f"{c['by_outcome']['not_upheld']:,} |",
        f"| median input length | {int(c['input_words_median'])} words |",
    ]
    types = " · ".join(f"{k} {v:,}" for k, v in t.items())
    prods = " · ".join(f"{k} {v:,}" for k, v in c["by_product"].items())
    return ("\n".join(rows) + f"\n\nComplaint types: {types}."
            f"\n\nProduct lines: {prods}.")


def block_grounds(a: dict) -> str:
    g = a["grounds"]
    lines = ["| ground | family | upheld decisions | share of upheld |",
             "|---|---|---:|---:|"]
    from bench import grounds as G
    for name, n in g["grounds"].items():
        fam = G.GROUNDS[name][0] if name in G.GROUNDS else "?"
        lines.append(f"| {name} | {fam} | {n:,} | {_pct(n / g['upheld_cases'])} |")
    lines.append(f"| *(untagged)* | — | {g['untagged']:,} | "
                 f"{_pct(g['untagged'] / g['upheld_cases'])} |")
    fam = g["families"]
    summary = " · ".join(f"{k} {v:,}" for k, v in fam.items())
    tagged = g["upheld_cases"] - g["untagged"]
    return ("\n".join(lines) +
            f"\n\nBy family: {summary}."
            f"\n\n**{_pct(g['handling_touched_share'])} of all upheld decisions fault the "
            f"handling** ({g['handling_touched']:,} of {g['upheld_cases']:,}); "
            f"{_pct(g['coverage_touched_share'])} fault the coverage decision "
            f"({g['coverage_touched']:,}). The two overlap — a decision can be both.\n\n"
            f"Read those against the untagged row, not past it. Of the {tagged:,} "
            f"decisions the tagger does place, "
            f"**{_pct(g['handling_touched'] / tagged)} fault the handling** and "
            f"{_pct(g['coverage_touched'] / tagged)} the coverage decision. The "
            f"{_pct(g['untagged'] / g['upheld_cases'])} it places nowhere is the "
            f"tagger's recall problem, not evidence of a third kind of fault, and "
            f"it means these shares are a floor rather than an estimate.")


def block_prediction(a: dict) -> str:
    p = a["prediction"]
    lines = ["| model | split | accuracy | balanced acc. | recall on upheld | "
             "predicted-upheld rate |", "|---|---|---:|---:|---:|---:|"]
    for split_name, key in (("grouped by respondent", "grouped_by_respondent"),
                            ("random", "random_split")):
        for model, s in p[key]["scores"].items():
            lines.append(
                f"| {model} | {split_name} | {_pct(s['accuracy'])} | "
                f"{_pct(s['balanced_accuracy'])} | {_pct(s['recall_upheld'])} | "
                f"{_pct(s['predicted_upheld_rate'])} |")
    base = p["grouped_by_respondent"]["scores"]["majority"]["base_rate"]
    return "\n".join(lines) + f"\n\nBase rate (share upheld): **{_pct(base)}**."


def block_triage(a: dict) -> str:
    rows = a["triage"]["precedent_knn"]["curve"]
    lines = ["| coverage | cases automated | accuracy | recall on upheld | "
             "overturns missed in the automated portion |",
             "|---:|---:|---:|---:|---:|"]
    for r in rows:
        lines.append(f"| {_pct(r['coverage'], 0)} | {r['n_automated']:,} | "
                     f"{_pct(r['accuracy'])} | {_pct(r['recall_upheld'])} | "
                     f"{r['upheld_missed_in_automated']:,} |")
    return "\n".join(lines)


def block_cost(a: dict) -> str:
    c = a["cost"]
    if not c.get("cases_with_award"):
        return "*No award figures recovered.*"
    return "\n".join([
        "| | |", "|---|---:|",
        f"| upheld decisions awarding compensation | {c['cases_with_award']:,} "
        f"({_pct(c['share_of_upheld_with_award'])} of upheld) |",
        f"| median compensation | £{c['median_award_gbp']:,.0f} |",
        f"| mean compensation | £{c['mean_award_gbp']:,.0f} |",
        f"| 90th percentile | £{c['p90_award_gbp']:,.0f} |",
        f"| largest in corpus | £{c['max_award_gbp']:,.0f} |",
        f"| ombudsman case fee, 2026/27, payable either way | "
        f"£{c['fos_case_fee_gbp_2026_27']} |",
        "",
        "Compensation for the experience only. The claim settlement is not extracted: "
        "remedy sections name the settlement, the earlier offer, the valuation and the "
        "policy limit in adjacent sentences, and picking between them reliably is not "
        "something a rule does well. So this column is the **smaller half** of what an "
        "overturned decision costs.",
    ])


def block_products(a: dict) -> str:
    """Rates from search totals, which our sampling does not touch."""
    path = ROOT / "results" / "uphold_rates.json"
    if not path.exists():
        return "*Run `python -m bench.rates` to populate this table.*"
    r = json.loads(path.read_text())
    lines = ["| search phrase | published decisions | upheld | rate | 95% CI |",
             "|---|---:|---:|---:|---|"]
    for phrase, v in sorted(r["by_phrase"].items(),
                            key=lambda kv: -(kv[1]["rate"] or 0)):
        lo, hi = v["ci95"]
        lines.append(f"| `{phrase}` | {v['total']:,} | {v['upheld']:,} | "
                     f"{_pct(v['rate'])} | {_pct(lo)} – {_pct(hi)} |")
    return ("\n".join(lines) +
            f"\n\nOver {r['date_from']} – {r['date_to']}. These are rates over "
            "**decisions matching a full-text phrase**, not over products and not "
            "over claims: the search has no product field, and a published "
            "decision is the tail of complaints that reached an ombudsman.")


def block_validation(a: dict) -> str:
    """The hand-labelled estimate, with the rule's measured accuracy beside it."""
    path = ROOT / "results" / "remedy_validation.json"
    if not path.exists():
        return "*No validation record.*"
    v = json.loads(path.read_text())
    p, r = v["pooled"], v["rule_on_clean_sample"]
    lo, hi = p["claim_decision_stood_ci95"]
    return "\n".join([
        f"**{_pct(p['claim_decision_stood_share'])} of upheld claim complaints left the "
        f"claim decision intact** (95% CI {_pct(lo)} – {_pct(hi)}, "
        f"{p['n']} decisions labelled by hand). The insurer\'s answer stood; it lost on "
        f"how it got there.",
        "",
        "| | |", "|---|---:|",
        f"| decisions hand-labelled | {p['n']} |",
        f"| claim decision changed | {p['disturbed']} ({_pct(p['share'])}) |",
        f"| claim decision stood | {p['n'] - p['disturbed']} "
        f"({_pct(p['claim_decision_stood_share'])}) |",
        f"| rule accuracy, clean sample (n={r['n']}) | {_pct(r['accuracy'])} |",
        f"| rule precision / recall | {_pct(r['precision'])} / {_pct(r['recall'])} |",
    ])


def block_example(a: dict) -> str:
    """The worked example, from its own results file."""
    path = ROOT / "results" / "worked_example.json"
    if not path.exists():
        return "*Run `python -m tools.worked_example`.*"
    from tools.worked_example import markdown
    return markdown(json.loads(path.read_text()))


def block_investigator(a: dict) -> str:
    ab = a["investigator_ablation"]
    if "note" in ab:
        return f"*{ab['note']}* ({ab['cases']} cases)"
    w = ab["with_investigator_view"]["scores"]
    wo = ab["without_investigator_view"]["scores"]
    lines = ["| model | without the adjudicator's view | with it | change |",
             "|---|---:|---:|---:|"]
    for model in wo:
        a1 = wo[model]["balanced_accuracy"]
        a2 = w[model]["balanced_accuracy"]
        lines.append(f"| {model} | {_pct(a1)} | {_pct(a2)} | "
                     f"{100 * (a2 - a1):+.1f} pts |")
    return ("\n".join(lines) +
            f"\n\nRun on the {ab['cases']:,} cases that carry one "
            f"({_pct(ab['share_of_corpus'])} of the benchmark). Balanced accuracy, "
            f"grouped split.")


def block_leakage(a: dict) -> str:
    lk = a["leakage"]
    h = a["composition"]["hash_check"]
    return "\n".join([
        "| | |", "|---|---:|",
        f"| inputs checked for verdict wording | {lk['cases_checked']:,} |",
        f"| inputs containing any | **{lk['inputs_containing_verdict_wording']}** |",
        f"| carrying the adjudicator's provisional view | "
        f"{lk['carrying_investigator_view']:,} "
        f"({_pct(lk['carrying_investigator_view_share'])}) |",
        f"| cached PDFs matching their recorded SHA-256 | "
        f"{h['ok']:,} / {h['rows']:,} (mismatches: {h['mismatch']}) |",
    ])


BLOCKS = {
    "COMPOSITION": block_composition,
    "LEAKAGE": block_leakage,
    "GROUNDS": block_grounds,
    "VALIDATION": block_validation,
    "COST": block_cost,
    "PREDICTION": block_prediction,
    "INVESTIGATOR": block_investigator,
    "TRIAGE": block_triage,
    "EXAMPLE": block_example,
    "PRODUCTS": block_products,
}


def render(readme: str, analysis: dict) -> str:
    for name, fn in BLOCKS.items():
        start, end = f"<!--AUTO:{name}-->", f"<!--/AUTO:{name}-->"
        i, j = readme.find(start), readme.find(end)
        if i == -1 or j == -1:
            continue
        readme = readme[:i + len(start)] + "\n" + fn(analysis) + "\n" + readme[j:]
    return readme


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()

    if not RESULTS.exists():
        raise SystemExit("no results/analysis.json — run `python -m bench.run`")
    analysis = json.loads(RESULTS.read_text())
    current = README.read_text()
    updated = render(current, analysis)

    if a.check:
        if current != updated:
            print("README is out of date — run `python -m bench.report`")
            return 1
        print("README matches results/analysis.json")
        return 0

    README.write_text(updated)
    print(f"rewrote {sum(1 for n in BLOCKS if f'<!--AUTO:{n}-->' in current)} blocks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
