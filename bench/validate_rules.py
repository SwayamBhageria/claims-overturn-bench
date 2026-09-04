"""How good is the ground tagger? Measured, not asserted.

The coverage/handling split is the headline finding, and it is produced by
regexes. A regex that quietly misses a third of the delay cases would move
that headline without anything failing, so the tagger is scored against a
hand-labelled sample the same way any other classifier would be.

    python -m bench.validate_rules --sample 60 --out data/labels_sample.json
        writes a blank labelling sheet: reference, the reasoning text, and an
        empty `grounds` list per case, in random order.

    python -m bench.validate_rules --score data/labels.json
        scores the rules against the filled-in sheet and prints per-ground
        precision and recall, plus the effect on the headline family split.

Labelling before running the tagger matters, so the sheet does not include the
rules' output. `--score` refuses a sheet that still has unlabelled rows, since
a half-filled sheet scores the rules on whichever cases were easy to label.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from bench import grounds
from bench.dataset import load_cases

ROOT = Path(__file__).resolve().parent.parent


def make_sheet(n: int, seed: int, out: Path) -> None:
    cases = [c for c in load_cases() if c.upheld]
    if len(cases) < n:
        raise SystemExit(f"only {len(cases)} upheld cases available")
    rng = random.Random(seed)
    sample = rng.sample(cases, n)

    sheet = {
        "instructions": (
            "For each case read `reasoning` and list every ground the "
            "ombudsman's own reasoning rests on, from: "
            + ", ".join(sorted(grounds.GROUNDS)) +
            ". Leave [] if none apply. Do not consult the tagger first."),
        "grounds_available": sorted(grounds.GROUNDS),
        "seed": seed,
        "cases": [{"drn": c.drn, "url": c.url, "product": c.product,
                   "reasoning": c.reasoning, "grounds": None} for c in sample],
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(sheet, indent=2))
    print(f"wrote {n} cases to {out}")


def score_sheet(path: Path) -> dict:
    sheet = json.loads(path.read_text())
    cases = sheet["cases"]
    unlabelled = [c["drn"] for c in cases if c.get("grounds") is None]
    if unlabelled:
        raise SystemExit(
            f"{len(unlabelled)} of {len(cases)} rows are unlabelled "
            f"(e.g. {unlabelled[0]}). Scoring a partly-filled sheet measures "
            f"the rules on whichever cases were easy to label.")

    per_ground: dict[str, dict] = {}
    for g in sorted(grounds.GROUNDS):
        tp = fp = fn = 0
        for c in cases:
            gold = set(c["grounds"])
            pred = set(grounds.tag(c["reasoning"]))
            if g in pred and g in gold:
                tp += 1
            elif g in pred:
                fp += 1
            elif g in gold:
                fn += 1
        per_ground[g] = {
            "tp": tp, "fp": fp, "fn": fn,
            "precision": tp / (tp + fp) if (tp + fp) else None,
            "recall": tp / (tp + fn) if (tp + fn) else None,
            "support": tp + fn,
        }

    fam_agree = sum(
        1 for c in cases
        if grounds.family(grounds.tag(c["reasoning"])) == grounds.family(c["grounds"]))
    gold_fams: dict[str, int] = {}
    pred_fams: dict[str, int] = {}
    for c in cases:
        gf = grounds.family(c["grounds"])
        pf = grounds.family(grounds.tag(c["reasoning"]))
        gold_fams[gf] = gold_fams.get(gf, 0) + 1
        pred_fams[pf] = pred_fams.get(pf, 0) + 1

    micro_tp = sum(v["tp"] for v in per_ground.values())
    micro_fp = sum(v["fp"] for v in per_ground.values())
    micro_fn = sum(v["fn"] for v in per_ground.values())

    return {
        "n": len(cases),
        "per_ground": per_ground,
        "micro_precision": micro_tp / (micro_tp + micro_fp) if (micro_tp + micro_fp) else None,
        "micro_recall": micro_tp / (micro_tp + micro_fn) if (micro_tp + micro_fn) else None,
        "family_agreement": fam_agree / len(cases),
        "family_counts_hand": gold_fams,
        "family_counts_rules": pred_fams,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sample", type=int, help="write a blank labelling sheet")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "labels_sample.json")
    ap.add_argument("--score", type=Path, help="score a filled-in sheet")
    a = ap.parse_args()

    if a.sample:
        make_sheet(a.sample, a.seed, a.out)
        return 0
    if a.score:
        r = score_sheet(a.score)
        print(json.dumps(r, indent=2))
        (ROOT / "results" / "rule_validation.json").write_text(json.dumps(r, indent=2))
        return 0
    ap.error("pass --sample N or --score PATH")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
