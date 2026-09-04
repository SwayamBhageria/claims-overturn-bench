"""Run the checker on one real held-out decision and emit markdown.

    python -m tools.worked_example >> /dev/null   # writes into the README block

The case is chosen deterministically — a fixed seed, the first case of the
held-out fold — and printed whatever the result. It is not selected for being
a good example, because an example selected for looking right tells a reader
nothing about the tool and they know it.

The index is fitted on the other cases only, so the decision being checked is
not among its own precedents.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from bench.dataset import load_cases
from checker import check as chk

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "worked_example.json"


def build(seed: int = 0, k: int = 6) -> dict:
    cases = load_cases()
    if len(cases) < 50:
        raise SystemExit("corpus too small — run `python -m corpus.build`")

    rng = random.Random(seed)
    held = rng.randrange(len(cases))
    case = cases[held]
    index = [c for i, c in enumerate(cases) if i != held]

    rep = chk.check(case.text, cases=index, k=k)
    return {
        "seed": seed,
        "case": {"drn": case.drn, "date": case.date, "business": case.business,
                 "product": case.product, "url": case.url,
                 "actual_outcome": "upheld" if case.upheld else "not upheld",
                 "facts": case.text},
        "report": rep.as_dict(),
        "render": rep.render(),
        "index_size": len(index),
    }


def markdown(x: dict) -> str:
    c, r = x["case"], x["report"]
    facts = " ".join(c["facts"].split())
    if len(facts) > 900:
        facts = facts[:900] + " […]"
    risk, actual = r["overturn_risk"], c["actual_outcome"]
    right = (risk >= 0.5) == (actual == "upheld")
    if abs(risk - 0.5) < 0.05:
        verdict = ("**That is a coin flip, and it is reported as one.** The risk sits "
                   "within five points of even, which is the checker saying it "
                   "cannot separate this case — exactly the kind that should reach "
                   "a person. Landing on the right side of 0.5 here is not a "
                   "result.")
    elif right:
        verdict = ("The checker put this on the correct side, and the precedents it "
                   "cites are the reason it can be argued with.")
    else:
        verdict = ("**The checker was wrong on this one.** It is here anyway: the "
                   "case is picked by seed, not by outcome. §4 is where the "
                   "question of how often that happens is answered.")

    lines = [
        f"Held-out decision **{c['drn']}** ({c['product']}, {c['date']}), checked "
        f"against the other {x['index_size']:,} cases. The facts, as a handler "
        f"would have held them:",
        "",
        f"> {facts}",
        "",
        "```",
        x["render"],
        "```",
        "",
        f"The ombudsman **{c['actual_outcome']}** this complaint "
        f"([{c['drn']}]({c['url']})). {verdict}",
    ]
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    x = build(a.seed)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(x, indent=2))
    print(markdown(x))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
