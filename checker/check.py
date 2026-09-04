"""Overturn check: run a draft claim decision past the published record.

    echo '{"facts": "...", "decision": "decline", "reason": "..."}' \
        | python -m checker.check

Given the facts of a claim and the decision about to be sent, this returns

  1. an overturn risk, from the outcomes of the most similar published
     decisions rather than from a model's opinion;
  2. the decisions themselves — reference, respondent, date, outcome,
     similarity and a link — so the number can be checked and, more to the
     point, so a file note can cite them;
  3. the grounds that carried the upheld neighbours, with the ombudsman's own
     words attached to each.

The design constraint that produced this shape: a claims decision has to be
explainable to the person it went against, to the client, and eventually to
the ombudsman. A score with no citations cannot do that job however accurate
it is, which is why the retrieval model is the one wearing the interface and
the stronger-scoring classifier is not.

It is a triage aid. It reports what happened in similar published cases; it
does not decide anything, and the abstention curve in the README is there to
say plainly what fraction of cases a system like this can be trusted on.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, asdict
from pathlib import Path

from bench import grounds
from bench.baselines import PrecedentKNN
from bench.dataset import Case, load_cases

ROOT = Path(__file__).resolve().parent.parent


@dataclass
class Precedent:
    drn: str
    date: str | None
    business: str | None
    upheld: bool
    similarity: float
    url: str
    grounds: list[str]


@dataclass
class Report:
    overturn_risk: float
    n_precedents: int
    upheld_among_precedents: int
    mean_similarity: float
    weak_match: bool
    weak_threshold: float
    precedents: list[Precedent]
    grounds_seen: dict[str, int]
    checks: list[str]

    def as_dict(self) -> dict:
        d = asdict(self)
        return d

    def render(self) -> str:
        lines = [
            f"overturn risk        {self.overturn_risk:.0%}",
            f"nearest precedents   {self.n_precedents} "
            f"({self.upheld_among_precedents} upheld), "
            f"mean similarity {self.mean_similarity:.2f}",
        ]
        if self.weak_match:
            lines.append(f"  ! weak match — mean similarity is below "
                         f"{self.weak_threshold:.2f}, the level nine out of ten "
                         f"real cases exceed against their own nearest "
                         f"precedent. Read this risk as close to the base rate, "
                         f"not as case-specific.")
        lines.append("")
        lines.append("grounds carried by the upheld precedents")
        if self.grounds_seen:
            for g, n in self.grounds_seen.items():
                fam = grounds.GROUNDS[g][0] if g in grounds.GROUNDS else "?"
                lines.append(f"  {n:>3}x  {g:<18} ({fam})")
        else:
            lines.append("   none tagged")
        lines.append("")
        lines.append("before sending this decision, check")
        for c in self.checks:
            lines.append(f"  [ ] {c}")
        lines.append("")
        lines.append("precedents")
        for p in self.precedents:
            mark = "UPHELD    " if p.upheld else "not upheld"
            lines.append(f"  {mark} {p.similarity:.2f}  {p.drn}  "
                         f"{p.date or '?':>11}  {p.business or '?'}")
            lines.append(f"             {p.url}")
        return "\n".join(lines)


# Checks keyed to the ground a neighbour was upheld on. Each is a question the
# ombudsman actually asked in decisions of that kind.
CHECKS: dict[str, str] = {
    "evidence": "Has every piece of evidence the policyholder supplied been "
                "considered, and has anything we are relying on absence of "
                "actually been requested in writing?",
    "delay": "How long has this claim been open, and can each period of "
             "inactivity be explained on the file?",
    "communication": "Has the reason for this outcome been put in terms the "
                     "policyholder can act on, including what would change it?",
    "quantum": "Is the deduction, depreciation or excess applied here "
               "evidenced, and would the sum offered actually put them back "
               "in the position they were in?",
    "process": "Has the claim been assessed under every section of cover that "
               "could respond, not just the one it was reported under?",
    "policy_wording": "Is the exclusion being relied on clearly worded, and "
                      "was it brought to the policyholder's attention at sale?",
    "misrepresentation": "If this turns on a misrepresentation: is it a "
                         "qualifying one under CIDRA, and is it careless "
                         "rather than deliberate? The remedy differs.",
    "fraud": "If fraud is alleged, does the evidence meet the standard — and "
             "is the allegation being made explicitly rather than implied by "
             "a declinature?",
}

GENERIC_CHECK = ("ICOBS 8.1: is this claim being handled promptly and fairly, "
                 "and is the declinature reasonable on the evidence held?")

WEAK_PERCENTILE = 10     # see `weak_threshold`


def weak_threshold(model: PrecedentKNN, cases: list[Case], sample: int = 200,
                   seed: int = 0) -> float:
    """The similarity below which a neighbourhood is not case-specific.

    This was a constant, and the constant was wrong. Cosine similarity between
    a short draft and a full published decision lands around 0.1 in this
    geometry, so a threshold of 0.2 flagged every case as a weak match,
    including ones whose neighbours were plainly on point — a warning that
    fires always carries no information.

    So it is measured instead: take real cases, and for each one compute the
    same statistic the report compares against it — the mean similarity of its
    top-k neighbours, excluding itself. "Weak" then means *less similar than
    nine out of ten real cases are to their own precedent set*, which is a
    statement about this corpus rather than a number someone picked.

    The statistic has to match on both sides. A first version took each case's
    single nearest neighbour and compared the report's *mean over ten* against
    it. The mean of ten is structurally below the maximum of ten, so every
    case was flagged weak, including a query that was itself a corpus case.
    """
    import numpy as np

    rng = np.random.default_rng(seed)
    idx = rng.choice(len(cases), size=min(sample, len(cases)), replace=False)
    means: list[float] = []
    for i in idx:
        # Skip the case itself, which is always its own nearest neighbour.
        sims = [s for j, s, _u in model.neighbours(cases[i].text) if j != i]
        if sims:
            means.append(float(np.mean(sims)))
    return float(np.percentile(means, WEAK_PERCENTILE)) if means else 0.0


def build_index(cases: list[Case] | None = None
                ) -> tuple[PrecedentKNN, list[Case], float]:
    cases = cases if cases is not None else load_cases()
    if not cases:
        raise SystemExit(
            "no corpus: run `python -m corpus.build` first (about 50 minutes, "
            "it fetches published decisions from the ombudsman's website)")
    model = PrecedentKNN(k=15).fit([c.text for c in cases], [c.upheld for c in cases])
    return model, cases, weak_threshold(model, cases)


def check(facts: str, decision: str = "decline", reason: str = "",
          k: int = 10, cases: list[Case] | None = None) -> Report:
    model, corpus, weak_at = build_index(cases)
    query = "\n".join(x for x in (facts, reason) if x)

    neigh = model.neighbours(query)[:k]
    sims = [s for _i, s, _u in neigh]
    upheld_idx = [i for i, _s, u in neigh if u]

    seen: dict[str, int] = {}
    precedents: list[Precedent] = []
    for i, s, u in neigh:
        c = corpus[i]
        g = c.grounds if u else []
        for name in g:
            seen[name] = seen.get(name, 0) + 1
        precedents.append(Precedent(
            drn=c.drn, date=c.date, business=c.business, upheld=u,
            similarity=round(s, 4), url=c.url, grounds=g,
        ))

    weight = sum(sims)
    risk = (sum(s for _i, s, u in neigh if u) / weight) if weight else 0.0
    mean_sim = (weight / len(sims)) if sims else 0.0

    ordered = sorted(seen.items(), key=lambda kv: -kv[1])
    checks = [CHECKS[g] for g, _n in ordered if g in CHECKS] or []
    checks.append(GENERIC_CHECK)

    return Report(
        overturn_risk=round(risk, 4),
        n_precedents=len(neigh),
        upheld_among_precedents=len(upheld_idx),
        mean_similarity=round(mean_sim, 4),
        weak_match=mean_sim < weak_at,
        weak_threshold=round(weak_at, 4),
        precedents=precedents,
        grounds_seen=dict(ordered),
        checks=checks,
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="Check a draft claim decision "
                                             "against published ombudsman decisions.")
    ap.add_argument("--facts", help="the claim facts; omit to read JSON on stdin")
    ap.add_argument("--decision", default="decline")
    ap.add_argument("--reason", default="")
    ap.add_argument("-k", type=int, default=10)
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    a = ap.parse_args()

    if a.facts:
        payload = {"facts": a.facts, "decision": a.decision, "reason": a.reason}
    else:
        payload = json.load(sys.stdin)

    if not payload.get("facts"):
        print("no facts given", file=sys.stderr)
        return 2

    rep = check(payload["facts"], payload.get("decision", "decline"),
                payload.get("reason", ""), k=a.k)
    print(json.dumps(rep.as_dict(), indent=2) if a.json else rep.render())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
