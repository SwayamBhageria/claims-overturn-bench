# claims-overturn-bench

**A claims platform decides to decline. What does the published record say about whether that
decision survives the ombudsman — and can the ones that won't be spotted before they are sent?**

The AI claims products I could find are sold on two numbers: speed and cost. Three times faster.
Sixty per cent lower cycle time. Ninety-nine per cent straight-through processing, 246% ROI.
Every one of those is measurable from the inside, on the day the claim closes, by the party with
an interest in the answer.

There is a third number and it arrives a year or more later, from outside, from someone with
statutory power to substitute their own view: the share of decisions that don't hold up. In the
UK that someone is the Financial Ombudsman Service.

The ombudsman already publishes the aggregate. Half-yearly complaints data gives uphold rates per
firm, and anyone can look up a carrier's record. What that cannot tell you is anything about a
*decision* — which of the ones on your desk this afternoon is the kind that loses. For that you
need the cases, and unusually, those are published too: every final decision since April 2013,
with the facts, the reasoning, the respondent, and the outcome. The outcome is a **search
filter**, which means the ground truth is queryable rather than something that has to be labelled.

This repository turns that into three things:

1. a **corpus** of real, adjudicated UK insurance claim disputes, built from the ombudsman's own
   search, with the answer strictly separated from the facts;
2. a **measurement** of what the ombudsman actually faults, which turns out not to be the thing
   the category is optimising for;
3. a **checker** that takes a draft decline and returns its nearest published precedents, an
   overturn risk, and the specific grounds those precedents turned on — with citations, because
   a claims decision has to be explainable to the person it went against.

**Everything here runs with no API key, no account, and no paid data.** The corpus is public
record; the models are local. That is the point rather than a limitation: the question is what
*anyone* can establish about claims-decision quality from the outside, and the answer turns out
to be quite a lot.

---

## What the corpus is

<!--AUTO:COMPOSITION-->
| | |
|---|---:|
| decisions harvested | 1,533 |
| date range | 2024-01-08 – 2026-07-21 |
| distinct respondent firms | 183 |
| **not about a claim** (premium, mis-sale, cancellation) | **126  (8.2%)** |
| benchmark cases (claim decisions only) | **1,349** |
| upheld / not upheld | 626 / 723 |
| median input length | 323 words |

Complaint types: claim 1,407 · other 80 · pricing 26 · sales 10 · admin 10.

Product lines: property 263 · unknown 241 · motor 220 · travel 177 · gadget 165 · pet 152 · health 57 · warranty 42 · commercial 21 · life 11.
<!--/AUTO:COMPOSITION-->

Not every published insurance decision is about a claim. Many are about a premium rise, a
mis-sale at the point of sale, or a mid-term cancellation, and scoring a claims model on those
pads the sample with cases where the model is answering a question nobody asked it. They are
classified out by reading each decision's own opening sentence, then counted and reported rather
than silently dropped.

The share excluded here is small, and that is a fact about the retrieval rather than about the
ombudsman: every query used to build this corpus contains the word "claim", so the set that comes
back is already claim-heavy. It is not an estimate of how much of the ombudsman's insurance
casework is about claims, and nothing here should be read as one.

The classifier reads the decision rather than trusting the query that retrieved it, because the
`Keyword` field is full-text: a search for "motor insurance claim" also returns travel decisions
that mention a hire car.

### The split, and why it has to be airtight

Each decision follows the same skeleton — `The complaint`, `What happened`, `What I've decided –
and why`, `My final decision`. The cut is made at the first reasoning heading. Above it is what a
handler holds when deciding; below it is the answer.

Two things could quietly destroy the benchmark, and both are handled rather than noted:

**Verdict wording.** If "I do not uphold this complaint" survives into an input, the model is
reading the answer. Every input is scanned, at build time and again at load time, and any case
that fails is dropped rather than repaired.

**The adjudicator's view.** `What happened` routinely ends with the first-stage adjudicator's
provisional conclusion — *"Our investigator thought Aviva had acted reasonably"*. That is a prior
adjudication of the same case, not a fact about the claim. A model that simply agrees with the
last opinion it was shown would score well here and be useless in production, where no such
opinion exists. It is removed from the input and kept aside, so the benchmark can be run both
ways and the difference measured. It is measured below.

<!--AUTO:LEAKAGE-->
| | |
|---|---:|
| inputs checked for verdict wording | 1,349 |
| inputs containing any | **0** |
| carrying the adjudicator's provisional view | 1,294 (95.9%) |
| cached PDFs matching their recorded SHA-256 | 1,533 / 1,533 (mismatches: 0) |
<!--/AUTO:LEAKAGE-->

---

## 1. What the ombudsman actually faults

<!--AUTO:GROUNDS-->
| ground | family | upheld decisions | share of upheld |
|---|---|---:|---:|
| process | handling | 125 | 20.0% |
| communication | handling | 121 | 19.3% |
| quantum | handling | 70 | 11.2% |
| misrepresentation | coverage | 47 | 7.5% |
| delay | handling | 30 | 4.8% |
| policy_wording | coverage | 16 | 2.6% |
| evidence | handling | 10 | 1.6% |
| fraud | coverage | 7 | 1.1% |
| *(untagged)* | — | 307 | 49.0% |

By family: untagged 307 · handling 250 · coverage 36 · both 33.

**45.2% of all upheld decisions fault the handling** (283 of 626); 11.0% fault the coverage decision (69). The two overlap — a decision can be both.

Read those against the untagged row, not past it. Of the 319 decisions the tagger does place, **88.7% fault the handling** and 21.6% the coverage decision. The 49.0% it places nowhere is the tagger's recall problem, not evidence of a third kind of fault, and it means these shares are a floor rather than an estimate.
<!--/AUTO:GROUNDS-->

Grounds are tagged by rules over the ombudsman's own reasoning, not by a model, so every tag
traces to the sentence that produced it — `bench.grounds.explain()` returns exactly that.

**This layer is the least validated thing on the page and the untagged row says so.** The
ombudsman writes findings in prose, and a decision that reasons its way to "this was unfair"
without using any of the recurring formulas is counted as untagged rather than guessed at. Its
precision has not been measured against hand labels; the harness for doing that is
`bench/validate_rules.py` and it is the obvious next thing. The remedy split in the next section
*has* been validated that way, and it is the one carrying a headline.

### The distinction the word "upheld" hides

"Upheld" covers two commercially different events. In one, the ombudsman decides the claim should
have been paid. In the other the declinature **stands**, and the insurer loses anyway — on delay,
on evidence it never asked for, on an explanation nobody could follow.

<!--AUTO:VALIDATION-->
**28.6% of upheld claim complaints left the claim decision intact** (95% CI 21.7% – 36.5%, 140 decisions labelled by hand). The insurer's answer stood; it lost on how it got there.

| | |
|---|---:|
| decisions hand-labelled | 140 |
| claim decision changed | 100 (71.4%) |
| claim decision stood | 40 (28.6%) |
| rule accuracy, clean sample (n=40) | 85.0% |
| rule precision / recall | 90.3% / 90.3% |
<!--/AUTO:VALIDATION-->

That distinction is the finding. The first kind is underwriting judgement and it is hard to
automate away. The second is operational, it sits inside the control of whoever administers the
claim, and it is the thing the category's speed metrics are already adjacent to without ever
claiming.

**This number was wrong until it was checked by hand, and it is worth saying how.** The split is
produced by rules over the operative directions. The first version of those rules put the
claim-stood share near half. Reading the decisions it had classified that way found "accept the
claim and repair the laptop", "increase the cash settlement to £355" and "pay Mr W's claim for
his pet's treatment" all counted as the claim being left alone. The rules were rewritten, and
then validated on a sample drawn and labelled *after* they were frozen, because the two samples
used to build them cannot test them.

---

## 2. What it costs

<!--AUTO:COST-->
| | |
|---|---:|
| upheld decisions awarding compensation | 422 (67.4% of upheld) |
| median compensation | £200 |
| mean compensation | £356 |
| 90th percentile | £795 |
| largest in corpus | £3,000 |
| ombudsman case fee, 2026/27, payable either way | £680 |

Compensation for the experience only. The claim settlement is not extracted: remedy sections name the settlement, the earlier offer, the valuation and the policy limit in adjacent sentences, and picking between them reliably is not something a rule does well. So this column is the **smaller half** of what an overturned decision costs.
<!--/AUTO:COST-->

The award is not the whole cost and is usually not the largest part of it. Reaching investigation
at all carries the ombudsman's case fee, and the ombudsman's own wording is the part worth
quoting: *"Respondent financial businesses pay a case fee regardless of the outcome of a
complaint."* The fee is **£680** where the outcome changes in the consumer's favour, against a
**£2,000 allowance for the financial year**; it falls to £500 only where a professional
representative referred the case and the business did not lose
([case fees](https://www.financial-ombudsman.org.uk/businesses/resolving-complaint/case-fees),
read 2026-09-04).

So the cost of a decision that does not hold up is the fee, plus the claim where it is then paid,
plus 8% simple interest where it was late, plus the handling time, plus the conversation with the
client whose policyholder it was. Only the first of those is on this page.

---

## 3. Can the overturned decisions be identified in advance?

<!--AUTO:PREDICTION-->
| model | split | accuracy | balanced acc. | recall on upheld | predicted-upheld rate |
|---|---|---:|---:|---:|---:|
| majority | grouped by respondent | 53.6% | 50.0% | 0.0% | 0.0% |
| prior | grouped by respondent | 53.6% | 50.0% | 0.0% | 0.0% |
| tfidf_lr | grouped by respondent | 68.9% | 68.7% | 64.9% | 44.8% |
| precedent_knn | grouped by respondent | 66.6% | 66.6% | 66.0% | 48.2% |
| precedent_lsa | grouped by respondent | 63.5% | 63.9% | 68.4% | 53.5% |
| majority | random | 53.6% | 50.0% | 0.0% | 0.0% |
| prior | random | 53.6% | 50.0% | 0.0% | 0.0% |
| tfidf_lr | random | 68.9% | 68.8% | 67.4% | 47.3% |
| precedent_knn | random | 65.6% | 65.5% | 64.1% | 47.4% |
| precedent_lsa | random | 64.9% | 65.2% | 70.1% | 53.8% |

Base rate (share upheld): **46.4%**.
<!--/AUTO:PREDICTION-->

Read that table by **balanced accuracy and recall on the upheld class**, not by accuracy. A
predictor that answers "not upheld" to everything scores the base rate and looks respectable
while being precisely the system that lets every wrong decline through; `majority` is in the
table to make that visible rather than arguable.

**The split is grouped by respondent firm** — no insurer appears on both sides. Without that, a
model can learn "this firm's travel claims get upheld" from one half and cash it in on the other,
which scores well and has learned nothing about claims. The random split is reported alongside so
the gap between them measures how much of the task is memorising defendants.

Here that gap is essentially nothing: 0.687 grouped against 0.688 random for the linear model,
across 183 respondent firms. Whatever signal there is does not come from recognising the insurer,
which is the result you want and not the one to assume. The terms the model leans on are printed
in `results/analysis.json` under `top_upheld_terms`, and they are conduct words — offers,
delays, interest — with one insurer's name among them.

### How much of it is agreeing with the adjudicator?

<!--AUTO:INVESTIGATOR-->
| model | without the adjudicator's view | with it | change |
|---|---:|---:|---:|
| majority | 50.0% | 50.0% | +0.0 pts |
| prior | 50.0% | 50.0% | +0.0 pts |
| tfidf_lr | 68.1% | 76.1% | +8.0 pts |
| precedent_knn | 64.9% | 68.3% | +3.4 pts |
| precedent_lsa | 63.9% | 66.5% | +2.6 pts |

Run on the 1,294 cases that carry one (95.9% of the benchmark). Balanced accuracy, grouped split.
<!--/AUTO:INVESTIGATOR-->

---

## 4. The number that decides whether this is a product

Nobody is proposing to automate the final call. The deployable question is triage: how much of
the book can a system take, at what accuracy, and how many overturns does it still miss?

<!--AUTO:TRIAGE-->
| coverage | cases automated | accuracy | recall on upheld | overturns missed in the automated portion |
|---:|---:|---:|---:|---:|
| 10% | 135 | 83.7% | 77.9% | 15 |
| 20% | 270 | 77.8% | 69.1% | 42 |
| 30% | 405 | 76.5% | 67.4% | 62 |
| 40% | 540 | 75.0% | 69.6% | 77 |
| 50% | 674 | 74.2% | 70.7% | 93 |
| 60% | 809 | 71.6% | 68.4% | 119 |
| 70% | 944 | 69.6% | 66.9% | 147 |
| 80% | 1,079 | 68.5% | 66.7% | 169 |
| 90% | 1,214 | 67.6% | 66.5% | 190 |
| 100% | 1,349 | 66.6% | 66.0% | 213 |
<!--/AUTO:TRIAGE-->

Cases are ranked by confidence and the least confident are handed to a human first. The last
column is the one with a price on it: overturned decisions that the automated portion got wrong.

---

## 5. Uphold rates, measured where our own sampling cannot reach them

The corpus is built by asking for upheld and not-upheld decisions in equal numbers, which makes
its class balance a choice we made, not an estimate of anything. So the rates come from
somewhere else: the search's own result totals, one request per outcome.

<!--AUTO:PRODUCTS-->
| search phrase | published decisions | upheld | rate | 95% CI |
|---|---:|---:|---:|---|
| `warranty claim` | 1,629 | 857 | 52.6% | 50.2% – 55.0% |
| `buildings insurance claim` | 3,561 | 1,654 | 46.4% | 44.8% – 48.1% |
| `contents insurance claim` | 7,750 | 3,569 | 46.1% | 44.9% – 47.2% |
| `motor insurance claim` | 4,885 | 2,173 | 44.5% | 43.1% – 45.9% |
| `travel insurance claim` | 2,466 | 1,074 | 43.6% | 41.6% – 45.5% |
| `pet insurance claim` | 856 | 372 | 43.5% | 40.2% – 46.8% |
| `home insurance claim` | 6,974 | 2,992 | 42.9% | 41.7% – 44.1% |
| `declined the claim` | 18,736 | 7,701 | 41.1% | 40.4% – 41.8% |
| `mobile phone insurance claim` | 551 | 223 | 40.5% | 36.5% – 44.6% |
| `gadget insurance claim` | 158 | 60 | 38.0% | 30.8% – 45.7% |

Over 2024-01-01 – 2026-08-31. These are rates over **decisions matching a full-text phrase**, not over products and not over claims: the search has no product field, and a published decision is the tail of complaints that reached an ombudsman.
<!--/AUTO:PRODUCTS-->

---

## The checker

```
$ echo '{"facts": "Policyholder fell ill abroad and stayed past the return date.
          Medical report requested but not received. Claim declined as the
          circumstances are not covered.", "decision": "decline"}' \
    | python -m checker.check
```

Returns an overturn risk, the nearest published decisions with references and links, the grounds
those precedents turned on, and a checklist drawn from them.

### A worked example

<!--AUTO:EXAMPLE-->
Held-out decision **DRN-5872505** (pet, 10 Nov 2025), checked against the other 1,348 cases. The facts, as a handler would have held them:

> The complaint Mr F complains that INTACT INSURANCE UK LIMITED has unfairly declined a claim under his pet insurance policy. Where I refer to Intact, this includes the actions of its agents and claims handlers for which it takes responsibility. What happened The detailed background to this complaint is well known to both parties, so I'll only summarise the key events here. • Mr F holds a pet insurance policy for his dog, underwritten by Intact and effective from 10 January 2025. • In March 2025, Mr F called Intact to see if he'd be covered for the cost to remove a skin tag. He was advised that as long as the tag hadn't been noted before the policy started, it would be covered. As Mr F believed it hadn't, he proceeded with the operation and made a claim. • Intact declined the claim on the basis the skin tag was a pre-existing condition because it was first noted in a vet appointment in Aug […]

```
overturn risk        51%
nearest precedents   6 (3 upheld), mean similarity 0.24

grounds carried by the upheld precedents
    3x  process            (handling)
    1x  misrepresentation  (coverage)

before sending this decision, check
  [ ] Has the claim been assessed under every section of cover that could respond, not just the one it was reported under?
  [ ] If this turns on a misrepresentation: is it a qualifying one under CIDRA, and is it careless rather than deliberate? The remedy differs.
  [ ] ICOBS 8.1: is this claim being handled promptly and fairly, and is the declinature reasonable on the evidence held?

precedents
  UPHELD     0.26  DRN-5598267  27 Jun 2025  HDI Global Specialty SE
             https://www.financial-ombudsman.org.uk/decision/DRN-5598267.pdf
  UPHELD     0.24  DRN-6261005  30 Mar 2026  Financial & Legal Insurance Company Ltd
             https://www.financial-ombudsman.org.uk/decision/DRN-6261005.pdf
  not upheld 0.24  DRN-6404120   5 Jun 2026  Red Sands Insurance Company (Europe) Limited
             https://www.financial-ombudsman.org.uk/decision/DRN-6404120.pdf
  not upheld 0.23  DRN-5742293  12 Dec 2025  INTACT INSURANCE UK LIMITED
             https://www.financial-ombudsman.org.uk/decision/DRN-5742293.pdf
  not upheld 0.23  DRN-5521061  23 Jun 2025  AmTrust Specialty Limited
             https://www.financial-ombudsman.org.uk/decision/DRN-5521061.pdf
  UPHELD     0.23  DRN-6436948   2 Jul 2026  Admiral Insurance (Gibraltar) Limited
             https://www.financial-ombudsman.org.uk/decision/DRN-6436948.pdf
```

The ombudsman **upheld** this complaint ([DRN-5872505](https://www.financial-ombudsman.org.uk/decision/DRN-5872505.pdf)). **That is a coin flip, and it is reported as one.** The risk sits within five points of even, which is the checker saying it cannot separate this case — exactly the kind that should reach a person. Landing on the right side of 0.5 here is not a result.
<!--/AUTO:EXAMPLE-->

The retrieval model wears the interface even where a classifier scores higher, and that is
deliberate. A claims decision has to be explainable to the policyholder, to the client, and
eventually to the ombudsman. A score with no citations cannot do that job however accurate it is,
whereas *"the four closest published decisions to this one were all upheld, here they are"* is
something that can go in a file note.

It is a triage aid. It reports what happened in similar published cases. It does not decide
anything, and §4 says plainly what fraction of cases a system of this kind can be trusted on.

---

## Reproducing it

```bash
pip install -r requirements.txt
python -m corpus.build      # ~65 min: searches, fetches and parses the decisions
python -m bench.rates       # ~1 min: uphold rates from search totals
python -m bench.run         # writes results/analysis.json
python -m bench.report      # rewrites every table above from that file
python -m pytest -q
```

**What is in the repository and what is not.** `data/corpus.jsonl` holds metadata and derived
fields — reference, date, respondent, outcome, product, complaint type, word counts, and the
SHA-256 of each source PDF. The decisions themselves are not redistributed. They are the
ombudsman's publications, already public at a stable URL, and a hash plus a fetcher reproduces
them byte for byte while proving nothing was edited on the way through:

```bash
python -m tools.verify_corpus
```

The harvester respects `robots.txt` (checked 2026-09-04: it disallows four PDF forms and permits
the rest) and sleeps a second between requests, which works out at roughly one decision every
two and a half seconds including the site's own latency.

### Validating the rules

Three layers here are rules rather than models — the complaint-type classifier, the ground
tagger, and the remedy split — because every number they produce has to be checkable by hand.
That does not make them correct, so they get scored against hand labels.

The remedy split carries the headline and has been done: `results/remedy_validation.json` holds
140 decisions labelled by reading their operative directions, in three samples, with only the
last of the three used to report accuracy. The other two built the rules and cannot test them.

For the ground tagger:

```bash
python -m bench.validate_rules --sample 60 --out data/labels_sample.json   # blank sheet
# label it by reading the reasoning, without looking at the tagger's output
python -m bench.validate_rules --score data/labels.json
```

Per-ground precision and recall land in `results/rule_validation.json`. `--score` refuses a
partly-filled sheet, because scoring one measures the rules on whichever cases were easy.

---

## What this does not establish

**Published decisions are the tail, not the book.** They are the fraction of complaints that
reached a final decision by an ombudsman — themselves a fraction of complaints, themselves a
fraction of claims. Nothing here estimates how often claims go wrong. It measures which disputes
fail to survive scrutiny once they get that far, which is a different and smaller question, and
the one attached to the £680.

**The facts are the ombudsman's account of the facts.** `What happened` is written after the
decision, by the person who made it, and is not the file a handler held on day one. It is the
closest public approximation and it is not the same thing. Any accuracy figure here should be
read as an upper bound on what the same model would do on a live file.

**Contamination is not ruled out.** These decisions are public and may sit in the training data
of any model evaluated against them. The local baselines are fit here, on this corpus, so they
cannot be contaminated; a hosted model can be, and the probe for it is in `bench/llm.py` and is
reported with the model results rather than assumed away.

**A rule is not a judgement.** The ground tagger finds the ombudsman's recurring formulas. A
decision that reasons its way to the same place in unusual words is counted as untagged, and the
untagged share is printed in the table above rather than hidden in a denominator.

---

## Sources

- Financial Ombudsman Service, [decisions database](https://www.financial-ombudsman.org.uk/businesses/resolving-complaint/ombudsman-decisions) — every figure about decisions
- Financial Ombudsman Service, [case fees](https://www.financial-ombudsman.org.uk/businesses/resolving-complaint/case-fees) — the £680
- FCA Handbook, [ICOBS 8.1](https://www.handbook.fca.org.uk/handbook/ICOBS/8/1.html) — the claims-handling rules the decisions apply
- Consumer Insurance (Disclosure and Representations) Act 2012 — the misrepresentation ground

MIT licensed. Built by [Swayam Bhageria](https://github.com/SwayamBhageria).
