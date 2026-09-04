# claims-overturn-bench

**A claims platform decides to decline. What does the published record say about whether that
decision survives the ombudsman — and can the ones that won't be spotted before they are sent?**

Every AI claims product in the market is sold on the same two numbers: speed and cost. Three
times faster. Sixty per cent lower cycle time. Ninety-nine per cent straight-through. All of
those are measurable from the inside on the day a claim closes.

There is a third number and it arrives eighteen months later, from outside, from someone with
statutory power to substitute their own view: the share of decisions that don't hold up. In the
UK that someone is the Financial Ombudsman Service, and unusually, it publishes its work. Every
final decision since 2013 is online with the case facts, the reasoning, the respondent firm, and
the outcome — and the outcome is a **search filter**, which means the ground truth is queryable.

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
*Run `python -m corpus.build` then `python -m bench.run`.*
<!--/AUTO:COMPOSITION-->

The first number that matters is the one about what these decisions are *about*. A large share
of published insurance decisions are not about a claim at all — they are about a premium rise, a
mis-sale, or a cancellation. Scoring a claims model on those would pad the sample with cases
where the model is answering a question nobody asked it, so they are classified out, counted,
and reported. The classifier reads the decision's own opening sentence rather than trusting the
search query that retrieved it, because the ombudsman's `Keyword` field is full-text: a search
for "motor insurance claim" returns travel decisions that mention a hire car.

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
*Run `python -m bench.run`.*
<!--/AUTO:LEAKAGE-->

---

## 1. What the ombudsman actually faults

<!--AUTO:GROUNDS-->
*Run `python -m bench.run`.*
<!--/AUTO:GROUNDS-->

Grounds are tagged by rules over the ombudsman's own reasoning, not by a model, so every tag
traces to the sentence that produced it — `bench.grounds.explain()` returns exactly that. The
tagger's accuracy against a hand-labelled sample is reported in
`results/rule_validation.json`; see *Validating the rules* below.

### The distinction the word "upheld" hides

"Upheld" covers two commercially different events. In one, the ombudsman decides the claim
should have been paid. In the other, the declinature stands and the insurer loses anyway — on
delay, on evidence it never asked for, on an explanation nobody could follow. Read
`claim_decision_stood` in `results/analysis.json` for how the corpus splits.

That distinction is the finding. The first kind is underwriting judgement and it is hard to
automate away. The second is operational, it is inside the control of whoever administers the
claim, and it is what the category's speed metrics are already adjacent to without ever claiming.

---

## 2. What it costs

<!--AUTO:COST-->
*Run `python -m bench.run`.*
<!--/AUTO:COST-->

The award is not the whole cost and is usually not the largest part of it. A complaint that goes
to investigation carries the ombudsman's **case fee of £680 in 2026/27** — payable by the
respondent business whether it wins or loses, above a £2,000 annual allowance
([case fees](https://www.financial-ombudsman.org.uk/businesses/resolving-complaint/case-fees)).
On top sit the claim itself where it is paid, 8% simple interest where it is late, the handling
time, and the client conversation.

---

## 3. Can the overturned decisions be identified in advance?

<!--AUTO:PREDICTION-->
*Run `python -m bench.run`.*
<!--/AUTO:PREDICTION-->

Read that table by **balanced accuracy and recall on the upheld class**, not by accuracy. A
predictor that answers "not upheld" to everything scores the base rate and looks respectable
while being precisely the system that lets every wrong decline through; `majority` is in the
table to make that visible rather than arguable.

**The split is grouped by respondent firm** — no insurer appears on both sides. Without that, a
model learns "this firm's travel claims get upheld" from one half and cashes it in on the other,
which scores well and has learned nothing about claims. The random split is reported alongside;
the gap between them is a measurement of how much of this task is memorising defendants.

### How much of it is agreeing with the adjudicator?

<!--AUTO:INVESTIGATOR-->
*Run `python -m bench.run`.*
<!--/AUTO:INVESTIGATOR-->

---

## 4. The number that decides whether this is a product

Nobody is proposing to automate the final call. The deployable question is triage: how much of
the book can a system take, at what accuracy, and how many overturns does it still miss?

<!--AUTO:TRIAGE-->
*Run `python -m bench.run`.*
<!--/AUTO:TRIAGE-->

Cases are ranked by confidence and the least confident are handed to a human first. The last
column is the one with a price on it: overturned decisions that the automated portion got wrong.

---

## 5. Uphold rates, measured where our own sampling cannot reach them

The corpus is built by asking for upheld and not-upheld decisions in equal numbers, which makes
its class balance a choice we made, not an estimate of anything. So the rates come from
somewhere else: the search's own result totals, one request per outcome.

<!--AUTO:PRODUCTS-->
*Run `python -m bench.rates`.*
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
python -m corpus.build      # ~30 min: searches, fetches and parses the decisions
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
the rest) and runs one request per second.

### Validating the rules

Two layers here are rules rather than models — the complaint-type classifier and the ground
tagger — because every number they produce has to be checkable by hand. That does not make them
correct, so they are scored:

```bash
python -m bench.validate_rules --sample 60 --out data/labels_sample.json   # blank sheet
# label it by reading the reasoning, without looking at the tagger's output
python -m bench.validate_rules --score data/labels.json
```

Per-ground precision and recall land in `results/rule_validation.json`.

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
