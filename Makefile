.PHONY: corpus rates run report test verify all

corpus:            ## fetch and parse the published decisions (~50 min)
	python -m corpus.build --date-from 2024-01-01 --date-to 2026-08-31

rates:             ## uphold rates from search totals (~1 min)
	python -m bench.rates

run:               ## every experiment -> results/analysis.json
	python -m bench.run

report:            ## rewrite the README tables from results/
	python -m bench.report

verify:            ## re-hash every cached decision against the corpus
	python -m tools.verify_corpus

test:
	python -m pytest -q

all: rates run report findings test

findings:          ## regenerate FINDINGS.md from results/
	python -m tools.findings
