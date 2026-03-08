# Benchmarking

This directory tracks public benchmark plans and comparable result tables for `patholib`.

## Phase 1

- Plan: [phase1_plan.md](phase1_plan.md)
- Result template: [results_template.csv](results_template.csv)
- BCData note: [bcdata.md](bcdata.md)
- HER2-IHC-40x note: [her2_ihc_40x.md](her2_ihc_40x.md)

## Scope

Phase 1 focuses on four public datasets that best match the current `patholib` feature set:

1. `PanNuke` for H&E nuclei detection/classification
2. `BCSS` for H&E tissue-region segmentation and area-ratio proxies
3. `BCData` for IHC nuclear positive/negative cell detection and Ki-67 index estimation
4. `HER2-IHC-40x` for IHC membrane scoring

## Output Convention

Store dataset-specific evaluation outputs outside git and keep only summaries, scripts, and metrics tables in this directory.

Suggested local layout:

```text
benchmarks/
├── README.md
├── phase1_plan.md
├── results_template.csv
├── scripts/              # local evaluation helpers
└── reports/              # aggregated markdown/csv summaries
```
