# Benchmarking

This directory tracks public benchmark plans and comparable result tables for `patholib`.

## Phase 1

- Plan: [phase1_plan.md](phase1_plan.md)
- Result template: [results_template.csv](results_template.csv)
- BCData note: [bcdata.md](bcdata.md)
- BCSS note: [bcss.md](bcss.md)
- HER2-IHC-40x note: [her2_ihc_40x.md](her2_ihc_40x.md)
- PanNuke note: [pannuke.md](pannuke.md)
- Aggregation script: `python3 benchmarks/scripts/aggregate_phase1.py --eval-json ... --output-dir ...`
- Orchestration script: `python3 benchmarks/scripts/run_phase1.py --output-dir ... --smoke`

## Dataset Access

Use the dataset-specific notes below for layout and evaluation details. Public acquisition entry points for the current phase-1 datasets are:

| Dataset | Primary public entry | Download / mirror | Notes |
| --- | --- | --- | --- |
| `BCData` | <https://sites.google.com/view/bcdataset> | Google Drive file: <https://drive.google.com/file/d/16W04QOR1E-G3ifc4061Be4eGpjRYDlkA/view?usp=sharing> | The public site documents the expected `images/` and `annotations/` layout. |
| `HER2-IHC-40x` | Zenodo record: <https://zenodo.org/records/15179608> | Preprocessing repo: <https://github.com/seraju77/HER2-IHC-40x-data-preprocessing> | The Zenodo record is the canonical dataset source. |
| `BCSS` | Repository: <https://github.com/PathologyDataScience/BCSS> | Google Drive folder: <https://drive.google.com/drive/folders/1zqbdkQF8i5cEmZOGmbdQm-EP8dRYtvss?usp=sharing> | Prefer the repository README if you need the authors' official download flow or metadata. |
| `PanNuke` | Official page: <https://warwick.ac.uk/fac/sci/dcs/research/tia/data/pannuke> | Hugging Face mirror: <https://huggingface.co/datasets/RationAI/PanNuke> | The current `patholib` scaffold expects `images.npy` / `masks.npy`; if you use the parquet mirror, convert it first. |

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
