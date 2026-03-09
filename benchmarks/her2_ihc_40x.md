# HER2-IHC-40x Benchmark Note

## Source

- Dataset record: <https://zenodo.org/records/15179608>
- Optional preprocessing helper: <https://github.com/seraju77/HER2-IHC-40x-data-preprocessing>
- Phase-1 plan: [phase1_plan.md](phase1_plan.md)

## Download

Use the Zenodo record as the canonical source for the dataset archive. The preprocessing repository above is optional and only useful if you need to reproduce the authors' patch preparation flow or inspect their extraction conventions.

The Zenodo record states that:

- `HER2-IHC-40x` is the WSI-split variant
- it contains `Train/` and `Test/` patch sets
- patches are `1024x1024`
- classes are `0`, `1+`, `2+`, `3+`
- labels are based on HER2 staining intensity

## Expected Local Layout

This helper accepts the common extracted layouts below:

```text
<dataset-root>/
├── Patches/
│   ├── Train/
│   │   ├── 0/
│   │   ├── 1+/
│   │   ├── 2+/
│   │   └── 3+/
│   └── Test/
│       ├── 0/
│       ├── 1+/
│       ├── 2+/
│       └── 3+/
```

or

```text
<dataset-root>/
├── Train/
└── Test/
```

with the same class subfolders underneath.

## Run patholib on HER2-IHC-40x

Example CPU baseline:

```bash
python3 benchmarks/scripts/her2_ihc_40x.py run \
  --dataset-root /data/HER2-IHC-40x \
  --split test \
  --output-dir /data/benchmarks/her2/patholib-watershed \
  --detection-method watershed
```

Example Cellpose run:

```bash
python3 benchmarks/scripts/her2_ihc_40x.py run \
  --dataset-root /data/HER2-IHC-40x \
  --split test \
  --output-dir /data/benchmarks/her2/patholib-cellpose \
  --detection-method cellpose \
  --use-gpu
```

Each image produces the standard `patholib` outputs:

- `*_ihc_report.json`
- `*_ihc_cells.csv`
- optional `*_ihc_overlay.png`

## Evaluate HER2-IHC-40x Predictions

```bash
python3 benchmarks/scripts/her2_ihc_40x.py eval \
  --dataset-root /data/HER2-IHC-40x \
  --split test \
  --reports-dir /data/benchmarks/her2/patholib-watershed
```

Default evaluation outputs:

- `her2_ihc_40x_test_eval_summary.json`
- `her2_ihc_40x_test_per_image.csv`

## Metrics

The current evaluator reports:

- `accuracy`
- `macro_f1`
- `quadratic_weighted_kappa`
- per-class `precision`, `recall`, `f1`
- full `4x4` confusion matrix

## Label Mapping Caveat

`patholib` currently produces membrane cell grades and patch-level summary statistics, but not a clinically validated HER2 scoring model. The evaluator therefore uses a benchmark heuristic:

- `0`: effectively no positive signal
- `1+`: low positive fraction
- `2+`: positive fraction above threshold with moderate-dominant signal
- `3+`: positive fraction above threshold with strong-dominant signal

Default heuristic parameters:

- `--zero-cutoff 1.0`
- `--positive-cutoff 10.0`
- `--strong-grade-cutoff 2.5`
- `--strong-fraction-cutoff 0.30`

These defaults reflect the dataset's public class definitions, but they are still only a proxy. If you tune them, record the changes in the benchmark report.

## Why This Is Still Useful

Even with the heuristic limitation, this benchmark is still useful for phase 1 because it answers a practical question:

- does the current `patholib` membrane-analysis pipeline produce stable enough patch-level outputs to separate `0/1+/2+/3+` on an external public dataset?

If the answer is no, that is still a strong and actionable result for the roadmap.
