# BCData Benchmark Note

## Source

- Dataset site: <https://sites.google.com/view/bcdataset>
- Phase-1 plan: [phase1_plan.md](phase1_plan.md)

The BCData site states that:

- images are stored under `BCData/images/<split>/`
- annotations are stored under `BCData/annotations/<split>/positive/` and `BCData/annotations/<split>/negative/`
- the annotation file format is `.h5`
- coordinates can be loaded from the `coordinates` dataset in each `.h5` file

## Expected Local Layout

```text
<dataset-root>/
├── images/
│   ├── train/
│   ├── validation/
│   └── test/
└── annotations/
    ├── train/
    │   ├── positive/
    │   └── negative/
    ├── validation/
    │   ├── positive/
    │   └── negative/
    └── test/
        ├── positive/
        └── negative/
```

## Run patholib on BCData

Example CPU baseline:

```bash
python3 benchmarks/scripts/bcdata.py run \
  --dataset-root /data/BCData \
  --split test \
  --output-dir /data/benchmarks/bcdata/patholib-watershed \
  --detection-method watershed
```

Example Cellpose run:

```bash
python3 benchmarks/scripts/bcdata.py run \
  --dataset-root /data/BCData \
  --split test \
  --output-dir /data/benchmarks/bcdata/patholib-cellpose \
  --detection-method cellpose \
  --use-gpu
```

Each image produces the standard `patholib` outputs:

- `*_ihc_report.json`
- `*_ihc_cells.csv`
- optional `*_ihc_overlay.png`

## Evaluate BCData Predictions

```bash
python3 benchmarks/scripts/bcdata.py eval \
  --dataset-root /data/BCData \
  --split test \
  --predictions-dir /data/benchmarks/bcdata/patholib-watershed
```

Default evaluation outputs:

- `bcdata_test_eval_summary.json`
- `bcdata_test_per_image.csv`

## Metrics

The current evaluator reports:

- positive-cell `precision`, `recall`, `f1`
- negative-cell `precision`, `recall`, `f1`
- `mean_f1`
- Ki-67 positive-percentage `mae`
- Ki-67 positive-percentage `rmse`
- Ki-67 positive-percentage `pearson_r`

## Matching Rule

Predicted and ground-truth cells are matched independently for positive and negative classes with greedy one-to-one matching inside a configurable pixel radius.

Default setting:

- `--match-radius 6`

If you need to align with a published reference implementation, change `--match-radius` to the value used in that scorer and record the choice in the benchmark report.

## Coordinate Order

The BCData site shows how to load `coordinates` from each `.h5` file but does not document the axis order on the page itself. This evaluator defaults to:

- `--coord-order xy`

If inspection of your downloaded files shows `(y, x)` ordering instead, rerun with:

```bash
python3 benchmarks/scripts/bcdata.py eval \
  --dataset-root /data/BCData \
  --split test \
  --predictions-dir /data/benchmarks/bcdata/patholib-watershed \
  --coord-order yx
```
