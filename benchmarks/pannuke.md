# PanNuke Benchmark Note

## Source

- Official dataset page: <https://warwick.ac.uk/fac/sci/dcs/research/tia/data/pannuke>
- Dataset card: <https://huggingface.co/datasets/RationAI/PanNuke>
- Official metrics repository: <https://github.com/TissueImageAnalytics/PanNuke-metrics>
- Phase-1 plan: [phase1_plan.md](phase1_plan.md)

## Download

Use the Warwick page as the canonical dataset entry point. In practice, the Hugging Face mirror is often easier to access today, but it is packaged as parquet rather than the `images.npy` / `masks.npy` files expected by the current `patholib` benchmark scaffold.

If you download from Hugging Face, plan to add a conversion step before running `python3 benchmarks/scripts/pannuke.py`.

`patholib` does not currently implement the full PanNuke 5-class nuclei taxonomy. The phase-1 scaffold therefore evaluates two things explicitly:

- all-nuclei instance quality
- inflammatory-nuclei proxy quality

This keeps the benchmark honest: it is useful for external validation, but it is not a drop-in reproduction of the official full-taxonomy leaderboard.

## Run patholib on PanNuke

```bash
python3 benchmarks/scripts/pannuke.py run \
  --images-npy /data/PanNuke/images.npy \
  --output-dir /data/benchmarks/pannuke/patholib-watershed \
  --detection-method watershed
```

Optional subset run:

```bash
python3 benchmarks/scripts/pannuke.py run \
  --images-npy /data/PanNuke/images.npy \
  --output-dir /data/benchmarks/pannuke/patholib-watershed \
  --start-index 0 \
  --limit 500
```

This writes:

- `pannuke_pred_instances.npy`
- `pannuke_pred_inflammatory_instances.npy`
- `pannuke_run_summary.json`

## Evaluate PanNuke Predictions

```bash
python3 benchmarks/scripts/pannuke.py eval \
  --masks-npy /data/PanNuke/masks.npy \
  --predictions-dir /data/benchmarks/pannuke/patholib-watershed
```

Default outputs:

- `pannuke_eval_summary.json`
- `pannuke_per_patch.csv`

## Metrics

The current evaluator reports:

- `binary_nuclei_dice`
- `aji`
- `pq`
- `all_nuclei_f1`
- `inflammatory_precision`
- `inflammatory_recall`
- `inflammatory_f1`

## Inflammatory Channel

PanNuke mask stacks are channel-based. This scaffold defaults to:

- `--inflammatory-channel 1`

If your local export uses a different channel order, override it explicitly during evaluation.

## Notes

- This benchmark is a proxy for `patholib`'s current H&E inflammation pipeline, not a full PanNuke taxonomy benchmark.
- If you want the official multi-class metrics, that requires a model that predicts PanNuke's complete class set.
