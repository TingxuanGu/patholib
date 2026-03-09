# BCSS Benchmark Note

## Source

- Dataset repository: <https://github.com/PathologyDataScience/BCSS>
- Phase-1 plan: [phase1_plan.md](phase1_plan.md)

`patholib` does not produce the full native BCSS label taxonomy today. The current phase-1 scaffold therefore evaluates a collapsed subset aligned with `patholib.analysis.he_area_ratio`:

- `tumor`
- `stroma`
- `necrosis`

It also reports:

- `tumor_ratio_mae`
- `necrosis_ratio_mae`

## Run patholib on BCSS Images

```bash
python3 benchmarks/scripts/bcss.py run \
  --images-dir /data/BCSS/images \
  --output-dir /data/benchmarks/bcss/patholib-threshold
```

This writes:

- `*_he_report.json`
- `*_he_segmentation.npy`
- `bcss_run_summary.json`

## Evaluate BCSS Predictions

If your downloaded BCSS masks are already collapsed to `patholib` class ids:

- `0 background`
- `1 normal`
- `2 tumor`
- `3 necrosis`
- `4 stroma`

then evaluation can run directly:

```bash
python3 benchmarks/scripts/bcss.py eval \
  --masks-dir /data/BCSS/masks \
  --predictions-dir /data/benchmarks/bcss/patholib-threshold
```

If your BCSS release uses a different mask encoding, pass a label map:

```bash
python3 benchmarks/scripts/bcss.py eval \
  --masks-dir /data/BCSS/masks \
  --predictions-dir /data/benchmarks/bcss/patholib-threshold \
  --label-map-json /data/BCSS/bcss_label_map.json
```

## Label Map Format

For integer masks:

```json
{
  "type": "int",
  "mapping": {
    "0": "background",
    "1": "tumor",
    "2": "stroma",
    "3": "necrosis",
    "255": "ignore"
  }
}
```

For RGB masks:

```json
{
  "type": "rgb",
  "mapping": {
    "255,0,0": "tumor",
    "0,255,0": "stroma",
    "0,0,255": "necrosis",
    "255,255,255": "background",
    "0,0,0": "ignore"
  }
}
```

## Metrics

The current evaluator reports:

- `tumor_dice`, `tumor_iou`
- `stroma_dice`, `stroma_iou`
- `necrosis_dice`, `necrosis_iou`
- `tumor_ratio_mae`
- `necrosis_ratio_mae`

## Notes

- The evaluator stores predicted masks as `*.npy` to avoid palette and compression ambiguity.
- If your BCSS release includes more classes than the three listed above, collapse them explicitly in the label map and document the rule in the benchmark report.
