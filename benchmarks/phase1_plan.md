# Phase 1 Benchmark Plan

## Goal

Establish a first public benchmark suite for `patholib` that covers its four current analysis directions:

- H&E nuclei / inflammatory-cell detection
- H&E tumor / necrosis / stroma area estimation
- IHC nuclear positive-percentage estimation
- IHC membrane scoring

The phase 1 target is not to beat every published model. The target is to produce a defensible external baseline for `patholib`, using public datasets, fixed metrics, and reproducible settings.

## Selected Datasets

| Priority | Dataset | Public Link | `patholib` mapping | Why it is phase 1 |
| --- | --- | --- | --- | --- |
| 1 | PanNuke | <https://huggingface.co/datasets/RationAI/PanNuke> | `analyze_he.py --mode inflammation` and nuclei detection backends | Broad H&E nuclei benchmark, includes inflammatory nuclei, standard public comparison point |
| 2 | BCSS | <https://github.com/PathologyDataScience/BCSS> | `analyze_he.py --mode area-ratio` | Directly tests region segmentation quality and area-ratio proxy quality |
| 3 | BCData | <https://sites.google.com/view/bcdataset> | `analyze_ihc.py --stain-type nuclear --marker Ki67` | Strong fit for positive/negative tumor-cell detection and Ki-67 index estimation |
| 4 | HER2-IHC-40x | <https://zenodo.org/records/15179608> | `analyze_ihc.py --stain-type membrane --marker HER2` | Best phase-1 public fit for membrane HER2 grading without full challenge infrastructure |

## Deferred To Phase 2

- `TIGER`: <https://tiger.grand-challenge.org/>
  - Strong match for H&E TIL evaluation, but heavier because it is WSI-level and challenge-oriented.
- `SHIDC-B-Ki-67`: <https://www.nature.com/articles/s41598-021-86912-w>
  - Good external Ki-67 validation set, but overlaps with `BCData` and is better as second-wave validation.
- `Warwick HER2 challenge`: <https://wrap.warwick.ac.uk/90840/>
  - Valuable for cross-checking HER2 scoring, but less convenient than `HER2-IHC-40x` for a first runnable pipeline.

## Comparison Matrix

Every phase-1 dataset should report these method groups:

1. `patholib/watershed`
   - Current classical CV baseline in this repo.
2. `patholib/cellpose`
   - Current DL-assisted baseline in this repo where applicable.
3. `external/reference` (optional for phase 1, required for phase 2)
   - Recommended: `HoVer-Net`, `StarDist`, `Cellpose`, or a task-specific public baseline from the dataset authors.

The immediate requirement for phase 1 is at least an internal `patholib` comparison between `watershed` and `cellpose` where both are supported.

## Dataset-Specific Evaluation

### 1. PanNuke

Use it as the primary H&E nuclei benchmark.

- Task mapping:
  - nuclei instance detection
  - inflammatory-vs-other cell discrimination where feasible
- Inputs:
  - 256x256 RGB patches
  - 0.25 um/pixel resolution from the dataset card
- Recommended metrics:
  - `PQ`
  - `AJI`
  - `Dice`
  - `F1@IoU=0.5`
  - `F1` for inflammatory nuclei if class mapping is implemented
- Notes:
  - `patholib` is not a native 5-class nuclei classifier today, so class-aware reporting should clearly distinguish "fully supported" from "proxy evaluation".
  - At minimum, report detection/segmentation metrics on all nuclei plus inflammatory-cell recall/precision if a mapping is added.

### 2. BCSS

Use it as the primary H&E region benchmark.

- Task mapping:
  - tumor / stroma / necrosis region segmentation
  - downstream ratio estimation
- Recommended metrics:
  - per-class `Dice`
  - per-class `IoU`
  - `tumor_ratio_mae`
  - `necrosis_ratio_mae`
- Notes:
  - The dataset uses encoded region masks and a "don't care" label that must be ignored during scoring.
  - If `patholib` outputs fewer classes than BCSS ground truth, publish the class-collapsing rules explicitly.

### 3. BCData

Use it as the primary Ki-67 nuclear benchmark.

- Task mapping:
  - positive/negative tumor-cell detection
  - Ki-67 positive percentage estimation
- Recommended metrics:
  - cell-level `precision`
  - cell-level `recall`
  - cell-level `F1`
  - `positive_percentage_mae`
  - `positive_percentage_rmse`
- Notes:
  - The public site describes separate positive and negative tumor-cell annotations in `.h5` files.
  - This is a good fit for evaluating both detection quality and final score stability.

### 4. HER2-IHC-40x

Use it as the primary membrane-scoring benchmark.

- Task mapping:
  - 4-way HER2 membrane grade prediction: `0`, `1+`, `2+`, `3+`
- Recommended metrics:
  - `accuracy`
  - `macro_f1`
  - `quadratic_weighted_kappa`
- Notes:
  - This dataset is patch-oriented and easier to operationalize than a hidden-label challenge.
  - Report confusion matrices because `1+` vs `2+` and `2+` vs `3+` errors are clinically more informative than raw accuracy alone.

## Standardized Run Settings

To keep comparisons interpretable, lock these settings per dataset before running:

1. Data split
   - Use the official public split when provided.
   - If a dataset offers multiple split strategies, pick one for phase 1 and keep it fixed in all tables.
2. Stain normalization
   - Run both `normalization=off` and `normalization=on` only if the added runtime is acceptable.
   - Otherwise default to `off` for baseline reproducibility and document the choice.
3. Resolution
   - Preserve original public resolution unless a conversion is strictly required by the algorithm.
4. Post-processing
   - Keep one default post-processing configuration per method family.
   - Do not retune thresholds independently per test image.

## Deliverables

Phase 1 is complete only when all of the following exist:

1. A dataset preparation note for each selected benchmark.
2. One reproducible command set per `patholib` method variant.
3. A single aggregated results table using `results_template.csv`.
4. One short markdown report summarizing wins, losses, and failure modes.

## Recommended Execution Order

Run in this order to reduce integration risk:

1. `BCData`
   - Fastest path to a clean end-to-end benchmark because it matches current nuclear IHC output well.
2. `HER2-IHC-40x`
   - Gives membrane-scoring coverage early and exposes IHC grading issues quickly.
3. `BCSS`
   - Validates area-ratio logic and class-mapping assumptions.
4. `PanNuke`
   - Highest scientific value for H&E nuclei, but likely needs the most evaluation glue.

## Success Criteria

Phase 1 is useful if it answers these questions clearly:

1. Is `cellpose` materially better than `watershed` on public H&E nuclei tasks?
2. Are `patholib` region outputs good enough to estimate tumor/necrosis ratios on public masks?
3. Does `patholib` produce stable Ki-67 positive-percentage estimates on public nuclear annotations?
4. Does membrane grading hold up on an external HER2 dataset, not just internal examples?

## Source Notes

Public dataset references checked for this plan:

- PanNuke dataset card: <https://huggingface.co/datasets/RationAI/PanNuke>
- BCSS repository: <https://github.com/PathologyDataScience/BCSS>
- BCData site: <https://sites.google.com/view/bcdataset>
- HER2-IHC-40x dataset record: <https://zenodo.org/records/15179608>
- TIGER challenge: <https://tiger.grand-challenge.org/>
- SHIDC-B-Ki-67 paper: <https://www.nature.com/articles/s41598-021-86912-w>
- Warwick HER2 challenge paper/archive: <https://wrap.warwick.ac.uk/90840/>
