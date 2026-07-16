# Data specification (authoritative)

Confirmed from the data files and the task deck
(`AI_wellbore_geology_prediction_task_en.pptx`). Units are **feet**; step is
**one foot**.

## Files

- `data/raw/train/<id>__horizontal_well.csv` (773 wells) - full lateral, WITH the
  answer `TVT` and train-only marker columns.
- `data/raw/train/<id>__typewell.csv` - the reference vertical well.
- `data/raw/train/<id>.png` - a rendering of the interpretation (NOT a model
  input; not extracted by default).
- `data/raw/test/<id>__horizontal_well.csv` (3 wells) - masked lateral, no `TVT`,
  no marker columns.
- `data/raw/test/<id>__typewell.csv` - reference well for each test lateral.
- `data/raw/sample_submission.csv` - 14,151 rows.

## Horizontal well columns

| Column | Train | Test | Meaning |
|--------|-------|------|---------|
| MD | yes | yes | measured depth (well length), one-foot steps |
| X, Y, Z | yes | yes | trajectory coordinates of each point |
| GR | yes | yes | gamma ray at each point (some NaN) |
| TVT_input | yes | yes | known TVT up to the Prediction Start (PS) point (heel); NaN after PS |
| TVT | yes | **no** | the target (geology position); predict this after PS |
| ANCC, ASTNU, ASTNL, EGFDU, EGFDL, BUDA | yes | **no** | top depth of each geological formation (train-only) |

`TVT_input` is non-null exactly for rows up to the PS point; rows after PS are the
prediction target.

## Type well columns

`TVT`, `GR`, `Geology` (formation label; some NaN). The type-well TVT is always
known. Correlating lateral GR against the type-well GR-over-TVT profile locates
stratigraphic position. Note (task deck): the lateral's own GR before PS often
correlates with its GR after PS better than the type well does.

## Submission

`id = <wellID>_<rowIndex>`, single prediction column `tvt`. Predict for the
post-PS (toe) rows listed in `sample_submission.csv`.

## Metric

RMSE of `dTVT = manualTVT - predictedTVT` over all predicted points, i.e. plain
RMSE on `tvt` in feet. See `shared/metric.py`.

## Leakage warning (important)

The 3 test well IDs (`000d7d20`, `00bbac68`, `00e12e8b`) are ALSO train wells, so
their true toe TVT is present in `train/`. The public leaderboard is therefore
near-meaningless / trivially gameable. **Trust the shared group-CV
(`shared/folds.csv`), not the public LB.** The scored private set is hidden.
