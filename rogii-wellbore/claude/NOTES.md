# Claude resume note - ROGII Wellbore Geology Prediction

Status as of 2026-07-16. Read this first when picking the competition back up, then
read `../COLLABORATION.md` (shared with Codex) and its handoff log.

## Where we are
- **Task:** predict `TVT` (ft) for the masked toe of each horizontal well.
  Metric: plain RMSE. Crux: **group-CV by well** (`../shared/folds.csv`); the
  public LB is leaked/meaningless (3 test wells are also train wells). Full spec:
  `../shared/DATA_SPEC.md`.
- **My best model: `src/03_gr_correlation_model.py`, OOF RMSE 15.249 ft.**
  GR->TVT inversion (nearest match vs type-log and the lateral's own pre-PS
  GR-vs-TVT) + LightGBM on `dTVT = TVT - TVT_PS`. `oof.csv`/`test_pred.csv`
  currently hold this model's output.
- **Codex** (independent, `../codex/`): ~14.36 ft standalone (geometry anchor +
  ridge, adding offset-well priors). Stronger solo than me.
- **Blend: `../shared/blend.py` -> 14.086 ft** (weight ~0.33 me / 0.67 Codex),
  beats both. Submission: `../outputs/submissions/blend_claude_codex_20260716.csv`.
  Rerun `blend.py` after either agent refreshes an `oof.csv`.

## What I learned (don't re-derive)
- "Flat" (toe TVT = TVT at PS) = 15.91 ft. That's the bar.
- Geometry is useless: `corr(dTVT, dZ) = -0.13`, extrapolating slope = 117 ft.
  Wells are geosteered, so Z swings ~88 ft while TVT moves only ~11 ft. **GR is the
  only signal.** dTVT is small (mean|dTVT| 11 ft, p95 32 ft).
- **Failed experiment:** `src/05_gr_context_model.py` (GR-context + heavier
  smoothing) **regressed to 15.76** - heavier smoothing lost resolution on easy
  folds. Kept for the record; do not use it as-is.

## Next levers (in priority order)
1. **Proper windowed cross-correlation.** My 03 uses a noisy per-point nearest-GR
   match (hence fold variance 13-18). The real geosteering method is a windowed
   NCC with a small (shift, apparent-dip) search per window - I never actually
   built this. It's the most likely standalone improvement.
2. **Fold-3 diagnostic.** Fold 3 is the hardest fold for BOTH me (17.7) and Codex
   (17.0). Figure out what's different about those 154 wells (geology? azimuth?
   poor type-well overlap? long toes?). A fix there lifts everyone.
3. Keep my model DIVERSE from Codex's (trees + GR inversion vs his linear/offset).
   The blend gain depends on that divergence.

## How to run
```
python claude/src/03_gr_correlation_model.py   # my best model -> oof.csv, test_pred.csv, submission
python shared/blend.py                         # blend with Codex -> blended submission
```
Data lives in `data/raw/` (git-ignored; re-download: `kaggle competitions download
-c rogii-wellbore-geology-prediction -p data/raw`, extract CSVs, skip PNGs).

## Git etiquette (two agents, one repo)
Commit only your own subdir (`git add rogii-wellbore/claude ... COLLABORATION.md`),
never `git add -A` (it sweeps Codex's live work). `oof.csv` is git-ignored (150 MB,
over GitHub's limit). Commit, then `git pull --rebase`, then push.
