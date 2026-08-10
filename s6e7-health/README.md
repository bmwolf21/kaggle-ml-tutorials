# S6E7 - multi-model blend and an information-ceiling study

Kaggle Playground Series S6E7 ("Predicting Student Health Risk"): an imbalanced 3-class
classification (at-risk 86% / unhealthy 8% / fit 6%) scored on **balanced accuracy**.
This project runs several independent modeling pipelines across different model families,
combines them under one honest cross-validation contract, and tests how far model
diversity can push the score. Short answer: not at all here, and the write-up explains why
that is the interesting result.

- **Task / metric / data:** `shared/DATA_SPEC.md`
- **Method, experiment, and findings:** `METHODOLOGY.md`
- **Shared contract:** `shared/folds.csv` (stratified 5-fold, seed 42),
  `shared/metric.py` (balanced accuracy + decision-rule tuner),
  `shared/blend.py` / `shared/stack.py` (the ensemble referee).

## Layout
```
data/          train/test/sample_submission (git-ignored; regenerate from Kaggle)
shared/        DATA_SPEC.md, metric.py, make_folds.py, folds.csv, blend.py, stack.py
lightgbm/      LightGBM baseline + linear (logreg) and kernel (Nystroem) probes
catboost/      CatBoost + engineered features, plus generative (QDA/NB) and kNN probes
neural_net/    neural tabular (entity-embedding MLP)
wildlife_translation/   ecological analogue (see its own README)
```

## Deliverable per pipeline
Each pipeline writes `<pipeline>/oof.csv` and `<pipeline>/test_proba.csv`, columns
`id,p_at-risk,p_fit,p_unhealthy`. Extra probes add tagged files `oof__<tag>.csv`. The
referee blends the out-of-fold probabilities, tunes one decision rule on OOF, applies it to
the blended test probabilities, and writes the submission (labels).

## Run
```bash
python shared/make_folds.py     # once; writes shared/folds.csv
python shared/metric.py         # self-test of the metric + decision-rule tuner
# train each pipeline, then:
python shared/blend.py          # blend + decision rule + submission
python shared/stack.py          # stacking meta-learner (comparison)
```

## Result
Single best model, weighted blend, and stacking meta-learner all reach OOF balanced
accuracy **0.9497**; public leaderboard **0.94967** (CV to LB gap 0.00003). The data sits
at an information ceiling. See `METHODOLOGY.md`.
