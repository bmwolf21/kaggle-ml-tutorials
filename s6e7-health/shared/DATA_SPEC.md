# DATA_SPEC - Playground Series S6E7 ("Predicting Student Health Risk")

Single source of truth for the data. If you notice something wrong here, fix it and
note it in METHODOLOGY.md - do not silently diverge.

## Task
Multiclass classification. Predict `health_condition` in {`at-risk`, `fit`, `unhealthy`}
for each `id` in the test set.

## Metric: BALANCED ACCURACY (macro-average of per-class recall)
- All three classes count equally despite the imbalance below.
- Predicting `at-risk` everywhere => 0.333. Majority-class shortcuts are worthless.
- Submission is a HARD LABEL, but model probabilities and tune the decision rule:
  `shared/metric.py` -> `tune_decision_weights()` (weighted argmax, fit on OOF only).

## Files (in `data/`, git-ignored)
| file | rows | cols |
|------|------|------|
| train.csv | 690,088 | 15 (id + target + 13 features) |
| test.csv  | 295,753 | 14 (id + 13 features) |
| sample_submission.csv | 295,753 | id, health_condition (string label) |

## Target balance (train)
| class | count | share |
|-------|------:|------:|
| at-risk   | 592,561 | 85.87% |
| unhealthy |  57,724 |  8.36% |
| fit       |  39,803 |  5.77% |

## Features (13)
**Numeric (7):** `sleep_duration`, `heart_rate`, `bmi`, `calorie_expenditure`,
`step_count`, `exercise_duration`, `water_intake`.
**Categorical (6):** `diet_type` (veg/non-veg/...), `stress_level` (low/.../high),
`sleep_quality` (poor/average/...), `physical_activity_level` (sedentary/moderate/active/...),
`smoking_alcohol` (yes/no), `gender` (male/female/other).

## Missingness (train, fraction NA) - NOT negligible, handle it deliberately
```
stress_level            0.120     water_intake             0.063
sleep_duration          0.110     physical_activity_level  0.053
sleep_quality           0.085     smoking_alcohol          0.041
calorie_expenditure     0.077     gender                   0.031
                                  step_count               0.020
                                  bmi                      0.020
```
`heart_rate`, `exercise_duration`, `diet_type` have little/no missingness.
Missingness itself may be informative - consider NA-indicator features.

## Shared contract (do not break)
- **Folds:** `shared/folds.csv` (id, fold) - stratified 5-fold, seed 42. LOAD it; never
  regenerate. No group structure exists, so this is the honest split.
- **Class order:** `shared/metric.py::CLASSES = ["at-risk","fit","unhealthy"]`. Every
  proba matrix uses these columns in this order.
- **OOF deliverable:** each agent writes `<agent>/oof.csv` (id, p_at-risk, p_fit,
  p_unhealthy) aligned to train ids, and `<agent>/test_proba.csv` (same columns) aligned
  to test ids. The blend referee combines OOF proba, tunes one decision rule, applies it
  to the blended test proba. Submit LABELS, never probabilities.
