# Wildlife Translation: From Spaceship Titanic to Species Detection

Angle 3 of the project. Here we take the exact workflow developed for the
Spaceship Titanic Kaggle competition and reapply it to a wildlife / environmental
problem: **predicting whether a target species is detected at a survey site**
from habitat covariates and survey effort.

Everything runs on a **simulated** survey dataset (`00_simulate_survey_data.R`),
so it is fully reproducible and involves no sensitive field data. The simulation
was designed so that every construct in the Kaggle data has a real ecological
counterpart.

## Why this translation is faithful

| Spaceship Titanic (Kaggle) | Wildlife detection survey (here) |
|----------------------------|----------------------------------|
| `Transported` (0/1) target | `detected` (species seen at site, 0/1) |
| `PassengerId = gggg_pp` (travel group) | `site_id = Txxxx_pp` (sites nested in a transect) |
| `Cabin = deck/num/side` | `location_code = unit/plot/aspect` |
| `CryoSleep` (asleep) | `passive_site` (camera-only, no active search) |
| 5 spend amenities | minutes on 5 active survey methods |
| spend = 0 whenever asleep (deterministic) | effort = 0 whenever passive (deterministic) |
| `HomePlanet` (constant within group) | `land_cover` (constant within transect) |
| `Age` (continuous) | `shrub_height_cm` (continuous habitat covariate) |
| ~2% pervasive missingness | ~2.5% pervasive missingness |
| group-mode imputation of HomePlanet | transect-mode imputation of land_cover |
| gradient boosting (LightGBM/XGBoost) | random forest (`ranger`, the SDM standard) |
| public leaderboard = noisy held-out signal | **spatial-block CV = honest held-out signal** |
| correlated spend hides CryoSleep importance | correlated canopy/NDVI split their importance |

## The scripts (mirror the Kaggle `src/` numbering)

| Wildlife | Kaggle analogue | Purpose |
|----------|-----------------|---------|
| `00_simulate_survey_data.R` | (Kaggle provides data) | build the reproducible survey dataset |
| `01_eda.R` | `src/01_eda.py` | missingness, balance, effort link, spatial pattern |
| `02_features.R` | `src/features.py` | decode IDs, deterministic + neighbor imputation |
| `03_detection_model.R` | `src/02`–`04` | ranger model, random vs spatial CV, importance |

Run in order:

```bash
Rscript 00_simulate_survey_data.R
Rscript 01_eda.R
Rscript 03_detection_model.R   # sources 02_features.R
```

## The three transferable lessons, restated for ecology

**1. Domain logic beats algorithmic tuning.** On Kaggle, decoding the compound
IDs and using the CryoSleep/spend rule moved us from ~0.79 to ~0.805, far more
than tuning did. Here the same structural moves (decode the transect/site
hierarchy; use the passive-site/effort rule; recover `land_cover` from
transect-mates) are what a model cannot invent on its own. Ecological knowledge
is the feature engineering.

**2. Trust the honest held-out signal.** On Kaggle the public leaderboard
(half the test set) was noisy, so a higher cross-validation score that came from
hyperparameter tuning did not generalize. In ecology the trap is sharper and has
a name: **spatial autocorrelation**. Random k-fold CV lets a test site sit right
next to a training site, so the model interpolates and the score is inflated.
Our run shows this directly:

| CV scheme | Accuracy | AUC |
|-----------|----------|-----|
| Random 5-fold | 0.74 | 0.81 |
| Spatial-block | 0.67 | 0.76 |

The ~0.07 accuracy gap is pure optimism from ignoring space. Report the
spatial-block number.

**3. Importance is not predictive value when features are correlated.** On
Kaggle, `CryoSleep` (the strongest single predictor in EDA) barely registered in
gain-based importance because the spend columns proxied it. Here `canopy_cover`
and `ndvi` correlate at ~0.90, so a random forest splits their shared signal
between them and neither looks as important as it truly is. Always read
importance alongside a correlation check.

## What is deliberately different

- **Model:** random forest (`ranger`) instead of gradient boosting, because RF is
  the long-standing default for species distribution and detection models and it
  was already available. The workflow (features, imputation, honest CV,
  importance) is identical; only the estimator changed.
- **Extra ecological structure:** real spatial coordinates and autocorrelation,
  which the tabular Kaggle problem did not have and which motivate spatial-block
  CV.

## Bridge to real field survey work

This mirrors the shape of transect-based ungulate survey data: sites nested in
transects, effort-dependent detection, patchy covariates, and strong spatial
structure. The same pipeline would apply to real detection / occupancy data,
with spatial-block CV as the honest performance estimate rather than a naive
random split.
