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

---

## Extracted patterns for the wildlife cookbook

These are the reusable, competition-agnostic ecological techniques harvested from
this project. They are written in a standard template so that once several
competitions exist, a separate wildlife-modeling cookbook repo can be assembled
by pulling these blocks (each links back to its source, and the cookbook will not
reproduce the Kaggle content).

Source competition for all entries below: **Spaceship Titanic**
(<https://github.com/bmwolf21/spaceship-titanic-tutorial>).

### Pattern: Spatial-block cross-validation
- **Ecological problem:** honest performance estimates when survey sites are
  spatially autocorrelated and a naive random split leaks between neighbors.
- **Technique:** tile the study area into a grid, assign whole blocks to folds so
  train and test regions never touch, and report the block-CV score. Compare to
  random CV to quantify the optimism (here: +0.07 accuracy).
- **Key code:** `03_detection_model.R` (block assignment + `evaluate_cv`).
- **Status:** ready to extract.

### Pattern: Effort-corrected detection via deterministic imputation
- **Ecological problem:** survey effort bounds what can be detected; a zero-effort
  site is a structural zero, not a true absence, and must not be imputed like an
  ordinary missing value.
- **Technique:** encode deterministic rules (any detection implies effort > 0; a
  passive / zero-effort site implies zero method minutes), separating structural
  from statistical missingness before any model-based imputation.
- **Key code:** `02_features.R` (deterministic effort block).
- **Status:** ready to extract.

### Pattern: Neighbor-based covariate imputation
- **Ecological problem:** covariates that are constant within a sampling unit
  (land cover within a transect) go missing at some sites.
- **Technique:** fill from the mode/mean of the site's transect-mates rather than
  a global summary, which is both more accurate and structurally justified.
- **Key code:** `02_features.R` (`fill_by_group_mode`).
- **Status:** ready to extract.

### Pattern: Hierarchical sampling-ID decoding
- **Ecological problem:** nested survey structure (transect -> site) is hidden
  inside compound identifiers instead of being available as covariates.
- **Technique:** decode IDs into features such as transect size and site
  isolation, giving the model the sampling hierarchy explicitly.
- **Key code:** `02_features.R` (transect-structure block).
- **Status:** ready to extract.

### Pattern: Correlated-covariate importance caution
- **Ecological problem:** collinear habitat covariates (canopy cover and NDVI at
  r = 0.9) split their shared signal, so an importance ranking understates each
  and can mislead interpretation.
- **Technique:** always read variable importance alongside a correlation check;
  consider grouping or dropping collinear covariates before drawing conclusions.
- **Key code:** `03_detection_model.R` (importance + correlation report).
- **Status:** ready to extract.
