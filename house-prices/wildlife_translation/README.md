# Wildlife Translation: From House Prices to Species Abundance

Angle 3 of the House Prices project. We take the full regression workflow built
for the Kaggle competition and reapply it to a wildlife problem: predicting
**animal density** at a survey site from habitat covariates.

Where the Spaceship Titanic translation was a **detection** problem (binary),
this is the **abundance / density** problem (continuous), which is the natural
ecological analogue of predicting a continuous `SalePrice`.

Everything runs on a **simulated** survey (`00_simulate_abundance_survey.R`), so
it is reproducible and uses no field data. The simulation was designed so each
House Prices construct has a real ecological counterpart.

## Why this translation is faithful

| House Prices (Kaggle) | Wildlife abundance survey (here) |
|-----------------------|----------------------------------|
| `SalePrice` (continuous, right-skewed) | `density` (continuous, right-skewed) |
| model `log(SalePrice)` | model `log(density)` |
| Garage / Pool present or absent | riparian zone / canopy present or absent |
| NA garage attrs when no garage | NA riparian attrs when no riparian zone |
| `ExterQual` Ex/Gd/TA/Fa (ordinal) | forage / cover quality Poor..Excellent (ordinal) |
| `LotFrontage` NA filled by Neighborhood | `soil_moisture` NA filled by region |
| `GrLivArea > 4000` outliers | aggregation-event (swarm) sites |
| Lasso/Ridge/ENet + GBR/XGB blend | glmnet Lasso + ranger + gbm blend |

## The scripts (mirror the Kaggle `src/` numbering)

| Wildlife | Kaggle analogue | Purpose |
|----------|-----------------|---------|
| `00_simulate_abundance_survey.R` | (Kaggle provides data) | build the reproducible survey |
| `01_eda.R` | `src/01_eda.py` | target skew, structural missingness, land cover |
| `02_features.R` | `src/features.py` | structural NA, ordinal, neighbor imputation |
| `03_abundance_model.R` | `src/03`, `src/04` | outliers, linear + tree blend, importance |

Run in order:

```bash
Rscript 00_simulate_abundance_survey.R
Rscript 01_eda.R
Rscript 03_abundance_model.R   # sources 02_features.R
```

## The transferable lessons, restated for ecology

**1. Distinguish structural from genuine missingness.** On House Prices, most NAs
meant "the house has no garage/pool," so they were filled with an explicit
"None"/0 rather than imputed. In a habitat survey the same trap appears: a missing
riparian-zone attribute usually means there is no riparian zone, not that it went
unrecorded. Filling those as "absent" (and their sizes as 0) is correct;
statistically imputing them invents habitat that is not there. Only genuinely
unmeasured covariates (here `soil_moisture`) get a real, neighbor-based fill.

**2. Encode ordinal habitat quality in its true order.** Forage and cover quality
(Poor < Fair < Good < Excellent) carry order that one-hot encoding throws away.
Mapping them to ordered integers keeps that signal, exactly as the House Prices
quality ratings were handled.

**3. Model the log of a right-skewed quantity, and handle anomalies honestly.**
Density, like sale price, is right-skewed, so we model `log(density)`. Transient
aggregation events (swarms) are the ecological version of the House Prices
mega-mansion outliers: we flag them with a standard Tukey rule and drop them from
the training side of each fold only, never from validation, so the cross-validated
error stays comparable to reality.

**4. Blend regularized linear models with trees.** On House Prices the big gain
came from combining Lasso/Ridge/ElasticNet with gradient boosting, because they
capture different structure. Here the same blend (glmnet Lasso + gbm) beats every
single model (blended CV RMSE 0.487 vs best single 0.502). Linear models are a
strong, underused baseline for ecological regression.

## What is deliberately different

- **Continuous target** (density), not the binary detection of the Spaceship
  Titanic translation, giving the cookbook a regression pattern to sit beside the
  classification one.
- **Models:** glmnet + ranger + gbm, the R analogues of the Kaggle stack.

## Bridge to real field survey work

This is the shape of density or abundance data: a right-skewed continuous
response, habitat features that are present at some sites and absent at others,
ordinal quality ratings, patchy covariates, and occasional aggregation anomalies.
The same pipeline applies to real density-estimation work, with log-scale
modeling, structural-NA handling, and honest anomaly treatment.

---

## Extracted patterns for the wildlife cookbook

Reusable, competition-agnostic techniques harvested from this project, in the
standard template so the separate wildlife-modeling cookbook repo can be
assembled by pulling these blocks.

Source competition for all entries below: **House Prices**
(competition #2 in this workspace).

### Pattern: Structural vs genuine missingness
- **Ecological problem:** a blank covariate can mean the feature is truly absent
  at the site (no water, no canopy) or merely unrecorded; treating them the same
  corrupts the model.
- **Technique:** fill "absent-feature" categoricals with an explicit "None" and
  their sizes with 0; reserve real (neighbor/median) imputation for genuinely
  unmeasured covariates.
- **Key code:** `02_features.R` (structural NA block).
- **Status:** ready to extract.

### Pattern: Ordinal habitat-quality encoding
- **Ecological problem:** ordered quality ratings (Poor..Excellent) lose their
  order under one-hot encoding.
- **Technique:** map to ordered integers so the model sees the ranking.
- **Key code:** `02_features.R` (QMAP).
- **Status:** ready to extract.

### Pattern: Log-scale density regression with honest anomaly handling
- **Ecological problem:** density/abundance is right-skewed and punctuated by
  aggregation events that distort fits.
- **Technique:** model `log(density)`; flag anomalies with a Tukey rule; drop them
  from the training side of each CV fold only, keeping validation intact so CV
  stays comparable to held-out reality.
- **Key code:** `02_features.R` (outlier flag), `03_abundance_model.R` (fold loop).
- **Status:** ready to extract.

### Pattern: Linear-plus-tree blending for ecological regression
- **Ecological problem:** no single learner captures all structure in habitat
  covariates.
- **Technique:** blend a regularized linear model (glmnet) with tree ensembles
  (ranger, gbm); weights tuned on out-of-fold predictions. Linear models are a
  strong, diverse partner to trees.
- **Key code:** `03_abundance_model.R` (blend section).
- **Status:** ready to extract.

### Pattern (recurring): Neighbor-based covariate imputation
- Also used here (`soil_moisture` filled by region median). First catalogued in
  the Spaceship Titanic translation; the cookbook will keep a single entry noting
  it recurs across competitions.
