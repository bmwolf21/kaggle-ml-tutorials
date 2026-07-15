# Spaceship Titanic — Tutorial Log

> Competition: <https://www.kaggle.com/competitions/spaceship-titanic>
> Goal: predict whether a passenger was **Transported** to an alternate
> dimension (binary classification). Metric: **classification accuracy**.
>
> This file is the running record of every step, kept in enough detail to be
> turned into a standalone tutorial later. Newest entries at the bottom.

## The three angles

1. **Compete** — build models and submit to the leaderboard.
2. **Document** — this file.
3. **Translate** — `wildlife_translation/` reframes the same pipeline as a
   species-detection problem (see that folder's README once we reach it).

## Why this competition maps well to ecology

Spaceship Titanic looks artificial but its data shape is exactly what wildlife
survey data looks like:

- **Binary outcome** (`Transported`) ≈ species detected / not detected at a site.
- **Missing values everywhere** ≈ covariates unrecorded at some survey points.
- **Compound ID fields** (`PassengerId = gggg_pp`, `Cabin = deck/num/side`)
  ≈ hierarchical sampling structure (group / transect / site).
- **Mixed numeric + categorical covariates** ≈ habitat + spend/behavior variables
  that need encoding and imputation before modeling.

---

## Step log

### Step 0 — Workspace & environment setup

- Created `Kaggle/spaceship-titanic/` with `data/`, `notebooks/`, `src/`,
  `outputs/`, and `wildlife_translation/`.
- Installed `kaggle`, `xgboost`, `lightgbm` into the anaconda base env.
- Set up `.gitignore` so raw data and credentials never get committed.
- **Pending:** Kaggle API credentials (`kaggle.json`) to enable data download.

<!-- Next steps get appended below as we go. -->
