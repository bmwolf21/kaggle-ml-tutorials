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

### Step 1 — Exploratory Data Analysis (`src/01_eda.py`)

Ran a documented EDA script. Figures saved to `outputs/figures/`. Key findings:

**Target is balanced** — `Transported` is 50.4% True / 49.6% False. Accuracy (the
competition metric) is meaningful here; no resampling needed.

**Missingness is light but pervasive** — every feature column is ~2.0–2.5%
missing, and **24% of rows have at least one missing value**. No single column is
catastrophic, but we can't drop incomplete rows (we'd lose a quarter of the
data). → Imputation is required, and "is-missing" flags may carry signal.

**Compound ID fields decode into strong features:**
- `PassengerId = gggg_pp` → **6,217 travel groups**; party sizes range 1–8
  (4,805 solo travelers, the rest in groups). People in a group likely share a
  fate → group-level features should help.
- `Cabin = deck/num/side` → decks F & G dominate; `side` is P(ort)/S(tarboard),
  ~50/50; 199 cabins missing.

**CryoSleep is the dominant predictor — and it's logically linked to spend:**
- CryoSleep=True → **81.8%** transported; CryoSleep=False → only 32.9%.
- CryoSleep passengers have **exactly 0** spend across all 5 amenities (they're
  asleep). → We can impute missing CryoSleep from spend (any spend ⇒ awake) and
  impute missing spend to 0 when CryoSleep=True. This is a rare case where
  imputation is *deterministic*, not just statistical.

**Other signal:** HomePlanet matters a lot (Europa 66% vs Earth 42% transported);
VIP passengers transported slightly *less* (38% vs 51%).

**Planned feature engineering (Step 2):** split `PassengerId`→Group/GroupSize,
`Cabin`→Deck/Num/Side; `TotalSpend` + per-amenity + log transforms; deterministic
CryoSleep/spend imputation; group-size and "traveling alone" flags.

### Step 2 — Feature engineering (`src/features.py`)

Built a shared feature module so train and test are transformed identically.
24 features from the raw 13, driven by the EDA:

- **ID decoding:** `Group`, `GroupSize`, `IsAlone` (from PassengerId);
  `Deck`, `Num`, `Side` (from Cabin); `Surname`, `FamilySize` (from Name).
- **Deterministic imputation** via the CryoSleep↔spend link: any spend ⇒ awake
  (fill CryoSleep=False); CryoSleep=True ⇒ fill missing amenities with 0.
- **Skew handling:** `log1p` of each amenity + `TotalSpend`; `HasSpend` flag.
- **Leak-safe group sizes:** computed on train+test *combined* (uses only IDs,
  never the target).
- Categoricals encoded as integer codes (NaN → its own code, which trees split on).

### Step 3 — Baseline model + first submission (`src/02_baseline_model.py`)

LightGBM, 5-fold stratified CV with early stopping.

| Fold | Accuracy |
|------|----------|
| 1 | 0.8114 |
| 2 | 0.8148 |
| 3 | 0.8131 |
| 4 | 0.8142 |
| 5 | 0.8003 |
| **CV mean** | **0.8108 ± 0.0053** |

**Submitted → public leaderboard score: 0.80547.** The tiny CV→LB gap (~0.005)
confirms the cross-validation is trustworthy (no leakage, no overfit). This is a
strong first result — competitive Spaceship Titanic scores sit around 0.80–0.82.

**Lesson for the tutorial:** most of the lift here came from *decoding compound
fields* and the *deterministic CryoSleep/spend imputation* — domain logic beat
raw model tuning. A plain model on the raw columns typically scores ~0.79.

**Next:** tune / add XGBoost + ensemble, engineer group-fate features, then the
wildlife translation (`wildlife_translation/`).

<!-- Next steps get appended below as we go. -->
