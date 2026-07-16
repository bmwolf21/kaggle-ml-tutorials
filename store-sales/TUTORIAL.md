# Store Sales - Time Series Forecasting - Tutorial Log

> Competition: <https://www.kaggle.com/competitions/store-sales-time-series-forecasting>
> Goal: forecast daily `sales` for each (store, product family) for the 16 days
> after the training period.
> Metric: **RMSLE** (root mean squared log error), so we model `log1p(sales)`.
>
> Running record, detailed enough to become a standalone tutorial. Competition #3,
> same three-angle playbook. This one adds the **time-series** technique class.

## The three angles

1. **Compete** - build forecasts and submit.
2. **Document** - this file.
3. **Translate** - `wildlife_translation/` reframes the pipeline as forecasting
   **population counts** across many monitoring series (site x species) over time.

## Why this competition maps well to ecology

Spaceship Titanic was detection (binary), House Prices was abundance (regression).
Store Sales is **temporal forecasting across many series**, the population-
monitoring analogue:

- **Many parallel series** (54 stores x 33 families = 1,782) is exactly a
  monitoring program with many site x species series.
- **Seasonality** (weekly and yearly sales cycles) mirrors biological seasonality
  (breeding, migration, phenology).
- **External drivers**: an oil-price series (Ecuador's economy) is the analogue of
  a climate driver (temperature, rainfall, NDVI); holidays/events are the analogue
  of episodic events (weather anomalies, food pulses).
- **The horizon rule (the key time-series lesson):** we forecast 16 days ahead, so
  at prediction time we do not know the most recent 16 days of sales. Any lag or
  rolling feature must be shifted by at least the horizon, or it leaks the future.
  This is the temporal analogue of spatial-block CV: respect the ordering, never
  let the future inform the past.

## Data

- `train.csv` 3.0M rows, daily sales 2013-01-01 to 2017-08-15.
- `test.csv` 28,512 rows, 2017-08-16 to 2017-08-31 (the 16-day horizon).
- `stores.csv` (city, state, type, cluster), `oil.csv` (daily price, has gaps),
  `holidays_events.csv`, `transactions.csv`.

## Step log

### Step 0 - Setup

- Downloaded all files to `data/raw/`. Standard project structure created.

### Step 1 - EDA (`src/01_eda.py`)

- **Target:** 31.3% of rows are zero sales; raw skew 7.36, log1p skew 0.41. Model
  `log1p(sales)`, which also matches the RMSLE metric.
- **Weekly seasonality:** weekends are highest (Sun 463, Sat 433) vs a midweek low
  (Thu 284).
- **Yearly seasonality:** December peaks (454) for holiday shopping; otherwise flat.
- **Promotions dominate:** mean sales rise monotonically with `onpromotion`
  (0 -> 158, 1-5 -> 645, 6-50 -> 1540, 50+ -> 3540). This will be a top feature.
- Figures: overall trend, day/month seasonality, an example series.

**Ecological read:** weekly/yearly cycles are biological seasonality; promotions
are like a management intervention (supplemental feeding) that spikes local counts.

### Step 2 - Horizon-aware features (`src/features.py`)

The central time-series decision. We forecast 16 days ahead, so at prediction
time the most recent 16 days of sales are unknown. Therefore **every lag/rolling
feature is shifted by at least 16 days** (`lag_16`, `lag_21`, `lag_28`, and
rolling means shifted by 16). Lags shorter than the horizon would leak future
values and inflate validation. Also merged: calendar parts, `onpromotion`, store
metadata (type/cluster/city/state), national-holiday flag, and oil price
(forward-filled over its weekend gaps).

### Step 3 - Baseline + the validation-to-leaderboard gap (`src/02_baseline_model.py`)

LightGBM on `log1p(sales)`, validated on the last 16 days of train (matching the
16-day horizon), never a random split.

- **Validation RMSLE: 0.400** vs **0.697 for naive lag-16 persistence**, so the
  model clearly adds signal on the holdout.
- **Public LB: 0.615.** A large gap, and notably worse than simple by-series
  seasonal-average benchmarks (~0.43-0.46) usually score here.

**This gap is the headline time-series lesson.** A big validation-to-test gap
that goes the wrong way (LB worse than a trivial benchmark) signals a real
weakness, not just a hard test window. Diagnosis:
1. **Trend/macro features do not extrapolate.** `dcoilwtico`, `year`, and
   `dayofyear` ranked high; tree models cannot predict beyond the range they saw,
   so using them as time proxies generalizes poorly to a future window.
2. **Holdout representativeness.** Even an honest last-16-days holdout can mislead
   if it is easier than the true forecast window, the temporal analogue of the
   spatial-CV lesson.

**Fix planned (Step 4):** drop or down-weight non-extrapolating trend features,
lean on cyclical seasonality (month, day-of-week, week-of-year) plus the
horizon-safe lags, and validate on multiple recent windows so the estimate is not
hostage to one period.

### Step 4 - Closing the gap: extrapolation-safe features (`src/03_seasonal_model.py`)

Dropped `year`, raw `dayofyear`, and `dcoilwtico`; added cyclical sin/cos
encodings of day-of-year, month, and day-of-week (which repeat into the future
instead of extrapolating). Validated on three consecutive 16-day windows.

- **Multi-window validation: 0.404 / 0.399 / 0.400, mean 0.40111 +/- 0.0022.**
  Rock-stable, and essentially identical to the baseline's validation.
- **Public LB: 0.491**, a large improvement from the baseline's 0.615.

**The definitive lesson.** The validation could not tell the two models apart
(both ~0.40) because every validation window lies inside the training range, where
the trend features still work. The difference showed up only on a truly future
window, the leaderboard. **In-range validation is blind to extrapolation failure.**
The fix was not a better validation scheme but a priori feature reasoning: keep
only features that repeat into the forecast window (seasonality, horizon-safe
lags), drop counters and macro trends that a tree cannot extend.

The residual 0.40-to-0.49 gap is the honest cost of forecasting a genuinely unseen
period, and it is as small as in-range tools can make visible.
