# Wildlife Translation: From Store Sales to Population Forecasting

Angle 3 of the Store Sales project. We take the multi-series time-series
forecasting workflow and reapply it to a wildlife problem: forecasting
**population counts** across many monitoring series (site x species) over time.

This is the temporal-monitoring analogue of forecasting sales for many
store-family series. Where Spaceship Titanic was detection and House Prices was
abundance, this adds the **time-series forecasting** technique class.

Everything runs on a **simulated** monitoring program
(`00_simulate_monitoring_series.R`), so it is reproducible and uses no field
data. Because it is simulated, we keep the true future values, letting us measure
something the Kaggle leaderboard could only hint at: how much harder a genuinely
future window is than in-range validation.

## Why this translation is faithful

| Store Sales (Kaggle) | Wildlife monitoring (here) |
|----------------------|----------------------------|
| daily sales per (store, family) | monthly counts per (site, species) |
| 16-day forecast horizon | 6-month forecast horizon |
| weekly / yearly seasonality | breeding / phenology seasonality |
| oil price (macro driver) | temperature (climate driver) |
| promotions | management interventions |
| model `log1p(sales)` (RMSLE) | model `log1p(count)` (RMSLE) |
| horizon-safe lags (>= 16 days) | horizon-safe lags (>= 6 months) |
| trust the leaderboard over CV | trust a true future window over in-range CV |

## The scripts (mirror the Kaggle `src/` numbering)

| Wildlife | Kaggle analogue | Purpose |
|----------|-----------------|---------|
| `00_simulate_monitoring_series.R` | (Kaggle provides data) | build the reproducible program |
| `01_eda.R` | `src/01_eda.py` | seasonality, zeros, multi-year trend |
| `02_features.R` | `src/features.py` | horizon-safe lags, cyclical encodings |
| `03_forecast_model.R` | `src/02`, `src/03` | in-range vs true-future comparison |

Run in order:

```bash
Rscript 00_simulate_monitoring_series.R
Rscript 01_eda.R
Rscript 03_forecast_model.R   # sources 02_features.R
```

## The transferable lessons, restated for ecology

**1. The horizon rule.** We forecast 6 months ahead, so at prediction time the
most recent 6 months of counts are unknown. Every lag and rolling feature is
shifted by at least the horizon (`lag_6`, `lag_12`, rolling means shifted by 6).
A lag shorter than the horizon would use counts we will not have when the forecast
is actually made, inflating validation and failing in deployment. This is the
temporal form of not letting the future inform the past.

**2. In-range validation is optimistic for forecasting.** We compared an in-range
validation window (inside the training period) with the true future window (the
last 6 months, genuinely beyond training). The true future was about **0.066 RMSLE
harder** than in-range validation predicted, for both feature sets. This mirrors
Store Sales, where validation said 0.40 but the leaderboard (a real future window)
said 0.49. A forecasting model must be judged on a truly out-of-time window, not
an in-range split, exactly as a spatial model must be judged on spatial-block CV.

**3. Cyclical seasonal encodings generalize; raw time counters do not.** We encode
month as sin/cos, which repeat into the forecast window, rather than leaning on a
raw time-index counter that a tree cannot extend past its training range. (In this
simulation the trend covariates were genuine drivers, so removing them did not
change much; on Store Sales the trend proxy (oil price) was largely spurious, so
removing it also improved the leaderboard. Either way, prefer features that repeat
into the future.)

## Bridge to real field survey work

This is the shape of a real monitoring program: many site x species count series,
strong seasonality, a slow population trend, patchy zeros, and episodic
interventions. Forecasting future counts (for early warning, harvest setting, or
survey planning) demands horizon-safe features and an out-of-time evaluation, not
a random or in-range split, or the reported accuracy will be optimistic.

---

## Extracted patterns for the wildlife cookbook

Source competition for all entries below: **Store Sales - Time Series Forecasting**
(competition #3 in this workspace).

### Pattern: Horizon-safe lag and rolling features
- **Ecological problem:** forecasting counts H periods ahead, when the most recent
  H periods are unavailable at prediction time.
- **Technique:** shift every lag / rolling feature by at least H periods so it
  never uses data from inside the forecast window.
- **Key code:** `02_features.R` (lag/rolling block).
- **Status:** ready to extract.

### Pattern: Out-of-time evaluation for forecasts
- **Ecological problem:** in-range cross-validation overstates how well a model
  will forecast a genuinely future period.
- **Technique:** evaluate on a held-out window that is strictly later than all
  training data (and, where possible, on several such windows); expect it to be
  meaningfully harder than in-range CV, and report that number.
- **Key code:** `03_forecast_model.R` (in-range vs true-future split).
- **Status:** ready to extract.

### Pattern: Extrapolation-safe temporal features
- **Ecological problem:** raw time counters and trending covariates cannot be
  extended by tree models past their training range, so they can mislead on a
  future window.
- **Technique:** encode seasonality cyclically (sin/cos) so it repeats forward;
  be wary of any feature whose forecast-window values fall outside the training
  range.
- **Key code:** `02_features.R` (cyclical encodings), `03_forecast_model.R`.
- **Status:** ready to extract.
