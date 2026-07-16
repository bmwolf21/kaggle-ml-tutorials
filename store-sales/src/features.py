"""
features.py - horizon-aware feature engineering for Store Sales.

The forecast horizon is 16 days, so at prediction time the last 16 days of sales
are unknown. Every lag / rolling feature is therefore shifted by >= 16 days, so it
only uses information available at prediction time. Shorter lags would leak the
future and inflate validation.

build_features() returns the combined (train+test) engineered frame plus the
train/test split index and the log target.
"""
import os
import numpy as np
import pandas as pd

HORIZON = 16
LAGS = [16, 21, 28]
ROLL = [7, 14, 30]


def build_features(raw_dir):
    train = pd.read_csv(os.path.join(raw_dir, "train.csv"), parse_dates=["date"])
    test = pd.read_csv(os.path.join(raw_dir, "test.csv"), parse_dates=["date"])
    stores = pd.read_csv(os.path.join(raw_dir, "stores.csv"))
    oil = pd.read_csv(os.path.join(raw_dir, "oil.csv"), parse_dates=["date"])
    hol = pd.read_csv(os.path.join(raw_dir, "holidays_events.csv"), parse_dates=["date"])

    n_train = len(train)
    both = pd.concat([train, test], ignore_index=True)
    both = both.sort_values(["store_nbr", "family", "date"]).reset_index(drop=True)

    # --- Target on log scale --------------------------------------------------
    both["y"] = np.log1p(both["sales"])

    # --- Calendar features ----------------------------------------------------
    d = both["date"].dt
    both["dow"] = d.dayofweek
    both["day"] = d.day
    both["month"] = d.month
    both["year"] = d.year
    both["dayofyear"] = d.dayofyear
    both["weekofyear"] = d.isocalendar().week.astype(int)
    both["is_weekend"] = (both["dow"] >= 5).astype(int)
    both["is_month_start"] = d.is_month_start.astype(int)
    both["is_month_end"] = d.is_month_end.astype(int)

    # Cyclical encodings: unlike raw year/dayofyear counters, these repeat every
    # cycle, so they generalize to a future window instead of extrapolating.
    both["doy_sin"] = np.sin(2 * np.pi * both["dayofyear"] / 365.25)
    both["doy_cos"] = np.cos(2 * np.pi * both["dayofyear"] / 365.25)
    both["month_sin"] = np.sin(2 * np.pi * both["month"] / 12)
    both["month_cos"] = np.cos(2 * np.pi * both["month"] / 12)
    both["dow_sin"] = np.sin(2 * np.pi * both["dow"] / 7)
    both["dow_cos"] = np.cos(2 * np.pi * both["dow"] / 7)

    # --- Store metadata -------------------------------------------------------
    both = both.merge(stores, on="store_nbr", how="left")

    # --- National holiday flag ------------------------------------------------
    nat = hol[(hol["locale"] == "National") & (~hol["transferred"]) &
              (hol["type"] != "Work Day")]
    holidays = set(nat["date"].dt.normalize())
    both["is_holiday"] = both["date"].isin(holidays).astype(int)

    # --- Oil price (forward/back fill over weekend gaps) ----------------------
    oil = oil.set_index("date").asfreq("D")
    oil["dcoilwtico"] = oil["dcoilwtico"].ffill().bfill()
    both = both.merge(oil, on="date", how="left")
    both["dcoilwtico"] = both["dcoilwtico"].ffill().bfill()

    # --- Horizon-aware lag & rolling features (per series) --------------------
    # both is sorted by [store_nbr, family, date] with a RangeIndex, so
    # groupby(...).rolling(...).values aligns positionally back onto `both`.
    grp = both.groupby(["store_nbr", "family"])["y"]
    for lag in LAGS:
        both[f"lag_{lag}"] = grp.shift(lag)
    both["_ylag"] = grp.shift(HORIZON)          # base for horizon-safe rolling
    roll = both.groupby(["store_nbr", "family"])["_ylag"]
    for w in ROLL:
        both[f"rmean_{w}"] = roll.rolling(w, min_periods=1).mean().values
        both[f"rstd_{w}"] = roll.rolling(w, min_periods=2).std().values
    both = both.drop(columns=["_ylag"])

    # --- Encode categoricals as codes ----------------------------------------
    for c in ["family", "city", "state", "type"]:
        both[c] = both[c].astype("category").cat.codes

    feature_cols = (
        ["store_nbr", "family", "onpromotion", "cluster", "city", "state",
         "type", "is_holiday", "dcoilwtico",
         "dow", "day", "month", "year", "dayofyear", "weekofyear",
         "is_weekend", "is_month_start", "is_month_end"]
        + [f"lag_{l}" for l in LAGS]
        + [f"rmean_{w}" for w in ROLL] + [f"rstd_{w}" for w in ROLL]
    )

    both = both.sort_values("id").reset_index(drop=True)
    train_out = both[both["id"] < 3000888].copy()
    test_out = both[both["id"] >= 3000888].copy()
    return train_out, test_out, feature_cols
