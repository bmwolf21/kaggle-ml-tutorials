"""
02_baseline_model.py - LightGBM forecast with honest temporal validation.

Validation is the LAST 16 days of train (matching the 16-day test horizon), never
a random split: the future must not leak into the past. Because we model
y = log1p(sales), the validation RMSE on y equals the competition RMSLE.

Run:  python src/02_baseline_model.py
"""
import os
import sys
import datetime as dt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
import lightgbm as lgb

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "src"))
from features import build_features  # noqa: E402

RAW = os.path.join(HERE, "data", "raw")
SUB = os.path.join(HERE, "outputs", "submissions")
os.makedirs(SUB, exist_ok=True)

print("Building features...")
train, test, cols = build_features(RAW)
print(f"train {train.shape} | test {test.shape} | features {len(cols)}")

# --- Temporal split: last 16 days as validation -----------------------------
cutoff = train["date"].max() - pd.Timedelta(days=15)  # last 16 days inclusive
tr = train[train["date"] < cutoff]
va = train[train["date"] >= cutoff]
print(f"train < {cutoff.date()}: {len(tr):,} rows | valid (last 16d): {len(va):,} rows")

params = dict(objective="regression", n_estimators=2000, learning_rate=0.05,
              num_leaves=127, subsample=0.8, subsample_freq=1,
              colsample_bytree=0.7, reg_lambda=1.0, min_child_samples=50,
              random_state=42, verbose=-1, n_jobs=-1)

model = lgb.LGBMRegressor(**params)
model.fit(tr[cols], tr["y"], eval_set=[(va[cols], va["y"])],
          callbacks=[lgb.early_stopping(100, verbose=False)])

va_pred = model.predict(va[cols])
rmsle = np.sqrt(mean_squared_error(va["y"], va_pred))
print(f"\nValidation RMSLE (last 16 days): {rmsle:.5f}  (best_iter {model.best_iteration_})")

# baseline sanity check: predict-with-lag_16 naive (persistence) on validation
naive = np.sqrt(mean_squared_error(va["y"], va["lag_16"].fillna(0)))
print(f"Naive lag-16 persistence RMSLE:  {naive:.5f}")

# --- Feature importance -----------------------------------------------------
imp = pd.Series(model.feature_importances_, index=cols).sort_values(ascending=False)
print("\nTop 10 features:")
print(imp.head(10).to_string())

# --- Refit on ALL training data, predict test -------------------------------
print("\nRefitting on all data for submission...")
final = lgb.LGBMRegressor(**{**params, "n_estimators": model.best_iteration_ or 1000})
final.fit(train[cols], train["y"])
test_pred = np.expm1(final.predict(test[cols]))
test_pred = np.clip(test_pred, 0, None)   # sales cannot be negative

tag = dt.datetime.now().strftime("%Y%m%d_%H%M")
sub = pd.DataFrame({"id": test["id"], "sales": test_pred})
path = os.path.join(SUB, f"submission_lgbm_{tag}.csv")
sub.to_csv(path, index=False)
print(f"Wrote {path}  (mean predicted sales {test_pred.mean():.1f})")
