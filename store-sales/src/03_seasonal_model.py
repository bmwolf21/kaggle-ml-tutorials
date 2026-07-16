"""
03_seasonal_model.py - close the validation-to-LB gap.

Two changes from the baseline:
1. Drop non-extrapolating trend features (year, raw dayofyear, oil price) and rely
   on cyclical seasonal encodings (sin/cos) that repeat into the future.
2. Validate on THREE consecutive 16-day windows, not one, so the estimate is not
   hostage to a single (possibly easy) period.

Run:  python src/03_seasonal_model.py
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
train, test, all_cols = build_features(RAW)

# Feature list: drop non-extrapolating trend proxies, add cyclical encodings.
DROP = {"year", "dayofyear", "dcoilwtico"}
cols = [c for c in all_cols if c not in DROP] + [
    "doy_sin", "doy_cos", "month_sin", "month_cos", "dow_sin", "dow_cos"]
print(f"features: {len(cols)} (dropped {sorted(DROP)}, added cyclical)")

params = dict(objective="regression", n_estimators=2000, learning_rate=0.05,
              num_leaves=127, subsample=0.8, subsample_freq=1,
              colsample_bytree=0.7, reg_lambda=1.0, min_child_samples=50,
              random_state=42, verbose=-1, n_jobs=-1)

# --- Multi-window temporal validation ---------------------------------------
max_date = train["date"].max()
rmsles, best_iters = [], []
for k in range(3):
    v_end = max_date - pd.Timedelta(days=16 * k)
    v_start = v_end - pd.Timedelta(days=15)
    tr = train[train["date"] < v_start]
    va = train[(train["date"] >= v_start) & (train["date"] <= v_end)]
    m = lgb.LGBMRegressor(**params)
    m.fit(tr[cols], tr["y"], eval_set=[(va[cols], va["y"])],
          callbacks=[lgb.early_stopping(100, verbose=False)])
    r = np.sqrt(mean_squared_error(va["y"], m.predict(va[cols])))
    rmsles.append(r); best_iters.append(m.best_iteration_ or 1000)
    print(f"  window {k} ({v_start.date()}..{v_end.date()}): RMSLE {r:.5f}")

print(f"\nMean multi-window RMSLE: {np.mean(rmsles):.5f} +/- {np.std(rmsles):.5f}")

# --- Refit on all data, predict test ----------------------------------------
n_est = int(np.mean(best_iters))
final = lgb.LGBMRegressor(**{**params, "n_estimators": n_est})
final.fit(train[cols], train["y"])
test_pred = np.clip(np.expm1(final.predict(test[cols])), 0, None)

tag = dt.datetime.now().strftime("%Y%m%d_%H%M")
sub = pd.DataFrame({"id": test["id"], "sales": test_pred})
path = os.path.join(SUB, f"submission_seasonal_{tag}.csv")
sub.to_csv(path, index=False)
print(f"Wrote {path}  (mean predicted sales {test_pred.mean():.1f})")
