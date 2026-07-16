"""
04_stack.py - honest CV + stacking for House Prices.

Fixes the evaluation flaw in 03: outliers are dropped only from the TRAINING
side of each fold, never from validation, so the CV RMSE is directly comparable
to the leaderboard again. Adds XGBoost for tree diversity and compares a weighted
blend against a Ridge stacker.

Run:  python src/04_stack.py
"""
import os
import sys
import datetime as dt
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LassoCV, RidgeCV, ElasticNetCV, Ridge
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import GradientBoostingRegressor
from scipy.optimize import minimize
import lightgbm as lgb
import xgboost as xgb

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "src"))
from features import engineer  # noqa: E402

RAW = os.path.join(HERE, "data", "raw")
SUB = os.path.join(HERE, "outputs", "submissions")
os.makedirs(SUB, exist_ok=True)

train = pd.read_csv(os.path.join(RAW, "train.csv"))
test = pd.read_csv(os.path.join(RAW, "test.csv"))

both, y, n_train = engineer(train, test)
y = y.values
# Outlier mask on the FULL training set (removed only from training folds).
outlier = (train["GrLivArea"] > 4000).values

tree = both.copy()
for c in tree.select_dtypes(include="category").columns:
    tree[c] = tree[c].cat.codes
Xtree, Xtree_test = tree.iloc[:n_train].values, tree.iloc[n_train:].values

lin = pd.get_dummies(both, dummy_na=False)
Xlin, Xlin_test = lin.iloc[:n_train].values, lin.iloc[n_train:].values

cv = KFold(n_splits=5, shuffle=True, random_state=42)
AL = np.logspace(-4, -1.5, 30)
AR = np.logspace(-1, 2, 30)


def make_models():
    return {
        "Lasso": ("lin", make_pipeline(RobustScaler(),
                  LassoCV(alphas=AL, cv=5, max_iter=8000, random_state=42))),
        "Ridge": ("lin", make_pipeline(RobustScaler(), RidgeCV(alphas=AR))),
        "ENet":  ("lin", make_pipeline(RobustScaler(),
                  ElasticNetCV(alphas=AL, l1_ratio=[0.1, 0.5, 0.9], cv=5,
                               max_iter=8000, random_state=42))),
        "LGBM":  ("tree", lgb.LGBMRegressor(
                  objective="regression", n_estimators=2000, learning_rate=0.02,
                  num_leaves=15, subsample=0.8, subsample_freq=1,
                  colsample_bytree=0.6, reg_lambda=1.0, min_child_samples=10,
                  random_state=42, verbose=-1)),
        "GBR":   ("tree", GradientBoostingRegressor(
                  n_estimators=1500, learning_rate=0.02, max_depth=3,
                  subsample=0.8, max_features=0.3, random_state=42)),
        "XGB":   ("tree", xgb.XGBRegressor(
                  n_estimators=2000, learning_rate=0.02, max_depth=3,
                  subsample=0.8, colsample_bytree=0.5, reg_lambda=1.0,
                  random_state=42, verbosity=0)),
    }


def data_for(kind):
    return (Xlin, Xlin_test) if kind == "lin" else (Xtree, Xtree_test)


names = list(make_models().keys())
oof = {n: np.zeros(n_train) for n in names}
test_pred = {n: np.zeros(len(Xtree_test)) for n in names}

for tr, va in cv.split(Xtree):
    tr_clean = tr[~outlier[tr]]           # drop outliers from TRAIN only
    for n, (kind, model) in make_models().items():
        Xtr, Xte = data_for(kind)
        model.fit(Xtr[tr_clean], y[tr_clean])
        oof[n][va] = model.predict(Xtr[va])
        test_pred[n] += model.predict(Xte) / cv.n_splits

rmse = lambda a, b: np.sqrt(mean_squared_error(a, b))
print("Individual CV RMSE(log) [honest: full validation]:")
for n in names:
    print(f"  {n:6s} {rmse(y, oof[n]):.5f}")

OOF = np.column_stack([oof[n] for n in names])
TEST = np.column_stack([test_pred[n] for n in names])

# --- Weighted blend (optimized on OOF) --------------------------------------
cons = ({"type": "eq", "fun": lambda w: w.sum() - 1},)
res = minimize(lambda w: rmse(y, OOF @ w), np.ones(len(names)) / len(names),
               method="SLSQP", bounds=[(0, 1)] * len(names), constraints=cons)
w = res.x
blend_rmse = rmse(y, OOF @ w)
print("\nBlend weights:", {n: round(wi, 3) for n, wi in zip(names, w)})
print(f"Weighted blend CV RMSE(log): {blend_rmse:.5f}")

# --- Ridge stacker (meta-model on OOF) --------------------------------------
meta = Ridge(alpha=1.0)
# honest stack CV: fit meta on OOF via the same folds
stack_oof = np.zeros(n_train)
for tr, va in cv.split(OOF):
    meta.fit(OOF[tr], y[tr])
    stack_oof[va] = meta.predict(OOF[va])
stack_rmse = rmse(y, stack_oof)
meta.fit(OOF, y)
print(f"Ridge stack   CV RMSE(log): {stack_rmse:.5f}")

# --- Pick the better and submit ---------------------------------------------
if stack_rmse < blend_rmse:
    final_log, tag_name, best = meta.predict(TEST), "stack", stack_rmse
else:
    final_log, tag_name, best = TEST @ w, "blend", blend_rmse
print(f"\nChosen: {tag_name} (CV {best:.5f})")

tag = dt.datetime.now().strftime("%Y%m%d_%H%M")
sub = pd.DataFrame({"Id": test["Id"], "SalePrice": np.expm1(final_log)})
path = os.path.join(SUB, f"submission_{tag_name}_{tag}.csv")
sub.to_csv(path, index=False)
print(f"Wrote {path}")
