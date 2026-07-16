"""
03_ensemble.py - outlier removal + diverse model blend for House Prices.

Angle 1 (compete): the two big honest gains on this dataset are (a) dropping the
known high-leverage outliers and (b) blending regularized linear models with
gradient-boosted trees, which capture different structure.

Models (shared 5-fold CV):
  linear (one-hot + RobustScaler): LassoCV, RidgeCV, ElasticNetCV
  trees  (integer codes):          LightGBM, GradientBoostingRegressor
Blend weights are optimized on out-of-fold predictions.

Run:  python src/03_ensemble.py
"""
import os
import sys
import datetime as dt
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LassoCV, RidgeCV, ElasticNetCV
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import GradientBoostingRegressor
from scipy.optimize import minimize
import lightgbm as lgb

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "src"))
from features import engineer  # noqa: E402

RAW = os.path.join(HERE, "data", "raw")
SUB = os.path.join(HERE, "outputs", "submissions")
os.makedirs(SUB, exist_ok=True)

train = pd.read_csv(os.path.join(RAW, "train.csv"))
test = pd.read_csv(os.path.join(RAW, "test.csv"))

# --- Outlier removal --------------------------------------------------------
# De Cock (the dataset author) recommends dropping homes with GrLivArea > 4000;
# a few are enormous houses sold cheaply and distort every model.
before = len(train)
train = train[train["GrLivArea"] <= 4000].reset_index(drop=True)
print(f"Dropped {before - len(train)} outliers (GrLivArea > 4000)")

# --- Build the two feature views --------------------------------------------
both, y, n_train = engineer(train, test)
y = y.values

# Tree view: integer codes
tree = both.copy()
for c in tree.select_dtypes(include="category").columns:
    tree[c] = tree[c].cat.codes
Xtree, Xtree_test = tree.iloc[:n_train].values, tree.iloc[n_train:].values

# Linear view: one-hot (scaling handled inside each pipeline)
lin = pd.get_dummies(both, dummy_na=False)
Xlin, Xlin_test = lin.iloc[:n_train].values, lin.iloc[n_train:].values
print(f"tree features: {tree.shape[1]} | linear (one-hot) features: {lin.shape[1]}")

cv = KFold(n_splits=5, shuffle=True, random_state=42)
ALPHAS_L = np.logspace(-4, -1.5, 30)
ALPHAS_R = np.logspace(-1, 2, 30)


def make_models():
    return {
        "Lasso":   ("lin", make_pipeline(RobustScaler(),
                    LassoCV(alphas=ALPHAS_L, cv=5, max_iter=8000, random_state=42))),
        "Ridge":   ("lin", make_pipeline(RobustScaler(),
                    RidgeCV(alphas=ALPHAS_R))),
        "ENet":    ("lin", make_pipeline(RobustScaler(),
                    ElasticNetCV(alphas=ALPHAS_L, l1_ratio=[0.1, 0.5, 0.9],
                                 cv=5, max_iter=8000, random_state=42))),
        "LGBM":    ("tree", lgb.LGBMRegressor(
                    objective="regression", n_estimators=2000, learning_rate=0.02,
                    num_leaves=15, subsample=0.8, subsample_freq=1,
                    colsample_bytree=0.6, reg_lambda=1.0, min_child_samples=10,
                    random_state=42, verbose=-1)),
        "GBR":     ("tree", GradientBoostingRegressor(
                    n_estimators=1500, learning_rate=0.02, max_depth=3,
                    subsample=0.8, max_features=0.3, random_state=42)),
    }


def data_for(kind):
    return (Xlin, Xlin_test) if kind == "lin" else (Xtree, Xtree_test)


names = list(make_models().keys())
oof = {n: np.zeros(n_train) for n in names}
test_pred = {n: np.zeros(len(Xtree_test)) for n in names}

for tr, va in cv.split(Xtree):
    models = make_models()
    for n, (kind, model) in models.items():
        Xtr, Xte = data_for(kind)
        model.fit(Xtr[tr], y[tr])
        oof[n][va] = model.predict(Xtr[va])
        test_pred[n] += model.predict(Xte) / cv.n_splits

rmse = lambda a, b: np.sqrt(mean_squared_error(a, b))
print("\nIndividual CV RMSE(log):")
for n in names:
    print(f"  {n:6s} {rmse(y, oof[n]):.5f}")

# --- Optimize blend weights on OOF ------------------------------------------
OOF = np.column_stack([oof[n] for n in names])
TEST = np.column_stack([test_pred[n] for n in names])


def neg_score(w):
    return rmse(y, OOF @ w)


w0 = np.ones(len(names)) / len(names)
cons = ({"type": "eq", "fun": lambda w: w.sum() - 1},)
bnds = [(0, 1)] * len(names)
res = minimize(neg_score, w0, method="SLSQP", bounds=bnds, constraints=cons)
w = res.x
print("\nOptimized blend weights:")
for n, wi in zip(names, w):
    print(f"  {n:6s} {wi:.3f}")
blend_rmse = rmse(y, OOF @ w)
print(f"\nBlended CV RMSE(log): {blend_rmse:.5f}")

# --- Submission -------------------------------------------------------------
tag = dt.datetime.now().strftime("%Y%m%d_%H%M")
final = np.expm1(TEST @ w)
sub = pd.DataFrame({"Id": test["Id"], "SalePrice": final})
path = os.path.join(SUB, f"submission_blend_{tag}.csv")
sub.to_csv(path, index=False)
print(f"Wrote {path}")
