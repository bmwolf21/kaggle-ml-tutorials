"""
02_baseline_model.py - first cross-validated model + leaderboard submission.

Angle 1 (compete): LightGBM with 5-fold stratified CV to estimate accuracy,
then refit on all data and predict the test set.
Angle 2 (document): prints per-fold and mean CV accuracy for TUTORIAL.md.

Run:  python src/02_baseline_model.py
Writes: outputs/submissions/submission_<tag>.csv
"""
import os
import sys
import datetime as dt
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
import lightgbm as lgb

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "src"))
from features import build_features  # noqa: E402

RAW = os.path.join(HERE, "data", "raw")
SUB = os.path.join(HERE, "outputs", "submissions")
os.makedirs(SUB, exist_ok=True)

train = pd.read_csv(os.path.join(RAW, "train.csv"))
test = pd.read_csv(os.path.join(RAW, "test.csv"))
X, y, X_test, cols = build_features(train, test)
print(f"Features ({len(cols)}): {cols}")

params = dict(
    objective="binary",
    n_estimators=600,
    learning_rate=0.03,
    num_leaves=31,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    random_state=42,
    verbose=-1,
)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof = np.zeros(len(X))
test_pred = np.zeros(len(X_test))
scores = []

for fold, (tr, va) in enumerate(cv.split(X, y), 1):
    model = lgb.LGBMClassifier(**params)
    model.fit(
        X.iloc[tr], y.iloc[tr],
        eval_set=[(X.iloc[va], y.iloc[va])],
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )
    p_va = model.predict_proba(X.iloc[va])[:, 1]
    oof[va] = p_va
    test_pred += model.predict_proba(X_test)[:, 1] / cv.n_splits
    acc = accuracy_score(y.iloc[va], (p_va > 0.5).astype(int))
    scores.append(acc)
    print(f"fold {fold}: acc = {acc:.4f}  (best_iter={model.best_iteration_})")

oof_acc = accuracy_score(y, (oof > 0.5).astype(int))
print(f"\nCV mean acc: {np.mean(scores):.4f} +/- {np.std(scores):.4f}")
print(f"OOF acc:     {oof_acc:.4f}")

# --- Write submission -------------------------------------------------------
tag = dt.datetime.now().strftime("%Y%m%d_%H%M")
sub = pd.DataFrame({
    "PassengerId": test["PassengerId"],
    "Transported": (test_pred > 0.5).astype(bool),
})
path = os.path.join(SUB, f"submission_lgbm_{tag}.csv")
sub.to_csv(path, index=False)
print(f"\nWrote {path}  ({sub['Transported'].mean()*100:.1f}% predicted Transported)")
