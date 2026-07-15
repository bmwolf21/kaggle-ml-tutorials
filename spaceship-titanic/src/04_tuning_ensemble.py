"""
04_tuning_ensemble.py - tuned LightGBM + a diverse 4-model ensemble.

Angle 1 (compete): one disciplined pass at the ~0.82 honest ceiling.
  1. Tune LightGBM hyperparameters with Optuna (5-fold CV accuracy objective).
  2. Train 4 diverse models: tuned LGBM, XGBoost, CatBoost, HistGradientBoosting.
  3. Blend by out-of-fold accuracy (equal-weight average - robust, avoids
     overfitting fragile per-model weights).
Angle 2 (document): prints tuned params, per-model CV, and the blend for TUTORIAL.md.

Run:  python src/04_tuning_ensemble.py
"""
import os
import sys
import datetime as dt
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.ensemble import HistGradientBoostingClassifier
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
import optuna

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "src"))
from features import build_features  # noqa: E402

RAW = os.path.join(HERE, "data", "raw")
SUB = os.path.join(HERE, "outputs", "submissions")
os.makedirs(SUB, exist_ok=True)

train = pd.read_csv(os.path.join(RAW, "train.csv"))
test = pd.read_csv(os.path.join(RAW, "test.csv"))
X, y, X_test, cols = build_features(train, test)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


# --- 1. Tune LightGBM with Optuna ------------------------------------------
def objective(trial):
    params = dict(
        objective="binary",
        n_estimators=trial.suggest_int("n_estimators", 300, 900),
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.08, log=True),
        num_leaves=trial.suggest_int("num_leaves", 15, 63),
        min_child_samples=trial.suggest_int("min_child_samples", 10, 60),
        subsample=trial.suggest_float("subsample", 0.6, 1.0),
        subsample_freq=1,
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
        reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        random_state=42, verbose=-1,
    )
    scores = cross_val_score(lgb.LGBMClassifier(**params), X, y,
                             cv=cv, scoring="accuracy")
    return scores.mean()


print("Tuning LightGBM with Optuna (40 trials)...")
study = optuna.create_study(direction="maximize",
                            sampler=optuna.samplers.TPESampler(seed=42))
study.optimize(objective, n_trials=40, show_progress_bar=False)
best = study.best_params
print(f"Best CV accuracy in tuning: {study.best_value:.4f}")
print(f"Best params: {best}")


# --- 2. Model factories -----------------------------------------------------
def make_lgb():
    return lgb.LGBMClassifier(objective="binary", subsample_freq=1,
                              random_state=42, verbose=-1, **best)


def make_xgb():
    return xgb.XGBClassifier(
        n_estimators=800, learning_rate=0.02, max_depth=5, subsample=0.8,
        colsample_bytree=0.8, reg_lambda=1.0, eval_metric="logloss",
        random_state=42, verbosity=0)


def make_cat():
    return CatBoostClassifier(
        iterations=800, learning_rate=0.03, depth=6, l2_leaf_reg=3.0,
        random_state=42, verbose=0)


def make_hist():
    return HistGradientBoostingClassifier(
        max_iter=600, learning_rate=0.03, max_leaf_nodes=31,
        l2_regularization=1.0, random_state=42)


def run_cv(make_model, name):
    oof = np.zeros(len(X))
    test_pred = np.zeros(len(X_test))
    for tr, va in cv.split(X, y):
        m = make_model()
        m.fit(X.iloc[tr], y.iloc[tr])
        oof[va] = m.predict_proba(X.iloc[va])[:, 1]
        test_pred += m.predict_proba(X_test)[:, 1] / cv.n_splits
    acc = accuracy_score(y, (oof > 0.5).astype(int))
    print(f"{name:16s} OOF acc {acc:.4f}")
    return oof, test_pred, acc


print("\nTraining diverse models with 5-fold CV...")
models = {
    "LightGBM(tuned)": make_lgb,
    "XGBoost": make_xgb,
    "CatBoost": make_cat,
    "HistGBM": make_hist,
}
oofs, tests, accs = {}, {}, {}
for name, fn in models.items():
    oofs[name], tests[name], accs[name] = run_cv(fn, name)

# --- 3. Equal-weight blend --------------------------------------------------
oof_blend = np.mean(list(oofs.values()), axis=0)
test_blend = np.mean(list(tests.values()), axis=0)
blend_acc = accuracy_score(y, (oof_blend > 0.5).astype(int))
print(f"\n{'Equal-weight blend':16s} OOF acc {blend_acc:.4f}")

# choose best of single-tuned-lgb vs blend (the two honest candidates)
if blend_acc >= accs["LightGBM(tuned)"]:
    final_name, final_test, final_acc = "blend4", test_blend, blend_acc
else:
    final_name, final_test, final_acc = "lgbm_tuned", tests["LightGBM(tuned)"], accs["LightGBM(tuned)"]
print(f"\nSubmitting: {final_name} (OOF {final_acc:.4f})")

tag = dt.datetime.now().strftime("%Y%m%d_%H%M")
sub = pd.DataFrame({"PassengerId": test["PassengerId"],
                    "Transported": (final_test > 0.5).astype(bool)})
path = os.path.join(SUB, f"submission_{final_name}_{tag}.csv")
sub.to_csv(path, index=False)
print(f"Wrote {path}")
