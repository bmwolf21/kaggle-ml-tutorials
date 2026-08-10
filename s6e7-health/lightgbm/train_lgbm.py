"""Honest LightGBM baseline for S6E7.

Angle (see METHODOLOGY.md): GBDT with native categorical + NaN handling, class
weighting for the imbalance, and OOF decision-rule tuning for balanced accuracy.
This sets the reference OOF the other two agents blend against.

Writes:
  lightgbm/oof.csv         id,p_at-risk,p_fit,p_unhealthy   (out-of-fold, train ids)
  lightgbm/test_proba.csv  id,p_at-risk,p_fit,p_unhealthy   (mean over folds, test ids)
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "shared"))
from metric import CLASSES, balanced_accuracy, labels_from_proba, tune_decision_weights  # noqa

NUM = ["sleep_duration", "heart_rate", "bmi", "calorie_expenditure",
       "step_count", "exercise_duration", "water_intake"]
CAT = ["diet_type", "stress_level", "sleep_quality", "physical_activity_level",
       "smoking_alcohol", "gender"]
# missingness itself may carry signal -> NA indicators for the leakier columns
NA_IND = ["stress_level", "sleep_duration", "sleep_quality", "calorie_expenditure",
          "water_intake", "physical_activity_level"]


def prep(df):
    X = df[NUM + CAT].copy()
    for c in CAT:
        X[c] = X[c].astype("category")
    for c in NA_IND:
        X[f"na_{c}"] = df[c].isna().astype("int8")
    return X


def main():
    tr = pd.read_csv(ROOT / "data" / "train.csv")
    te = pd.read_csv(ROOT / "data" / "test.csv")
    folds = pd.read_csv(ROOT / "shared" / "folds.csv").set_index("id")["fold"]
    tr["fold"] = tr["id"].map(folds)
    assert tr["fold"].notna().all() and set(tr["fold"]) == set(range(5))

    y = tr["health_condition"].values
    Xtr, Xte = prep(tr), prep(te)
    cat_idx = [Xtr.columns.get_loc(c) for c in CAT]

    oof = np.zeros((len(tr), 3))
    test = np.zeros((len(te), 3))
    # PERF: force_row_wise avoids LightGBM's per-call row/col-wise probe (which
    # thrashed -> 141 min); num_threads=8 beats 16 here (8 physical cores, HT
    # oversubscription hurts). This config runs 5 folds in ~2 min. See METHODOLOGY.md.
    params = dict(n_estimators=600, learning_rate=0.03, num_leaves=63,
                  subsample=0.8, subsample_freq=1, colsample_bytree=0.7,
                  reg_lambda=2.0, min_child_samples=100, class_weight="balanced",
                  objective="multiclass", num_class=3, num_threads=8, n_jobs=8,
                  force_row_wise=True, random_state=42, verbose=-1)
    for k in range(5):
        tri = tr["fold"].values != k
        vai = tr["fold"].values == k
        m = LGBMClassifier(**params)
        m.fit(Xtr[tri], y[tri], categorical_feature=cat_idx)
        cls = list(m.classes_)
        order = [cls.index(c) for c in CLASSES]  # align to canonical order
        oof[vai] = m.predict_proba(Xtr[vai])[:, order]
        test += m.predict_proba(Xte)[:, order] / 5
        fold_ba = balanced_accuracy(y[vai], labels_from_proba(oof[vai]))
        print(f"  fold {k}: argmax bal-acc {fold_ba:.4f}")

    ba_argmax = balanced_accuracy(y, labels_from_proba(oof))
    w, ba_tuned = tune_decision_weights(oof, y)
    print(f"\nOOF bal-acc  argmax : {ba_argmax:.4f}")
    print(f"OOF bal-acc  tuned  : {ba_tuned:.4f}   weights={w.round(3)}")

    pd.DataFrame({"id": tr["id"], "p_at-risk": oof[:, 0], "p_fit": oof[:, 1],
                  "p_unhealthy": oof[:, 2]}).to_csv(ROOT / "lightgbm" / "oof.csv", index=False)
    pd.DataFrame({"id": te["id"], "p_at-risk": test[:, 0], "p_fit": test[:, 1],
                  "p_unhealthy": test[:, 2]}).to_csv(ROOT / "lightgbm" / "test_proba.csv", index=False)
    print("\nwrote lightgbm/oof.csv, lightgbm/test_proba.csv")


if __name__ == "__main__":
    main()
