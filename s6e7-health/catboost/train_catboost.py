"""CatBoost pipeline for S6E7.

Writes:
  catboost/oof.csv         id,p_at-risk,p_fit,p_unhealthy
  catboost/test_proba.csv  id,p_at-risk,p_fit,p_unhealthy
"""
import os
import sys
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEPS = ROOT / "catboost" / ".deps"
if DEPS.exists():
    sys.path.insert(0, str(DEPS))
sys.path.insert(0, str(ROOT / "shared"))

os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool

from metric import CLASSES, balanced_accuracy, labels_from_proba, tune_decision_weights


NUM = [
    "sleep_duration",
    "heart_rate",
    "bmi",
    "calorie_expenditure",
    "step_count",
    "exercise_duration",
    "water_intake",
]
CAT = [
    "diet_type",
    "stress_level",
    "sleep_quality",
    "physical_activity_level",
    "smoking_alcohol",
    "gender",
]
TARGET = "health_condition"
N_FOLDS = 5
SEED = 42


def _safe_div(num, den):
    return num / den.replace(0, np.nan)


def add_features(df):
    x = df[NUM + CAT].copy()

    for c in NUM + CAT:
        if df[c].isna().any():
            x[f"na_{c}"] = df[c].isna().astype("int8")

    sleep = df["sleep_duration"]
    steps = df["step_count"]
    exercise = df["exercise_duration"]
    calories = df["calorie_expenditure"]
    water = df["water_intake"]
    bmi = df["bmi"]
    heart = df["heart_rate"]

    x["sleep_debt_8h"] = (8.0 - sleep).clip(lower=0)
    x["sleep_excess_8h"] = (sleep - 8.0).clip(lower=0)
    x["activity_minutes"] = exercise.fillna(0) + steps.fillna(0) / 1000.0
    x["steps_per_exercise_min"] = _safe_div(steps, exercise + 1.0)
    x["calories_per_1k_steps"] = _safe_div(calories, steps / 1000.0 + 1.0)
    x["calories_per_exercise_min"] = _safe_div(calories, exercise + 1.0)
    x["water_per_bmi"] = _safe_div(water, bmi)
    x["water_per_calorie_1k"] = _safe_div(water, calories / 1000.0)
    x["bmi_x_heart_rate"] = bmi * heart
    x["sleep_x_stress_missing"] = sleep.isna().astype("int8") * df["stress_level"].isna().astype("int8")

    x["bmi_band"] = pd.cut(
        bmi,
        bins=[-np.inf, 18.5, 25.0, 30.0, np.inf],
        labels=["under", "normal", "over", "obese"],
    ).astype("object")
    x["sleep_band"] = pd.cut(
        sleep,
        bins=[-np.inf, 5.5, 7.0, 9.0, np.inf],
        labels=["very_low", "low", "normal", "high"],
    ).astype("object")
    x["steps_band"] = pd.cut(
        steps,
        bins=[-np.inf, 2500, 7500, 12000, np.inf],
        labels=["very_low", "low", "medium", "high"],
    ).astype("object")
    x["exercise_band"] = pd.cut(
        exercise,
        bins=[-np.inf, 15, 40, 70, np.inf],
        labels=["very_low", "low", "medium", "high"],
    ).astype("object")
    x["heart_rate_band"] = pd.cut(
        heart,
        bins=[-np.inf, 60, 75, 90, np.inf],
        labels=["low", "normal", "elevated", "high"],
    ).astype("object")

    x["stress_sleep"] = (
        df["stress_level"].fillna("__MISSING__").astype(str)
        + "_"
        + df["sleep_quality"].fillna("__MISSING__").astype(str)
    )
    x["diet_activity"] = (
        df["diet_type"].fillna("__MISSING__").astype(str)
        + "_"
        + df["physical_activity_level"].fillna("__MISSING__").astype(str)
    )

    cat_cols = CAT + [
        "bmi_band",
        "sleep_band",
        "steps_band",
        "exercise_band",
        "heart_rate_band",
        "stress_sleep",
        "diet_activity",
    ]
    for c in cat_cols:
        x[c] = x[c].astype("object").where(x[c].notna(), "__MISSING__").astype(str)

    x = x.replace([np.inf, -np.inf], np.nan)
    return x, cat_cols


def validate_contract(train, test, folds):
    assert train["id"].is_unique and test["id"].is_unique
    assert folds.index.is_unique
    assert train["id"].map(folds["fold"]).notna().all()
    assert set(train["id"].map(folds["fold"]).astype(int)) == set(range(N_FOLDS))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--folds",
        default="0,1,2,3,4",
        help="Comma-separated folds to train. Cached folds are reused.",
    )
    parser.add_argument("--combine-only", action="store_true")
    parser.add_argument("--tune-decision", action="store_true")
    args = parser.parse_args()
    selected_folds = [int(x) for x in args.folds.split(",") if x != ""]

    train = pd.read_csv(ROOT / "data" / "train.csv")
    test = pd.read_csv(ROOT / "data" / "test.csv")
    folds = pd.read_csv(ROOT / "shared" / "folds.csv").set_index("id")
    validate_contract(train, test, folds)

    train["fold"] = train["id"].map(folds["fold"]).astype(int)
    y = train[TARGET].to_numpy()

    x_train, cat_cols = add_features(train)
    x_test, _ = add_features(test)
    cat_idx = [x_train.columns.get_loc(c) for c in cat_cols]

    class_counts = train[TARGET].value_counts().to_dict()
    class_weights = [len(train) / (len(CLASSES) * class_counts[c]) for c in CLASSES]

    params = dict(
        loss_function="MultiClass",
        eval_metric="MultiClass",
        classes_count=len(CLASSES),
        class_names=CLASSES,
        class_weights=class_weights,
        iterations=300,
        learning_rate=0.075,
        depth=6,
        l2_leaf_reg=6.0,
        random_strength=0.8,
        bagging_temperature=0.5,
        bootstrap_type="Bayesian",
        one_hot_max_size=8,
        od_type="Iter",
        od_wait=40,
        allow_writing_files=False,
        thread_count=8,
        random_seed=SEED,
        verbose=50,
    )

    oof = np.zeros((len(train), len(CLASSES)), dtype=float)
    test_proba = np.zeros((len(test), len(CLASSES)), dtype=float)
    cache_dir = ROOT / "catboost" / "fold_cache"
    cache_dir.mkdir(exist_ok=True)

    for fold in selected_folds:
        cache_path = cache_dir / f"fold_{fold}.npz"
        if cache_path.exists():
            print(f"fold {fold}: using cached {cache_path}", flush=True)
            continue
        if args.combine_only:
            raise FileNotFoundError(f"missing cached fold: {cache_path}")

        is_valid = train["fold"].to_numpy() == fold
        print(
            f"fold {fold}: train={int((~is_valid).sum())} valid={int(is_valid.sum())}",
            flush=True,
        )
        tr_pool = Pool(x_train.loc[~is_valid], y[~is_valid], cat_features=cat_idx)
        va_pool = Pool(x_train.loc[is_valid], y[is_valid], cat_features=cat_idx)
        te_pool = Pool(x_test, cat_features=cat_idx)

        model = CatBoostClassifier(**params)
        model.fit(tr_pool, eval_set=va_pool, use_best_model=True)

        model_classes = [str(c) for c in model.classes_]
        order = [model_classes.index(c) for c in CLASSES]
        valid_proba = model.predict_proba(va_pool)[:, order]
        fold_test_proba = model.predict_proba(te_pool)[:, order]

        fold_score = balanced_accuracy(y[is_valid], labels_from_proba(valid_proba))
        print(f"fold {fold}: argmax balanced-accuracy {fold_score:.6f}")
        np.savez_compressed(
            cache_path,
            valid_idx=np.flatnonzero(is_valid),
            valid_proba=valid_proba,
            test_proba=fold_test_proba,
            score=fold_score,
        )
        print(f"fold {fold}: cached {cache_path}", flush=True)

    missing = [k for k in range(N_FOLDS) if not (cache_dir / f"fold_{k}.npz").exists()]
    if missing:
        print(f"missing folds {missing}; not writing final CSVs yet")
        return

    for fold in range(N_FOLDS):
        cached = np.load(cache_dir / f"fold_{fold}.npz")
        oof[cached["valid_idx"]] = cached["valid_proba"]
        test_proba += cached["test_proba"] / N_FOLDS

    out_dir = ROOT / "catboost"
    pd.DataFrame(
        {
            "id": train["id"],
            "p_at-risk": oof[:, 0],
            "p_fit": oof[:, 1],
            "p_unhealthy": oof[:, 2],
        }
    ).to_csv(out_dir / "oof.csv", index=False)
    pd.DataFrame(
        {
            "id": test["id"],
            "p_at-risk": test_proba[:, 0],
            "p_fit": test_proba[:, 1],
            "p_unhealthy": test_proba[:, 2],
        }
    ).to_csv(out_dir / "test_proba.csv", index=False)
    print("wrote catboost/oof.csv and catboost/test_proba.csv")

    argmax_score = balanced_accuracy(y, labels_from_proba(oof))
    print(f"OOF argmax balanced-accuracy: {argmax_score:.6f}", flush=True)
    if args.tune_decision:
        weights, tuned_score = tune_decision_weights(oof, y, n_iter=1200, seed=SEED)
        print(f"OOF tuned  balanced-accuracy: {tuned_score:.6f} weights={weights.round(4)}")


if __name__ == "__main__":
    main()
