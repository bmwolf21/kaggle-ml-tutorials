"""Phase-2 generative members: QDA and GaussianNB.

Writes:
  catboost/oof__qda.csv, catboost/test_proba__qda.csv
  catboost/oof__nb.csv,  catboost/test_proba__nb.csv
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEPS = ROOT / "catboost" / ".deps"
if DEPS.exists():
    sys.path.insert(0, str(DEPS))
sys.path.insert(0, str(ROOT / "shared"))

os.environ.setdefault("OMP_NUM_THREADS", "3")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "3")
os.environ.setdefault("MKL_NUM_THREADS", "3")

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.impute import SimpleImputer
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from metric import CLASSES, balanced_accuracy, labels_from_proba


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
N_FOLDS = 5


def _safe_div(num, den):
    return num / den.replace(0, np.nan)


def make_features(df):
    x = df[NUM + CAT].copy()
    for c in NUM + CAT:
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

    x = x.replace([np.inf, -np.inf], np.nan)
    return x


def make_preprocessor(x):
    cat_cols = [c for c in x.columns if x[c].dtype == "object" or str(x[c].dtype) == "category"]
    num_cols = [c for c in x.columns if c not in cat_cols]
    return ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]
                ),
                num_cols,
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="constant", fill_value="__MISSING__")),
                        (
                            "onehot",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float32),
                        ),
                    ]
                ),
                cat_cols,
            ),
        ],
        sparse_threshold=0.0,
        n_jobs=3,
    )


def align_proba(model, proba):
    model_classes = [str(c) for c in model.classes_]
    order = [model_classes.index(c) for c in CLASSES]
    out = np.clip(proba[:, order], 1e-12, None)
    return out / out.sum(axis=1, keepdims=True)


def write_member(tag, train_ids, test_ids, oof, test_proba):
    out_dir = ROOT / "catboost"
    pd.DataFrame(
        {
            "id": train_ids,
            "p_at-risk": oof[:, 0],
            "p_fit": oof[:, 1],
            "p_unhealthy": oof[:, 2],
        }
    ).to_csv(out_dir / f"oof__{tag}.csv", index=False)
    pd.DataFrame(
        {
            "id": test_ids,
            "p_at-risk": test_proba[:, 0],
            "p_fit": test_proba[:, 1],
            "p_unhealthy": test_proba[:, 2],
        }
    ).to_csv(out_dir / f"test_proba__{tag}.csv", index=False)


def main():
    train = pd.read_csv(ROOT / "data" / "train.csv")
    test = pd.read_csv(ROOT / "data" / "test.csv")
    folds = pd.read_csv(ROOT / "shared" / "folds.csv").set_index("id")["fold"]
    train["fold"] = train["id"].map(folds).astype(int)
    assert train["fold"].notna().all() and set(train["fold"]) == set(range(N_FOLDS))

    y = train["health_condition"].to_numpy()
    x_train = make_features(train)
    x_test = make_features(test)

    members = {
        "qda": QuadraticDiscriminantAnalysis(priors=np.ones(3) / 3, reg_param=0.25),
        "nb": GaussianNB(priors=np.ones(3) / 3, var_smoothing=1e-8),
    }
    oofs = {tag: np.zeros((len(train), 3), dtype=np.float32) for tag in members}
    tests = {tag: np.zeros((len(test), 3), dtype=np.float32) for tag in members}

    for fold in range(N_FOLDS):
        valid = train["fold"].to_numpy() == fold
        print(f"fold {fold}: transform train={int((~valid).sum())} valid={int(valid.sum())}", flush=True)

        prep = make_preprocessor(x_train)
        x_tr = prep.fit_transform(x_train.loc[~valid]).astype(np.float32, copy=False)
        x_va = prep.transform(x_train.loc[valid]).astype(np.float32, copy=False)
        x_te = prep.transform(x_test).astype(np.float32, copy=False)
        y_tr = y[~valid]

        for tag, model in members.items():
            print(f"fold {fold}: fitting {tag}", flush=True)
            model.fit(x_tr, y_tr)
            oofs[tag][valid] = align_proba(model, model.predict_proba(x_va))
            tests[tag] += align_proba(model, model.predict_proba(x_te)) / N_FOLDS
            score = balanced_accuracy(y[valid], labels_from_proba(oofs[tag][valid]))
            print(f"fold {fold}: {tag} argmax balanced-accuracy {score:.6f}", flush=True)

    for tag in members:
        score = balanced_accuracy(y, labels_from_proba(oofs[tag]))
        write_member(tag, train["id"], test["id"], oofs[tag], tests[tag])
        print(f"{tag}: OOF argmax balanced-accuracy {score:.6f}")
        print(f"wrote catboost/oof__{tag}.csv and catboost/test_proba__{tag}.csv")


if __name__ == "__main__":
    main()
