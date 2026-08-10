"""Phase-2 alternative-bias members (see METHODOLOGY.md Phase 2).

Two non-tree biases to probe for decorrelated errors against the GBDTs:
  logreg    - multinomial logistic regression on standardized numerics + numeric
              interactions + one-hot categoricals (LINEAR decision boundaries)
  nystroem  - Nystroem RBF feature map -> linear SGD (KERNEL-APPROX smooth boundaries)

Both use shared/folds.csv and write the tagged extra-member files the referee discovers:
  lightgbm/oof__<tag>.csv, lightgbm/test_proba__<tag>.csv  (id,p_at-risk,p_fit,p_unhealthy)

Usage: python lightgbm/altbias.py [logreg|nystroem]
Cap threads while other pipelines train:  OMP_NUM_THREADS=3 python lightgbm/altbias.py logreg
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.kernel_approximation import Nystroem

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "shared"))
from metric import CLASSES, balanced_accuracy, labels_from_proba, tune_decision_weights  # noqa

NUM = ["sleep_duration", "heart_rate", "bmi", "calorie_expenditure",
       "step_count", "exercise_duration", "water_intake"]
CAT = ["diet_type", "stress_level", "sleep_quality", "physical_activity_level",
       "smoking_alcohol", "gender"]
N_THREADS = 3  # other pipelines may be using the cores; be a good neighbour


def pre(poly):
    """Numeric: median-impute (+NA flag) -> scale -> optional interaction terms.
       Categorical: constant-impute -> one-hot (unknown-safe)."""
    num_steps = [("imp", SimpleImputer(strategy="median", add_indicator=True)),
                 ("sc", StandardScaler())]
    if poly:
        num_steps.append(("pf", PolynomialFeatures(degree=2, interaction_only=True,
                                                    include_bias=False)))
    return ColumnTransformer([
        ("num", Pipeline(num_steps), NUM),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="constant", fill_value="__NA__")),
                          ("oh", OneHotEncoder(handle_unknown="ignore"))]), CAT),
    ])


def build(model):
    if model == "logreg":
        clf = LogisticRegression(C=0.5, class_weight="balanced", solver="lbfgs",
                                 multi_class="multinomial", max_iter=400, n_jobs=N_THREADS)
        return Pipeline([("pre", pre(poly=True)), ("clf", clf)])
    if model == "nystroem":
        # RBF feature map, then a fast linear log-loss head (SGD scales to 690k x 300)
        clf = SGDClassifier(loss="log", class_weight="balanced", alpha=1e-5,
                            max_iter=30, random_state=42, n_jobs=N_THREADS)
        return Pipeline([("pre", pre(poly=False)),
                         ("rbf", Nystroem(kernel="rbf", n_components=300, random_state=42)),
                         ("clf", clf)])
    raise SystemExit(f"unknown model {model!r}; use logreg|nystroem")


def main():
    model = sys.argv[1] if len(sys.argv) > 1 else "logreg"
    tr = pd.read_csv(ROOT / "data" / "train.csv")
    te = pd.read_csv(ROOT / "data" / "test.csv")
    folds = pd.read_csv(ROOT / "shared" / "folds.csv").set_index("id")["fold"]
    tr["fold"] = tr["id"].map(folds)
    y = tr["health_condition"].values

    oof = np.zeros((len(tr), 3))
    test = np.zeros((len(te), 3))
    for k in range(5):
        tri, vai = tr["fold"].values != k, tr["fold"].values == k
        pipe = build(model)
        pipe.fit(tr[NUM + CAT][tri], y[tri])
        order = [list(pipe.classes_).index(c) for c in CLASSES]
        oof[vai] = pipe.predict_proba(tr[NUM + CAT][vai])[:, order]
        test += pipe.predict_proba(te[NUM + CAT])[:, order] / 5
        print(f"  fold {k}: bal-acc {balanced_accuracy(y[vai], labels_from_proba(oof[vai])):.4f}")

    ba = balanced_accuracy(y, labels_from_proba(oof))
    w, ba_t = tune_decision_weights(oof, y, n_iter=2000)
    print(f"\n{model}: OOF bal-acc argmax {ba:.4f} | tuned {ba_t:.4f}  w={w.round(2)}")

    pd.DataFrame({"id": tr["id"], "p_at-risk": oof[:, 0], "p_fit": oof[:, 1],
                  "p_unhealthy": oof[:, 2]}).to_csv(ROOT / "lightgbm" / f"oof__{model}.csv", index=False)
    pd.DataFrame({"id": te["id"], "p_at-risk": test[:, 0], "p_fit": test[:, 1],
                  "p_unhealthy": test[:, 2]}).to_csv(ROOT / "lightgbm" / f"test_proba__{model}.csv", index=False)
    print(f"wrote lightgbm/oof__{model}.csv, lightgbm/test_proba__{model}.csv")


if __name__ == "__main__":
    main()
