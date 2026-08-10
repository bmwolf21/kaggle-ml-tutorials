"""Stacking meta-learner - the last lever on the S6E7 blend.

Instead of a linear weighted blend, train a meta-learner on the members' OOF
probabilities (level-1 features) to learn WHERE each model is trustworthy. Cross-fitted
on shared/folds.csv so the meta-OOF score is honest (meta for fold k is trained only on
the other folds' OOF rows - no leakage). Compares a logistic meta and a LightGBM meta
against the plain weighted blend (0.9497) and the best single member.

Writes lightgbm/submission_stack.csv from the better meta (decision rule tuned on meta-OOF).
Usage: python lightgbm/stack.py
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "shared"))
sys.path.insert(0, str(ROOT / "shared"))
from blend import _discover, _norm, PCOLS            # reuse referee discovery
from metric import CLASSES, CLASS_TO_IDX, balanced_accuracy, labels_from_proba, tune_decision_weights  # noqa

NT = 3  # thread cap (nystroem may still be on the cores)


def _load(oof_p, test_p, ids_tr, ids_te):
    o = _norm(pd.read_csv(oof_p).set_index("id").reindex(ids_tr)[PCOLS].values)
    t = _norm(pd.read_csv(test_p).set_index("id").reindex(ids_te)[PCOLS].values)
    return o, t


def main():
    tr = pd.read_csv(ROOT / "data" / "train.csv")[["id", "health_condition"]]
    te_ids = pd.read_csv(ROOT / "data" / "test.csv")["id"]
    fold = pd.read_csv(ROOT / "shared" / "folds.csv").set_index("id")["fold"].reindex(tr["id"]).values
    y = tr["health_condition"].values
    yi = np.array([CLASS_TO_IDX[c] for c in y])

    labels, Xo_parts, Xt_parts = [], [], []
    for label, oof_p, test_p in _discover():
        o, t = _load(oof_p, test_p, tr["id"], te_ids)
        if np.isnan(o).any() or np.isnan(t).any():
            print(f"  skip {label}: id misalignment"); continue
        labels.append(label); Xo_parts.append(o); Xt_parts.append(t)
    Xo = np.hstack(Xo_parts)          # (n_train, 3*M)
    Xt = np.hstack(Xt_parts)          # (n_test, 3*M)
    print(f"stacking {len(labels)} members -> {Xo.shape[1]} meta-features: {labels}")

    single_best = max(balanced_accuracy(y, labels_from_proba(o)) for o in Xo_parts)

    def cross_fit(make):
        oof = np.zeros((len(tr), 3)); test = np.zeros((len(te_ids), 3))
        for k in range(5):
            tri, vai = fold != k, fold == k
            m = make(); m.fit(Xo[tri], y[tri])
            order = [list(m.classes_).index(c) for c in CLASSES]
            oof[vai] = m.predict_proba(Xo[vai])[:, order]
            test += m.predict_proba(Xt)[:, order] / 5
        return oof, test

    metas = {
        "logistic": lambda: LogisticRegression(C=1.0, class_weight="balanced",
                     solver="lbfgs", multi_class="multinomial", max_iter=500, n_jobs=NT),
        "lightgbm": lambda: LGBMClassifier(n_estimators=300, learning_rate=0.03,
                     num_leaves=15, min_child_samples=200, reg_lambda=5.0,
                     class_weight="balanced", objective="multiclass", num_class=3,
                     n_jobs=NT, num_threads=NT, force_row_wise=True, random_state=42, verbose=-1),
    }
    results = {}
    print(f"\nbaselines: best single member {single_best:.4f} | weighted blend 0.9497")
    for name, make in metas.items():
        oof, test = cross_fit(make)
        ba = balanced_accuracy(y, labels_from_proba(oof))
        w, ba_t = tune_decision_weights(oof, y, n_iter=2500)
        results[name] = (ba_t, oof, test, w)
        print(f"  meta={name:9s}: meta-OOF bal-acc argmax {ba:.4f} | tuned {ba_t:.4f}  w={w.round(2)}")

    best = max(results, key=lambda n: results[n][0])
    ba_t, oof, test, w = results[best]
    lift = ba_t - max(0.9497, single_best)
    print(f"\n==> best meta: {best}  tuned {ba_t:.4f}   lift vs blend/single: {lift:+.4f}")
    if lift <= 0.0005:
        print("    (within noise of the linear blend - stacking does not beat the ceiling)")

    sub = pd.DataFrame({"id": te_ids, "health_condition": labels_from_proba(test, w)})
    out = ROOT / "submission_stack.csv"
    sub.to_csv(out, index=False)
    print(f"\nwrote {out}  ({len(sub)} rows)")
    print(sub["health_condition"].value_counts(normalize=True).round(4).to_string())


if __name__ == "__main__":
    main()
