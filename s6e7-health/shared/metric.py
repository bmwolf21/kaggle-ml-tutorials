"""Shared metric + decision-rule utilities for S6E7.

Competition metric: BALANCED ACCURACY = macro-average of per-class recall.
Classes are heavily imbalanced (at-risk 86% / unhealthy 8% / fit 6%), but balanced
accuracy weights all three equally, so predicting the majority class everywhere
scores only 0.333. Minority recall is everything.

Every agent MUST:
  - build probability matrices in CLASSES order (below),
  - score with balanced_accuracy(),
  - turn proba into labels with labels_from_proba() (weighted argmax), and tune the
    weights on OOF only via tune_decision_weights().
Plain argmax is almost never optimal here: up-weighting the minority classes trades
a little majority recall for a lot of minority recall, which balanced accuracy rewards.
"""
import numpy as np
from sklearn.metrics import balanced_accuracy_score

# Canonical column order for ALL probability matrices. Do not reorder.
CLASSES = ["at-risk", "fit", "unhealthy"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}


def balanced_accuracy(y_true, y_pred):
    """y_true, y_pred: array-like of string labels (or matching int indices)."""
    return balanced_accuracy_score(np.asarray(y_true), np.asarray(y_pred))


def labels_from_proba(proba, weights=None):
    """proba: (n, 3) array in CLASSES order. weights: optional (3,) multiplicative
    prior applied before argmax. Returns string labels."""
    P = np.asarray(proba, dtype=float)
    if weights is not None:
        P = P * np.asarray(weights, dtype=float)[None, :]
    return np.asarray(CLASSES)[P.argmax(axis=1)]


def tune_decision_weights(proba, y_true, n_iter=6000, seed=42):
    """Search per-class multiplicative weights that maximize balanced accuracy on
    OOF proba (weighted argmax). Returns (weights, score). weights normalized so
    weights[0] == 1 (the rule is scale-invariant). Fit on OOF, then freeze and apply
    the SAME weights to the test proba - never tune on test."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y_true)
    best_w = np.ones(3)
    best_s = balanced_accuracy(y, labels_from_proba(proba, best_w))
    for _ in range(n_iter):  # random search in log space
        w = np.exp(rng.normal(0.0, 0.7, size=3))
        s = balanced_accuracy(y, labels_from_proba(proba, w))
        if s > best_s:
            best_s, best_w = s, w
    step = 0.5  # local coordinate polish
    for _ in range(300):
        improved = False
        for j in range(3):
            for d in (step, -step):
                w = best_w.copy()
                w[j] *= np.exp(d)
                s = balanced_accuracy(y, labels_from_proba(proba, w))
                if s > best_s:
                    best_s, best_w, improved = s, w, True
        if not improved:
            step *= 0.5
            if step < 1e-3:
                break
    return best_w / best_w[0], best_s


if __name__ == "__main__":  # self-test on random proba
    rng = np.random.default_rng(0)
    y = rng.choice(CLASSES, size=5000, p=[0.86, 0.06, 0.08])
    P = rng.dirichlet(np.ones(3), size=5000)
    w, s = tune_decision_weights(P, y, n_iter=1000)
    print("argmax bal-acc:", round(balanced_accuracy(y, labels_from_proba(P)), 4))
    print("tuned  bal-acc:", round(s, 4), "weights:", w.round(3))
