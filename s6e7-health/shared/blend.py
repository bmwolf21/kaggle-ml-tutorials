"""Referee - N-way OOF blend + single decision-rule tune + submission.

The blend referee (see METHODOLOGY.md). It:
  1. discovers whichever members have written BOTH oof.csv and test_proba.csv,
  2. reports each member solo (balanced accuracy, argmax and decision-tuned),
  3. shows a decorrelation diagnostic (members must DISAGREE for a blend to help),
  4. blends OOF proba (equal-weight, and a guarded weight search), tunes ONE decision
     rule on the blended OOF, applies the SAME rule to the blended test proba,
  5. writes lightgbm/submission.csv (id, health_condition labels).

Pure I/O + numpy - no model training, so it is safe to run while the other two agents
are using the cores. Uses a fast vectorized balanced-accuracy for the inner search
(bincount, not sklearn) so it stays quick even under CPU contention; the canonical
shared metric is used for the final reported numbers. Re-run whenever a member lands.

Usage: python lightgbm/blend.py
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "shared"))
from metric import CLASSES, CLASS_TO_IDX, balanced_accuracy, labels_from_proba  # noqa

PCOLS = [f"p_{c}" for c in CLASSES]
AGENTS = ["lightgbm", "catboost", "neural_net"]
NC = len(CLASSES)


# ---- fast inner-loop helpers (bincount-based; ~10x faster than sklearn per eval) ----
def _ba_fast(y_idx, pred_idx):
    correct = np.bincount(y_idx[y_idx == pred_idx], minlength=NC).astype(float)
    total = np.bincount(y_idx, minlength=NC).astype(float)
    total[total == 0] = 1.0
    return float((correct / total).mean())


def _tune_fast(P, y_idx, n_iter=4000, seed=42):
    """Weighted-argmax decision-rule search maximizing balanced accuracy on OOF.
    Returns (weights normalized so w[0]==1, score)."""
    rng = np.random.default_rng(seed)
    best_w = np.ones(NC)
    best_s = _ba_fast(y_idx, (P * best_w).argmax(1))
    for _ in range(n_iter):
        w = np.exp(rng.normal(0.0, 0.7, size=NC))
        s = _ba_fast(y_idx, (P * w).argmax(1))
        if s > best_s:
            best_s, best_w = s, w
    step = 0.5
    for _ in range(300):
        improved = False
        for j in range(NC):
            for d in (step, -step):
                w = best_w.copy()
                w[j] *= np.exp(d)
                s = _ba_fast(y_idx, (P * w).argmax(1))
                if s > best_s:
                    best_s, best_w, improved = s, w, True
        if not improved:
            step *= 0.5
            if step < 1e-3:
                break
    return best_w / best_w[0], best_s


def _discover():
    """Every (label, oof_path, test_path) across all subdirs. A subdir's primary is
    oof.csv (label=subdir); tagged extras are oof__<tag>.csv (label=subdir:<tag>), each
    needing a matching test_proba[__tag].csv."""
    found = []
    for name in AGENTS:
        d = ROOT / name
        pairs = [("", d / "oof.csv", d / "test_proba.csv")]
        for p in sorted(d.glob("oof__*.csv")):
            tag = p.stem[len("oof__"):]
            pairs.append((tag, p, d / f"test_proba__{tag}.csv"))
        for tag, op, tp in pairs:
            if op.exists() and tp.exists():
                found.append((name if not tag else f"{name}:{tag}", op, tp))
    return found


def _load_member(oof_p, test_p):
    oof, test = pd.read_csv(oof_p), pd.read_csv(test_p)
    if not (set(PCOLS) <= set(oof.columns) and set(PCOLS) <= set(test.columns)):
        print(f"  ! {oof_p.name}: missing proba columns {PCOLS}; skipping")
        return None
    return oof, test


def _norm(P):
    P = np.clip(np.asarray(P, float), 1e-9, None)
    return P / P.sum(1, keepdims=True)


def main():
    tr = pd.read_csv(ROOT / "data" / "train.csv")[["id", "health_condition"]]
    te_ids = pd.read_csv(ROOT / "data" / "test.csv")["id"]
    y = tr["health_condition"].values
    y_idx = np.array([CLASS_TO_IDX[c] for c in y])

    members, oofs, tests, solo = [], {}, {}, {}
    print("=== members present ===")
    for label, oof_p, test_p in _discover():
        got = _load_member(oof_p, test_p)
        if got is None:
            continue
        oof, test = got
        o = _norm(oof.set_index("id").reindex(tr["id"])[PCOLS].values)
        t = _norm(test.set_index("id").reindex(te_ids)[PCOLS].values)
        if np.isnan(o).any() or np.isnan(t).any():
            print(f"  {label:18s} -  ! id misalignment (NaNs after reindex); skipping")
            continue
        members.append(label)
        oofs[label], tests[label] = o, t
        ba = balanced_accuracy(y, labels_from_proba(o))
        w, ba_t = _tune_fast(o, y_idx)
        solo[label] = ba
        print(f"  {label:18s} -  OOF bal-acc argmax {ba:.4f} | tuned {ba_t:.4f}  w={w.round(2)}")

    if not members:
        print("\nNo members ready. Re-run once an agent writes oof.csv + test_proba.csv.")
        return
    if len(members) == 1:
        print(f"\nOnly '{members[0]}' present - nothing to blend yet. Referee stands by.")
        return

    # --- decorrelation: blend gains come from DISAGREEMENT ---
    print("\n=== decorrelation (argmax-label agreement between members) ===")
    preds = {m: labels_from_proba(oofs[m]) for m in members}
    for i, a in enumerate(members):
        for b in members[i + 1:]:
            agree = float(np.mean(preds[a] == preds[b]))
            corr = float(np.corrcoef(oofs[a].ravel(), oofs[b].ravel())[0, 1])
            tag = "decorrelated - good" if agree < 0.98 else "very similar - limited gain"
            print(f"  {a} vs {b}: label agreement {agree:.3f} | proba corr {corr:.3f}   ({tag})")

    O = np.stack([oofs[m] for m in members])   # (M, n, 3)
    T = np.stack([tests[m] for m in members])

    # --- equal-weight blend (robust default) ---
    eq_oof = _norm(O.mean(0))
    w_eq, ba_eq = _tune_fast(eq_oof, y_idx)
    print(f"\nequal-weight blend: OOF tuned bal-acc {ba_eq:.4f}  decision-w={w_eq.round(2)}")

    # --- guarded member-weight search (decoupled: weights on argmax, THEN tune rule) ---
    rng = np.random.default_rng(42)
    best_mw, best_s = np.ones(len(members)) / len(members), -1.0
    for _ in range(3000):
        mw = rng.dirichlet(np.ones(len(members)))
        s = _ba_fast(y_idx, np.tensordot(mw, O, axes=(0, 0)).argmax(1))
        if s > best_s:
            best_s, best_mw = s, mw
    sw_oof = _norm(np.tensordot(best_mw, O, axes=(0, 0)))
    w_sw, ba_sw = _tune_fast(sw_oof, y_idx)
    print(f"weight-search blend: OOF tuned bal-acc {ba_sw:.4f}  "
          f"member-w={dict(zip(members, best_mw.round(3)))}")

    # --- choose: prefer equal-weight unless search clearly wins (optimizer's curse) ---
    if ba_sw > ba_eq + 0.001:
        chosen, mw, dw, ba = "weight-search", best_mw, w_sw, ba_sw
    else:
        chosen, mw, dw, ba = "equal-weight", np.ones(len(members)) / len(members), w_eq, ba_eq
    best_member = max(solo, key=solo.get)
    print(f"\n==> chosen blend: {chosen}  (OOF tuned bal-acc {ba:.4f})")
    print(f"    best single member: {best_member} {solo[best_member]:.4f} (argmax)"
          f" | blend lift {ba - solo[best_member]:+.4f}")

    # --- apply frozen rule to blended TEST proba, write submission (labels) ---
    blend_test = _norm(np.tensordot(mw, T, axes=(0, 0)))
    sub = pd.DataFrame({"id": te_ids, "health_condition": labels_from_proba(blend_test, dw)})
    out = ROOT / "submission.csv"
    sub.to_csv(out, index=False)
    print(f"\nwrote {out}  ({len(sub)} rows)")
    print("test label distribution:")
    print(sub["health_condition"].value_counts(normalize=True).round(4).to_string())


if __name__ == "__main__":
    main()
