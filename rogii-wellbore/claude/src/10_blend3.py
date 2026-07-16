"""
10_blend3.py  (Claude's pipeline)

Does adding the CNN (a different family) as a THIRD model improve the blend beyond
the current 03+Codex (13.873)? Compares 2-way vs 3-way, with honest out-of-fold
weights. If the 3-way wins, writes claude/oof.csv = the internal 03+CNN blend so
shared/blend.py picks it up.
"""
import os
import sys
import numpy as np
import pandas as pd
from itertools import product

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "shared"))
from metric import rmse  # noqa: E402

RAW = os.path.join(ROOT, "data", "raw")


def load(path, name):
    return pd.read_csv(path).rename(columns={"tvt_pred": name})


d = load(os.path.join(HERE, "oof_03_best.csv"), "m03")
d = d.merge(load(os.path.join(HERE, "oof.csv"), "cnn"), on=["well_id", "row_index"])
d = d.merge(load(os.path.join(ROOT, "codex", "oof.csv"), "codex"), on=["well_id", "row_index"])
truth = []
for wid, g in d.groupby("well_id", sort=False):
    tv = pd.read_csv(os.path.join(RAW, "train", f"{wid}__horizontal_well.csv"), usecols=["TVT"])["TVT"].values
    truth.append(pd.Series(tv[g["row_index"].values], index=g.index))
d["y"] = pd.concat(truth)
d = d.merge(pd.read_csv(os.path.join(ROOT, "shared", "folds.csv"))[["well_id", "fold"]], on="well_id")

for m in ["m03", "cnn", "codex"]:
    print(f"  {m:6s} {rmse(d['y'], d[m]):.4f}")

grid = [w / 20 for w in range(21)]


def best_weights(idx, models):
    """honest: for each fold pick simplex weights on other folds, apply to fold."""
    oof = np.zeros(len(d))
    for fld in range(5):
        tr, va = d["fold"] != fld, d["fold"] == fld
        combos = [w for w in product(grid, repeat=len(models)) if abs(sum(w) - 1) < 1e-9]
        best = min(combos, key=lambda w: rmse(d.loc[tr, "y"],
                   sum(wi * d.loc[tr, m] for wi, m in zip(w, models))))
        oof[va.values] = sum(wi * d.loc[va, m] for wi, m in zip(best, models))
    return rmse(d["y"], oof)


print(f"\n2-way (03+codex):     {best_weights(None, ['m03','codex']):.4f}  (was 13.873)")
print(f"3-way (03+cnn+codex): {best_weights(None, ['m03','cnn','codex']):.4f}")
