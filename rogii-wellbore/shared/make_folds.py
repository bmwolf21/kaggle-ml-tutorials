"""
make_folds.py - build the canonical shared cross-validation folds.

Group-aware: each WELL is assigned wholly to one fold (never split across folds),
so validation never leaks within-well autocorrelation. Wells are stratified by a
label combining median TVT, drilling-azimuth sign, and spatial X bin, following
the domain-expert recommendation, so folds are balanced on geology and geometry.

Both agents MUST use the resulting shared/folds.csv. Deterministic (fixed seed).

Run:  python shared/make_folds.py
Writes: shared/folds.csv  (well_id, fold, median_tvt, azimuth_sign, x_bin)
"""
import os
import glob
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
TRAIN = os.path.join(ROOT, "data", "raw", "train")
N_FOLDS = 5
SEED = 42

rows = []
for f in sorted(glob.glob(os.path.join(TRAIN, "*__horizontal_well.csv"))):
    wid = os.path.basename(f).replace("__horizontal_well.csv", "")
    df = pd.read_csv(f, usecols=lambda c: c in {"X", "Y", "TVT"})
    x, y = df["X"].values, df["Y"].values
    heading = np.arctan2(y[-1] - y[0], x[-1] - x[0])   # overall drilling direction
    rows.append({
        "well_id": wid,
        "median_tvt": float(np.nanmedian(df["TVT"])),
        "azimuth_sign": int(np.sign(np.sin(heading)) or 1),  # +1/-1 (drilling N/S component)
        "x_centroid": float(np.nanmean(x)),
    })
wells = pd.DataFrame(rows)
print(f"wells: {len(wells)}")

# Stratum label: 3 TVT bins x azimuth sign x 2 spatial bins.
wells["tvt_bin"] = pd.qcut(wells["median_tvt"], 3, labels=False, duplicates="drop")
wells["x_bin"] = pd.qcut(wells["x_centroid"], 2, labels=False, duplicates="drop")
wells["stratum"] = (wells["tvt_bin"].astype(str) + "_" +
                    wells["azimuth_sign"].astype(str) + "_" +
                    wells["x_bin"].astype(str))

# Assign whole wells to folds, balanced across strata.
wells["fold"] = -1
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
for fold, (_, va) in enumerate(skf.split(wells, wells["stratum"])):
    wells.loc[wells.index[va], "fold"] = fold

out = wells[["well_id", "fold", "median_tvt", "azimuth_sign", "x_bin"]].sort_values("well_id")
out.to_csv(os.path.join(HERE, "folds.csv"), index=False)
print("fold sizes:", out["fold"].value_counts().sort_index().to_dict())
print(f"wrote {os.path.join(HERE, 'folds.csv')}")
