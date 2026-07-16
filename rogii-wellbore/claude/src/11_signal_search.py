"""
11_signal_search.py  (Claude, helping Codex's line)

Principled search for NEW signal: predict the current blend's RESIDUALS from
untapped features, out-of-fold. If a feature reduces residual RMSE, it carries
information the current models (GR + geometry/offset) do not - genuinely new
signal worth adding. If nothing helps, we've confirmed the well is dry.

Candidate families tested (all available at inference from MD,X,Y,Z,GR):
  A. Steering dynamics: inclination, build/drop rate, dogleg, azimuth turn -
     the driller's RESPONSE to geology (may be partly independent of GR).
  B. GR texture: local roughness/variance (not the GR value 03 already uses).
Reports how much each family reduces the blend residual, plus feature importance.
"""
import os
import sys
import glob
import numpy as np
import pandas as pd
import lightgbm as lgb

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "shared"))
from metric import rmse  # noqa: E402

RAW = os.path.join(ROOT, "data", "raw")
folds = pd.read_csv(os.path.join(ROOT, "shared", "folds.csv"))
fold_of = dict(zip(folds["well_id"], folds["fold"]))


def roll(a, w, fn):
    return getattr(pd.Series(a).rolling(w, center=True, min_periods=1), fn)().values


def well_feats(wid):
    hz = pd.read_csv(os.path.join(RAW, "train", f"{wid}__horizontal_well.csv"))
    ps = np.where(hz["TVT_input"].notna().values)[0].max()
    md, x, y, z = hz["MD"].values, hz["X"].values, hz["Y"].values, hz["Z"].values
    gr = pd.Series(hz["GR"].values).interpolate(limit_direction="both").fillna(0).values
    dmd = np.gradient(md) + 1e-6
    incl = roll(np.gradient(z) / dmd, 15, "mean")               # A: inclination
    curv = roll(np.gradient(incl) / dmd, 15, "mean")            # A: build/drop rate
    az = np.unwrap(np.arctan2(np.gradient(y), np.gradient(x)))
    daz = roll(np.gradient(az) / dmd, 15, "mean")               # A: azimuth turn
    dogleg = roll(np.sqrt(np.gradient(incl) ** 2 + (np.gradient(az) * np.sin(incl)) ** 2) / dmd, 15, "mean")
    gr_std_s = roll(gr, 11, "std")                              # B: GR texture (short)
    gr_std_l = roll(gr, 51, "std")                              # B: GR texture (long)
    toe = np.arange(ps + 1, len(hz))
    return pd.DataFrame({
        "well_id": wid, "row_index": toe,
        "incl": incl[toe], "curv": curv[toe], "daz": daz[toe], "dogleg": dogleg[toe],
        "incl_cum": roll(np.abs(curv), 101, "mean")[toe],        # recent steering activity
        "gr_std_s": gr_std_s[toe], "gr_std_l": gr_std_l[toe],
        "toe_frac": (toe - ps) / (len(hz) - ps),
    })


print("Building candidate features...")
wids = [os.path.basename(f).replace("__horizontal_well.csv", "")
        for f in sorted(glob.glob(os.path.join(RAW, "train", "*__horizontal_well.csv")))]
F = pd.concat([well_feats(w) for w in wids], ignore_index=True)
F["fold"] = F["well_id"].map(fold_of)

# current blend residual (03 + codex, honest per-fold weights)
c = pd.read_csv(os.path.join(HERE, "oof_03_best.csv")).rename(columns={"tvt_pred": "m03"})
x = pd.read_csv(os.path.join(ROOT, "codex", "oof.csv")).rename(columns={"tvt_pred": "codex"})
d = c.merge(x, on=["well_id", "row_index"])
truth = []
for wid, g in d.groupby("well_id", sort=False):
    tv = pd.read_csv(os.path.join(RAW, "train", f"{wid}__horizontal_well.csv"), usecols=["TVT"])["TVT"].values
    truth.append(pd.Series(tv[g["row_index"].values], index=g.index))
d["y"] = pd.concat(truth)
d = d.merge(folds[["well_id", "fold"]], on="well_id")
ws = np.linspace(0, 1, 41)
blend = np.zeros(len(d))
for f in range(5):
    tr, va = d["fold"] != f, d["fold"] == f
    w = min(ws, key=lambda w: rmse(d.loc[tr, "y"], w * d.loc[tr, "m03"] + (1 - w) * d.loc[tr, "codex"]))
    blend[va.values] = w * d.loc[va, "m03"] + (1 - w) * d.loc[va, "codex"]
d["resid"] = d["y"] - blend
F = F.merge(d[["well_id", "row_index", "resid"]], on=["well_id", "row_index"])
base = rmse(d["y"], blend)
print(f"current blend RMSE: {base:.4f}  (residual std {F['resid'].std():.4f})")

# correlation of each candidate with the residual
print("\ncorrelation with residual:")
for col in ["incl", "curv", "daz", "dogleg", "incl_cum", "gr_std_s", "gr_std_l"]:
    print(f"  {col:9s} {np.corrcoef(F[col].fillna(0), F['resid'])[0,1]:+.3f}")

# can a GBM predict the residual out-of-fold from these features? (= new signal)
FEATS = ["incl", "curv", "daz", "dogleg", "incl_cum", "gr_std_s", "gr_std_l", "toe_frac"]
params = dict(objective="regression", n_estimators=400, learning_rate=0.05, num_leaves=31,
              min_child_samples=500, reg_lambda=5.0, subsample=0.7, subsample_freq=1,
              random_state=42, verbose=-1, n_jobs=-1)
oof = np.zeros(len(F))
for f in range(5):
    tr, va = F["fold"] != f, F["fold"] == f
    m = lgb.LGBMRegressor(**params).fit(F.loc[tr, FEATS].fillna(0), F.loc[tr, "resid"])
    oof[va.values] = m.predict(F.loc[va, FEATS].fillna(0))
corrected = d["y"].values - (blend + oof)     # residual after subtracting predicted residual
print(f"\nblend + residual-model RMSE: {rmse(d['y'], blend + oof):.4f}  (vs {base:.4f})")
imp = pd.Series(lgb.LGBMRegressor(**params).fit(F[FEATS].fillna(0), F['resid']).feature_importances_, index=FEATS)
print("residual-model importance:", imp.sort_values(ascending=False).round(0).to_dict())
