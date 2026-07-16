"""
09_cnn_model.py  (Claude's pipeline)

A NEW model family: a 1-D CNN over the GR sequence. Trees (my 03) and ridge
(Codex) both consume hand-crafted features; a conv net instead learns GR *shape*
directly from a window of gamma-ray around each toe point - the pattern-matching
my nearest-GR match does badly. Even if its standalone is only so-so, a genuinely
different function approximator should add blend diversity.

Input per toe point: a per-well-normalized GR window (length L) + a few scalar
hints (toe fraction, geometry, and the 03 GR-implied dTVT). Output: dTVT.
Trained with the shared group folds. Writes oof.csv (CNN), test_pred, submission.
"""
import os
import sys
import glob
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

torch.manual_seed(42)
np.random.seed(42)
torch.set_num_threads(16)

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "shared"))
from metric import rmse  # noqa: E402

RAW = os.path.join(ROOT, "data", "raw")
folds = pd.read_csv(os.path.join(ROOT, "shared", "folds.csv"))
fold_of = dict(zip(folds["well_id"], folds["fold"]))
L = 161
HALF = L // 2
SHIFTS = np.arange(-40.0, 40.001, 0.5)
STRIDE = 2                      # subsample toe points for training speed


def roll(a, w, fn):
    return getattr(pd.Series(a).rolling(w, center=True, min_periods=1), fn)().values


def implied(gr_pt, ref_tvt, ref_gr, tvt_ps):
    if len(ref_tvt) < 3 or not np.isfinite(ref_gr).any():
        return np.zeros_like(gr_pt)
    o = np.argsort(ref_tvt)
    grid = np.interp(tvt_ps + SHIFTS, np.asarray(ref_tvt)[o], np.asarray(ref_gr)[o])
    return SHIFTS[np.argmin(np.abs(grid[None, :] - gr_pt[:, None]), axis=1)]


def well_arrays(hz, tw, is_train):
    ps = np.where(hz["TVT_input"].notna().values)[0].max()
    tvt_ps = float(hz["TVT_input"].iloc[ps])
    md_ps, z_ps = float(hz["MD"].iloc[ps]), float(hz["Z"].iloc[ps])
    gr = pd.Series(hz["GR"].values).interpolate(limit_direction="both").fillna(0).values
    gr_s = roll(gr, 5, "mean")
    hm, hsd = np.nanmean(gr_s[:ps + 1]), np.nanstd(gr_s[:ps + 1]) + 1e-6
    grn = (gr_s - hm) / hsd
    grn_pad = np.pad(grn, HALF, mode="edge")
    toe = np.arange(ps + 1, len(hz))
    d_tw = implied(gr_s[toe], tw["TVT"].values, roll(tw["GR"].values, 5, "mean"), tvt_ps)
    d_self = implied(gr_s[toe], hz["TVT_input"].values[:ps + 1], gr_s[:ps + 1], tvt_ps)
    win = np.stack([grn_pad[i:i + L] for i in toe]).astype(np.float32)     # (M, L)
    scal = np.stack([
        (toe - ps) / (len(hz) - ps),
        (hz["MD"].values[toe] - md_ps) / 3000.0,
        (hz["Z"].values[toe] - z_ps) / 100.0,
        (hz["Z"].values[toe] - z_ps) / (hz["MD"].values[toe] - md_ps + 1e-6),
        (gr_s[toe] - hm) / hsd,
        d_tw / 40.0, d_self / 40.0, roll(0.5 * (d_tw + d_self), 41, "median") / 40.0,
    ], axis=1).astype(np.float32)
    y = (hz["TVT"].values[toe] - tvt_ps).astype(np.float32) if is_train else np.zeros(len(toe), np.float32)
    wid = np.array([hz["well_id"].iloc[0]] * len(toe))
    return win, scal, y, wid, toe, np.full(len(toe), tvt_ps, np.float32)


def load(split, wid):
    hz = pd.read_csv(os.path.join(RAW, split, f"{wid}__horizontal_well.csv")); hz["well_id"] = wid
    tw = pd.read_csv(os.path.join(RAW, split, f"{wid}__typewell.csv"))
    return hz, tw


class Net(nn.Module):
    def __init__(self, n_scalar):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 16, 7, padding=3), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(16, 32, 5, padding=2), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 32, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool1d(1))
        self.head = nn.Sequential(nn.Linear(32 + n_scalar, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, w, s):
        c = self.conv(w.unsqueeze(1)).squeeze(-1)
        return self.head(torch.cat([c, s], 1)).squeeze(-1)


print("Building arrays...")
wids = [os.path.basename(f).replace("__horizontal_well.csv", "")
        for f in sorted(glob.glob(os.path.join(RAW, "train", "*__horizontal_well.csv")))]
W, Sc, Y, WID, ROW, TP = [], [], [], [], [], []
for w in wids:
    a = well_arrays(*load("train", w), True)
    W.append(a[0]); Sc.append(a[1]); Y.append(a[2]); WID.append(a[3]); ROW.append(a[4]); TP.append(a[5])
W = np.concatenate(W); Sc = np.concatenate(Sc); Y = np.concatenate(Y)
WID = np.concatenate(WID); ROW = np.concatenate(ROW); TP = np.concatenate(TP)
FOLD = np.array([fold_of[w] for w in WID])
print(f"samples {len(Y):,} | window {W.shape} | scalars {Sc.shape}")


def train_predict(tr_idx, va_idx, epochs=4):
    mu, sd = Sc[tr_idx].mean(0), Sc[tr_idx].std(0) + 1e-6
    net = Net(Sc.shape[1])
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-5)
    lossf = nn.MSELoss()
    sub = tr_idx[::STRIDE]
    Wt = torch.from_numpy(W[sub]); St = torch.from_numpy((Sc[sub] - mu) / sd); Yt = torch.from_numpy(Y[sub])
    n, bs = len(sub), 8192
    for ep in range(epochs):
        net.train(); perm = torch.randperm(n)
        for b in range(0, n, bs):
            j = perm[b:b + bs]
            opt.zero_grad()
            loss = lossf(net(Wt[j], St[j]), Yt[j]); loss.backward(); opt.step()
    net.eval()
    with torch.no_grad():
        Wv = torch.from_numpy(W[va_idx]); Sv = torch.from_numpy((Sc[va_idx] - mu) / sd)
        out = np.concatenate([net(Wv[b:b+16384], Sv[b:b+16384]).numpy() for b in range(0, len(va_idx), 16384)])
    return out, net, (mu, sd)


oof = np.zeros(len(Y))
nets = []
for k in range(5):
    tr = np.where(FOLD != k)[0]; va = np.where(FOLD == k)[0]
    pred, net, sc = train_predict(tr, va)
    oof[va] = pred; nets.append((net, sc))
    print(f"  fold {k}: RMSE {rmse(Y[va] + TP[va], pred + TP[va]):.3f} ft")
print(f"\nCNN OOF RMSE: {rmse(Y + TP, oof + TP):.3f} ft  (03=15.249, flat 15.91)")

pd.DataFrame({"well_id": WID, "row_index": ROW, "tvt_pred": oof + TP}).to_csv(
    os.path.join(HERE, "oof.csv"), index=False)

# test
twids = [os.path.basename(f).replace("__horizontal_well.csv", "")
         for f in sorted(glob.glob(os.path.join(RAW, "test", "*__horizontal_well.csv")))]
tW, tSc, tWID, tROW, tTP = [], [], [], [], []
for w in twids:
    a = well_arrays(*load("test", w), False)
    tW.append(a[0]); tSc.append(a[1]); tWID.append(a[3]); tROW.append(a[4]); tTP.append(a[5])
tW = np.concatenate(tW); tSc = np.concatenate(tSc)
tWID = np.concatenate(tWID); tROW = np.concatenate(tROW); tTP = np.concatenate(tTP)
preds = []
for net, (mu, sd) in nets:
    net.eval()
    with torch.no_grad():
        Wv = torch.from_numpy(tW); Sv = torch.from_numpy((tSc - mu) / sd)
        preds.append(np.concatenate([net(Wv[b:b+16384], Sv[b:b+16384]).numpy() for b in range(0, len(tW), 16384)]))
tp = np.mean(preds, 0) + tTP
pd.DataFrame({"well_id": tWID, "row_index": tROW, "tvt_pred": tp}).to_csv(os.path.join(HERE, "test_pred.csv"), index=False)
pd.DataFrame({"id": pd.Series(tWID) + "_" + pd.Series(tROW).astype(str), "tvt": tp}).to_csv(
    os.path.join(ROOT, "outputs", "submissions", "claude_cnn_20260716.csv"), index=False)
print("wrote oof.csv, test_pred.csv, submission")
