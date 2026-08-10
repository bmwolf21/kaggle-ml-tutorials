"""Generate the SHARED stratified 5-fold split. Run ONCE; all three agents load
shared/folds.csv and never regenerate it. Fold drift between agents silently
corrupts every OOF blend, so this is the single source of truth.

Stratified on the target because the classes are imbalanced (fit ~6%): unstratified
folds would leave some folds thin on minority classes and make CV noisy.
There is NO group structure in this data (rows are independent synthetic records),
so plain StratifiedKFold is the honest split - unlike ROGII, there is no group crux.
"""
from pathlib import Path
import pandas as pd
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
tr = pd.read_csv(ROOT / "data" / "train.csv")

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
tr["fold"] = -1
for k, (_, va) in enumerate(skf.split(tr, tr["health_condition"])):
    tr.loc[va, "fold"] = k

out = tr[["id", "fold"]].copy()
out.to_csv(ROOT / "shared" / "folds.csv", index=False)
print("wrote shared/folds.csv:", out.shape)
print("\nfold sizes:")
print(out["fold"].value_counts().sort_index().to_string())
print("\nper-fold class proportions (should be ~identical across folds):")
print(tr.groupby("fold")["health_condition"].value_counts(normalize=True).round(4).unstack().to_string())
