"""
01_eda.py — Exploratory Data Analysis for Spaceship Titanic

Angle 1 (compete): understand the data before modeling.
Angle 2 (document): every finding printed here feeds TUTORIAL.md.

Run:  python src/01_eda.py
Saves figures to outputs/figures/, prints a findings summary to stdout.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless: save figures, don't try to open windows
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(HERE, "data", "raw")
FIG = os.path.join(HERE, "outputs", "figures")
os.makedirs(FIG, exist_ok=True)

train = pd.read_csv(os.path.join(RAW, "train.csv"))
test = pd.read_csv(os.path.join(RAW, "test.csv"))

SPEND = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]

print("=" * 70)
print("SHAPES")
print("=" * 70)
print(f"train: {train.shape}   test: {test.shape}")

# --- Target balance ---------------------------------------------------------
print("\n" + "=" * 70)
print("TARGET BALANCE (Transported)")
print("=" * 70)
print(train["Transported"].value_counts(normalize=True).round(4).to_string())

# --- Missingness ------------------------------------------------------------
print("\n" + "=" * 70)
print("MISSINGNESS (% missing, train / test)")
print("=" * 70)
miss = pd.DataFrame({
    "train_%": (train.isna().mean() * 100).round(2),
    "test_%": (test.reindex(columns=train.columns).isna().mean() * 100).round(2),
})
print(miss.to_string())
print(f"\nRows with >=1 missing value: train {train.isna().any(axis=1).mean()*100:.1f}% | "
      f"test {test.isna().any(axis=1).mean()*100:.1f}%")

# --- Compound fields --------------------------------------------------------
# PassengerId = gggg_pp  (group / number-within-group)
# Cabin       = deck/num/side
print("\n" + "=" * 70)
print("COMPOUND FIELDS")
print("=" * 70)
grp = train["PassengerId"].str.split("_", expand=True)[0]
gsize = grp.map(grp.value_counts())
print(f"PassengerId groups: {grp.nunique()} unique groups among {len(train)} passengers")
print("group-size distribution (passengers travel in parties):")
print(gsize.value_counts().sort_index().to_string())

cab = train["Cabin"].str.split("/", expand=True)
cab.columns = ["deck", "num", "side"]
print("\nCabin deck counts:")
print(cab["deck"].value_counts(dropna=False).to_string())
print("\nCabin side counts (P=port, S=starboard):")
print(cab["side"].value_counts(dropna=False).to_string())

# --- Categorical vs target --------------------------------------------------
print("\n" + "=" * 70)
print("TRANSPORTED RATE BY KEY CATEGORICALS")
print("=" * 70)
for col in ["HomePlanet", "CryoSleep", "Destination", "VIP"]:
    rate = train.groupby(col)["Transported"].mean().round(3)
    print(f"\n{col}:")
    print(rate.to_string())

# CryoSleep is the big one — check spend behavior under CryoSleep
print("\n" + "=" * 70)
print("SPEND vs CRYOSLEEP (asleep passengers can't spend)")
print("=" * 70)
train["_total_spend"] = train[SPEND].sum(axis=1)
print(train.groupby("CryoSleep")["_total_spend"].agg(["mean", "median", "max"]).round(1).to_string())

# --- Figures ----------------------------------------------------------------
# 1. Missingness bar
ax = miss["train_%"].sort_values().plot.barh(figsize=(7, 5), color="#4C72B0")
ax.set_title("Missingness by column (train)")
ax.set_xlabel("% missing")
plt.tight_layout(); plt.savefig(os.path.join(FIG, "01_missingness.png"), dpi=120); plt.close()

# 2. Spend distributions (log1p) split by Transported
fig, axes = plt.subplots(1, len(SPEND), figsize=(18, 3.2))
for ax, col in zip(axes, SPEND):
    for val, color in [(True, "#55A868"), (False, "#C44E52")]:
        s = np.log1p(train.loc[train["Transported"] == val, col].dropna())
        ax.hist(s, bins=30, alpha=0.55, color=color, label=f"T={val}")
    ax.set_title(col); ax.set_xlabel("log1p(spend)")
axes[0].legend()
plt.tight_layout(); plt.savefig(os.path.join(FIG, "01_spend_by_target.png"), dpi=120); plt.close()

# 3. Age distribution by target
fig, ax = plt.subplots(figsize=(7, 4))
for val, color in [(True, "#55A868"), (False, "#C44E52")]:
    ax.hist(train.loc[train["Transported"] == val, "Age"].dropna(),
            bins=40, alpha=0.55, color=color, label=f"Transported={val}")
ax.set_title("Age by Transported"); ax.set_xlabel("Age"); ax.legend()
plt.tight_layout(); plt.savefig(os.path.join(FIG, "01_age_by_target.png"), dpi=120); plt.close()

print("\n" + "=" * 70)
print("FIGURES SAVED to outputs/figures/: "
      "01_missingness.png, 01_spend_by_target.png, 01_age_by_target.png")
print("=" * 70)
