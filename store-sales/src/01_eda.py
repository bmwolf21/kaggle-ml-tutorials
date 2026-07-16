"""
01_eda.py - Exploratory Data Analysis for Store Sales.

Angle 1 (compete): understand seasonality, zeros, trends, and driver effects.
Angle 2 (document): findings feed TUTORIAL.md.

Run:  python src/01_eda.py
Saves figures to outputs/figures/.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(HERE, "data", "raw")
FIG = os.path.join(HERE, "outputs", "figures")
os.makedirs(FIG, exist_ok=True)

train = pd.read_csv(os.path.join(RAW, "train.csv"), parse_dates=["date"])

print("=" * 70)
print(f"train {train.shape} | {train['date'].min().date()} to {train['date'].max().date()}")
print(f"series: {train['store_nbr'].nunique()} stores x {train['family'].nunique()} families")
print("=" * 70)

# --- Target: skew + zeros ---------------------------------------------------
s = train["sales"]
print("\nTARGET (sales):")
print(f"  median {s.median():.1f} | mean {s.mean():.1f} | max {s.max():,.0f}")
print(f"  zero-sales rows: {(s == 0).mean()*100:.1f}%")
print(f"  skew raw {s.skew():.2f} | skew log1p {np.log1p(s).skew():.2f}")
print("  -> heavy zeros + right skew; model log1p(sales) (matches RMSLE).")

# --- Seasonality: day-of-week and month -------------------------------------
train["dow"] = train["date"].dt.dayofweek
train["month"] = train["date"].dt.month
dow = train.groupby("dow")["sales"].mean()
mon = train.groupby("month")["sales"].mean()
print("\nMean sales by day-of-week (0=Mon):")
print(dow.round(1).to_string())
print("\nMean sales by month:")
print(mon.round(1).to_string())

# --- Promotion effect -------------------------------------------------------
promo = train.groupby("onpromotion" if train["onpromotion"].nunique() < 3
                       else pd.cut(train["onpromotion"], [-1, 0, 5, 50, 1e9]))["sales"].mean()
print("\nMean sales by promotion bucket:")
print(promo.round(1).to_string())

# --- Overall trend (total daily sales) --------------------------------------
daily = train.groupby("date")["sales"].sum()

# --- Figures ----------------------------------------------------------------
plt.figure(figsize=(12, 4))
daily.rolling(7).mean().plot(color="#4C72B0")
plt.title("Total daily sales (7-day rolling mean)"); plt.ylabel("sales")
plt.tight_layout(); plt.savefig(os.path.join(FIG, "01_trend.png"), dpi=120); plt.close()

fig, ax = plt.subplots(1, 2, figsize=(12, 4))
dow.plot.bar(ax=ax[0], color="#55A868"); ax[0].set_title("Mean sales by day-of-week")
mon.plot.bar(ax=ax[1], color="#C44E52"); ax[1].set_title("Mean sales by month")
plt.tight_layout(); plt.savefig(os.path.join(FIG, "01_seasonality.png"), dpi=120); plt.close()

# one example series (store 1, GROCERY I) to show series-level structure
ex = train[(train["store_nbr"] == 1) & (train["family"] == "GROCERY I")]
plt.figure(figsize=(12, 3.5))
plt.plot(ex["date"], ex["sales"], color="#4C72B0", lw=0.6)
plt.title("Example series: store 1, GROCERY I"); plt.tight_layout()
plt.savefig(os.path.join(FIG, "01_example_series.png"), dpi=120); plt.close()

print("\nFIGURES SAVED: 01_trend.png, 01_seasonality.png, 01_example_series.png")
