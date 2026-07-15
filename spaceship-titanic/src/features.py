"""
features.py — shared feature engineering for Spaceship Titanic.

Imported by the modeling scripts so train and test get identical treatment.
Design choices are driven by the EDA (see TUTORIAL.md Step 1):

- Decode compound IDs: PassengerId -> Group/GroupSize; Cabin -> Deck/Num/Side.
- Deterministic imputation using the CryoSleep<->spend link.
- Group-level features (people travel — and are transported — together).
"""
import numpy as np
import pandas as pd

SPEND = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
CATEGORICAL = ["HomePlanet", "Destination", "Deck", "Side", "CryoSleep", "VIP"]


def build_features(train: pd.DataFrame, test: pd.DataFrame):
    """Return (X_train, y_train, X_test) with engineered, model-ready features.

    train/test are the raw CSVs. We concatenate them for ID-structure features
    (group size, family size) — this uses only the ID columns, never the target,
    so it does not leak.
    """
    y = train["Transported"].astype(int)

    both = pd.concat([train.drop(columns=["Transported"]), test], ignore_index=True)
    n_train = len(train)

    # --- Decode PassengerId = gggg_pp -------------------------------------
    both["Group"] = both["PassengerId"].str.split("_").str[0]
    both["GroupSize"] = both["Group"].map(both["Group"].value_counts())
    both["IsAlone"] = (both["GroupSize"] == 1).astype(int)

    # --- Decode Cabin = deck/num/side -------------------------------------
    cab = both["Cabin"].str.split("/", expand=True)
    both["Deck"] = cab[0]
    both["Num"] = pd.to_numeric(cab[1], errors="coerce")
    both["Side"] = cab[2]

    # --- Surname (family) from Name ---------------------------------------
    both["Surname"] = both["Name"].str.split().str[-1]
    both["FamilySize"] = both["Surname"].map(both["Surname"].value_counts())

    # --- Deterministic CryoSleep <-> spend logic --------------------------
    both["TotalSpend"] = both[SPEND].sum(axis=1, min_count=1)
    spent = both["TotalSpend"] > 0
    # Anyone who spent money was awake.
    both.loc[spent, "CryoSleep"] = both.loc[spent, "CryoSleep"].fillna(False)
    # Asleep passengers spend nothing -> fill their missing amenities with 0.
    asleep = both["CryoSleep"] == True  # noqa: E712
    both.loc[asleep, SPEND] = both.loc[asleep, SPEND].fillna(0)
    # Recompute total after filling.
    both["TotalSpend"] = both[SPEND].sum(axis=1, min_count=1)
    both["HasSpend"] = (both["TotalSpend"] > 0).astype("float")

    # --- Log-transform skewed spend ---------------------------------------
    for c in SPEND:
        both[c + "_log"] = np.log1p(both[c])
    both["TotalSpend_log"] = np.log1p(both["TotalSpend"])

    # --- Simple imputation for the rest -----------------------------------
    for c in SPEND + [c + "_log" for c in SPEND] + ["TotalSpend", "TotalSpend_log"]:
        both[c] = both[c].fillna(0)
    both["Age"] = both["Age"].fillna(both["Age"].median())
    both["Num"] = both["Num"].fillna(both["Num"].median())

    # Categoricals: encode as pandas 'category' codes (LightGBM-friendly,
    # NaN -> its own code -1 which trees can split on).
    for c in CATEGORICAL:
        both[c] = both[c].astype("category").cat.codes

    feature_cols = (
        ["Age", "GroupSize", "IsAlone", "FamilySize", "Num", "HasSpend",
         "TotalSpend", "TotalSpend_log"]
        + SPEND + [c + "_log" for c in SPEND]
        + CATEGORICAL
    )

    X = both[feature_cols]
    X_train = X.iloc[:n_train].reset_index(drop=True)
    X_test = X.iloc[n_train:].reset_index(drop=True)
    return X_train, y, X_test, feature_cols
