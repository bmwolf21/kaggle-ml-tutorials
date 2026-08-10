import sys
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

def diagnose():
    print("Loading data...")
    train = pd.read_csv(ROOT / "data" / "train.csv")
    test = pd.read_csv(ROOT / "data" / "test.csv")
    
    print("\nTrain shape:", train.shape)
    print("Test shape:", test.shape)
    
    num_cols = ["sleep_duration", "heart_rate", "bmi", "calorie_expenditure",
                "step_count", "exercise_duration", "water_intake"]
    cat_cols = ["diet_type", "stress_level", "sleep_quality", "physical_activity_level",
                "smoking_alcohol", "gender"]
    
    print("\nCategorical Column Unique Values & NAs (Train):")
    for col in cat_cols:
        uniques = train[col].dropna().unique()
        nas = train[col].isna().sum()
        test_nas = test[col].isna().sum()
        print(f"  {col}: {len(uniques)} unique values {list(uniques)} | Train NA: {nas} ({nas/len(train):.2%}) | Test NA: {test_nas} ({test_nas/len(test):.2%})")

    print("\nNumeric Column Ranges & NAs (Train):")
    for col in num_cols:
        nas = train[col].isna().sum()
        test_nas = test[col].isna().sum()
        mean_val = train[col].mean()
        std_val = train[col].std()
        print(f"  {col}: mean={mean_val:.2f}, std={std_val:.2f} | Train NA: {nas} ({nas/len(train):.2%}) | Test NA: {test_nas} ({test_nas/len(test):.2%})")

    print("\nTarget Class Distribution:")
    counts = train["health_condition"].value_counts(dropna=False)
    for k, v in counts.items():
        print(f"  {k}: {v} ({v/len(train):.2%})")

if __name__ == "__main__":
    diagnose()
