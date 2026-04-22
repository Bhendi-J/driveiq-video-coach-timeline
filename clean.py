import pandas as pd
import numpy as np
import re

df = pd.read_csv("eco_driving_cv_dataset_30k.csv")

print("Before cleaning:", df.shape)

# Fix eco_score formatting
def fix_numeric(val):
    if pd.isna(val):
        return np.nan
    val = str(val)
    val = re.sub(r'(\d)\s+(\d)', r'\1\2', val)
    val = re.sub(r'[^0-9.]', '', val)
    try:
        return float(val)
    except:
        return np.nan

df["eco_score"] = df["eco_score"].apply(fix_numeric)

# Convert all to numeric
for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna()

# --- IQR filtering (fixed version) ---
numeric_cols = df.select_dtypes(include=[np.number]).columns
mask = pd.Series(True, index=df.index)

for col in numeric_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    mask &= (df[col] >= lower) & (df[col] <= upper)

df = df[mask]

# Clip ranges
df["eco_score"] = df["eco_score"].clip(0, 100)
df["braking_ratio"] = df["braking_ratio"].clip(0, 1)
df["lane_change_ratio"] = df["lane_change_ratio"].clip(0, 1)
df["proximity_score"] = df["proximity_score"].clip(0, 1)
df["pedestrian_ratio"] = df["pedestrian_ratio"].clip(0, 1)
df["low_motion_ratio"] = df["low_motion_ratio"].clip(0, 1)
df["vehicle_density"] = df["vehicle_density"].clip(0, 10)

# Consistency fixes
df.loc[df["mean_flow"] < 0.5, "flow_variance"] = np.minimum(df["flow_variance"], 20)
df.loc[df["braking_ratio"] > 0.6, "flow_variance"] = np.maximum(df["flow_variance"], 40)
df.loc[df["low_motion_ratio"] > 0.7, "mean_flow"] = np.minimum(df["mean_flow"], 3)

# Remove extremes
df = df[(df["eco_score"] > 5) & (df["eco_score"] < 95)]

# Save
df.to_csv("eco_driving_cv_dataset_clean.csv", index=False)

print("After cleaning:", df.shape)

# Validation
print("\nSummary:\n", df.describe())
print("\nCorrelation:\n", df.corr(numeric_only=True)["eco_score"].sort_values())