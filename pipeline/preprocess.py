"""
Phase 1: Feature Engineering & Preprocessing
Reads eco_driving_score.csv → normalizes, encodes, splits → saves to data/processed/

Run: python pipeline/preprocess.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT / "eco_driving_score.csv"
DATA_OUT = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"

DATA_OUT.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Canonical schema used by BOTH preprocess.py and cv_pipeline.feature_vector_for_xgb
XGB_FEATURE_SCHEMA = [
    "vehicle_count",
    "proximity_score",
    "pedestrian_flag",
    "mean_flow",
    "flow_variance",
    "braking_flag",
    "lane_change_flag",
    "road_type_id",
    "weather_id",
]

TARGET_COL = "eco_score"  # The column we predict (0–100)


def load_and_clean(path: Path):
    """Returns (df, target_col_name)."""
    print(f"[preprocess] Loading {path} ...")
    df = pd.read_csv(path)
    print(f"[preprocess] Raw shape: {df.shape}")
    print(f"[preprocess] Columns: {df.columns.tolist()}")

    # Forward-fill nulls (temporal data — carry last known value)
    df = df.ffill()

    target_col = TARGET_COL

    # If the expected target column is absent, auto-detect
    if target_col not in df.columns:
        candidates = [c for c in df.columns if "eco" in c.lower() or "score" in c.lower()]
        if candidates:
            target_col = candidates[0]
            print(f"[preprocess] Using '{target_col}' as target column")
        else:
            raise ValueError(
                f"Target column '{TARGET_COL}' not found. Available: {df.columns.tolist()}"
            )

    df = df.dropna(subset=[target_col])
    print(f"[preprocess] Clean shape: {df.shape}")
    return df, target_col


def _series_or_default(df: pd.DataFrame, name: str, default: float = 0.0) -> pd.Series:
    """Return a numeric series if the column exists, otherwise a default-filled series."""
    if name in df.columns:
        return pd.to_numeric(df[name], errors="coerce").fillna(default)
    return pd.Series(default, index=df.index, dtype=float)


def build_canonical_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Build the canonical vision-first schema expected by runtime inference."""
    feat = pd.DataFrame(index=df.index)

    rpm_variation = _series_or_default(df, "rpm_variation", 0.0)
    accel_smooth = _series_or_default(df, "acceleration_smoothness", 0.0)
    fuel_consumption = _series_or_default(df, "fuel_consumption", 0.0)
    harsh_brake_count = _series_or_default(df, "harsh_braking_count", 0.0)
    idling_time = _series_or_default(df, "idling_time", 0.0)

    # Build stable CV-style proxy features from available tabular signals.
    feat["braking_flag"] = (harsh_brake_count > 0).astype(float)
    feat["mean_flow"] = (np.abs(accel_smooth) * 0.7 + np.abs(rpm_variation) * 0.3).clip(0.0, 10.0)
    feat["flow_variance"] = np.abs(rpm_variation).clip(0.0, 10.0)
    feat["lane_change_flag"] = (np.abs(rpm_variation) > np.abs(rpm_variation).quantile(0.70)).astype(float)

    prox = (fuel_consumption / max(float(fuel_consumption.max()), 1.0)).clip(0.0, 1.0)
    feat["proximity_score"] = prox
    feat["vehicle_count"] = np.round((idling_time / 15.0) + (prox * 4.0)).clip(0.0, 12.0)
    feat["pedestrian_flag"] = (idling_time > idling_time.quantile(0.85)).astype(float)

    if "road_type_id" in df.columns:
        feat["road_type_id"] = _series_or_default(df, "road_type_id", 0.0)
    elif "road_type" in df.columns:
        road_map = {"city": 0.0, "highway": 1.0, "rural": 2.0}
        feat["road_type_id"] = (
            df["road_type"].astype(str).str.lower().map(road_map).fillna(0.0)
        )
    else:
        feat["road_type_id"] = 0.0

    if "weather_id" in df.columns:
        feat["weather_id"] = _series_or_default(df, "weather_id", 0.0)
    elif "weather" in df.columns:
        weather_map = {"clear": 0.0, "rain": 1.0, "fog": 2.0}
        feat["weather_id"] = (
            df["weather"].astype(str).str.lower().map(weather_map).fillna(0.0)
        )
    else:
        feat["weather_id"] = 0.0

    return feat[XGB_FEATURE_SCHEMA].astype(float)


def build_features_and_target(df: pd.DataFrame, target_col: str):
    """Build canonical feature frame and target without scaling."""
    feat = build_canonical_feature_frame(df)
    y = df[target_col].values.astype(float)
    return feat, y, XGB_FEATURE_SCHEMA


def split_scale_and_save(X_df: pd.DataFrame, y: np.ndarray, feature_cols: list[str]):
    """70/15/15 split, fit scaler on train only, transform val/test, then save CSVs."""
    X_train_df, X_tmp_df, y_train, y_tmp = train_test_split(
        X_df, y, test_size=0.30, random_state=42
    )
    X_val_df, X_test_df, y_val, y_test = train_test_split(
        X_tmp_df, y_tmp, test_size=0.50, random_state=42
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_df)
    X_val = scaler.transform(X_val_df)
    X_test = scaler.transform(X_test_df)

    splits = {
        "X_train": X_train,
        "X_val": X_val,
        "X_test": X_test,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
    }

    for name, arr in splits.items():
        out_path = DATA_OUT / f"{name}.csv"
        cols = feature_cols if name.startswith("X_") else None
        pd.DataFrame(arr, columns=cols).to_csv(out_path, index=False)
        print(f"[preprocess] Saved {name}.csv → {out_path}  shape={arr.shape}")

    return X_train, X_val, X_test, y_train, y_val, y_test, scaler


def main():
    df, target_col = load_and_clean(DATA_RAW)
    X_df, y, feature_cols = build_features_and_target(df, target_col)

    print(f"\n[preprocess] Feature columns used: {feature_cols}")
    print(f"[preprocess] X shape: {X_df.shape}  |  y shape: {y.shape}")
    print(f"[preprocess] y range: [{y.min():.2f}, {y.max():.2f}]")

    _, _, _, _, _, _, scaler = split_scale_and_save(X_df, y, feature_cols)

    # Save scaler and encoder metadata
    scaler_path = MODELS_DIR / "scaler.pkl"
    joblib.dump(
        {
            "scaler": scaler,
            "feature_cols": feature_cols,
            "schema": "xgb_v2_vision_9cols",
            "target_col": target_col,
        },
        scaler_path,
    )
    print(f"\n[preprocess] Saved scaler.pkl → {scaler_path}")
    print("[preprocess] ✅ Done — Phase 1 preprocessing complete.")


if __name__ == "__main__":
    main()
