"""Build coaching dataset using a dedicated CSV-only coaching model.

Run:
    python pipeline/build_coaching_dataset.py
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor


ROOT = Path(__file__).resolve().parent.parent
DATA_CSV = ROOT / "eco_driving_score.csv"
COACH_MODEL_PATH = ROOT / "models" / "coaching_xgb_scorer.pkl"
COACH_SCALER_PATH = ROOT / "models" / "coaching_scaler.pkl"
OUT_PATH = ROOT / "data" / "coaching_dataset.csv"

BASE_COLUMNS = [
    "rpm_variation",
    "harsh_braking_count",
    "idling_time",
    "fuel_consumption",
    "acceleration_smoothness",
    "eco_score",
]

FEATURE_COLS = [
    "rpm_variation",
    "harsh_braking_count",
    "idling_time",
    "fuel_consumption",
    "acceleration_smoothness",
]
TARGET_COL = "eco_score"


COACHING_TEXT_MAP: dict[tuple[str, str], str] = {
    # rpm_variation
    ("rpm_variation", "green"): "RPM is stable in this segment; keep early upshifts and maintain a steady throttle to preserve efficiency.",
    ("rpm_variation", "yellow"): "RPM is slightly erratic; shift one gear earlier during acceleration and avoid throttle surges.",
    ("rpm_variation", "red"): "RPM is highly erratic here; shift up earlier and hold a smooth throttle to stop repeated rev spikes.",
    # harsh_braking_count
    ("harsh_braking_count", "green"): "Braking behavior is controlled; continue scanning ahead so deceleration stays progressive.",
    ("harsh_braking_count", "yellow"): "Harsh braking appears occasionally; increase following distance and lift off earlier before stops.",
    ("harsh_braking_count", "red"): "Frequent harsh braking is hurting this segment; anticipate traffic sooner and brake in a single smooth phase.",
    # idling_time
    ("idling_time", "green"): "Idling is low; keep reducing standstill engine time when traffic allows.",
    ("idling_time", "yellow"): "Idling is moderate; avoid unnecessary engine-on waiting and move off promptly when safe.",
    ("idling_time", "red"): "Excessive idling is dominating this segment; minimize stationary engine time and plan smoother gap entry.",
    # fuel_consumption
    ("fuel_consumption", "green"): "Fuel use is controlled; maintain this pace with gentle throttle transitions.",
    ("fuel_consumption", "yellow"): "Fuel use is elevated; reduce aggressive acceleration and keep speed changes gradual.",
    ("fuel_consumption", "red"): "Fuel use is very high here; immediately smooth acceleration and avoid rapid stop-go cycles.",
    # acceleration_smoothness
    ("acceleration_smoothness", "green"): "Acceleration profile is smooth; continue building speed progressively without sudden pedal inputs.",
    ("acceleration_smoothness", "yellow"): "Acceleration smoothness is inconsistent; apply throttle in smaller, steadier increments.",
    ("acceleration_smoothness", "red"): "Acceleration is highly uneven in this segment; stabilize pedal input and avoid repeated hard throttle pulses.",
}


def _ensure_file(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing {label}: {path}")


def _severity_from_score(score: float) -> str:
    if score > 65:
        return "green"
    if score > 40:
        return "yellow"
    return "red"


def _coaching_text(top_issue: str, severity: str) -> str:
    key = (top_issue, severity)
    if key not in COACHING_TEXT_MAP:
        raise ValueError(
            f"No coaching text mapping for issue/severity: {top_issue}/{severity}. "
            "Add an explicit deterministic mapping before generating dataset."
        )
    return COACHING_TEXT_MAP[key]


def _normalize_shap_values(shap_values: np.ndarray | list[np.ndarray]) -> np.ndarray:
    """Normalize SHAP output to a 2D array of shape (n_rows, n_features)."""
    if isinstance(shap_values, list):
        if len(shap_values) != 1:
            raise ValueError(
                f"Unexpected SHAP list output length for regressor: {len(shap_values)}"
            )
        shap_values = shap_values[0]

    arr = np.asarray(shap_values)
    if arr.ndim != 2:
        raise ValueError(f"Unexpected SHAP output shape: {arr.shape}")
    return arr


def _load_or_train_coaching_artifacts(X: pd.DataFrame, y: pd.Series) -> tuple[StandardScaler, XGBRegressor]:
    """Load coaching-only scaler/model if present, otherwise train and save them."""
    if COACH_SCALER_PATH.exists() and COACH_MODEL_PATH.exists():
        scaler_bundle = joblib.load(COACH_SCALER_PATH)
        scaler = scaler_bundle.get("scaler")
        scaler_cols = scaler_bundle.get("feature_cols")
        if scaler is None or scaler_cols != FEATURE_COLS:
            raise ValueError(
                "Existing coaching scaler is invalid or has mismatched feature_cols. "
                "Delete coaching artifacts and rerun to retrain."
            )
        model = joblib.load(COACH_MODEL_PATH)
        print(f"Loaded coaching scaler: {COACH_SCALER_PATH}")
        print(f"Loaded coaching model: {COACH_MODEL_PATH}")
        return scaler, model

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        eval_metric="rmse",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_scaled, y)

    COACH_SCALER_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"scaler": scaler, "feature_cols": FEATURE_COLS}, COACH_SCALER_PATH)
    joblib.dump(model, COACH_MODEL_PATH)
    print(f"Trained and saved coaching scaler: {COACH_SCALER_PATH}")
    print(f"Trained and saved coaching model: {COACH_MODEL_PATH}")
    return scaler, model


def main() -> None:
    _ensure_file(DATA_CSV, "input CSV")

    df = pd.read_csv(DATA_CSV)
    df = df.dropna().ffill()

    missing_base = [c for c in BASE_COLUMNS if c not in df.columns]
    if missing_base:
        raise ValueError(
            f"Input CSV missing required base columns: {missing_base}. "
            f"Available columns: {df.columns.tolist()}"
        )

    X = df.loc[:, FEATURE_COLS].apply(pd.to_numeric, errors="coerce")
    if X.isna().any().any():
        raise ValueError("Feature matrix contains non-numeric values after coercion in CSV feature columns")
    y = pd.to_numeric(df[TARGET_COL], errors="coerce")
    if y.isna().any():
        raise ValueError("Target column eco_score contains non-numeric values")

    scaler, model = _load_or_train_coaching_artifacts(X, y)
    X_scaled_arr = scaler.transform(X)
    X_scaled = pd.DataFrame(X_scaled_arr, columns=FEATURE_COLS)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_scaled)
    shap_arr = _normalize_shap_values(shap_values)

    abs_shap = np.abs(shap_arr)
    ranked_idx = np.argsort(-abs_shap, axis=1)
    top_idx = ranked_idx[:, 0]
    top_2_idx = ranked_idx[:, 1]

    predicted_score = model.predict(X_scaled)
    severity = [_severity_from_score(float(s)) for s in predicted_score]
    top_issue = [FEATURE_COLS[int(i)] for i in top_idx]
    top_2_issue = [FEATURE_COLS[int(i)] for i in top_2_idx]
    coaching_text = [_coaching_text(i, sev) for i, sev in zip(top_issue, severity)]

    out_df = df.loc[:, BASE_COLUMNS].copy()
    out_df["predicted_score"] = predicted_score.astype(float)
    out_df["top_issue"] = top_issue
    out_df["top_2_issue"] = top_2_issue
    out_df["severity"] = severity
    out_df["coaching_text"] = coaching_text

    # Enforce exact output column order.
    out_df = out_df[
        [
            "rpm_variation",
            "harsh_braking_count",
            "idling_time",
            "fuel_consumption",
            "acceleration_smoothness",
            "eco_score",
            "predicted_score",
            "top_issue",
            "top_2_issue",
            "severity",
            "coaching_text",
        ]
    ]

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(OUT_PATH, index=False)

    print(f"Saved: {OUT_PATH}")
    print(f"shape: {out_df.shape}")
    print("severity distribution:")
    print(out_df["severity"].value_counts(dropna=False))
    print("top_issue distribution:")
    print(out_df["top_issue"].value_counts(dropna=False))
    print("first 3 rows:")
    print(out_df.head(3).to_string(index=False))


if __name__ == "__main__":
    main()
