"""
backend/model_loader.py
Loads and caches all trained models at Flask startup.
"""

from __future__ import annotations

import joblib
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"

_cache = {}


def _validate_runtime_schema(feature_cols: list[str]) -> tuple[bool, str | None, list[str]]:
    """Validate trained feature schema against runtime scorer schema."""
    try:
        from cv.cv_pipeline import XGB_FEATURE_SCHEMA  # Runtime single source of truth.
        runtime_cols = list(XGB_FEATURE_SCHEMA)
    except Exception as e:
        return False, f"Runtime schema import failed: {e}", []

    if not feature_cols:
        return False, "Scaler metadata missing feature_cols", runtime_cols

    if list(feature_cols) != runtime_cols:
        return (
            False,
            f"Schema mismatch. trained={feature_cols} runtime={runtime_cols}",
            runtime_cols,
        )

    return True, None, runtime_cols


def load_models() -> dict:
    """
    Load all DriveIQ models once and return a dict.
    Subsequent calls return the cached dict.

    Returns:
        {
            "xgb":       XGBRegressor | None,
            "scaler":    StandardScaler | None,
            "feature_cols": list[str],
        }
    """
    if _cache:
        return _cache

    print("[model_loader] Loading models ...")

    _cache["schema_valid"] = False
    _cache["schema_error"] = "schema_not_checked"
    _cache["runtime_feature_cols"] = []
    _cache["trained_feature_cols"] = []

    # ── XGBoost ───────────────────────────────────────────────────────────────
    xgb_path = MODELS_DIR / "xgb_scorer.pkl"
    if xgb_path.exists():
        _cache["xgb"] = joblib.load(xgb_path)
        print(f"[model_loader] ✅ XGBoost loaded from {xgb_path}")
    else:
        _cache["xgb"] = None
        print(f"[model_loader] ⚠️  XGBoost not found at {xgb_path} — /api/score will return mock")

    # ── Scaler ────────────────────────────────────────────────────────────────
    scaler_path = MODELS_DIR / "scaler.pkl"
    if scaler_path.exists():
        bundle = joblib.load(scaler_path)
        _cache["scaler"]       = bundle.get("scaler")
        _cache["feature_cols"] = bundle.get("feature_cols", [])
        _cache["trained_feature_cols"] = list(_cache["feature_cols"])
        print(f"[model_loader] ✅ Scaler loaded — features: {_cache['feature_cols']}")
    else:
        _cache["scaler"]       = None
        _cache["feature_cols"] = []
        _cache["trained_feature_cols"] = []
        print(f"[model_loader] ⚠️  Scaler not found — run pipeline/preprocess.py first")

    schema_valid, schema_error, runtime_cols = _validate_runtime_schema(_cache.get("feature_cols", []))
    _cache["schema_valid"] = schema_valid
    _cache["schema_error"] = schema_error
    _cache["runtime_feature_cols"] = runtime_cols

    if schema_valid:
        print("[model_loader] ✅ Runtime schema matches trained scaler metadata")
    else:
        print(f"[model_loader] ❌ Runtime schema invalid: {schema_error}")
        # Fail closed: never use a potentially incompatible model+scaler pair.
        _cache["xgb"] = None
        _cache["scaler"] = None

    print("[model_loader] ✅ All models initialised.")
    return _cache
