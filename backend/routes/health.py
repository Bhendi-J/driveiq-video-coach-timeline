"""
backend/routes/health.py
GET /api/health — liveness probe
"""
import os
from flask import Blueprint, jsonify
from flask import current_app
from backend.routes.coach import get_coach_status

health_bp = Blueprint("health", __name__)


@health_bp.route("/api/health", methods=["GET"])
def health():
    models = current_app.config.get("MODELS", {})
    # Current primary path: XGBoost scoring with schema validation.
    core_required = ["xgb", "scaler"]

    core_models_loaded = all(models.get(k) is not None for k in core_required)
    schema_valid = bool(models.get("schema_valid", False))

    # Core readiness now requires both model presence and schema compatibility.
    models_loaded = core_models_loaded and schema_valid
    coach_state = get_coach_status()

    score_ready = models_loaded
    coach_ready = bool(coach_state.get("ready", False) or coach_state.get("disabled", False))
    overall_ready = score_ready and coach_ready
    degraded = not overall_ready

    return jsonify({
        "status": "ok",
        "service": "DriveIQ API",
        "models_loaded": models_loaded,
        "core_models_loaded": core_models_loaded and schema_valid,
        "schema_valid": schema_valid,
        "schema_error": models.get("schema_error"),
        "optional_models_loaded": True,
        "missing_core_models": [k for k in core_required if models.get(k) is None],
        "missing_optional_models": [],
        "score_ready": score_ready,
        "coach_ready": coach_ready,
        "coach_status": coach_state.get("status", "idle"),
        "coach_disabled": coach_state.get("disabled", False),
        "coach_error": coach_state.get("load_error"),
        "ready": overall_ready,
        "degraded": degraded,
        "version": os.environ.get("DRIVEIQ_API_VERSION", "v1"),
    })
