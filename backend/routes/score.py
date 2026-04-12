"""
backend/routes/score.py
POST /api/score
Body: { "telemetry": { speed, rpm, throttle_position, gear, acceleration, fuel_rate },
        "frame_b64": "<base64-encoded jpg>" (optional) }

Returns: { "score": float, "features": dict }
"""

from __future__ import annotations

import base64
import numpy as np
from flask import Blueprint, request, jsonify, current_app

score_bp = Blueprint("score", __name__)

MAX_SESSION_FRAMES = 128
_PREV_FRAME_BY_SESSION: dict[str, np.ndarray] = {}


def _decode_frame(b64_str: str) -> np.ndarray | None:
    """Decode a base64 JPEG/PNG string into a numpy BGR array."""
    try:
        import cv2
        img_bytes = base64.b64decode(b64_str)
        arr = np.frombuffer(img_bytes, np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)
    except Exception:
        return None


def _vision_like_features_from_telemetry(telemetry: dict) -> dict:
    """Synthesize CV-like features when only telemetry is available."""
    speed = float(telemetry.get("speed", 0.0))
    accel = float(telemetry.get("acceleration", 0.0))
    throttle = float(telemetry.get("throttle_position", 0.0))

    proximity = max(0.0, min(1.0, (speed - 40.0) / 100.0))
    mean_flow = abs(accel) * 0.8 + throttle / 120.0
    flow_variance = abs(accel) * 0.5

    return {
        "vehicle_count": int(max(0, min(12, round(1 + speed / 30.0)))),
        "proximity_score": round(proximity, 4),
        "pedestrian_flag": 0,
        "mean_flow": round(mean_flow, 4),
        "flow_variance": round(flow_variance, 4),
        "braking_flag": int(accel < -0.8),
        "lane_change_flag": int(abs(accel) > 1.6),
        "road_type_id": 1 if speed > 70 else 0,
        "weather_id": 0,
    }


def _heuristic_score_from_features(features: dict) -> float:
    """Fallback eco score based on vision-like behavior signals."""
    score = 85.0
    score -= float(features.get("proximity_score", 0.0)) * 30.0
    score -= float(features.get("mean_flow", 0.0)) * 8.0
    score -= float(features.get("flow_variance", 0.0)) * 10.0
    score -= float(features.get("braking_flag", 0.0)) * 12.0
    score -= float(features.get("lane_change_flag", 0.0)) * 10.0
    score -= float(features.get("pedestrian_flag", 0.0)) * 5.0
    return max(0.0, min(100.0, score))


def _session_key(payload: dict) -> str | None:
    """Resolve an explicit per-client key for previous-frame continuity.

    We intentionally avoid implicit IP-based fallback to prevent cross-user
    cache contamination behind NAT/proxies.
    """
    sid = payload.get("session_id")
    if sid:
        return str(sid)

    hdr = request.headers.get("X-Session-Id")
    if hdr:
        return hdr

    return None


def _cache_prev_frame(session_id: str | None, frame: np.ndarray) -> None:
    """Store previous frame with a small bounded cache to avoid unbounded growth."""
    if not session_id:
        return

    if len(_PREV_FRAME_BY_SESSION) >= MAX_SESSION_FRAMES and session_id not in _PREV_FRAME_BY_SESSION:
        oldest_key = next(iter(_PREV_FRAME_BY_SESSION))
        _PREV_FRAME_BY_SESSION.pop(oldest_key, None)
    _PREV_FRAME_BY_SESSION[session_id] = frame


@score_bp.route("/api/score", methods=["POST"])
def score():
    data = request.get_json(silent=True) or {}
    telemetry = data.get("telemetry", {})
    frame_b64 = data.get("frame_b64")
    prev_frame_b64 = data.get("prev_frame_b64")
    session_id = _session_key(data)

    models = current_app.config["MODELS"]
    xgb    = models.get("xgb")
    scaler = models.get("scaler")

    if models.get("schema_valid") is False:
        return jsonify({
            "error": "schema_mismatch",
            "message": "Runtime feature schema does not match trained artifacts",
            "details": models.get("schema_error"),
            "trained_feature_cols": models.get("trained_feature_cols", []),
            "runtime_feature_cols": models.get("runtime_feature_cols", []),
        }), 503

    # ── If frame provided, run CV pipeline ───────────────────────────────────
    cv_features = _vision_like_features_from_telemetry(telemetry)
    if frame_b64:
        try:
            from cv.cv_pipeline import cv_pipeline, feature_vector_for_xgb

            frame = _decode_frame(frame_b64)
            if frame is None:
                raise ValueError("invalid frame_b64")

            # Use explicit previous frame when provided, otherwise use cached session frame
            # only when session_id is explicitly supplied.
            if prev_frame_b64:
                prev_frame = _decode_frame(prev_frame_b64)
            elif session_id:
                prev_frame = _PREV_FRAME_BY_SESSION.get(session_id)
            else:
                prev_frame = None

            cv_features = cv_pipeline(frame, prev_frame, telemetry)
            _cache_prev_frame(session_id, frame)
        except Exception as e:
            cv_features = {**cv_features, "cv_error": str(e)}

    # ── Build feature vector for XGBoost ─────────────────────────────────────
    feature_dict = {**telemetry, **cv_features}

    if xgb is not None and scaler is not None:
        try:
            from cv.cv_pipeline import feature_vector_for_xgb
            x = feature_vector_for_xgb(feature_dict, scaler)
            score_val = float(xgb.predict(x)[0])
            # Clip to 0–100 range
            score_val = max(0.0, min(100.0, score_val))
        except Exception:
            score_val = _heuristic_score_from_features(feature_dict)
    else:
        # Mock score when model not yet trained
        score_val = _heuristic_score_from_features(feature_dict)

    return jsonify({
        "score":    round(score_val, 2),
        "features": feature_dict,
    })
