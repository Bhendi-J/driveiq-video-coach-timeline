"""
Phase 3: Combined CV Pipeline
Merges YOLO + optical flow features into a single feature vector.

Usage:
    from cv.cv_pipeline import cv_pipeline
    feature_dict = cv_pipeline(curr_frame_bgr, prev_frame_bgr, telemetry_dict)
"""

from __future__ import annotations

import numpy as np
import cv2


# Must stay in sync with pipeline/preprocess.py:XGB_FEATURE_SCHEMA
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


def classify_scene(frame_bgr: np.ndarray) -> dict:
    """Return deterministic scene defaults (legacy MobileNet path removed)."""
    _ = frame_bgr
    return {
        "road_type": "unknown",
        "weather": "unknown",
        "road_type_id": -1,
        "weather_id": -1,
    }


def cv_pipeline(
    curr_frame: np.ndarray,
    prev_frame: np.ndarray | None = None,
    telemetry: dict | None = None,
) -> dict:
    """
    Full computer-vision feature extraction pipeline.

    Args:
        curr_frame:  Current video frame (BGR, uint8)
        prev_frame:  Previous frame for optical flow (BGR, uint8), or None
        telemetry:   Dict with keys: speed, rpm, throttle_position, gear, acceleration, fuel_rate

    Returns:
        Combined feature dict ready for model inference.
    """
    from cv.yolo_pipeline import extract_yolo_features
    from cv.optical_flow  import extract_flow_features

    # ── YOLO features ────────────────────────────────────────────────────────
    yolo_feats = extract_yolo_features(curr_frame)

    # ── Optical flow features ─────────────────────────────────────────────────
    if prev_frame is not None:
        flow_feats = extract_flow_features(prev_frame, curr_frame)
    else:
        flow_feats = {
            "mean_flow": 0.0, "variance": 0.0,
            "braking_flag": 0, "lane_change_flag": 0,
        }

    # ── Scene classification ──────────────────────────────────────────────────
    scene_feats = classify_scene(curr_frame)

    # ── Telemetry passthrough ─────────────────────────────────────────────────
    tele = telemetry or {}

    feature_vector = {
        # YOLO
        "vehicle_count":      yolo_feats["vehicle_count"],
        "proximity_score":    yolo_feats["proximity_score"],
        "pedestrian_flag":    yolo_feats["pedestrian_flag"],
        # Optical flow
        "mean_flow":          flow_feats["mean_flow"],
        "flow_variance":      flow_feats["variance"],
        "braking_flag":       flow_feats["braking_flag"],
        "lane_change_flag":   flow_feats["lane_change_flag"],
        # Scene
        "road_type":          scene_feats["road_type"],
        "weather":            scene_feats["weather"],
        "road_type_id":       scene_feats["road_type_id"],
        "weather_id":         scene_feats["weather_id"],
        # Telemetry
        "speed":              tele.get("speed", 0),
        "rpm":                tele.get("rpm", 0),
        "throttle_position":  tele.get("throttle_position", 0),
        "gear":               tele.get("gear", 1),
        "acceleration":       tele.get("acceleration", 0),
        "fuel_rate":          tele.get("fuel_rate", 0),
    }

    return feature_vector


def feature_vector_for_xgb(features: dict, scaler=None) -> np.ndarray:
    """
    Convert the full feature dict into the numeric vector expected by XGBoost.
    Matches the column order from preprocess.py.
    """
    row = {
        "vehicle_count": float(features.get("vehicle_count", 0.0)),
        "proximity_score": float(features.get("proximity_score", 0.0)),
        "pedestrian_flag": float(features.get("pedestrian_flag", 0.0)),
        "mean_flow": float(features.get("mean_flow", 0.0)),
        "flow_variance": float(features.get("flow_variance", 0.0)),
        "braking_flag": float(features.get("braking_flag", 0.0)),
        "lane_change_flag": float(features.get("lane_change_flag", 0.0)),
        "road_type_id": float(features.get("road_type_id", 0.0)),
        "weather_id": float(features.get("weather_id", 0.0)),
    }

    # Build row in the canonical order used during scaler training.
    v = np.array([[row[c] for c in XGB_FEATURE_SCHEMA]], dtype=np.float32)

    if scaler is not None:
        # Use DataFrame when possible so sklearn can align by feature names.
        try:
            import pandas as pd
            col_order = list(getattr(scaler, "feature_names_in_", XGB_FEATURE_SCHEMA))
            x_df = pd.DataFrame([{c: row.get(c, 0.0) for c in col_order}], columns=col_order)
            v = scaler.transform(x_df)
        except Exception:
            v = scaler.transform(v)

    return v


# ── Quick sanity test ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys, cv2

    if len(sys.argv) < 2:
        print("Usage: python cv/cv_pipeline.py path/to/video.mp4")
        sys.exit(0)

    cap = cv2.VideoCapture(sys.argv[1])
    ret, prev = cap.read()
    ret, curr = cap.read()
    cap.release()

    if not ret:
        print("❌ Could not read frames from video")
    else:
        result = cv_pipeline(curr, prev, telemetry={"speed": 60, "rpm": 2000})
        for k, v in result.items():
            print(f"  {k:<25} = {v}")
