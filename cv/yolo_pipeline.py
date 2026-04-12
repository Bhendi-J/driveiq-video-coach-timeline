"""
Phase 3: YOLOv8 Feature Extractor
Extracts vehicle_count, proximity_score, pedestrian_flag from a single frame.

Usage (standalone):
    python cv/yolo_pipeline.py --frame path/to/frame.jpg

Usage (as module):
    from cv.yolo_pipeline import extract_yolo_features
    features = extract_yolo_features(frame_bgr)  # numpy array HxWx3
"""

import numpy as np
import argparse
from pathlib import Path

# YOLOv8 class IDs (COCO)
VEHICLE_CLASS_IDS = {2, 3, 5, 7}   # car, motorbike, bus, truck
PERSON_CLASS_ID   = 0

# Lazy-load model to avoid import cost when used as a module
_yolo_model = None


def get_yolo_model():
    global _yolo_model
    if _yolo_model is None:
        from ultralytics import YOLO
        # Use nano model (fastest) to meet latency target
        _yolo_model = YOLO("yolov8n.pt")
    return _yolo_model


def extract_yolo_features(frame: np.ndarray, conf_threshold: float = 0.4) -> dict:
    """
    Run YOLOv8 on a single BGR frame and return eco-driving features.

    Args:
        frame: numpy array shape (H, W, 3), BGR format (OpenCV default).
        conf_threshold: minimum detection confidence.

    Returns:
        dict with keys:
            vehicle_count   (int)   — number of vehicles detected
            proximity_score (float) — area of closest vehicle bbox as % of frame area
            pedestrian_flag (int)   — 1 if any pedestrian detected, else 0
    """
    model = get_yolo_model()
    frame_area = frame.shape[0] * frame.shape[1]

    results = model(frame, conf=conf_threshold, verbose=False)[0]
    boxes   = results.boxes

    vehicle_count   = 0
    max_bbox_area   = 0.0
    pedestrian_flag = 0

    for box in boxes:
        cls  = int(box.cls[0].item())
        conf = float(box.conf[0].item())
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        bbox_area = (x2 - x1) * (y2 - y1)

        if cls in VEHICLE_CLASS_IDS:
            vehicle_count += 1
            if bbox_area > max_bbox_area:
                max_bbox_area = bbox_area

        if cls == PERSON_CLASS_ID:
            pedestrian_flag = 1

    proximity_score = round(max_bbox_area / frame_area, 4) if frame_area > 0 else 0.0

    return {
        "vehicle_count":   vehicle_count,
        "proximity_score": proximity_score,
        "pedestrian_flag": pedestrian_flag,
    }


# ── CLI entry point ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    import cv2

    parser = argparse.ArgumentParser(description="YOLOv8 feature extractor")
    parser.add_argument("--frame", required=True, help="Path to image or video frame")
    args = parser.parse_args()

    frame = cv2.imread(args.frame)
    if frame is None:
        print(f"❌ Could not read: {args.frame}")
    else:
        feats = extract_yolo_features(frame)
        print(feats)
