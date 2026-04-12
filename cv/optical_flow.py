"""
Phase 3: Farneback Optical Flow Feature Extractor
Extracts mean_flow, variance, braking_flag, lane_change_flag from consecutive frames.

Usage (as module):
    from cv.optical_flow import extract_flow_features
    features = extract_flow_features(prev_frame_bgr, curr_frame_bgr)
"""

import cv2
import numpy as np

# Farneback parameters (good balance of speed vs quality)
FARNEBACK_PARAMS = dict(
    pyr_scale=0.5,
    levels=3,
    winsize=15,
    iterations=3,
    poly_n=5,
    poly_sigma=1.2,
    flags=0,
)

# Thresholds — tune these based on your video resolution / frame rate
BRAKING_THRESH      = -1.5   # negative vertical flow (camera tilts forward when braking)
LANE_CHANGE_THRESH  =  2.0   # horizontal flow magnitude


def extract_flow_features(
    prev_frame: np.ndarray,
    curr_frame: np.ndarray,
) -> dict:
    """
    Compute Farneback optical flow between two BGR frames.

    Args:
        prev_frame: previous frame (BGR, uint8)
        curr_frame: current frame  (BGR, uint8)

    Returns:
        dict with keys:
            mean_flow         (float)  — mean magnitude of all flow vectors
            variance          (float)  — variance of flow magnitudes
            braking_flag      (int)    — 1 if strong negative y-flow (deceleration)
            lane_change_flag  (int)    — 1 if strong x-flow asymmetry
    """
    # Convert to grayscale
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

    # Compute dense optical flow
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, curr_gray, None, **FARNEBACK_PARAMS
    )  # shape: (H, W, 2) — [x-flow, y-flow]

    fx = flow[..., 0]   # horizontal component
    fy = flow[..., 1]   # vertical component

    # Magnitude of each flow vector
    magnitude = np.sqrt(fx ** 2 + fy ** 2)

    mean_flow = float(np.mean(magnitude))
    variance  = float(np.var(magnitude))

    # Braking: strong uniform negative y-flow in the central crop
    h, w = fy.shape
    central_fy = fy[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4]
    braking_flag = int(np.mean(central_fy) < BRAKING_THRESH)

    # Lane change: asymmetric horizontal flow (left vs right halves differ)
    left_fx  = fx[:, : w // 2]
    right_fx = fx[:, w // 2 :]
    x_asymmetry = abs(float(np.mean(left_fx)) - float(np.mean(right_fx)))
    lane_change_flag = int(x_asymmetry > LANE_CHANGE_THRESH)

    return {
        "mean_flow":        round(mean_flow, 4),
        "variance":         round(variance, 4),
        "braking_flag":     braking_flag,
        "lane_change_flag": lane_change_flag,
    }


# ── Quick demo ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python cv/optical_flow.py path/to/video.mp4")
        sys.exit(1)

    cap = cv2.VideoCapture(sys.argv[1])
    ret, prev = cap.read()
    if not ret:
        print("❌ Could not read video")
        sys.exit(1)

    frame_count = 0
    while frame_count < 5:
        ret, curr = cap.read()
        if not ret:
            break
        features = extract_flow_features(prev, curr)
        print(f"Frame {frame_count}: {features}")
        prev = curr
        frame_count += 1

    cap.release()
