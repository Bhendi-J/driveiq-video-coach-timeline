import numpy as np
import pandas as pd

np.random.seed(42)
N = 30000

def generate_features():
    mean_flow = np.clip(np.random.normal(6, 2.5), 0, 13)
    flow_variance = np.clip(np.random.gamma(2.5, 20), 0, 200)

    braking_ratio = np.clip(np.random.beta(2, 5), 0, 1)
    lane_change_ratio = np.clip(np.random.beta(1.5, 6), 0, 1)
    proximity_score = np.clip(np.random.beta(2, 3), 0, 1)

    vehicle_density = int(np.clip(np.random.poisson(3), 0, 10))
    pedestrian_ratio = np.clip(np.random.beta(1, 8), 0, 1)
    low_motion_ratio = np.clip(np.random.beta(2, 4), 0, 1)

    # --- 🔥 Add REALISTIC DEPENDENCIES ---
    if vehicle_density > 6:
        braking_ratio += np.random.uniform(0.1, 0.3)
        mean_flow -= np.random.uniform(1, 3)

    if proximity_score > 0.5:
        braking_ratio += np.random.uniform(0.1, 0.2)

    braking_ratio = np.clip(braking_ratio, 0, 1)
    mean_flow = np.clip(mean_flow, 0, 13)

    return {
        "mean_flow": mean_flow,
        "flow_variance": flow_variance,
        "braking_ratio": braking_ratio,
        "lane_change_ratio": lane_change_ratio,
        "proximity_score": proximity_score,
        "vehicle_density": vehicle_density,
        "pedestrian_ratio": pedestrian_ratio,
        "low_motion_ratio": low_motion_ratio
    }

def compute_eco_score(f):
    eco_score = 100

    # Braking penalty
    if f["braking_ratio"] > 0.6:
        eco_score -= 35
    elif f["braking_ratio"] > 0.3:
        eco_score -= 20
    elif f["braking_ratio"] > 0.1:
        eco_score -= 10

    # Smoothness
    if f["flow_variance"] > 120:
        eco_score -= 25
    elif f["flow_variance"] > 60:
        eco_score -= 15
    elif f["flow_variance"] > 30:
        eco_score -= 8

    # Idling
    eco_score -= 20 * f["low_motion_ratio"]

    # Proximity
    if f["proximity_score"] > 0.6:
        eco_score -= 25
    elif f["proximity_score"] > 0.4:
        eco_score -= 15
    elif f["proximity_score"] > 0.2:
        eco_score -= 5

    # Lane discipline
    eco_score -= 20 * f["lane_change_ratio"]

    # Traffic context
    eco_score -= 2 * f["vehicle_density"]
    eco_score -= 10 * f["pedestrian_ratio"]

    # Speed efficiency
    if 3 < f["mean_flow"] < 7:
        eco_score += 10
    elif f["mean_flow"] > 10:
        eco_score -= 10

    # --- Interactions ---
    if f["braking_ratio"] > 0.4 and f["proximity_score"] > 0.4:
        eco_score -= 20

    if f["flow_variance"] > 80 and f["vehicle_density"] > 5:
        eco_score -= 15

    if f["lane_change_ratio"] > 0.3 and f["vehicle_density"] > 6:
        eco_score -= 10

    # Noise
    eco_score += np.random.normal(0, 4)

    return np.clip(eco_score, 0, 100)

# Generate dataset
rows = []
for _ in range(N):
    f = generate_features()
    f["eco_score"] = compute_eco_score(f)
    rows.append(f)

df = pd.DataFrame(rows)
df.to_csv("eco_driving_cv_dataset_30k.csv", index=False)

print("Generated dataset:", df.shape)