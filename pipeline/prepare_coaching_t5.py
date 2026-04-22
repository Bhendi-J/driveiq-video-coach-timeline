"""
pipeline/prepare_coaching_t5.py

Converts the coaching_dataset.csv into a HuggingFace-ready JSONL dataset
for fine-tuning Flan-T5-Small as a driving coach.

Input format (CSV):
  rpm_variation, harsh_braking_count, idling_time, fuel_consumption,
  acceleration_smoothness, eco_score, predicted_score, top_issue,
  top_2_issue, severity, coaching_text

Output format (JSONL):
  {"input": "...", "target": "..."}

Usage:
  python -m pipeline.prepare_coaching_t5
"""

import csv
import json
import os
import random

INPUT_CSV = os.path.join(os.path.dirname(__file__), "..", "data", "coaching_dataset.csv")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "coaching_t5")
TRAIN_FILE = os.path.join(OUTPUT_DIR, "train.jsonl")
VAL_FILE = os.path.join(OUTPUT_DIR, "val.jsonl")

# Use 90/10 train/val split
VAL_RATIO = 0.10
SEED = 42



# Map old dataset issue names → new CV-based event names
# so the model learns terms matching the live system
ISSUE_REMAP = {
    "harsh_braking_count": "hard braking",
    "fuel_consumption": "high fuel usage",
    "acceleration_smoothness": "erratic acceleration",
    "rpm_variation": "unstable RPM",
    "idling_time": "excessive idling",
    "smooth_driving": "smooth driving",
    # CV event names (pass through if already new format)
    "hard_braking": "hard braking",
    "tailgating": "tailgating",
    "lane_swerving": "lane swerving",
    "pedestrian_risk": "pedestrian risk",
    "erratic_speed": "erratic speed",
}


def remap_issue(raw: str) -> str:
    """Normalize issue names to the new CV-based event vocabulary."""
    return ISSUE_REMAP.get(raw.strip(), raw.replace("_", " "))


def build_prompt(row: dict) -> str:
    """Build a feature-agnostic instruction prompt.
    
    Only uses score, severity, and issue types — fields that exist
    in both the old telemetry dataset and the new CV-based system.
    This ensures the trained model works at inference time when
    called with live CV features.
    """
    score = row.get("eco_score", "50")
    severity = row.get("severity", "yellow")
    top_issue = remap_issue(row.get("top_issue", "smooth_driving"))
    secondary = remap_issue(row.get("top_2_issue", ""))

    prompt = (
        f"You are an expert driving coach. "
        f"The driver scored {score} out of 100 (severity: {severity}). "
        f"Primary issue: {top_issue}."
    )
    if secondary and secondary != top_issue:
        prompt += f" Secondary issue: {secondary}."
    prompt += " Give one specific, actionable coaching tip."
    return prompt


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    rows = []
    with open(INPUT_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            coaching_text = row.get("coaching_text", "").strip()
            if not coaching_text:
                continue
            prompt = build_prompt(row)
            rows.append({"input": prompt, "target": coaching_text})

    random.seed(SEED)
    random.shuffle(rows)

    split_idx = int(len(rows) * (1.0 - VAL_RATIO))
    train_rows = rows[:split_idx]
    val_rows = rows[split_idx:]

    with open(TRAIN_FILE, "w", encoding="utf-8") as f:
        for item in train_rows:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    with open(VAL_FILE, "w", encoding="utf-8") as f:
        for item in val_rows:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"✅ Dataset prepared:")
    print(f"   Train: {len(train_rows)} examples → {TRAIN_FILE}")
    print(f"   Val:   {len(val_rows)} examples → {VAL_FILE}")


if __name__ == "__main__":
    main()
