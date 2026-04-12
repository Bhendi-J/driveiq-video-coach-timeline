"""Prepare Flan-T5 fine-tuning splits from coaching_dataset.csv.

Input:
    data/coaching_dataset.csv

Outputs:
    data/flan_t5/train.jsonl
    data/flan_t5/val.jsonl
    data/flan_t5/test.jsonl

Each JSONL row contains:
    {"input_text": "...", "target_text": "..."}

Run:
    python pipeline/prepare_flan_t5_dataset.py
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parent.parent
INPUT_CSV = ROOT / "data" / "coaching_dataset.csv"
OUT_DIR = ROOT / "data" / "flan_t5"

REQUIRED_COLUMNS = [
    "rpm_variation",
    "harsh_braking_count",
    "idling_time",
    "fuel_consumption",
    "acceleration_smoothness",
    "predicted_score",
    "top_issue",
    "top_2_issue",
    "severity",
    "coaching_text",
]


def _build_input_text(row: pd.Series) -> str:
    return (
        "Task: Generate one concise and actionable eco-driving coaching sentence.\n"
        f"predicted_score: {float(row['predicted_score']):.2f}\n"
        f"top_issue: {row['top_issue']}\n"
        f"top_2_issue: {row['top_2_issue']}\n"
        f"severity: {row['severity']}\n"
        f"rpm_variation: {float(row['rpm_variation']):.4f}\n"
        f"harsh_braking_count: {float(row['harsh_braking_count']):.4f}\n"
        f"idling_time: {float(row['idling_time']):.4f}\n"
        f"fuel_consumption: {float(row['fuel_consumption']):.4f}\n"
        f"acceleration_smoothness: {float(row['acceleration_smoothness']):.4f}\n"
        "Response:"
    )


def _to_records(df: pd.DataFrame) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    for _, row in df.iterrows():
        records.append(
            {
                "input_text": _build_input_text(row),
                "target_text": str(row["coaching_text"]).strip(),
            }
        )
    return records


def _write_jsonl(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> None:
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Missing input dataset: {INPUT_CSV}")

    df = pd.read_csv(INPUT_CSV)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Input dataset missing required columns: {missing}. "
            f"Available: {df.columns.tolist()}"
        )

    df = df[REQUIRED_COLUMNS].dropna().copy()
    if df.empty:
        raise ValueError("Input dataset became empty after dropna")

    # 80/10/10 with severity-stratified splits.
    train_df, tmp_df = train_test_split(
        df,
        test_size=0.20,
        random_state=42,
        stratify=df["severity"],
    )

    val_df, test_df = train_test_split(
        tmp_df,
        test_size=0.50,
        random_state=42,
        stratify=tmp_df["severity"],
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    train_path = OUT_DIR / "train.jsonl"
    val_path = OUT_DIR / "val.jsonl"
    test_path = OUT_DIR / "test.jsonl"

    train_records = _to_records(train_df)
    val_records = _to_records(val_df)
    test_records = _to_records(test_df)

    _write_jsonl(train_path, train_records)
    _write_jsonl(val_path, val_records)
    _write_jsonl(test_path, test_records)

    print("Prepared Flan-T5 dataset splits")
    print(f"train: {len(train_records)} -> {train_path}")
    print(f"val:   {len(val_records)} -> {val_path}")
    print(f"test:  {len(test_records)} -> {test_path}")
    print("severity distribution (train):")
    print(train_df["severity"].value_counts(dropna=False))
    print("severity distribution (val):")
    print(val_df["severity"].value_counts(dropna=False))
    print("severity distribution (test):")
    print(test_df["severity"].value_counts(dropna=False))
    print("sample record:")
    print(json.dumps(train_records[0], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
