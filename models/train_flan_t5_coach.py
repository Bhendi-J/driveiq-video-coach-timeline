"""Train Flan-T5 for coaching text generation.

Expected data files:
    data/flan_t5/train.jsonl
    data/flan_t5/val.jsonl
    data/flan_t5/test.jsonl

Each JSONL row:
    {"input_text": "...", "target_text": "..."}

Run (safe default):
    python models/train_flan_t5_coach.py

Example (larger run):
    python models/train_flan_t5_coach.py --epochs 3 --batch-size 8 --lr 2e-5
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from datasets import load_dataset
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)


ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "flan_t5"
OUT_DIR = ROOT / "models" / "flan_t5_coach"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train Flan-T5 on coaching dataset")
    p.add_argument("--model", default="google/flan-t5-small", help="HF model name")
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--eval-batch-size", type=int, default=2)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--max-input-len", type=int, default=192)
    p.add_argument("--max-target-len", type=int, default=48)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--warmup-ratio", type=float, default=0.05)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--fp16", action="store_true", help="Enable fp16 training when supported")
    p.add_argument("--bf16", action="store_true", help="Enable bf16 training when supported")
    p.add_argument("--use-cpu", action="store_true", help="Force CPU training to avoid MPS OOM")
    p.add_argument(
        "--gradient-checkpointing",
        action="store_true",
        help="Enable gradient checkpointing to reduce memory usage",
    )
    p.add_argument(
        "--dataloader-pin-memory",
        action="store_true",
        help="Enable pinned memory in dataloaders (usually keep OFF on MPS)",
    )
    p.add_argument(
        "--reset-output",
        action="store_true",
        help="Delete output directory before training (replacement for old overwrite_output_dir).",
    )
    return p.parse_args()


def _check_data_files() -> tuple[str, str, str]:
    train = DATA_DIR / "train.jsonl"
    val = DATA_DIR / "val.jsonl"
    test = DATA_DIR / "test.jsonl"
    missing = [str(p) for p in [train, val, test] if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing Flan-T5 split files. Run pipeline/prepare_flan_t5_dataset.py first. "
            f"Missing: {missing}"
        )
    return str(train), str(val), str(test)


def main() -> None:
    args = parse_args()
    train_path, val_path, test_path = _check_data_files()

    print("[flan-t5] Loading dataset splits ...")
    ds = load_dataset(
        "json",
        data_files={
            "train": train_path,
            "validation": val_path,
            "test": test_path,
        },
    )

    required_cols = {"input_text", "target_text"}
    for split in ["train", "validation", "test"]:
        cols = set(ds[split].column_names)
        missing = required_cols - cols
        if missing:
            raise ValueError(f"Split '{split}' missing required columns: {sorted(missing)}")

    print(f"[flan-t5] Model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model)

    def preprocess(batch: dict) -> dict:
        model_inputs = tokenizer(
            batch["input_text"],
            max_length=args.max_input_len,
            truncation=True,
        )
        labels = tokenizer(
            text_target=batch["target_text"],
            max_length=args.max_target_len,
            truncation=True,
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    print("[flan-t5] Tokenizing ...")
    tokenized = ds.map(
        preprocess,
        batched=True,
        num_proc=args.num_workers,
        remove_columns=ds["train"].column_names,
        desc="Tokenizing",
    )

    if args.reset_output and OUT_DIR.exists():
        print(f"[flan-t5] Removing existing output dir: {OUT_DIR}")
        shutil.rmtree(OUT_DIR)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(OUT_DIR),
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=100,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.grad_accum,
        warmup_steps=0,
        predict_with_generate=True,
        generation_max_length=args.max_target_len,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=2,
        seed=args.seed,
        fp16=args.fp16,
        bf16=args.bf16,
        use_cpu=args.use_cpu,
        gradient_checkpointing=args.gradient_checkpointing,
        dataloader_pin_memory=args.dataloader_pin_memory,
        report_to="none",
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        processing_class=tokenizer,
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
    )

    print("[flan-t5] Training ...")
    train_result = trainer.train()
    print("[flan-t5] Training complete")

    print("[flan-t5] Evaluating on validation split ...")
    val_metrics = trainer.evaluate(tokenized["validation"])
    print(f"[flan-t5] validation metrics: {val_metrics}")

    print("[flan-t5] Evaluating on test split ...")
    test_metrics = trainer.evaluate(tokenized["test"])
    print(f"[flan-t5] test metrics: {test_metrics}")

    print(f"[flan-t5] Saving model + tokenizer -> {OUT_DIR}")
    trainer.save_model(str(OUT_DIR))
    tokenizer.save_pretrained(str(OUT_DIR))

    # Save train summary in trainer state for quick review.
    print(f"[flan-t5] train metrics: {train_result.metrics}")
    print("[flan-t5] Done.")


if __name__ == "__main__":
    main()
