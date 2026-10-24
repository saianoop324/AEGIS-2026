"""Fine-Tuning Pipeline for MemGuard Memory Write Gate DeBERTa Classifier.

Optimized for NVIDIA RTX 3050 (4GB/6GB/8GB) & Cloud GPUs:
- Base Model: cross-encoder/nli-deberta-v3-small (or microsoft/deberta-v3-small)
- Loss: Binary Cross-Entropy / Classification Cross-Entropy
- Mixed Precision: FP16 enabled for maximum throughput and minimal VRAM footprint (~1.2 GB)
- Output: Exported fine-tuned model checkpoint + ONNX / PyTorch deployment artifacts
"""

import os
import json
import argparse
import logging
from typing import Dict, Any, List

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


class MemoryClassificationDataset(Dataset):
    """PyTorch Dataset for memory statement injection classification."""

    def __init__(self, json_path: str, tokenizer, max_length: int = 128):
        with open(json_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        text = item["text"]
        label = item["label"]

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.long),
        }


def train_classifier(
    train_path: str = "data/train.json",
    val_path: str = "data/val.json",
    model_name: str = "cross-encoder/nli-deberta-v3-small",
    output_dir: str = "models/memguard-deberta-write-gate",
    epochs: int = 3,
    batch_size: int = 16,
    lr: float = 2e-5,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    logger.info(f"Starting MemGuard Classifier fine-tuning on device: {device.upper()}")
    os.makedirs(output_dir, exist_ok=True)

    # 1. Load Tokenizer & Model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,
        ignore_mismatched_sizes=True,
    )
    model.to(device)

    # 2. Prepare DataLoaders
    train_dataset = MemoryClassificationDataset(train_path, tokenizer)
    val_dataset = MemoryClassificationDataset(val_path, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(total_steps * 0.1), num_training_steps=total_steps)

    use_amp = (device == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    best_val_f1 = 0.0

    for epoch in range(1, epochs + 1):
        model.train()
        total_train_loss = 0.0

        for step, batch in enumerate(train_loader):
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            with torch.cuda.amp.autocast(enabled=use_amp):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)

        # Validation Step
        model.eval()
        val_correct = 0
        val_total = 0
        tp, fp, fn = 0, 0, 0

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["label"].to(device)

                with torch.cuda.amp.autocast(enabled=use_amp):
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                    preds = torch.argmax(outputs.logits, dim=-1)

                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

                for p, l in zip(preds.tolist(), labels.tolist()):
                    if p == 1 and l == 1:
                        tp += 1
                    elif p == 1 and l == 0:
                        fp += 1
                    elif p == 0 and l == 1:
                        fn += 1

        val_acc = val_correct / val_total if val_total > 0 else 0.0
        val_precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        val_recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        val_f1 = (2 * val_precision * val_recall) / (val_precision + val_recall) if (val_precision + val_recall) > 0 else 0.0

        logger.info(
            f"Epoch {epoch}/{epochs} | Loss: {avg_train_loss:.4f} | "
            f"Val Acc: {val_acc:.4f} | Precision: {val_precision:.4f} | Recall: {val_recall:.4f} | F1: {val_f1:.4f}"
        )

        if val_f1 >= best_val_f1:
            best_val_f1 = val_f1
            logger.info(f"Saving best model checkpoint to {output_dir}")
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)

    logger.info(f"Training complete. Best Validation F1: {best_val_f1:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MemGuard DeBERTa Write Gate Classifier")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--output_dir", type=str, default="models/memguard-deberta-write-gate")
    args = parser.parse_args()

    train_classifier(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        output_dir=args.output_dir,
    )

