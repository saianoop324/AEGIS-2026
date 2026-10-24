"""Evaluation script for fine-tuned MemGuard DeBERTa classifier on held-out test data."""

import os
import json
import argparse
import time
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


def evaluate_test_set(
    model_dir: str = "models/memguard-deberta-write-gate",
    test_path: str = "data/test.json",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    print(f"Evaluating fine-tuned model from '{model_dir}' on {device.upper()}...")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device)
    model.eval()

    with open(test_path, "r", encoding="utf-8") as f:
        test_data = json.load(f)

    tp, fp, tn, fn = 0, 0, 0, 0
    latencies = []

    for item in test_data:
        text = item["text"]
        label = item["label"]

        t0 = time.perf_counter()
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(device)
        with torch.no_grad():
            logits = model(**inputs).logits
            pred = torch.argmax(logits, dim=-1).item()
        lat_ms = (time.perf_counter() - t0) * 1000.0
        latencies.append(lat_ms)

        if pred == 1 and label == 1:
            tp += 1
        elif pred == 1 and label == 0:
            fp += 1
        elif pred == 0 and label == 0:
            tn += 1
        elif pred == 0 and label == 1:
            fn += 1

    total = len(test_data)
    acc = (tp + tn) / total
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

    latencies.sort()
    p50 = latencies[len(latencies) // 2]
    p95 = latencies[int(len(latencies) * 0.95)]

    print("=" * 60)
    print("MEMGUARD WRITE-GATE TEST EVALUATION REPORT")
    print("=" * 60)
    print(f"Total Test Samples: {total}")
    print(f"Accuracy:          {acc * 100:.2f}%")
    print(f"Precision:         {prec:.4f}")
    print(f"Recall:            {rec:.4f}")
    print(f"F1-Score:          {f1:.4f}")
    print(f"Latency P50:       {p50:.2f} ms")
    print(f"Latency P95:       {p95:.2f} ms  (Target: < 25 ms)")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", default="models/memguard-deberta-write-gate")
    parser.add_argument("--test_path", default="data/test.json")
    args = parser.parse_args()
    evaluate_test_set(args.model_dir, args.test_path)

