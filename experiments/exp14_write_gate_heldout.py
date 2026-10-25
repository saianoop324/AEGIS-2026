"""Experiment 14: write-gate precision and recall on a public labelled set none of our gates was tuned on.

Uses deepset/prompt-injections (Hugging Face, train + test = 662 texts, label 1 = injection), so the
texts are not ours. Every text is written through the write gate as if it came from an untrusted
channel (web_retrieval), each into its own empty tenant so items cannot affect each other.
quarantined = predicted injection. Run once per classifier, e.g.

  WRITE_GATE_CLASSIFIER=deberta DEVICE=cuda python experiments/exp14_write_gate_heldout.py --name deberta
  WRITE_GATE_CLASSIFIER=llm WRITE_GATE_LLM_API_BASE=http://localhost:8012/v1 WRITE_GATE_LLM_MODEL=qwen3.5-4b \
      NVIDIA_NIM_API_KEY=local python experiments/exp14_write_gate_heldout.py --name qwen3.5-4b
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import ChannelType, settings  # noqa: E402
from firewall.write_gate import MemGuardWriteGate  # noqa: E402
from store.vector_store import QdrantVectorStoreWrapper  # noqa: E402


def pctl(values, q):
    v = sorted(values)
    return round(v[max(0, int(round(q * len(v))) - 1)], 1) if v else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, help="label for this classifier in the output file")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    from datasets import load_dataset
    data = load_dataset("deepset/prompt-injections")
    items = [{"split": split, "text": r["text"], "label": int(r["label"])} for split in ("train", "test") for r in data[split]]

    vs = QdrantVectorStoreWrapper()
    wg = MemGuardWriteGate(vector_store=vs)
    rows = []
    for i, it in enumerate(items):
        t0 = time.perf_counter()
        d = wg.evaluate_and_write(tenant_id=f"heldout_{a.name}_{i}", content=it["text"],
                                  source_channel=ChannelType.WEB_RETRIEVAL)
        rows.append({**it, "quarantined": d.is_quarantined, "ms": round((time.perf_counter() - t0) * 1000, 1),
                     "source": getattr(wg.classifier, "last_source", None), "score": round(d.instruction_score, 3)})

    tp = sum(r["quarantined"] and r["label"] == 1 for r in rows)
    fp = sum(r["quarantined"] and r["label"] == 0 for r in rows)
    fn = sum(not r["quarantined"] and r["label"] == 1 for r in rows)
    pos, neg = sum(r["label"] == 1 for r in rows), sum(r["label"] == 0 for r in rows)
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / pos if pos else None
    summary = {
        "classifier": a.name, "write_gate_classifier": settings.WRITE_GATE_CLASSIFIER,
        "llm_model": settings.WRITE_GATE_LLM_MODEL, "n": len(rows), "injections": pos, "benign": neg,
        "tp": tp, "fp": fp, "fn": fn,
        "precision": round(precision, 3) if precision is not None else None,
        "recall": round(recall, 3) if recall is not None else None,
        "f1": round(2 * precision * recall / (precision + recall), 3) if precision and recall else None,
        "false_quarantine_rate": round(fp / neg, 3) if neg else None,
        "caught_by_fast_path": sum(r["quarantined"] and r["source"] == "fast_path" for r in rows),
        "write_p50_ms": pctl([r["ms"] for r in rows], 0.5), "write_p95_ms": pctl([r["ms"] for r in rows], 0.95),
    }
    out = a.out or os.path.join(os.path.dirname(os.path.abspath(__file__)), f"exp14_heldout_{a.name}.json")
    json.dump({"summary": summary, "rows": rows}, open(out, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
