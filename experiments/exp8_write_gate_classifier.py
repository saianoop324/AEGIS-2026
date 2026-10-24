"""Experiment 8: write-gate classifier, DeBERTa vs an LLM (Qwen3.5-9B), on the existing gauntlet.

Runs MemGauntlet's 30 attacks and 10 benign controls through the write gate with the classifier
chosen by WRITE_GATE_CLASSIFIER ("deberta" or "llm"), and records for every payload whether it was
quarantined, why (fast-path regex, semantic instruction check, or contradiction) and how long the
write took. Run once per classifier from a scratch directory:

  WRITE_GATE_CLASSIFIER=deberta DEVICE=cuda python experiments/exp8_write_gate_classifier.py
  WRITE_GATE_CLASSIFIER=llm NVIDIA_NIM_API_BASE=http://localhost:8011/v1 NVIDIA_NIM_MODEL=qwen3.5-9b \
      NVIDIA_NIM_API_KEY=local DEVICE=cuda python experiments/exp8_write_gate_classifier.py
"""

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import settings  # noqa: E402
from store.vector_store import QdrantVectorStoreWrapper  # noqa: E402
from firewall.write_gate import MemGuardWriteGate  # noqa: E402
from firewall.read_gate import MemGuardReadGate  # noqa: E402
from firewall.lineage import LineageTracker  # noqa: E402
from memgauntlet.harness import MemGauntletHarness  # noqa: E402


def main() -> None:
    kind = settings.WRITE_GATE_CLASSIFIER.lower()
    vs = QdrantVectorStoreWrapper()
    wg = MemGuardWriteGate(vector_store=vs)
    rg = MemGuardReadGate(vector_store=vs, embedding_function=wg._embedding_func)
    harness = MemGauntletHarness(wg, rg, LineageTracker(vector_store=vs, db_path=f"lineage_{kind}.json"))

    # Record which path produced each instruction score.
    source_by_text = {}
    original = wg.classifier.classify_instruction_vs_fact

    def traced(text, channel=None):
        score = original(text, channel=channel)
        source_by_text[text] = (wg.classifier.last_source, round(score, 3))
        return score

    wg.classifier.classify_instruction_vs_fact = traced

    t0 = time.perf_counter()
    attacks = harness.execute_gauntlet_run(tenant_id="exp8_attacks")
    benign = harness.execute_benign_controls(tenant_id="exp8_benign")
    wall = time.perf_counter() - t0

    def why(r):
        if not r.quarantined:
            return "admitted"
        reason = r.quarantine_reason or ""
        if "contradiction" in reason.lower():
            return "contradiction"
        return "fast_path" if source_by_text.get(r.content, ("",))[0] == "fast_path" else f"{kind}_instruction"

    rows = [{"kind": "attack" if r.attack_vector != "benign_control" else "benign",
             "vector": r.attack_vector, "channel": r.channel, "content": r.content,
             "quarantined": r.quarantined, "why": why(r),
             "instruction_score": source_by_text.get(r.content, (None, None))[1],
             "latency_ms": round(r.latency_ms, 1)} for r in attacks + benign]
    lat = sorted(x["latency_ms"] for x in rows)
    summary = {
        "classifier": kind, "model": settings.WRITE_GATE_LLM_MODEL or settings.NVIDIA_NIM_MODEL if kind == "llm"
        else settings.DEBERTA_NLI_MODEL_NAME,
        "attacks_caught": sum(r["quarantined"] for r in rows if r["kind"] == "attack"),
        "attacks_total": sum(r["kind"] == "attack" for r in rows),
        "false_positives": sum(r["quarantined"] for r in rows if r["kind"] == "benign"),
        "benign_total": sum(r["kind"] == "benign" for r in rows),
        "caught_by": {k: sum(1 for r in rows if r["kind"] == "attack" and r["why"] == k)
                      for k in ("fast_path", f"{kind}_instruction", "contradiction")},
        "by_vector": {},
        "write_latency_ms": {"p50": lat[len(lat) // 2], "p95": lat[int(len(lat) * 0.95) - 1], "max": lat[-1]},
        "wall_s": round(wall, 1),
    }
    for r in rows:
        if r["kind"] == "attack":
            v = summary["by_vector"].setdefault(r["vector"], [0, 0])
            v[0] += int(r["quarantined"])
            v[1] += 1
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"exp8_write_gate_{kind}_results.json")
    json.dump({"summary": summary, "rows": rows}, open(out, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2))
    print("saved", out)


if __name__ == "__main__":
    main()
