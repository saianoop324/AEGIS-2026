"""Experiment 9: write-gate classifier, DeBERTa vs an LLM (Qwen3.5-9B), on the existing gauntlet.

Runs MemGauntlet's attacks and benign controls through the write gate with the classifier chosen by
WRITE_GATE_CLASSIFIER ("deberta" or "llm"). For every payload it records each check that ran:
the regex fast path, the semantic instruction check (with the LLM's raw verdict), the on-topic
neighbours the contradiction check compared against (with similarity), the contradiction verdict,
the final decision and reasons, and the write latency. Run once per classifier from a scratch dir:

  WRITE_GATE_CLASSIFIER=deberta DEVICE=cuda python experiments/exp9_write_gate_classifier.py
  WRITE_GATE_CLASSIFIER=llm NVIDIA_NIM_API_BASE=http://localhost:8011/v1 NVIDIA_NIM_MODEL=qwen3.5-9b \
      NVIDIA_NIM_API_KEY=local DEVICE=cuda python experiments/exp9_write_gate_classifier.py

Then rebuild the detail page: python experiments/build_exp9_details.py
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
    clf = wg.classifier

    # ---- tracing: wrap the classifier and the neighbour search without changing behaviour ----
    trace = {}          # payload text -> detail dict
    last_search = []    # most recent neighbour search (the write gate's contradiction pre-step)
    last_raw = {}       # most recent raw LLM verdict

    if hasattr(clf, "_ask"):
        orig_ask = clf._ask

        def ask(system, user, schema):
            out = orig_ask(system, user, schema)
            last_raw.clear()
            last_raw.update(out)
            return out
        clf._ask = ask

    orig_classify = clf.classify_instruction_vs_fact

    def classify(text, channel=None):
        last_raw.clear()
        score = orig_classify(text, channel=channel)
        trace.setdefault(text, {})["instruction"] = {
            "source": clf.last_source, "score": round(score, 3),
            "threshold": settings.INSTRUCTION_DETECTION_THRESHOLD,
            "raw": dict(last_raw) if clf.last_source == "llm" else None,
        }
        return score
    clf.classify_instruction_vs_fact = classify

    orig_search = vs.search_memories

    def search(*args, **kwargs):
        res = orig_search(*args, **kwargs)
        last_search.clear()
        last_search.extend(res)
        return res
    vs.search_memories = search

    orig_contra = clf.check_contradiction

    def contradiction(candidate, existing):
        last_raw.clear()
        score, conflict = orig_contra(candidate, existing)
        sims = {h["payload"].get("content", ""): round(float(h.get("score", 0.0)), 3) for h in last_search}
        trace.setdefault(candidate, {})["contradiction"] = {
            "compared_with": [{"text": t, "similarity": sims.get(t)} for t in existing],
            "score": round(float(score), 3), "threshold": settings.CONTRADICTION_THRESHOLD,
            "conflicting_memory": conflict, "raw": dict(last_raw) if last_raw else None,
        }
        return score, conflict
    clf.check_contradiction = contradiction

    # ---- run ----
    t0 = time.perf_counter()
    attacks = harness.execute_gauntlet_run(tenant_id="exp9_attacks")
    benign = harness.execute_benign_controls(tenant_id="exp9_benign")
    wall = time.perf_counter() - t0
    payloads = harness.generate_attack_matrix()
    descriptions = {p.content: p.description for p in payloads}

    def why(r, detail):
        if not r.quarantined:
            return "admitted"
        if "contradiction" in (r.quarantine_reason or "").lower():
            return "contradiction"
        return "fast_path" if detail.get("instruction", {}).get("source") == "fast_path" else f"{kind}_instruction"

    rows = []
    for i, r in enumerate(attacks + benign):
        detail = trace.get(r.content, {})
        rows.append({
            "id": f"A{i + 1:02d}" if r.attack_vector != "benign_control" else f"B{i - len(attacks) + 1:02d}",
            "kind": "attack" if r.attack_vector != "benign_control" else "benign",
            "vector": r.attack_vector, "channel": r.channel, "content": r.content,
            "description": descriptions.get(r.content, "Legitimate memory; quarantining it is a false positive."),
            "quarantined": r.quarantined, "why": why(r, detail),
            "reasons": r.quarantine_reason, "instruction_score": detail.get("instruction", {}).get("score"),
            "checks": detail, "latency_ms": round(r.latency_ms, 1),
        })

    lat = sorted(x["latency_ms"] for x in rows)
    summary = {
        "classifier": kind,
        "model": (settings.WRITE_GATE_LLM_MODEL or settings.NVIDIA_NIM_MODEL) if kind == "llm"
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
        "settings": {"instruction_threshold": settings.INSTRUCTION_DETECTION_THRESHOLD,
                     "contradiction_threshold": settings.CONTRADICTION_THRESHOLD,
                     "contradiction_min_similarity": settings.CONTRADICTION_MIN_SIMILARITY,
                     "embedding_model": settings.EMBEDDING_MODEL_NAME, "device": settings.DEVICE},
        "baseline_facts": [
            "The corporate tax rate for the company's domestic filings is 21%, per standing fiscal policy.",
            "The current CEO of Apple Inc. is Tim Cook.",
            "Port 22 SSH access is permanently disabled for anonymous or guest connections; "
            "exceptions require written executive approval.",
        ],
    }
    for r in rows:
        if r["kind"] == "attack":
            v = summary["by_vector"].setdefault(r["vector"], [0, 0])
            v[0] += int(r["quarantined"])
            v[1] += 1
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"exp9_write_gate_{kind}_results.json")
    json.dump({"summary": summary, "rows": rows}, open(out, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(json.dumps({k: summary[k] for k in ("classifier", "attacks_caught", "false_positives", "caught_by",
                                              "write_latency_ms")}, indent=2))
    print("saved", out)


if __name__ == "__main__":
    main()
