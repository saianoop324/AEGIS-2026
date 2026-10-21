# MemGuard: AI Agent Memory Governance & Provenance Firewall Platform

MemGuard is a production-grade memory security firewall designed to protect autonomous LLM agents from memory poisoning, indirect prompt injections, authority spoofing, sleeper triggers, slow drift, and cross-tenant data leakage.

---

## 🏛️ System Architecture

```mermaid
flowchart TD
    subgraph Ingestion["Ingestion Channels"]
        C1["User Chat"]
        C2["Documents"]
        C3["Tool Outputs"]
        C4["Web Retrieval"]
        C5["Peer Agents"]
    end

    subgraph WriteGate["Write Gate (&lt;25ms local path)"]
        W1["DeBERTa Classifier (Instruction vs Fact)"]
        W2["Contradiction Screening"]
        W3["HMAC-SHA256 Provenance Envelope Signing"]
        W4{"Security Decision"}
        W5["Quarantine Zone (Hard Pre-Filter Flag)"]
    end

    subgraph VectorStore["Qdrant Vector Store"]
        V1[("Tenant Namespaces (Hard Pre-Filtering)")]
    end

    subgraph ReadGate["Read Gate &amp; Inference"]
        R1["Trust-Weighted Recall"]
        R2["Strict Quoted Untrusted Data Wrapping"]
        R3["NeMo Guardrails Prompt Hooks"]
        R4["NVIDIA NIM Inference"]
    end

    subgraph Lineage["Lineage &amp; Blast Radius"]
        L1["NetworkX Directed Acyclic Graph"]
        L2["Root Memory ➔ Derived Memory ➔ Action"]
        L3["Recursive Cascade Rollback"]
    end

    Ingestion --> W1
    W1 --> W2 --> W3 --> W4
    W4 -- "Hostile/Contradictory" --> W5
    W4 -- "Vetted" --> V1
    W5 --> V1

    V1 --> R1 --> R2 --> R3 --> R4
    W4 -.-> L1
    R4 -.-> L1
    L1 -.-> L3
```

---

## 🚀 Key Modules & Directory Structure

- **`config.py`**: Centralized configuration management using Pydantic Settings (Qdrant endpoints, DeBERTa model, NVIDIA NIM keys, and SLA thresholds).
- **`store/vector_store.py`**: High-performance Qdrant vector store wrapper featuring per-tenant namespace isolation, dense vector storage, cryptographic provenance payloads, and hard metadata pre-filtering.
- **`firewall/write_gate.py`**: Sub-25ms fast-path write gate. DeBERTa instruction-vs-fact classifier, neighbor contradiction detection, and HMAC-SHA256 cryptographic provenance signature generation.
- **`firewall/read_gate.py`**: Trust-weighted recall gate. Automatically wraps untrusted or provisional memories into strict `<UNTRUSTED_RECALLED_DATA>` quarantine envelopes.
- **`firewall/lineage.py`**: NetworkX-backed DAG tracking complete memory provenance (`Root Memory -> Derived Memory -> Agent Action`) with BFS blast-radius calculation and cascading rollback.
- **`memgauntlet/harness.py`**: Automated red-team test harness covering 5 channels and 6 attack vectors (False Fact, Embedded Instruction, Authority Spoofing, Sleeper Trigger, Slow Drift, Cross-Tenant Probe).
- **`evaluator/metrics.py`**: Security and utility scoring computing ASR (Attack Success Rate), Persistence Rate, Precision/Recall/F1, and Latency SLAs.
- **`agent/memguard_agent.py`**: Stateful `LangGraph` agent orchestrating memory reads, NVIDIA NIM inference, NeMo Guardrails wrappers, candidate memory evaluation, and external action logging.
- **`app.py`**: Streamlit observability dashboard featuring executive scorecards, live red-team runner, quarantine inspectors, and lineage DAG graphs.
- **`server.py`**: FastAPI REST API exposing `/api/v1/write`, `/api/v1/query`, `/api/v1/lineage`, and `/api/v1/rollback`.

---

## 🛠️ Getting Started

### 1. Installation
```bash
pip install -r requirements.txt
```

### 2. Run Streamlit Observability Dashboard
```bash
streamlit run app.py
```

### 3. Run FastAPI Security Microservice
```bash
uvicorn server:app --host 0.0.0.0 --port 8000
```

