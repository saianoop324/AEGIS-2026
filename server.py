"""MemGuard FastAPI Service & REST Endpoints.

Provides:
  - POST /api/v1/write: Write Gate validation, cryptographic signing & quarantine.
  - POST /api/v1/query: Read Gate trust-weighted recall & NeMo sandbox wrapping.
  - GET  /api/v1/lineage/{tenant_id}: Lineage DAG export.
  - POST /api/v1/rollback: Recursive blast-radius invalidation.
  - POST /api/v1/redteam/run: MemGauntlet benchmark trigger.
"""

import uuid
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field

from config import settings, ChannelType, TrustTier
from store.vector_store import QdrantVectorStoreWrapper
from firewall.write_gate import MemGuardWriteGate, WriteGateDecision
from firewall.read_gate import MemGuardReadGate, ReadGateResponse
from firewall.lineage import LineageTracker
from memgauntlet.harness import MemGauntletHarness
from evaluator.metrics import EvaluatorEngine

app = FastAPI(
    title=settings.APP_NAME,
    description="AI Agent Memory Governance & Provenance Firewall Platform",
    version="1.0.0",
)

# Initialize singletons
vector_store = QdrantVectorStoreWrapper()
lineage_tracker = LineageTracker(vector_store=vector_store)
write_gate = MemGuardWriteGate(vector_store=vector_store)
read_gate = MemGuardReadGate(vector_store=vector_store)
harness = MemGauntletHarness(write_gate, read_gate, lineage_tracker)


class WriteRequest(BaseModel):
    tenant_id: str
    content: str
    channel: ChannelType = ChannelType.USER_CHAT
    derived_from: Optional[List[str]] = None


class ReadRequest(BaseModel):
    tenant_id: str
    query: str
    top_k: int = 5
    strict_sandboxing: bool = True


class RollbackRequest(BaseModel):
    memory_id: str


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "app": settings.APP_NAME,
        "env": settings.APP_ENV,
        "qdrant_location": settings.QDRANT_LOCATION,
    }


@app.post("/api/v1/write")
def write_memory(req: WriteRequest):
    """Screen candidate memory through DeBERTa Write Gate and sign provenance envelope."""
    decision: WriteGateDecision = write_gate.evaluate_and_write(
        tenant_id=req.tenant_id,
        content=req.content,
        source_channel=req.channel,
        derived_from=req.derived_from,
    )
    # Log into lineage DAG
    mem_id = decision.envelope.memory_id if decision.envelope else "unknown"
    lineage_tracker.record_root_memory(
        memory_id=mem_id,
        tenant_id=req.tenant_id,
        content=req.content,
        channel=req.channel.value,
        trust_tier=decision.trust_tier,
        is_quarantined=decision.is_quarantined,
    )
    return decision.model_dump()


@app.post("/api/v1/query")
def query_memory(req: ReadRequest):
    """Retrieve memories through trust-weighted Read Gate with strict quotation wrapping."""
    response: ReadGateResponse = read_gate.query_memories(
        tenant_id=req.tenant_id,
        query=req.query,
        top_k=req.top_k,
        strict_sandboxing=req.strict_sandboxing,
    )
    return response.model_dump()


@app.get("/api/v1/lineage/{tenant_id}")
def get_lineage(tenant_id: str):
    """Retrieve DAG nodes and edges for blast-radius visualization."""
    return lineage_tracker.get_subgraph_data_for_tenant(tenant_id)


@app.post("/api/v1/rollback")
def rollback_memory_blast_radius(req: RollbackRequest):
    """Cascade rollback poisoned memory and downstream derived entities."""
    result = lineage_tracker.rollback_memory(req.memory_id)
    if not result.get("success"):
        raise HTTPException(status_code=404, detail=result.get("message"))
    return result


@app.post("/api/v1/redteam/run")
def run_memgauntlet(tenant_id: str = "tenant_alpha"):
    """Run MemGauntlet's attacks through MemGuard, its benign controls, and the same attacks with MemGuard off.

    The scorecard is computed from these three passes only (see evaluator/metrics.py). Each call writes into a
    fresh tenant (<tenant_id>-redteam-<id>, returned as run_tenant_id): re-adding the same texts to one store makes
    earlier copies compete for the recalled slots and skews a repeated run.
    """
    run_tenant = f"{tenant_id}-redteam-{uuid.uuid4().hex[:8]}"
    results = harness.execute_gauntlet_run(tenant_id=run_tenant)
    benign = harness.execute_benign_controls(tenant_id=run_tenant)
    baseline = harness.execute_unprotected_baseline(tenant_id=run_tenant)
    scorecard = EvaluatorEngine.evaluate_results(results, benign_results=benign, baseline_results=baseline)
    return {
        "run_tenant_id": run_tenant,
        "scorecard": scorecard.model_dump(),
        "total_results": len(results),
        "results": [r.model_dump() for r in results],
        "benign_results": [r.model_dump() for r in benign],
        "baseline_results": [r.model_dump() for r in baseline],
    }

