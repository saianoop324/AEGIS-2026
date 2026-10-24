"""Universal Memory Store Adapter Interface for MemGuard.

Provides plug-and-play adapters for external agent memory frameworks:
1. Mem0 (mem0ai) adapter
2. LangChain VectorStore adapter
3. ChromaDB local/remote adapter
4. Native Qdrant adapter

Allows MemGuard's Write Gate, Read Gate, and Lineage DAG to govern ANY
external agent memory store without altering the agent's application code.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
import time
import uuid

from config import settings, ChannelType, TrustTier
from store.vector_store import QdrantVectorStoreWrapper, ProvenanceEnvelope
from firewall.write_gate import MemGuardWriteGate, WriteGateDecision
from firewall.read_gate import MemGuardReadGate, ReadGateResponse
from firewall.lineage import LineageTracker


class BaseMemoryStoreAdapter(ABC):
    """Abstract interface that all memory store adapters implement."""

    @abstractmethod
    def add(
        self,
        messages_or_text: Any,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        run_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Intercept write request, evaluate via Write Gate, and commit with provenance."""
        pass

    @abstractmethod
    def search(
        self,
        query: str,
        user_id: Optional[str] = None,
        limit: int = 5,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Intercept recall query, evaluate via Read Gate, and wrap untrusted content."""
        pass

    @abstractmethod
    def delete(self, memory_id: str) -> bool:
        """Purge memory and trigger lineage cascade rollback."""
        pass


class Mem0StoreAdapter(BaseMemoryStoreAdapter):
    """Drop-in adapter matching the official Mem0 (mem0.ai) Memory API.
    
    Usage:
        memory = Mem0StoreAdapter(write_gate, read_gate, lineage_tracker)
        memory.add("Customer mentioned their new address is 742 Evergreen Terr", user_id="cust_042")
        results = memory.search("Where does the customer live?", user_id="cust_042")
    """

    def __init__(
        self,
        write_gate: MemGuardWriteGate,
        read_gate: MemGuardReadGate,
        lineage_tracker: LineageTracker,
        default_tenant: str = "tenant_alpha",
    ):
        self.write_gate = write_gate
        self.read_gate = read_gate
        self.lineage_tracker = lineage_tracker
        self.default_tenant = default_tenant

    def add(
        self,
        messages_or_text: Any,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        run_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        tenant_id = user_id or self.default_tenant
        meta = metadata or {}
        channel_str = meta.get("channel", ChannelType.USER_CHAT.value)
        try:
            channel = ChannelType(channel_str)
        except ValueError:
            channel = ChannelType.USER_CHAT

        text_to_save = messages_or_text
        if isinstance(messages_or_text, list):
            # Mem0 format: list of dicts [{"role": "user", "content": "..."}]
            text_to_save = " ".join([m.get("content", "") for m in messages_or_text if isinstance(m, dict)])
        elif isinstance(messages_or_text, dict):
            text_to_save = messages_or_text.get("content", str(messages_or_text))

        decision: WriteGateDecision = self.write_gate.evaluate_and_write(
            tenant_id=tenant_id,
            content=str(text_to_save),
            source_channel=channel,
            extra_attributes=meta,
        )

        mem_id = decision.envelope.memory_id if decision.envelope else str(uuid.uuid4())
        self.lineage_tracker.record_root_memory(
            memory_id=mem_id,
            tenant_id=tenant_id,
            content=str(text_to_save),
            channel=channel.value,
            trust_tier=decision.trust_tier,
            is_quarantined=decision.is_quarantined,
        )

        return {
            "id": mem_id,
            "status": "quarantined" if decision.is_quarantined else "admitted",
            "trust_tier": decision.trust_tier,
            "allowed": decision.allowed,
            "quarantine_reasons": decision.reasons,
            "sha256_signature": decision.envelope.sha256_signature if decision.envelope else None,
        }

    def search(
        self,
        query: str,
        user_id: Optional[str] = None,
        limit: int = 5,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        tenant_id = user_id or self.default_tenant
        resp: ReadGateResponse = self.read_gate.query_memories(
            tenant_id=tenant_id,
            query=query,
            top_k=limit,
            strict_sandboxing=True,
        )
        return [
            {
                "id": it.memory_id,
                "memory": it.original_content,
                "sandboxed_content": it.formatted_content,
                "trust_tier": it.trust_tier,
                "score": it.relevance_score,
                "is_sandboxed": it.is_sandboxed,
            }
            for it in resp.items
        ]

    def delete(self, memory_id: str) -> bool:
        res = self.lineage_tracker.rollback_memory(memory_id)
        return res.get("success", False)


class ChromaStoreAdapter(BaseMemoryStoreAdapter):
    """Adapter for ChromaDB integration with MemGuard provenance enforcement."""

    def __init__(
        self,
        write_gate: MemGuardWriteGate,
        read_gate: MemGuardReadGate,
        lineage_tracker: LineageTracker,
        chroma_client: Optional[Any] = None,
    ):
        self.write_gate = write_gate
        self.read_gate = read_gate
        self.lineage_tracker = lineage_tracker
        self.chroma_client = chroma_client

    def add(
        self,
        messages_or_text: Any,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        run_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        tenant = user_id or "default_tenant"
        text = str(messages_or_text)
        decision = self.write_gate.evaluate_and_write(
            tenant_id=tenant,
            content=text,
            source_channel=ChannelType.DOCUMENTS,
            extra_attributes=metadata or {},
        )
        mem_id = decision.envelope.memory_id if decision.envelope else str(uuid.uuid4())
        self.lineage_tracker.record_root_memory(
            memory_id=mem_id,
            tenant_id=tenant,
            content=text,
            channel=ChannelType.DOCUMENTS.value,
            trust_tier=decision.trust_tier,
            is_quarantined=decision.is_quarantined,
        )
        return {"id": mem_id, "quarantined": decision.is_quarantined}

    def search(
        self,
        query: str,
        user_id: Optional[str] = None,
        limit: int = 5,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        tenant = user_id or "default_tenant"
        resp = self.read_gate.query_memories(tenant_id=tenant, query=query, top_k=limit)
        return [{"id": it.memory_id, "content": it.formatted_content} for it in resp.items]

    def delete(self, memory_id: str) -> bool:
        return self.lineage_tracker.rollback_memory(memory_id).get("success", False)

