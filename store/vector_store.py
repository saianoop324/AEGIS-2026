"""Qdrant Vector Store wrapper with per-tenant namespaces and hard metadata pre-filtering."""

import logging
from typing import Any, Dict, List, Optional
import numpy as np
from pydantic import BaseModel, Field
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
    Range,
)

from config import settings

logger = logging.getLogger(__name__)


class ProvenanceEnvelope(BaseModel):
    """Cryptographically signed metadata envelope for persisted memory points."""
    memory_id: str
    tenant_id: str
    content: str
    sha256_signature: str
    trust_tier: str
    source_channel: str
    timestamp: float
    is_quarantined: bool = False
    quarantine_reason: Optional[str] = None
    derived_from: List[str] = Field(default_factory=list)
    access_roles: List[str] = Field(default_factory=lambda: ["user", "agent"])
    extra_attributes: Dict[str, Any] = Field(default_factory=dict)


class QdrantVectorStoreWrapper:
    """Production Qdrant interface enforcing per-tenant partitioning & hard pre-filtering."""

    def __init__(
        self,
        collection_name: str = settings.DEFAULT_COLLECTION_NAME,
        dim: int = settings.EMBEDDING_DIM,
        location: Optional[str] = None,
        url: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        self.collection_name = collection_name
        self.dim = dim
        self._target_location = location or settings.QDRANT_LOCATION

        if url or settings.QDRANT_URL:
            self.client = QdrantClient(
                url=url or settings.QDRANT_URL,
                api_key=api_key or settings.QDRANT_API_KEY,
            )
        elif self._target_location == ":memory:":
            self.client = QdrantClient(location=":memory:")
        else:
            self.client = QdrantClient(
                host=settings.QDRANT_HOST,
                port=settings.QDRANT_PORT,
                api_key=api_key or settings.QDRANT_API_KEY,
            )

        self._ensure_collection_and_indexes()

    def _ensure_collection_and_indexes(self) -> None:
        """Initialize collection and create pre-filter payload indexes."""
        try:
            collections_res = self.client.get_collections()
            existing_names = [col.name for col in collections_res.collections]
            if self.collection_name not in existing_names:
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(size=self.dim, distance=Distance.COSINE),
                )
                logger.info(f"Initialized collection '{self.collection_name}' with dim {self.dim}")

            # Create payload indexes for hard pre-filtering performance
            for field_name in ["tenant_id", "trust_tier", "is_quarantined", "source_channel"]:
                try:
                    self.client.create_payload_index(
                        collection_name=self.collection_name,
                        field_name=field_name,
                        field_schema=models.PayloadSchemaType.KEYWORD,
                    )
                except Exception as index_err:
                    # Index might already exist, safe to proceed
                    logger.debug(f"Payload index on {field_name}: {index_err}")

        except Exception as e:
            logger.error(f"Failed to verify/create Qdrant collection: {e}")
            raise

    def store_memory(
        self,
        envelope: ProvenanceEnvelope,
        embedding: List[float],
    ) -> bool:
        """Persist a memory vector along with its complete cryptographic provenance envelope."""
        try:
            payload = envelope.model_dump()
            point = PointStruct(
                id=envelope.memory_id,
                vector=embedding,
                payload=payload,
            )
            self.client.upsert(
                collection_name=self.collection_name,
                points=[point],
                wait=True,
            )
            logger.info(
                f"Memory stored: id={envelope.memory_id}, tenant={envelope.tenant_id}, "
                f"quarantined={envelope.is_quarantined}, tier={envelope.trust_tier}"
            )
            return True
        except Exception as e:
            logger.error(f"Failed to upsert memory point {envelope.memory_id}: {e}")
            raise

    def search_memories(
        self,
        tenant_id: str,
        query_vector: List[float],
        top_k: int = 5,
        allowed_trust_tiers: Optional[List[str]] = None,
        include_quarantined: bool = False,
        source_channels: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Hard pre-filtered search restricted strictly to tenant boundaries and trust states."""
        try:
            must_filters = [
                FieldCondition(
                    key="tenant_id",
                    match=MatchValue(value=tenant_id),
                )
            ]

            if not include_quarantined:
                must_filters.append(
                    FieldCondition(
                        key="is_quarantined",
                        match=MatchValue(value=False),
                    )
                )

            if allowed_trust_tiers:
                tier_conditions = [
                    FieldCondition(key="trust_tier", match=MatchValue(value=tier))
                    for tier in allowed_trust_tiers
                ]
                must_filters.append(Filter(should=tier_conditions))

            if source_channels:
                channel_conditions = [
                    FieldCondition(key="source_channel", match=MatchValue(value=ch))
                    for ch in source_channels
                ]
                must_filters.append(Filter(should=channel_conditions))

            query_filter = Filter(must=must_filters)

            search_response = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                query_filter=query_filter,
                limit=top_k,
                with_payload=True,
            )

            results: List[Dict[str, Any]] = []
            for hit in search_response:
                results.append({
                    "id": hit.id,
                    "score": hit.score,
                    "payload": hit.payload or {},
                })
            return results

        except Exception as e:
            logger.error(f"Search failed for tenant {tenant_id}: {e}")
            raise

    def delete_memory(self, memory_id: str) -> bool:
        """Purge a point by ID (used during recursive blast-radius rollbacks)."""
        try:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=models.PointIdsList(points=[memory_id]),
                wait=True,
            )
            logger.info(f"Purged memory {memory_id} from vector store.")
            return True
        except Exception as e:
            logger.error(f"Failed to delete memory {memory_id}: {e}")
            raise

    def retrieve_by_id(self, memory_id: str) -> Optional[Dict[str, Any]]:
        """Fetch exact record and payload by memory ID."""
        try:
            records = self.client.retrieve(
                collection_name=self.collection_name,
                ids=[memory_id],
                with_payload=True,
                with_vectors=True,
            )
            if records:
                rec = records[0]
                return {
                    "id": rec.id,
                    "payload": rec.payload,
                    "vector": rec.vector,
                }
            return None
        except Exception as e:
            logger.error(f"Failed retrieval for id {memory_id}: {e}")
            return None

    def list_all_tenant_memories(self, tenant_id: str, limit: int = 100) -> List[Dict[str, Any]]:
        """List memories for an operational tenant."""
        try:
            scroll_res = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=Filter(
                    must=[FieldCondition(key="tenant_id", match=MatchValue(value=tenant_id))]
                ),
                limit=limit,
                with_payload=True,
            )
            points, _ = scroll_res
            return [{"id": p.id, "payload": p.payload} for p in points]
        except Exception as e:
            logger.error(f"Failed to list memories for tenant {tenant_id}: {e}")
            return []

