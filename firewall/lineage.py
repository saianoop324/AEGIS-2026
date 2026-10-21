"""Provenance & Lineage Tracking DAG powered by NetworkX.

Tracks:
  Memory (Root) -> Derived Memory -> Agent Action

Supports:
  - Blast-radius identification (forward BFS/DFS traversal).
  - Recursive rollback (purging contaminated memories & cascading invalidation).
  - Provenance export to JSON/Graph formats for Streamlit visualization.
"""

import os
import json
import time
import logging
from typing import Dict, Any, List, Set, Optional, Tuple
from enum import Enum
import networkx as nx
from pydantic import BaseModel, Field

from config import settings
from store.vector_store import QdrantVectorStoreWrapper

logger = logging.getLogger(__name__)


class NodeType(str, Enum):
    ROOT_MEMORY = "root_memory"
    DERIVED_MEMORY = "derived_memory"
    AGENT_ACTION = "agent_action"
    QUARANTINED_EVENT = "quarantined_event"


class NodeMetadata(BaseModel):
    node_id: str
    node_type: NodeType
    tenant_id: str
    label: str
    details: Dict[str, Any] = Field(default_factory=dict)
    timestamp: float = Field(default_factory=time.time)
    status: str = "active"  # active, rolled_back, quarantined


class LineageTracker:
    """NetworkX-powered lineage graph tracker with persistence and rollback capabilities."""

    def __init__(
        self,
        vector_store: Optional[QdrantVectorStoreWrapper] = None,
        db_path: str = settings.LINEAGE_DB_PATH,
    ):
        self.vector_store = vector_store
        self.db_path = db_path
        self.graph = nx.DiGraph()
        self._load_from_disk()

    def _load_from_disk(self) -> None:
        """Load lineage graph state if file exists."""
        if os.path.exists(self.db_path):
            try:
                with open(self.db_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.graph = nx.node_link_graph(data, edges="links")
                logger.info(f"Loaded lineage graph with {self.graph.number_of_nodes()} nodes.")
            except Exception as e:
                logger.warning(f"Could not load lineage graph from {self.db_path}: {e}")
                self.graph = nx.DiGraph()

    def save_to_disk(self) -> None:
        """Persist current lineage DAG to JSON."""
        try:
            data = nx.node_link_data(self.graph, edges="links")
            with open(self.db_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to persist lineage graph: {e}")

    def record_root_memory(
        self,
        memory_id: str,
        tenant_id: str,
        content: str,
        channel: str,
        trust_tier: str,
        is_quarantined: bool = False,
    ) -> None:
        """Register a base memory ingestion node into the DAG."""
        node_type = NodeType.QUARANTINED_EVENT if is_quarantined else NodeType.ROOT_MEMORY
        self.graph.add_node(
            memory_id,
            node_type=node_type.value,
            tenant_id=tenant_id,
            label=f"Memory [{channel}]",
            content=content,
            channel=channel,
            trust_tier=trust_tier,
            status="quarantined" if is_quarantined else "active",
            timestamp=time.time(),
        )
        self.save_to_disk()

    def record_derived_memory(
        self,
        derived_id: str,
        parent_memory_ids: List[str],
        tenant_id: str,
        summary_or_insight: str,
    ) -> None:
        """Register a synthesized or derived memory that originates from other memories."""
        self.graph.add_node(
            derived_id,
            node_type=NodeType.DERIVED_MEMORY.value,
            tenant_id=tenant_id,
            label="Derived Synthesis",
            content=summary_or_insight,
            status="active",
            timestamp=time.time(),
        )
        for parent_id in parent_memory_ids:
            if not self.graph.has_node(parent_id):
                # Auto-create stub node if parent wasn't formally registered
                self.graph.add_node(
                    parent_id,
                    node_type=NodeType.ROOT_MEMORY.value,
                    tenant_id=tenant_id,
                    label="Implicit Parent Memory",
                    status="active",
                    timestamp=time.time(),
                )
            self.graph.add_edge(parent_id, derived_id, relation="derives")
        self.save_to_disk()

    def record_action(
        self,
        action_id: str,
        consumed_memory_ids: List[str],
        tenant_id: str,
        action_name: str,
        action_payload: Dict[str, Any],
    ) -> None:
        """Register an external effect or agent tool execution triggered by recall."""
        self.graph.add_node(
            action_id,
            node_type=NodeType.AGENT_ACTION.value,
            tenant_id=tenant_id,
            label=f"Action: {action_name}",
            action_name=action_name,
            payload=json.dumps(action_payload),
            status="executed",
            timestamp=time.time(),
        )
        for mem_id in consumed_memory_ids:
            if not self.graph.has_node(mem_id):
                self.graph.add_node(
                    mem_id,
                    node_type=NodeType.ROOT_MEMORY.value,
                    tenant_id=tenant_id,
                    label="Implicit Consumed Memory",
                    status="active",
                    timestamp=time.time(),
                )
            self.graph.add_edge(mem_id, action_id, relation="influences")
        self.save_to_disk()

    def calculate_blast_radius(self, memory_id: str) -> Dict[str, Any]:
        """Compute complete downstream impact of a compromised or poisoned memory."""
        if not self.graph.has_node(memory_id):
            return {
                "root_id": memory_id,
                "found": False,
                "affected_nodes_count": 0,
                "derived_memories": [],
                "tainted_actions": [],
            }

        # BFS / Descendants traversal
        descendants: Set[str] = nx.descendants(self.graph, memory_id)
        all_affected = descendants.union({memory_id})

        derived_memories: List[Dict[str, Any]] = []
        tainted_actions: List[Dict[str, Any]] = []

        for node in all_affected:
            data = self.graph.nodes[node]
            ntype = data.get("node_type")
            if ntype == NodeType.DERIVED_MEMORY.value:
                derived_memories.append({
                    "id": node,
                    "content": data.get("content", ""),
                    "status": data.get("status", "active"),
                })
            elif ntype == NodeType.AGENT_ACTION.value:
                tainted_actions.append({
                    "id": node,
                    "action_name": data.get("action_name", ""),
                    "payload": data.get("payload", ""),
                    "status": data.get("status", "executed"),
                })

        return {
            "root_id": memory_id,
            "found": True,
            "affected_nodes_count": len(all_affected),
            "all_affected_node_ids": list(all_affected),
            "derived_memories": derived_memories,
            "tainted_actions": tainted_actions,
        }

    def rollback_memory(self, memory_id: str) -> Dict[str, Any]:
        """Cascade rollback: purge memory from vector store and revoke downstream artifacts in DAG."""
        blast = self.calculate_blast_radius(memory_id)
        if not blast["found"]:
            return {"success": False, "message": f"Memory {memory_id} not found in graph."}

        purged_memories = []
        revoked_actions = []

        # Iterate over all affected nodes
        for node in blast["all_affected_node_ids"]:
            node_data = self.graph.nodes[node]
            ntype = node_data.get("node_type")

            if ntype in (NodeType.ROOT_MEMORY.value, NodeType.DERIVED_MEMORY.value, NodeType.QUARANTINED_EVENT.value):
                # Purge from vector store if vector_store handle provided
                if self.vector_store:
                    try:
                        self.vector_store.delete_memory(node)
                    except Exception as e:
                        logger.warning(f"Vector store deletion error for {node}: {e}")
                node_data["status"] = "rolled_back"
                purged_memories.append(node)

            elif ntype == NodeType.AGENT_ACTION.value:
                node_data["status"] = "revoked_compensating_action_needed"
                revoked_actions.append(node)

        self.save_to_disk()
        logger.warning(
            f"Rollback executed for {memory_id}: purged {len(purged_memories)} memories, "
            f"revoked {len(revoked_actions)} actions."
        )

        return {
            "success": True,
            "root_target": memory_id,
            "purged_memory_ids": purged_memories,
            "revoked_action_ids": revoked_actions,
            "total_compromised": blast["affected_nodes_count"],
        }

    def get_subgraph_data_for_tenant(self, tenant_id: str) -> Dict[str, Any]:
        """Export serialized graph data tailored for rendering in frontend dashboards."""
        nodes = []
        links = []

        for n, data in self.graph.nodes(data=True):
            if data.get("tenant_id") == tenant_id or tenant_id == "*":
                nodes.append({
                    "id": n,
                    "label": data.get("label", n[:8]),
                    "type": data.get("node_type", "unknown"),
                    "status": data.get("status", "active"),
                    "content": data.get("content", ""),
                    "channel": data.get("channel", "internal"),
                    "timestamp": data.get("timestamp", 0),
                })

        tenant_node_ids = {n["id"] for n in nodes}
        for u, v, data in self.graph.edges(data=True):
            if u in tenant_node_ids and v in tenant_node_ids:
                links.append({
                    "source": u,
                    "target": v,
                    "relation": data.get("relation", "connects"),
                })

        return {"nodes": nodes, "links": links}

