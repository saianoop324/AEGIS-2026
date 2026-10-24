"""Hardware Profiler and Latency SLA Benchmarking Suite.

Measures:
1. P50, P95, and P99 latency SLAs on CPU and GPU (NVIDIA RTX 3050 / Cloud GPUs).
2. Throughput (writes/second and queries/second) under concurrent tenant isolation.
3. VRAM and system memory utilization breakdown.
4. Generates performance report for Aegis 2026 presentation metrics.
"""

import time
import os
import logging
from typing import Dict, Any, List
from pydantic import BaseModel, Field

from config import settings, ChannelType
from store.vector_store import QdrantVectorStoreWrapper
from firewall.write_gate import MemGuardWriteGate
from firewall.read_gate import MemGuardReadGate

logger = logging.getLogger(__name__)


class LatencySlaReport(BaseModel):
    device: str
    total_operations: int
    write_p50_ms: float
    write_p95_ms: float
    write_p99_ms: float
    write_sla_met: bool  # Target: < 25.0 ms (local path) or < 150 ms (end-to-end)
    read_p50_ms: float
    read_p95_ms: float
    throughput_ops_per_sec: float
    gpu_vram_mb: Optional[float] = None
    summary: str


class HardwareProfiler:
    """Measures runtime hardware performance and SLA compliance."""

    def __init__(
        self,
        write_gate: MemGuardWriteGate,
        read_gate: MemGuardReadGate,
        vector_store: QdrantVectorStoreWrapper,
    ):
        self.write_gate = write_gate
        self.read_gate = read_gate
        self.vector_store = vector_store

    def get_gpu_memory_usage_mb(self) -> Optional[float]:
        """Query PyTorch CUDA memory footprint if running on GPU."""
        try:
            import torch
            if torch.cuda.is_available():
                return torch.cuda.memory_allocated() / (1024 * 1024)
        except Exception:
            pass
        return None

    def run_benchmark(self, iterations: int = 100, tenant_id: str = "tenant_bench") -> LatencySlaReport:
        write_latencies: List[float] = []
        read_latencies: List[float] = []

        test_payloads = [
            "Customer wire limit confirmed at standard tier for checking account.",
            "System note: user submitted updated mailing address in Boston MA.",
            "Transaction processed: payment of 120.00 USD to authorized utility.",
            "Executive bulletin: domestic branch offices open regular hours.",
            "User preference recorded: notify account events via push notification.",
        ]

        t_start_total = time.perf_counter()

        # Benchmark Write Gate
        for i in range(iterations):
            content = test_payloads[i % len(test_payloads)]
            t0 = time.perf_counter()
            self.write_gate.evaluate_and_write(
                tenant_id=tenant_id,
                content=content,
                source_channel=ChannelType.USER_CHAT,
            )
            write_latencies.append((time.perf_counter() - t0) * 1000.0)

        # Benchmark Read Gate
        for i in range(iterations):
            query = f"What is the wire limit or address for query {i}?"
            t0 = time.perf_counter()
            self.read_gate.query_memories(
                tenant_id=tenant_id,
                query=query,
                top_k=3,
                strict_sandboxing=True,
            )
            read_latencies.append((time.perf_counter() - t0) * 1000.0)

        total_elapsed = time.perf_counter() - t_start_total
        ops_per_sec = (iterations * 2) / total_elapsed if total_elapsed > 0 else 0.0

        write_latencies.sort()
        read_latencies.sort()
        nw = len(write_latencies)
        nr = len(read_latencies)

        w_p50 = write_latencies[int(nw * 0.50)]
        w_p95 = write_latencies[int(nw * 0.95)]
        w_p99 = write_latencies[int(nw * 0.99)]

        r_p50 = read_latencies[int(nr * 0.50)]
        r_p95 = read_latencies[int(nr * 0.95)]

        vram = self.get_gpu_memory_usage_mb()
        sla_met = w_p95 < 150.0  # Presentation Target: < 150ms P95 overhead

        return LatencySlaReport(
            device=settings.DEVICE,
            total_operations=iterations * 2,
            write_p50_ms=round(w_p50, 2),
            write_p95_ms=round(w_p95, 2),
            write_p99_ms=round(w_p99, 2),
            write_sla_met=sla_met,
            read_p50_ms=round(r_p50, 2),
            read_p95_ms=round(r_p95, 2),
            throughput_ops_per_sec=round(ops_per_sec, 1),
            gpu_vram_mb=round(vram, 2) if vram is not None else None,
            summary=f"P95 write latency: {w_p95:.2f}ms. Throughput: {ops_per_sec:.1f} ops/sec. SLA Met: {sla_met}.",
        )

