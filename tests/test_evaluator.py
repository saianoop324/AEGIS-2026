"""MemGauntlet scorecard: every number comes from a pass that was run. No LLM needed.

Run: python -m pytest tests/ -q
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluator.metrics import EvaluatorEngine  # noqa: E402
from firewall.lineage import LineageTracker  # noqa: E402
from firewall.read_gate import MemGuardReadGate  # noqa: E402
from firewall.write_gate import MemGuardWriteGate  # noqa: E402
from memgauntlet.harness import MemGauntletHarness, RedTeamTestResult  # noqa: E402
from store.vector_store import QdrantVectorStoreWrapper  # noqa: E402


def result(quarantined=False, exposed=False, quoted=False, vector="false_fact", channel="documents", ms=10.0):
    return RedTeamTestResult(payload_id="p", attack_vector=vector, channel=channel, tenant_id="t", content="x",
                             intercepted_by_write_gate=quarantined, quarantined=quarantined, quarantine_reason=None,
                             read_gate_sandboxed=quoted or quarantined, persisted_safely=not exposed, latency_ms=ms)


def test_rates_come_from_the_passes_given():
    attacks = [result(quarantined=True)] * 3 + [result(quoted=True)] * 5 + [result(exposed=True)] * 2
    benign = [result()] * 9 + [result(quarantined=True)]
    baseline = [result(exposed=True)] * 8 + [result()] * 2
    s = EvaluatorEngine.evaluate_results(attacks, benign_results=benign, baseline_results=baseline)
    assert s.memguard_attack_success_rate == 20.0 and s.baseline_attack_success_rate == 80.0
    assert s.attack_mitigation_rate == 75.0                       # 1 - 20/80
    assert s.write_gate_recall == 0.3 and s.write_gate_precision == 0.75  # 3 of 3+1 quarantined texts were attacks
    assert s.task_utility_retention == 90.0 and s.benign_quarantined_count == 1
    assert s.persistence_rate_memguard == 70.0 and s.persistence_rate_baseline == 100.0
    assert s.sandboxed_count == 5
    assert s.vector_breakdown["false_fact"]["exposed_baseline"] == 8


def test_missing_passes_give_none_not_a_guess():
    s = EvaluatorEngine.evaluate_results([result(quarantined=True), result(exposed=True)])
    assert s.baseline_attack_success_rate is None and s.attack_mitigation_rate is None
    assert s.write_gate_precision is None and s.write_gate_f1 is None and s.task_utility_retention is None
    assert s.memguard_attack_success_rate == 50.0 and s.write_gate_recall == 0.5


def test_unprotected_baseline_runs_the_same_attacks_without_gates(tmp_path):
    vs = QdrantVectorStoreWrapper()
    wg = MemGuardWriteGate(vector_store=vs)
    rg = MemGuardReadGate(vector_store=vs, embedding_function=wg._embedding_func, verifier=wg.verify_envelope)
    harness = MemGauntletHarness(wg, rg, LineageTracker(vector_store=vs, db_path=str(tmp_path / "lineage.json")))
    protected = harness.execute_gauntlet_run(tenant_id="ev")
    baseline = harness.execute_unprotected_baseline(tenant_id="ev")
    assert [r.content for r in baseline] == [r.content for r in protected]
    assert not any(r.quarantined or r.read_gate_sandboxed for r in baseline)
    assert {r.tenant_id for r in baseline} == {"ev__unprotected"}
    # every attack was stored; the protected tenant is untouched by the baseline pass
    assert len(vs.list_all_tenant_memories("ev__unprotected", limit=1000)) == len(baseline) + 3
