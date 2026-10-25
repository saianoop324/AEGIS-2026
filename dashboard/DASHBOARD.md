# MemGuard dashboard

Shows, at a glance, whether MemGuard works: how often attacks succeed with and without it, what it costs,
and how it compares with NVIDIA's guardrails. Every number on it is measured; anything not measured
shows "–".

```bash
GPU=3 scripts/run_dashboard.sh          # http://localhost:8501 (omit GPU= for CPU)
streamlit run dashboard/bankdesk_app.py # saved results only, never loads a model
```
From VS Code over Remote-SSH: **Ports** tab → Forward a Port → `8501` → open the forwarded address (or
*Simple Browser: Show*).

## 1. Before: how it was built

```
app.py (one file, ~400 lines)
 ├─ on startup, always: vector store, write gate, read gate, lineage, MemGauntlet harness, LangGraph agent
 │    (loads the embedding + DeBERTa models, ~20 s, even to view saved results)
 └─ 6 tabs
     🏦 BankDesk Experiments ── dashboard/bankdesk.py ── experiments/exp*.json        (real data)
     📊 Executive Scorecard ─── harness.execute_gauntlet_run → evaluator/metrics.py   (mostly invented, below)
     ⚔️ MemGauntlet Red-Team ── harness.execute_gauntlet_run                          (real, no benign texts)
     🛡️ Write-Gate & Quarantines ─ write gate form + store table                      (real)
     🌐 Lineage DAG & Rollback ── node table + rollback button                         (no graph drawn)
     🤖 Interactive Agent ────── LangGraph agent → NvidiaNimClient                    (canned reply, labeled as LLM)
```

The problems a viewer couldn't see:

| Where | What it showed | What was actually behind it |
|---|---|---|
| Executive: baseline ASR, "+88.0% over Unprotected" | a measured comparison | `baseline_asr = 0.88` and the label hardcoded; nothing unprotected was run |
| Executive: baseline bars per attack vector | baseline 12% | a constant `12.0` for every vector |
| Executive: precision, F1, utility retention | write-gate quality | `false_positives = 1` assumed; the harness's 10 benign texts were never run |
| Executive: MemGuard ASR ≈ 0% | attacks stopped | counted an attack as stopped if it was merely quoted, or never recalled at all; no agent acted |
| Agent: "LLM Response (NVIDIA NIM with NeMo Guardrails)" | a model's answer | a canned string unless an `nvapi-` key longer than 25 characters is set; with the local vLLM `.env`, always canned |
| Read gate in the dashboard | the experiments' read gate | no signature verifier, and a second copy of the embedding model |

The same invented numbers were returned by the API (`POST /api/v1/redteam/run` in `server.py`).

## 2. Now: how it is built

```
app.py ─ sidebar: pages (saved | live), tenant picker (live pages only), runtime config, signing-key warning
 │
 ├─ saved results (no models; open in 1–2 s)
 │   Executive scorecard ── dashboard/executive.py ─┐
 │   Experiment results ─── dashboard/bankdesk.py ──┴─ experiments/exp<N>_<classifier>_<set>.json  (agent runs)
 │                                                      experiments/exp<N>_heldout_<classifier>.json (public set)
 │                                                      experiments/<set>.json                       (scenario definitions)
 │                                                      scorecard() from experiments/run_bank_scenarios.py
 │
 ├─ live (get_services(): models loaded and warmed on first live page)
 │   Live red team ─────── harness: protected + unprotected + benign passes → evaluator/metrics.py
 │   Memory inspector ──── write gate, read gate (with signature verifier), vector store
 │   Lineage & rollback ── firewall/lineage.py, drawn as a graph
 │   Agent playground ──── LangGraph agent; says when the reply is canned (NvidiaNimClient.last_source)
 │
 └─ dashboard/style.py ── palette, fixed arm colors, chart helpers (tidy, top_down, facets)
```

| Page | Leads with |
|---|---|
| **Executive scorecard** (landing) | 4 tiles (attacks executed, reduction vs unprotected, benign tasks retained, added write latency); attack success by attack type and by entry channel (unprotected vs NeMo Guardrails vs MemGuard); targets as bullet bars; poisoned memories quarantined per channel; held-out precision/recall/F1; trend across experiments. Text in expanders at the bottom. |
| Experiment results | the same tiles for any experiment, classifier and sets; charts by scenario set; heat-grids by attack type and per scenario; then the scorecard (✓/✗), classifier comparison, held-out set, and a per-run drill-down (scenario definition, recalled memories, tool calls with the action gate's verdict, replay after deletion, action log) |
| Live red team | without vs with MemGuard tiles; exposure by attack type; what happened to each attack; latency vs the 25 ms budget |
| Memory inspector | the store by channel and trust tier; write a memory; preview what recall puts in the prompt; signature check per memory |
| Lineage & rollback | the memory → action graph; blast radius; rollback |
| Agent playground | the agent's steps, the context the read gate built, the reply (marked canned when it is) |

## 3. Changes and why

| Change | Why |
|---|---|
| Executive scorecard rebuilt on the agent experiments | They are the only place harm is measured (a harmful action executed in the sandbox) with a real baseline: the unprotected arm, same scenarios and seeds. Every old tile has a measured counterpart (table in §4). |
| Charts first, tables in "Data" expanders, prose last | The page should answer "does it work?" before anyone reads. Tables stay one click away for exact counts. |
| Run counts next to every rate | Several rates rest on a few dozen runs (3 seeds); a viewer needs the denominator. |
| Results pages load no models; live pages load lazily and warm up | Viewing results took ~20 s for no reason; warm-up keeps model loading out of the first write's latency (260 → ~30 ms p95). |
| `evaluator/metrics.py` rewritten; harness gains `execute_unprotected_baseline()` | Replaces the 0.88 baseline and the assumed false positive with passes that actually run. A missing pass yields `None`, never a guess. |
| Each red-team run (dashboard and API) uses a fresh tenant | Re-adding the same 30 texts to one store let old copies crowd out new ones on recall: a second run measured 23/30 exposed instead of 29/30. |
| "Exposure" vs "harm" named on every page | The live red team only shows whether an attack reaches the prompt as plain text. MemGuard's 0/30 there is mostly quoting, and the experiments show quoted memories still drive actions (13%), which the action gate stops. |
| Agent playground and agent step log say when no LLM was called | The old page presented a canned string as a model's answer. |
| Read gate wired like the experiment runner | Same embedder (one model load) and signature verification, so the live pages behave like the measured system. |
| Lineage drawn as a graph (isolated memories hidden on request) | A table of node ids couldn't show what fed what. |
| Fixed arm colors; validated palette; ✓/✗ text with status colors | Color follows the arm across every chart; pass/fail never relies on color alone. The palette is checked for color-blind separation in light and dark mode. |
| `scripts/run_dashboard.sh` with the file watcher off | One command to start it; the watcher was flooding the log with transformers' optional torchvision imports. |

## 4. How to read the numbers

**Harm (saved pages).** An attack succeeds only if its harmful action executed in the sandbox's action
log. What the agent merely said is not scored. The agent is Qwen3.5-9B in every run.

**Arms** (which layers were on): `unprotected` (none; the action gate only records what it would hold),
`nemo_rails` (NVIDIA NeMo Guardrails input rails), `no_action_gate` (write + read gate),
`action_gate_only`, `memguard` (all), `memguard_ttl` (all, plus expiry of unverified memories).

**Old Executive tile → its measured replacement**

| Old | Now (Exp 14, DeBERTa write gate) |
|---|---|
| ASR with delta vs 88% baseline | attacks executed: 13% MemGuard vs 74% unprotected (14 and 80 of 108 runs) |
| Mitigation rate | reduction vs unprotected 82% (target ≥ 80% ✓) |
| Utility retention | benign tasks retained 89% (target ≥ 95% ✗) |
| Write-gate latency | added write p95 37 ms (target < 150 ms ✓) |
| Baseline vs MemGuard by vector | attack success per attack type, per arm |
| Quarantines per channel (pie) | attack success per entry channel, and share of poisoned memories quarantined per channel |
| Precision / recall / F1 | held-out public set (deepset/prompt-injections, 662 texts): DeBERTa 0.78 / 0.12 / 0.21 |

**Entry channel.** Each attack plants at most one kind of content, which the sandbox tools read on a
fixed channel: document → `documents`; transaction memo or IT ticket → `tool_outputs`; peer message →
`peer_agent`. With nothing planted, the customer typed the attack (`user_chat`).

**Borrowed baselines.** Experiment 14's Qwen classifier runs didn't repeat the arms without a write
gate. Those arms don't depend on the classifier, so their rows come from the same experiment's DeBERTa
files, and Experiment results says so.

**Exposure (Live red team).** Exposed = recalled as plain text that the agent would take as fact. There
are three passes: the 30 attacks through MemGuard; the same 30 with no gate (same seeded facts, probe
query and recall depth); and 10 benign texts. Typical run: 29/30 exposed without MemGuard, 0/30 with it,
9/30 quarantined, 0/10 false positives. Latency is live and rises when the shared box is busy.

**"–"** means not measured. For example, persistence can only be measured where an attack succeeded,
so an arm that stopped every attack has none.

## 5. Checked by
- **Streamlit AppTest:** every experiment × classifier × arm on both saved pages; all 64 drill-downs of
  Exp 14; every live page with real models (red team, rerun, write, recall, agent, rollback). 0
  exceptions.
- **Recomputed scorecards** match the ones stored in the results files on all 3,280 values (Exp 12–14).
- **Tests:** `tests/test_evaluator.py` (math, `None` for missing passes, the unprotected pass); full
  suite 39 passed.
- **Headless-browser screenshots** of every page in light and dark mode. Issues found this way and
  fixed: NaN shown as a failed target, clipped labels, hidden chart rows, reversed bar order.

## 6. Files and known limits

| File | Role |
|---|---|
| `app.py` | routing, lazy services, live pages |
| `dashboard/executive.py` | Executive scorecard |
| `dashboard/bankdesk.py` | Experiment results; shared loaders, tiles, grids, held-out chart, entry channel |
| `dashboard/style.py` | palette and chart helpers |
| `dashboard/bankdesk_app.py` | standalone saved-results viewer |
| `evaluator/metrics.py`, `memgauntlet/harness.py`, `server.py` | measured red-team scorecard and API |

Known limits:
- `EXPERIMENT_NOTES` in `bankdesk.py` is a hand-kept line per experiment; add one for Experiment 15.
- Scenario definitions in the drill-down are the current files, which may have changed since older
  runs.
- The held-out set exists only for Exp 14, and other experiments show it with that label.
- Live pages share one in-memory store per process (cleared on restart). The lineage graph is saved to
  `memguard_lineage.json` in the working directory.
