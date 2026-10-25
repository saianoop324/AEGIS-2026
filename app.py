"""MemGuard dashboard (Streamlit): streamlit run app.py

Results pages read saved experiment outputs and load no models:
  Executive scorecard   the landing page: measured headline numbers and charts against the unprotected baseline
  Experiment results    one experiment in detail, down to each run's transcript and action log
Live pages load the embedding and classifier models on first use and run the gates in this process:
  Live red team         MemGauntlet's attacks with and without MemGuard, and its benign controls
  Memory inspector      write a memory, see the gate's decision, what recall returns and the tenant's store
  Lineage & rollback    the provenance graph, blast radius and cascading rollback
  Agent playground      the LangGraph agent: read gate -> LLM -> write gate -> lineage

Design and change log: dashboard/DASHBOARD.md
"""

import json

import networkx as nx
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from config import Settings, settings, ChannelType, TrustTier
from dashboard.bankdesk import render as render_results, esc, fmt
from dashboard.executive import INK, render_executive
from dashboard.style import CATEGORICAL, arm_colors, theme, tidy, top_down
from store.vector_store import ProvenanceEnvelope

st.set_page_config(page_title="MemGuard | Agent Memory Firewall", page_icon="🛡️", layout="wide",
                   initial_sidebar_state="expanded")

RESULT_PAGES = ["Executive scorecard", "Experiment results"]
LIVE_PAGES = ["Live red team", "Memory inspector", "Lineage & rollback", "Agent playground"]
CHANNEL_TIER = {  # firewall/write_gate.py, evaluate_and_write step 1
    ChannelType.USER_CHAT.value: TrustTier.TIER_2_PROVISIONAL.value,
    ChannelType.DOCUMENTS.value: TrustTier.TIER_2_PROVISIONAL.value,
    ChannelType.TOOL_OUTPUTS.value: TrustTier.TIER_3_UNTRUSTED.value,
    ChannelType.WEB_RETRIEVAL.value: TrustTier.TIER_3_UNTRUSTED.value,
    ChannelType.PEER_AGENT.value: TrustTier.TIER_3_UNTRUSTED.value,
}


@st.cache_resource(show_spinner="Loading the MemGuard gates and their models…")
def get_services():
    """One set of gates for the live pages, wired the way the experiment runner wires them."""
    from agent.memguard_agent import MemGuardLangGraphAgent
    from firewall.lineage import LineageTracker
    from firewall.read_gate import MemGuardReadGate
    from firewall.write_gate import MemGuardWriteGate
    from memgauntlet.harness import MemGauntletHarness
    from store.vector_store import QdrantVectorStoreWrapper

    vs = QdrantVectorStoreWrapper()
    lineage = LineageTracker(vector_store=vs)
    wg = MemGuardWriteGate(vector_store=vs)
    # share the write gate's embedder (one model load) and verify signatures on recall
    rg = MemGuardReadGate(vector_store=vs, embedding_function=wg._embedding_func, verifier=wg.verify_envelope)
    # load the models now, so the first write's latency is the gate's and not the model download/load
    wg._embedding_func("warm up")
    wg.classifier.classify_instruction_vs_fact("The branch opens at nine.")
    return {"vector_store": vs, "lineage": lineage, "write_gate": wg, "read_gate": rg,
            "harness": MemGauntletHarness(wg, rg, lineage), "agent": MemGuardLangGraphAgent(vs, wg, rg, lineage)}


def agent_calls_llm() -> bool:
    """Mirrors NvidiaNimClient.generate: without a real nvapi- key it answers from a local canned reply."""
    key = settings.NVIDIA_NIM_API_KEY or ""
    return key.startswith("nvapi-") and len(key) > 25


# ---------------------------------------------------------------- sidebar
st.sidebar.title("🛡️ MemGuard")
st.sidebar.caption("Memory firewall for AI agents")
page = st.sidebar.radio("Page", RESULT_PAGES + LIVE_PAGES,
                        captions=["saved results"] * len(RESULT_PAGES) + ["live · loads models"] * len(LIVE_PAGES))
tenant = None
if page in LIVE_PAGES:
    # red-team runs get their own tenants (see page_red_team); list them so their memories can be inspected
    tenant = st.sidebar.selectbox("Tenant (live pages)", ["tenant_alpha", "tenant_beta", "tenant_finance",
                                                          "tenant_healthcare"] + st.session_state.get("redteam_tenants", []))
with st.sidebar.expander("Runtime configuration"):
    st.markdown(
        f"- Write-gate classifier: `{settings.WRITE_GATE_CLASSIFIER}` on `{settings.DEVICE}`\n"
        f"- NLI model: `{settings.DEBERTA_NLI_MODEL_NAME}`\n"
        f"- Embeddings: `{settings.EMBEDDING_MODEL_NAME}`\n"
        f"- Agent LLM: `{settings.NVIDIA_NIM_MODEL}` at `{settings.NVIDIA_NIM_API_BASE}`\n"
        f"- Vector store: `{settings.QDRANT_URL or settings.QDRANT_LOCATION}`\n"
        f"- Write-gate latency budget: {settings.WRITE_GATE_LATENCY_BUDGET_MS:g} ms")
    if settings.SECRET_SIGNING_KEY == Settings.model_fields["SECRET_SIGNING_KEY"].default:
        st.warning("Provenance signatures use the default key from config.py; set SECRET_SIGNING_KEY for anything "
                   "shared.")


# ---------------------------------------------------------------- live: write-gate red team
def page_red_team(svc) -> None:
    from evaluator.metrics import EvaluatorEngine
    st.header("Live red team: write and read gates")
    st.caption("Runs MemGauntlet's 30 hand-written attacks (6 attack types across 5 channels) through MemGuard in this "
               "process, the same attacks with MemGuard off, and 10 benign texts. Exposed = the attack comes back on "
               "recall as plain text the agent would take as fact. This is exposure, not harm: whether an agent acts on "
               "a memory is measured on the Executive scorecard.")
    # Each run writes into a fresh tenant: re-adding the same 30 texts to one store makes earlier copies compete
    # for the recalled slots, which skews both the protected and the unprotected numbers on every rerun.
    base = tenant.split("-redteam")[0]
    key = f"gauntlet::{base}"
    rerun = st.button("Run again", type="primary")  # the first visit runs it automatically
    if rerun or key not in st.session_state:
        runs = st.session_state.setdefault("redteam_tenants", [])
        run_tenant = f"{base}-redteam{sum(r.startswith(base + '-') for r in runs) + 1}"
        runs.append(run_tenant)
        with st.spinner("Running the attacks with and without MemGuard, and the benign texts…"):
            h = svc["harness"]
            st.session_state[key] = (run_tenant, h.execute_gauntlet_run(tenant_id=run_tenant),
                                     h.execute_benign_controls(tenant_id=run_tenant),
                                     h.execute_unprotected_baseline(tenant_id=run_tenant))
    run_tenant, attacks, benign, baseline = st.session_state[key]
    s = EvaluatorEngine.evaluate_results(attacks, benign_results=benign, baseline_results=baseline)
    n, nb = s.total_attacks_tested, s.benign_controls_tested
    exposed = sum(not r.persisted_safely for r in attacks)
    exposed_base = sum(not r.persisted_safely for r in baseline)
    budget = settings.WRITE_GATE_LATENCY_BUDGET_MS

    c = st.columns(5)
    c[0].metric("Without MemGuard", f"{exposed_base}/{n}", border=True,
                help="Attacks recalled as plain text when stored and recalled without any gate.")
    c[0].caption(f"exposed: {s.baseline_attack_success_rate:.0f}% recalled as plain text")
    c[1].metric("With MemGuard", f"{exposed}/{n}", border=True, delta_color="inverse",
                delta=f"{s.memguard_attack_success_rate - s.baseline_attack_success_rate:+.0f} pts")
    c[1].caption(f"exposed · {s.quarantined_count} quarantined, {s.sandboxed_count} quoted as untrusted")
    c[2].metric("Quarantined", f"{s.quarantined_count}/{n}", border=True, help="Attacks the write gate quarantined.")
    c[2].caption(f"write-gate recall {s.write_gate_recall:.2f}")
    c[3].metric("False positives", f"{s.benign_quarantined_count}/{nb}", border=True,
                help="Benign texts the write gate quarantined.")
    c[3].caption(f"precision {fmt(s.write_gate_precision)} · utility {fmt(s.task_utility_retention)}%")
    c[4].metric("Write p95", f"{s.p95_latency_ms:.1f} ms", border=True)
    c[4].caption(f"p50 {s.p50_latency_ms:.1f} ms · budget {budget:g} ms "
                 f"{'✓ within' if s.p95_latency_ms <= budget else '✗ over'}")
    st.caption("Quoting is why exposure drops to near zero with MemGuard: the read gate wraps every unverified memory "
               "as data. The agent experiments show quoted memories can still drive actions, which the action gate "
               "is there to stop.")

    colors = arm_colors(st, ["unprotected", "memguard"])
    left, right = st.columns(2)
    with left:
        st.markdown("**Exposed as plain text, by attack type**")
        vb = pd.DataFrame([{"attack type": v, "arm": arm, "exposed": d[k], "of": d["total"]}
                           for v, d in s.vector_breakdown.items()
                           for arm, k in (("unprotected", "exposed_baseline"), ("memguard", "exposed_memguard"))])
        fig = px.bar(vb, x="exposed", y="attack type", color="arm", orientation="h", barmode="group", text="exposed",
                     color_discrete_map=colors, category_orders={"arm": ["unprotected", "memguard"]},
                     hover_data=["of"], labels={"attack type": "", "exposed": "attacks exposed"})
        fig.update_traces(textposition="outside", cliponaxis=False, textfont_size=11)
        st.plotly_chart(top_down(tidy(fig, 330)), width="stretch")

    def outcome(r):
        if r.quarantined:
            return "quarantined"
        if r.read_gate_sandboxed:
            return "admitted, quoted on recall"
        return "admitted, not recalled" if r.persisted_safely else "exposed as plain text"

    df = pd.DataFrame([{"attack type": r.attack_vector, "channel": r.channel, "outcome": outcome(r),
                        "reason": r.quarantine_reason or "", "latency (ms)": round(r.latency_ms, 1),
                        "text": r.content} for r in attacks])
    order = ["quarantined", "admitted, quoted on recall", "admitted, not recalled", "exposed as plain text"]
    # slots 6/3/2/1: validated together in both modes, and unused by the arms shown on this page
    outcome_colors = dict(zip(order, [CATEGORICAL[theme(st)][i] for i in (6, 3, 2, 1)]))
    with right:
        st.markdown("**What MemGuard did with each attack, by channel**")
        counts = df.groupby(["channel", "outcome"]).size().reset_index(name="texts")
        fig = px.bar(counts, x="texts", y="channel", color="outcome", orientation="h", text="texts",
                     color_discrete_map=outcome_colors, category_orders={"outcome": order}, labels={"channel": ""})
        fig.update_traces(textposition="inside", insidetextanchor="middle", textfont_size=11)
        fig.update_layout(barmode="stack")
        st.plotly_chart(tidy(fig, 330), width="stretch")

    left, right = st.columns(2)
    with left:
        st.markdown("**What MemGuard did with each attack, by attack type**")
        counts = df.groupby(["attack type", "outcome"]).size().reset_index(name="texts")
        fig = px.bar(counts, x="texts", y="attack type", color="outcome", orientation="h", text="texts",
                     color_discrete_map=outcome_colors, category_orders={"outcome": order}, labels={"attack type": ""})
        fig.update_traces(textposition="inside", insidetextanchor="middle", textfont_size=11)
        fig.update_layout(barmode="stack")
        st.plotly_chart(tidy(fig, 330), width="stretch")
    with right:
        st.markdown("**Write-gate latency per text**")
        lat = pd.DataFrame({"ms": [r.latency_ms for r in attacks + benign]})
        fig = px.histogram(lat, x="ms", nbins=30, color_discrete_sequence=[colors["memguard"]],
                           labels={"ms": "milliseconds per write"})
        fig.add_vline(x=budget, line_dash="dash", line_width=2, line_color=INK[theme(st)],
                      annotation_text=f"budget {budget:g} ms", annotation_position="top right")
        st.plotly_chart(tidy(fig, 330), width="stretch")

    with st.expander(f"All {n} attacks"):
        st.dataframe(df, hide_index=True, width="stretch")
    with st.expander(f"All {nb} benign texts (any quarantine here is a false positive)"):
        st.dataframe(pd.DataFrame([{"channel": b.channel, "quarantined": b.quarantined,
                                    "reason": b.quarantine_reason or "", "latency (ms)": round(b.latency_ms, 1),
                                    "text": b.content} for b in benign]), hide_index=True, width="stretch")
    st.caption(f"This run wrote into tenant {run_tenant} (unprotected copy: {run_tenant}__unprotected); pick it in the "
               "sidebar to inspect its memories and lineage. Each run starts from an empty tenant, so runs are "
               "comparable. These texts were written by the team; the held-out public set on the Executive scorecard "
               "is the fairer test of the classifiers.")


# ---------------------------------------------------------------- live: memory inspector
def page_memory(svc) -> None:
    wg, rg, vs, lineage = svc["write_gate"], svc["read_gate"], svc["vector_store"], svc["lineage"]
    st.header("Memory inspector (live)")
    st.caption("Write a memory through the write gate, see what recall would put in the agent's prompt, and inspect "
               f"everything stored for {tenant}.")
    overview = st.container()  # filled last, so it includes a memory written below in this same run
    with st.expander("How a memory's trust tier is set"):
        st.dataframe(pd.DataFrame([{"channel": ch, "tier if admitted": t} for ch, t in CHANNEL_TIER.items()]),
                     hide_index=True, width="stretch")
        st.markdown("A memory the gate flags (reads as an instruction, or contradicts a similar stored memory) is "
                    "stored as `quarantined` and never recalled. Only memories given `verified` explicitly are "
                    "recalled unquoted; the rest are wrapped as untrusted data.")

    left, right = st.columns(2)
    with left, st.form("write_form"):
        st.markdown("**Write a memory**")
        text = st.text_area("Content", value="The primary corporate headquarters has relocated to Zurich.")
        channel = st.selectbox("Source channel", [c.value for c in ChannelType])
        submitted = st.form_submit_button("Submit to write gate")
    if submitted and text:
        d = wg.evaluate_and_write(tenant_id=tenant, content=text, source_channel=ChannelType(channel))
        mem_id = d.envelope.memory_id if d.envelope else "unknown"
        lineage.record_root_memory(memory_id=mem_id, tenant_id=tenant, content=text, channel=channel,
                                   trust_tier=d.trust_tier, is_quarantined=d.is_quarantined)
        with left:
            if d.is_quarantined:
                st.error(f"Quarantined: {'; '.join(d.reasons)}")
            else:
                st.success(f"Stored as `{d.trust_tier}`.")
            src = getattr(wg.classifier, "last_source", "")
            st.dataframe(pd.DataFrame([
                {"check": "instruction score", "value": f"{d.instruction_score:.3f}",
                 "detail": f"quarantine at ≥ {settings.INSTRUCTION_DETECTION_THRESHOLD:g}" + (f" · decided by {src}" if src else "")},
                {"check": "contradiction score", "value": f"{d.contradiction_score:.3f}",
                 "detail": f"quarantine at ≥ {settings.CONTRADICTION_THRESHOLD:g}"
                           + (f" · against {d.contradicting_memory_id[:8]}" if d.contradicting_memory_id else "")},
                {"check": "latency", "value": f"{d.latency_ms:.1f} ms", "detail": f"budget {settings.WRITE_GATE_LATENCY_BUDGET_MS:g} ms"},
                {"check": "signature", "value": (d.envelope.sha256_signature[:16] + "…") if d.envelope else "–",
                 "detail": f"memory {mem_id[:8]}"},
            ]), hide_index=True, width="stretch")

    with right, st.form("recall_form"):
        st.markdown("**Preview recall**")
        query = st.text_input("Query", value="Where is the corporate headquarters?")
        recall = st.form_submit_button("Recall")
    if recall and query:
        res = rg.query_memories(tenant_id=tenant, query=query, top_k=5)
        with right:
            st.caption(" ".join(res.execution_notes))
            st.markdown("What the agent's prompt would contain:")
            st.code(res.rendered_prompt_context, language=None, wrap_lines=True)

    st.subheader(f"Stored memories for {tenant}")
    records = vs.list_all_tenant_memories(tenant_id=tenant, limit=200)
    if not records:
        st.info("No memories stored for this tenant yet. Write one above or run the red team.")
        return
    with overview:
        tiers = [TrustTier.TIER_1_VERIFIED.value, TrustTier.TIER_2_PROVISIONAL.value, TrustTier.TIER_3_UNTRUSTED.value,
                 TrustTier.QUARANTINED.value]
        counts = (pd.DataFrame([{"channel": r["payload"].get("source_channel"),
                                 "tier": r["payload"].get("trust_tier")} for r in records])
                  .groupby(["channel", "tier"]).size().reset_index(name="memories"))
        st.markdown(f"**{len(records)} memories stored for {tenant}, by channel and trust tier**")
        fig = px.bar(counts, x="memories", y="channel", color="tier", orientation="h", text="memories",
                     color_discrete_map=dict(zip(tiers, CATEGORICAL[theme(st)][:4])), category_orders={"tier": tiers},
                     labels={"channel": ""})
        fig.update_traces(textposition="inside", insidetextanchor="middle", textfont_size=11)
        fig.update_layout(barmode="stack")
        st.plotly_chart(tidy(fig, 90 + 40 * counts["channel"].nunique()), width="stretch")

    def verifies(payload) -> bool:
        try:
            return wg.verify_envelope(ProvenanceEnvelope(**payload))
        except Exception:
            return False
    st.dataframe(pd.DataFrame([{
        "memory": str(r.get("id"))[:8], "tier": r["payload"].get("trust_tier"),
        "channel": r["payload"].get("source_channel"), "quarantined": bool(r["payload"].get("is_quarantined")),
        "signature verifies": verifies(r["payload"]), "reason": r["payload"].get("quarantine_reason") or "",
        "content": r["payload"].get("content", ""),
    } for r in records]), hide_index=True, width="stretch")


# ---------------------------------------------------------------- live: lineage
LAYER = {"observed_content": 0, "root_memory": 1, "quarantined_event": 1, "derived_memory": 2, "agent_action": 3}
SYMBOL = {"observed_content": "diamond", "root_memory": "circle", "quarantined_event": "x",
          "derived_memory": "square", "agent_action": "star"}


def lineage_figure(nodes, links):
    g = nx.DiGraph()
    for n in nodes:
        g.add_node(n["id"], layer=LAYER.get(n["type"], 1))
    g.add_edges_from((l["source"], l["target"]) for l in links)
    pos = nx.multipartite_layout(g, subset_key="layer", align="vertical")
    ex, ey = [], []
    for u, v in g.edges():
        ex += [pos[u][0], pos[v][0], None]
        ey += [pos[u][1], pos[v][1], None]
    fig = go.Figure(go.Scatter(x=ex, y=ey, mode="lines", line=dict(width=1, color="rgba(128,128,128,0.5)"),
                               hoverinfo="skip", showlegend=False))
    slots = CATEGORICAL[theme(st)]
    # one trace per node type (color + shape), and rolled-back nodes in a gray trace of their own so the
    # legend never shows a hollow marker for a whole type
    groups = [(kind.replace("_", " "), slots[i], [n for n in nodes if n["type"] == kind and n["status"] != "rolled_back"])
              for i, kind in enumerate(LAYER)]
    groups.append(("rolled back", "rgba(128,128,128,0.9)", [n for n in nodes if n["status"] == "rolled_back"]))
    for name, color, ns in groups:
        if not ns:
            continue
        fig.add_trace(go.Scatter(
            x=[pos[n["id"]][0] for n in ns], y=[pos[n["id"]][1] for n in ns], mode="markers", name=name,
            marker=dict(size=12, color=color, symbol=[SYMBOL.get(n["type"], "circle") for n in ns],
                        line=dict(width=1, color="rgba(255,255,255,0.6)")),
            customdata=[[n["id"][:12], n["type"], n["status"], n["channel"], (n["content"] or n["label"])[:140]]
                        for n in ns],
            hovertemplate="%{customdata[0]} · %{customdata[1]} · %{customdata[2]} · %{customdata[3]}"
                          "<br>%{customdata[4]}<extra></extra>"))
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return tidy(fig, 420)


def page_lineage(svc) -> None:
    lineage = svc["lineage"]
    st.header("Lineage & rollback (live)")
    st.caption("Every memory, the tool output it came from and the actions it fed, as a directed graph "
               f"(left to right: tool output read → memory → derived memory → action). Tenant {tenant}; the graph is "
               f"saved to {settings.LINEAGE_DB_PATH} in the directory the dashboard was started from.")
    sub = lineage.get_subgraph_data_for_tenant(tenant)
    nodes, links = sub["nodes"], sub["links"]
    if not nodes:
        st.info("No lineage recorded for this tenant yet. Run the red team, write a memory or use the agent "
                "playground first.")
        return
    counts = pd.Series([n["type"] for n in nodes]).value_counts()
    c = st.columns(4)
    c[0].metric("Nodes", len(nodes))
    c[1].metric("Edges", len(links))
    c[2].metric("Actions", int(counts.get("agent_action", 0)))
    c[3].metric("Rolled back", sum(n["status"] == "rolled_back" for n in nodes))
    linked = {l["source"] for l in links} | {l["target"] for l in links}
    if not linked:
        st.info("No memory has fed an action or a derived memory yet, so there are no links to draw; every memory is "
                "listed below. Run the agent playground with a prompt that fires its action (e.g. one asking for a "
                "financial summary) to create some.")
        shown = []
    else:
        show_all = st.toggle(f"Include the {len(nodes) - len(linked)} memories with no links", value=False)
        shown = nodes if show_all else [n for n in nodes if n["id"] in linked]
    if shown:
        st.plotly_chart(lineage_figure(shown, links), width="stretch")
        st.caption("Shape and color show the node type; gray nodes have been rolled back. Hover a node for its content.")

    left, right = st.columns([3, 2])
    with left:
        st.dataframe(pd.DataFrame(nodes)[["id", "type", "status", "channel", "content"]], hide_index=True,
                     width="stretch", height=360)
    with right:
        st.markdown("**Blast radius & rollback**")
        target = st.selectbox("Memory", [n["id"] for n in nodes if n["type"] != "agent_action"],
                              format_func=lambda i: next(f"{i[:8]} · {n['type']} · {n['content'][:40]}"
                                                         for n in nodes if n["id"] == i))
        if target:
            blast = lineage.calculate_blast_radius(target)
            st.markdown(f"Reaches **{blast['affected_nodes_count']}** nodes: "
                        f"{len(blast['derived_memories'])} derived memories, {len(blast['tainted_actions'])} actions.")
            for a in blast["tainted_actions"]:
                st.markdown(f"- action `{a['id'][:12]}`")
            st.caption("Rollback purges this memory and everything derived from it from the vector store and marks "
                       "the actions it reached as revoked. It cannot be undone.")
            if st.button("Roll back", type="primary"):
                res = lineage.rollback_memory(target)
                st.session_state["rollback_msg"] = (
                    f"Purged {len(res['purged_memory_ids'])} memories and revoked {len(res['revoked_action_ids'])} "
                    "actions." if res.get("success") else f"Rollback failed: {res.get('message')}")
                st.rerun()
            if st.session_state.get("rollback_msg"):
                st.success(st.session_state.pop("rollback_msg"))


# ---------------------------------------------------------------- live: agent playground
def page_agent(svc) -> None:
    agent = svc["agent"]
    st.header("Agent playground (live)")
    st.caption("One pass of the LangGraph demo agent: read gate → LLM → write gate → lineage. This is the "
               "original demo agent, not BankDesk: it has no tools. It writes a memory only when the prompt contains "
               "“update” or “remember”, and its one action (generate_financial_report) fires on the words "
               "“execute” in the reply or “financial summary” in the prompt, not on a model decision (Experiment 7).")
    if agent_calls_llm():
        st.info(f"Answers come from `{settings.NVIDIA_NIM_MODEL}` at `{settings.NVIDIA_NIM_API_BASE}`.")
    else:
        st.warning("No NVIDIA NIM key (nvapi-…) is configured, so the agent answers with a canned local reply, not "
                   "an LLM. Set NVIDIA_NIM_API_KEY to a real key to see model answers.")
    prompt = st.text_input("Prompt", value="What is our official company tax policy?")
    channel = st.selectbox("Channel", [c.value for c in ChannelType])
    if st.button("Run agent", type="primary"):
        with st.spinner("Running the agent graph…"):
            out = agent.run(tenant_id=tenant, user_prompt=prompt, incoming_channel=ChannelType(channel))
        source = agent.nim_client.last_source
        st.subheader("Steps")
        st.markdown("\n".join(f"1. {esc(s)}" for s in out.get("step_history", [])))
        st.subheader("Context the read gate built")
        st.code(out.get("recalled_context", ""), language=None, wrap_lines=True)
        st.subheader("Response" + (" (canned local reply, no LLM)" if source != "llm" else f" ({settings.NVIDIA_NIM_MODEL})"))
        st.markdown(esc(out.get("final_response", "")))
        if out.get("action_to_take"):
            st.markdown(f"**Action fired:** `{out['action_to_take']}` {json.dumps(out.get('action_payload', {}))}")
        if out.get("write_decision"):
            st.subheader("Write gate on the learned memory")
            st.json(out["write_decision"])


# ---------------------------------------------------------------- route
if page == "Executive scorecard":
    render_executive(st)
elif page == "Experiment results":
    render_results(st)
else:
    services = get_services()
    {"Live red team": page_red_team, "Memory inspector": page_memory, "Lineage & rollback": page_lineage,
     "Agent playground": page_agent}[page](services)
