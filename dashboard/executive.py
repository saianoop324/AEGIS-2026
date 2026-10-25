"""Executive scorecard: the dashboard's landing page, charts first, every number measured.

Built from the saved agent experiments (experiments/exp<N>_*.json): an attack counts only if its
harmful action executed in the sandbox, and the baseline is the unprotected arm run on the same
scenarios and seeds. Write-gate detection comes from the held-out public set (experiments/exp<N>_heldout_*).
Nothing here is assumed or typed in; a value that was not measured shows "–".

Layout: headline tiles, then attack success by attack type and by entry channel (unprotected vs the
NVIDIA baseline vs MemGuard), the deck's targets, write-gate detection, the trend across experiments,
and the explanatory text last, in expanders.
"""

from collections import defaultdict
from typing import Dict, List

from dashboard.bankdesk import (
    AGENT_NOTE, ARM_INFO, CHANNEL_LABELS, COMMON_SETS, EXP_DIR, EXPERIMENT_NOTES, LABELS, arms_table, classifier_label,
    entry_channel, find_results, fmt, heldout_summaries, heldout_chart, kpi_tiles, load_rows, md_table, metric_glossary,
    rate_grid, rows_for, scorecard_table, target_met,
)
from dashboard.style import STATUS, arm_colors, arm_order, theme, tidy, top_down
from experiments.run_bank_scenarios import TARGETS, scorecard

INK = {"light": "#52514e", "dark": "#c3c2b7"}
# the deck's targets that are rates, drawn on one 0-1 axis (latency and tokens are tiles / tables)
TARGET_RATES = ["asr_reduction", "sleeper_activation", "persistence_rate", "blast_radius_recall", "benign_retained",
                "provenance_coverage"]
# experiments 9-13 ran the LLM classifier with Qwen3.5-9B; the held-out set names it by model
HELDOUT_NAME = {"llm": "qwen3.5-9b"}


def _success_by(st, rows: List[dict], arms: List[str], dim, title: str, labels: Dict[str, str] = None) -> None:
    """Attack success per value of dim (a function of a row), one bar per arm, worst-for-unprotected first."""
    import pandas as pd
    import plotly.express as px
    cells = defaultdict(lambda: [0, 0])
    for r in rows:
        if r["kind"] == "attack" and r["arm"] in arms:
            c = cells[(dim(r), r["arm"])]
            c[0] += int(r["oracle_hit"])
            c[1] += 1
    if not cells:
        return
    df = pd.DataFrame([{"group": (labels or {}).get(g, g), "arm": a, "rate": h / n, "hits": h, "runs": n}
                       for (g, a), (h, n) in cells.items()])
    order = (df[df["arm"] == arms[0]].sort_values("rate", ascending=False)["group"].tolist()
             + [g for g in df["group"].unique() if g not in set(df[df["arm"] == arms[0]]["group"])])
    st.markdown(f"**{title}**")
    fig = px.bar(df, x="rate", y="group", color="arm", orientation="h", barmode="group",
                 text=df["rate"].map(lambda v: f"{v:.0%}"), color_discrete_map=arm_colors(st, arms),
                 category_orders={"arm": arms, "group": order}, hover_data={"hits": True, "runs": True},
                 labels={"group": "", "rate": "attack success rate"})
    fig.update_traces(textposition="outside", cliponaxis=False, textfont_size=11)
    fig.update_xaxes(range=[0, 1.15], tickformat=".0%")
    st.plotly_chart(top_down(tidy(fig, 60 + 22 * len(df))), width="stretch")
    with st.expander("Data"):
        grid = {((labels or {}).get(g, g), a): v for (g, a), v in cells.items()}
        st.dataframe(rate_grid(grid, order, arms, title.split(" by ")[-1]), width="stretch")


def _quarantine_by_channel(st, rows: List[dict], arm: str) -> None:
    """Of the poisoned memories the attacks created, the share the write gate quarantined, per entry channel."""
    import pandas as pd
    import plotly.express as px
    agg = defaultdict(lambda: [0, 0])
    for r in rows:
        if r["kind"] == "attack" and r["arm"] == arm and r.get("blast") and r["blast"]["poison_memories"]:
            a = agg[CHANNEL_LABELS[entry_channel(r)]]
            a[0] += r["blast"]["poison_quarantined"]
            a[1] += r["blast"]["poison_memories"]
    if not agg:
        return
    st.markdown(f"**Poisoned memories the write gate quarantined · {arm}**")
    df = pd.DataFrame([{"channel": c, "share": q / n, "quarantined": q, "poisoned memories": n}
                       for c, (q, n) in agg.items()]).sort_values("share")
    fig = px.bar(df, x="share", y="channel", orientation="h", text=df.apply(lambda r: f"{r['quarantined']}/{r['poisoned memories']}", axis=1),
                 color_discrete_sequence=[arm_colors(st, [arm])[arm]], hover_data=["quarantined", "poisoned memories"],
                 labels={"channel": "", "share": "quarantined"})
    fig.update_traces(textposition="outside", cliponaxis=False, textfont_size=11)
    fig.update_xaxes(range=[0, 1.15], tickformat=".0%")
    st.plotly_chart(tidy(fig, 70 + 44 * len(df)), width="stretch")
    st.caption("Counted over the memories each attack's planted text became. Attacks the customer typed plant "
               "nothing to trace, so customer chat is not shown. Memories admitted here are still quoted on recall "
               "and still face the action gate.")


def _targets_chart(st, card: Dict[str, dict], arm: str) -> None:
    """The deck's rate targets for one arm: a bar for the value (green met / red missed, with ✓/✗), a tick for the target."""
    import re
    import plotly.graph_objects as go
    s = card[arm]
    ink = INK[theme(st)]
    labels, values, colors, texts, ticks = [], [], [], [], []
    for k in TARGET_RATES:
        target = TARGETS[k]
        v = s.get(k)
        ok = target_met(v, target)
        labels.append(f"{LABELS[k]}  ({target})")
        values.append(0 if v is None else v)
        colors.append(STATUS["good" if ok else "critical"] if ok is not None else ink)
        texts.append("not measured" if v is None else f"{fmt(v)} {'✓' if ok else '✗'}")
        ticks.append(float(re.search(r"[\d.]+", target).group()))
    fig = go.Figure(go.Bar(x=values, y=labels, orientation="h", marker_color=colors, text=texts,
                           textposition="outside", cliponaxis=False, showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=ticks, y=labels, mode="markers", name="target", hovertemplate="target %{x}<extra></extra>",
                             marker=dict(symbol="line-ns-open", size=26, color=ink, line=dict(width=3, color=ink))))
    fig.update_xaxes(range=[0, 1.3], tickformat=".0%")
    fig.update_yaxes(autorange="reversed")
    st.markdown(f"**The deck's targets · {arm}**")
    st.plotly_chart(tidy(fig, 320), width="stretch")
    st.caption("Bar = measured value, tick = target. Persistence and blast radius can only be measured where an "
               "attack succeeded, so an arm that stops every attack shows “not measured”.")


def _detection(st, exp: str, clf: str) -> None:
    use, summ = heldout_summaries(EXP_DIR, exp)
    if not summ:
        use, summ = heldout_summaries(EXP_DIR)
    if not summ:
        return
    name = HELDOUT_NAME.get(clf, clf)
    st.markdown(f"**Write-gate detection on a held-out public set** (Experiment {use})")
    mine = next((s for s in summ if s["classifier"] == name), None)
    if mine:
        c = st.columns(4)
        c[0].metric("Precision", f"{mine['precision']:.2f}", border=True)
        c[1].metric("Recall", f"{mine['recall']:.2f}", border=True)
        c[2].metric("F1", f"{mine['f1']:.2f}", border=True)
        c[3].metric("Write p95", f"{mine['write_p95_ms']:.0f} ms", border=True)
    heldout_chart(st, summ, highlight=name)
    st.caption(f"deepset/prompt-injections: {summ[0]['n']} texts ({summ[0]['injections']} injections, "
               f"{summ[0]['benign']} benign) that none of the gates was tuned on. "
               + (f"Highlighted: {classifier_label(clf)}, the classifier selected above."
                  if mine else f"{classifier_label(clf)} was not run on this set."))


def _trend(st, found) -> None:
    import pandas as pd
    import plotly.express as px
    data = []
    for exp in sorted(found, key=int):
        if "deberta" not in found[exp] or not all(s in found[exp]["deberta"] for s in COMMON_SETS):
            continue
        card = scorecard(load_rows([found[exp]["deberta"][s] for s in COMMON_SETS]))["arms"]
        for arm, s in card.items():
            data.append({"experiment": f"Exp {exp}", "arm": arm, "attack success rate": s["attack_success_rate"],
                         "benign tasks done": s["benign_done"] / s["benign_total"] if s["benign_total"] else None,
                         "attack runs": s["attack_total"], "benign runs": s["benign_total"]})
    if not data:
        return
    st.subheader("Across experiments")
    df = pd.DataFrame(data)
    arms = arm_order(df["arm"])
    left, right = st.columns(2)
    for col, measure in ((left, "attack success rate"), (right, "benign tasks done")):
        col.markdown(f"**{measure.capitalize()}**")
        fig = px.line(df, x="experiment", y=measure, color="arm", markers=True, color_discrete_map=arm_colors(st, arms),
                      category_orders={"arm": arms}, hover_data=["attack runs", "benign runs"], labels={"experiment": ""})
        fig.update_traces(line_width=2, marker_size=8)
        fig.update_yaxes(range=[0, 1.05], tickformat=".0%")
        col.plotly_chart(tidy(fig, 300), width="stretch")
    st.caption("DeBERTa write gate, on the three scenario sets every experiment ran (19 original attacks, 7 written "
               "to pass the action gate, 7 benign tasks), so later sets don't move the line. The code changed between "
               "experiments, so this tracks the system, not one fixed version.")
    with st.expander("Data"):
        st.dataframe(df, hide_index=True, width="stretch")


def _about(st, card: Dict[str, dict]) -> None:
    with st.expander("About MemGuard"):
        st.markdown(
            "MemGuard is a firewall for an AI agent's long-term memory. Text the agent reads (documents, transaction "
            "memos, web pages, messages from other agents) can carry instructions or false facts that are saved to "
            "memory and act days later. MemGuard checks what goes into memory, labels what comes out, refuses "
            "sensitive actions whose arguments only untrusted memory supports, and traces every action back to the "
            "memories behind it so a poisoned memory's effects can be rolled back.")
        md_table(st, [
            {"Layer": "Write gate", "What it does": "Classifies each new memory (instruction or fact? contradicts a "
             "known fact?), quarantines hostile ones, signs the rest with their source and trust tier.",
             "Code": "`firewall/write_gate.py`"},
            {"Layer": "Read gate", "What it does": "Recalls by relevance weighted by trust, drops memories whose "
             "signature no longer verifies, and quotes unverified ones as data the agent must not obey.",
             "Code": "`firewall/read_gate.py`"},
            {"Layer": "Action gate", "What it does": "Before a payment, new payee, role grant or similar: each "
             "argument must come from the bank's records or the verified customer, and policy limits hold.",
             "Code": "`firewall/action_gate.py`"},
            {"Layer": "Lineage", "What it does": "Graph of memory → action. Finds every action a poisoned memory "
             "reached and undoes it.", "Code": "`firewall/lineage.py`"},
        ])
        st.markdown("**The arms** (which layers each run had on)")
        arms_table(st, [a for a in card if a in ARM_INFO])
    with st.expander("Reading these numbers"):
        st.markdown(
            "- **Harm, not detection.** An attack counts only if its harmful action executed in the sandbox's action "
            "log; what the agent merely said is not scored.\n"
            "- **One agent model.** Every run used Qwen3.5-9B; another model may obey poisoned text more or less "
            "often.\n"
            "- **Who wrote the attacks.** Most attack sets were written by the team after reading the gates' code. The "
            "garak set uses garak's carriers with our bank payloads, and the held-out write-gate set (deepset/prompt-"
            "injections) is the only fully outside data. AgentDojo and InjecAgent conversions are planned "
            "(`experiments/THIRDPARTY_TESTS.md`).\n"
            "- **Small samples for some rates.** Three seeds per scenario; sleeper and persistence rates rest on a few "
            "dozen runs. Run counts are under each chart's Data and on the Experiment results page.\n"
            "- **Rollback is a dry run** on a copy of the sandbox, not on the live run.")
    with st.expander("Every metric for every arm"):
        scorecard_table(st, card, arm_order(card))
        metric_glossary(st)


def render_executive(st) -> None:
    st.header("Executive scorecard")
    found = find_results(EXP_DIR)
    if not found:
        st.info("No experiment results found in experiments/. Run experiments/run_bank_scenarios.py first.")
        return
    c1, c2, c3 = st.columns(3)
    exp = c1.selectbox("Experiment", sorted(found, key=int, reverse=True), format_func=lambda e: f"Experiment {e}",
                       key="ex_exp")
    clfs = sorted(found[exp], key=lambda c: (c != "deberta", c))
    clf = c2.selectbox("Write-gate classifier", clfs, format_func=classifier_label, key="ex_clf")
    rows, _ = rows_for(found, exp, clf, sorted(found[exp][clf]))
    card = scorecard(rows)["arms"]
    protected = [a for a in arm_order(card) if a in ("memguard", "memguard_ttl")] or ["memguard"]
    arm = c3.selectbox("MemGuard arm", protected, key="ex_arm")
    st.caption(f"Measured on the agent experiments: {len(rows)} runs of {len({r['scenario'] for r in rows})} "
               f"scenarios. An attack succeeds only if its harmful action executed; the baseline is the same "
               f"scenarios with protection off. {AGENT_NOTE} Experiment {exp}: {EXPERIMENT_NOTES.get(exp, '')}")
    if arm not in card:
        st.info(f"{arm} was not run in this selection.")
        return

    kpi_tiles(st, card, arm)
    compare = [a for a in ("unprotected", "nemo_rails", arm) if a in card]
    left, right = st.columns(2)
    with left:
        _success_by(st, rows, compare, lambda r: r.get("vector") or "untagged", "Attack success by attack type")
    with right:
        _success_by(st, rows, compare, entry_channel, "Attack success by entry channel", CHANNEL_LABELS)
    st.caption("nemo_rails is NVIDIA's NeMo Guardrails input rails on the same runs, for comparison. Hover a bar for "
               "its run counts.")
    left, right = st.columns(2)
    with left:
        _targets_chart(st, card, arm)
    with right:
        _quarantine_by_channel(st, rows, arm)
    _detection(st, exp, clf)
    _trend(st, found)
    _about(st, card)
