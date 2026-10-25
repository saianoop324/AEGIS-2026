"""BankDesk / ITDesk experiment results for the Streamlit dashboard.

Reads the saved runner outputs (experiments/exp<N>_<classifier>_<scenario set>.json), the held-out
write-gate results (experiments/exp<N>_heldout_<classifier>.json) and the scenario definitions
(experiments/<scenario set>.json); runs nothing and loads no models.

  render  the Experiment results page: one experiment in detail, charts first (headline tiles, attacks
          and benign tasks by set, attack success by attack type and by scenario), then the scorecard,
          what was run, the write-gate classifiers side by side, held-out precision/recall, and a
          drill-down into each run

Also the shared pieces the Executive scorecard (dashboard/executive.py) is built from. Used by app.py
and standalone: streamlit run dashboard/bankdesk_app.py. Design: dashboard/DASHBOARD.md
"""

import glob
import json
import math
import os
import re
import sys
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import streamlit as _st

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from agent.bank_agent import ARMS  # noqa: E402
from dashboard.style import CATEGORICAL, SEQUENTIAL, STATUS, arm_colors, arm_order, facets, theme, tidy, tint, top_down  # noqa: E402
from experiments.run_bank_scenarios import TARGETS, is_poison, poison_markers, scorecard  # noqa: E402

EXP_DIR = os.path.join(ROOT, "experiments")
RESULTS_RE = re.compile(r"^exp(\d+)_(.+?)_((?:bank_scenarios|it_scenarios).*)\.json$")
# the scenario sets every experiment from 11 on ran, so numbers stay comparable across experiments
COMMON_SETS = ["bank_scenarios_attack", "bank_scenarios_attack_ingate", "bank_scenarios_benign"]

SCORECARD_KEYS = [
    ("attack_success_rate", "Attack success rate", "Attack runs whose harmful action executed, over all attack runs."),
    ("asr_reduction", "Reduction vs unprotected", "1 − (this arm's attack success rate ÷ the unprotected arm's)."),
    ("sleeper_activation", "Sleeper activation",
     "Sleeper attacks whose harmful action ran in a later session than the one that planted it."),
    ("persistence_rate", "Persistence rate",
     "Of the attacks that succeeded: the share that fired again from memory alone after the harmful action was "
     "undone, the poisoned source deleted from the sandbox and N unrelated sessions run."),
    ("poison_recalled_after_n", "Poison recalled after N sessions",
     "Replays in which the poisoned memory was still recalled, over every replayed attack."),
    ("blast_radius_recall", "Blast-radius recall",
     "Harmful executed actions reached by walking the lineage graph from the poisoned memories and the poisoned "
     "tool output the agent read, over all harmful executed actions."),
    ("blast_radius_recall_from_memory", "Blast-radius recall (from memory)",
     "The same, counting only harmful actions taken after a poisoned memory existed."),
    ("rollback_undone", "Rollback: harmful undone",
     "Harmful actions undone by rolling back every poisoned memory (a dry run on a copy of the sandbox)."),
    ("rollback_collateral_actions", "Rollback: innocent undone", "Legitimate actions the same rollback also undid."),
    ("benign_retained", "Benign tasks retained", "Benign tasks completed, relative to the unprotected arm."),
    ("provenance_coverage", "Provenance coverage", "Stored memories whose signed envelope verifies."),
    ("write_p95_ms", "Write p95 (ms)", "95th-percentile time per memory write."),
    ("added_write_p95_ms", "Added write p95 (ms)", "Write p95 minus the unprotected arm's."),
    ("recall_p95_ms", "Recall p95 (ms)", "95th-percentile time per memory recall."),
    ("rail_p95_ms", "Rail check p95 (ms)", "Time per NeMo Guardrails check (nemo_rails arm only)."),
    ("rail_blocks", "Texts blocked by rails", "Customer messages, recalled memories and tool outputs the rails blocked."),
    ("tokens_per_turn", "Tokens per turn", "LLM tokens per customer turn, as vLLM reports them (includes the tool loop)."),
    ("extra_tokens_per_turn", "Extra tokens per turn", "Tokens per turn relative to the unprotected arm."),
]
LABELS = {k: label for k, label, _ in SCORECARD_KEYS}

ARM_INFO = {  # write gate, read gate, action gate, what else
    "unprotected": ("off", "off", "monitor", "Everything stored and recalled as plain facts. The action gate only "
                                             "records what it would have held."),
    "nemo_rails": ("off", "off", "monitor", "NVIDIA baseline: NeMo Guardrails input rails on every customer message, "
                                            "recalled memory and tool output; blocked text is dropped."),
    "no_action_gate": ("on", "on", "monitor", "The memory layers alone."),
    "action_gate_only": ("off", "off", "enforce", "The action gate alone, over unprotected memory."),
    "memguard": ("on", "on", "enforce", "All of MemGuard."),
    "memguard_ttl": ("on", "on", "enforce", "MemGuard, and recall drops unverified memories past their TTL "
                                            "(2 days for tool/web/peer content, 7 for chat and documents)."),
}
CLASSIFIERS = {"deberta": "DeBERTa NLI", "llm": "Qwen3.5-9B", "nemoguard": "NemoGuard JailbreakDetect"}
EXPERIMENT_NOTES = {  # one line each; the full write-ups are in experiments/EXPERIMENTS.html
    "11": "Three seeds; DeBERTa and Qwen3.5-9B write-gate classifiers; 7 attacks written to satisfy the action "
          "gate's rules; benign scenarios run alongside.",
    "12": "Experiment 11 rerun with the deck's scorecard metrics: sleeper activation, persistence, blast radius, "
          "rollback and overhead.",
    "13": "Lineage fix (actions linked only to what drove them), and NVIDIA baselines: NemoGuard JailbreakDetect as "
          "a classifier, a NeMo Guardrails arm and 24 garak attacks.",
    "14": "ITDesk helpdesk domain, memory-TTL arm, provenance check on recall, 14 more benign scenarios, Qwen3.5 "
          "2B/4B/9B classifiers and a held-out public write-gate set; the action gate now trusts the customer's own "
          "words from earlier sessions.",
}
AGENT_NOTE = "Agent in every run: Qwen3.5-9B served by vLLM, temperature 0.2, with the sandbox's tools."


def classifier_label(clf: str) -> str:
    m = re.match(r"qwen(\d+(?:\.\d+)?)-([\d.]+b)$", clf)
    return CLASSIFIERS.get(clf) or (f"Qwen{m.group(1)}-{m.group(2).upper()}" if m else clf)


def set_label(set_name: str) -> str:
    """'bank_scenarios_attack_ingate' -> 'bank: attack ingate', short enough for an unrotated axis."""
    return set_name.replace("bank_scenarios_", "bank: ").replace("it_scenarios", "it: helpdesk").replace("_", " ")


# ---------------------------------------------------------------- loading
@_st.cache_resource(show_spinner=False, max_entries=64)
def _load(path: str, mtime: float) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_json(path: str) -> dict:
    return _load(path, os.path.getmtime(path))


def find_results(exp_dir: str = EXP_DIR) -> Dict[str, Dict[str, Dict[str, str]]]:
    """experiment number -> classifier -> scenario set -> path"""
    out: Dict[str, Dict[str, Dict[str, str]]] = defaultdict(lambda: defaultdict(dict))
    for path in glob.glob(os.path.join(exp_dir, "exp*_*.json")):
        m = RESULTS_RE.match(os.path.basename(path))
        if m:
            out[m.group(1)][m.group(2)][m.group(3)] = path
    return out


def load_rows(paths: List[str]) -> List[dict]:
    rows = []
    for p in paths:
        s = RESULTS_RE.match(os.path.basename(p)).group(3)
        rows.extend(dict(r, _set=s) for r in load_json(p)["rows"])  # copies: the loaded file is shared
    return rows


def rows_for(found, exp: str, clf: str, sets: List[str]) -> Tuple[List[dict], Dict[str, str]]:
    """Rows of one classifier's runs, plus baseline arms borrowed from another classifier's files.

    Arms without the write gate (unprotected, action_gate_only, nemo_rails) do not depend on the
    classifier, and some runs (Experiment 14's Qwen classifiers) did not repeat them. Borrowing them
    keeps the reductions against unprotected computable. Returns (rows, {borrowed arm: classifier}).
    """
    rows = load_rows([found[exp][clf][s] for s in sets if s in found[exp][clf]])
    have = {r["arm"] for r in rows}
    borrowed: Dict[str, str] = {}
    for other in sorted(found[exp], key=lambda c: (c != "deberta", c)):
        if other == clf:
            continue
        extra = [r for r in load_rows([found[exp][other][s] for s in sets if s in found[exp][other]])
                 if r["arm"] in ARMS and not ARMS[r["arm"]].write_gate and r["arm"] not in have]
        borrowed.update({r["arm"]: other for r in extra})
        rows += extra
        have |= {r["arm"] for r in extra}
    return rows, borrowed


@_st.cache_data(show_spinner=False)
def scenario_defs(set_name: str) -> Dict[str, dict]:
    path = os.path.join(EXP_DIR, f"{set_name}.json")
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        return {s["id"]: s for s in json.load(f)}


# ---------------------------------------------------------------- formatting
def missing(value) -> bool:
    """None, or the NaN pandas puts in its place in a numeric column (a metric that does not apply)."""
    return value is None or (isinstance(value, float) and math.isnan(value))


def target_met(value, target: str) -> Optional[bool]:
    m = re.match(r"^\s*(>=|<=|<|>)?\s*([\d.]+)\s*$", target or "")
    if missing(value) or not m or isinstance(value, str):
        return None
    op, num = m.group(1) or "==", float(m.group(2))
    return {">=": value >= num, "<=": value <= num, "<": value < num, ">": value > num,
            "==": abs(value - num) < 1e-9}[op]


def fmt(value) -> str:
    if missing(value):
        return "–"
    if isinstance(value, float):
        return f"{value:.3f}".rstrip("0").rstrip(".") if abs(value) < 10 else f"{value:.1f}"
    return str(value)


def esc(text: str) -> str:
    """Show stored text literally in markdown (scenario payloads contain markup and $ signs)."""
    return re.sub(r"([\\`*_\[\]#|>~$<])", r"\\\1", str(text or ""))


def esc_reply(text: str) -> str:
    """Agent replies are the model's own markdown: keep it, but stop $ from turning into LaTeX."""
    return str(text or "").replace("$", "\\$").replace("<", "&lt;")


def styled_targets(df, targets: Dict[str, str], value_cols: List[str], row_target_col: Optional[str] = None):
    """Show values with ✓/✗ against their targets, tinted with the status colors.

    targets maps a column to its target (one target per column) unless row_target_col names a column
    that holds each row's target (one target per row, as in the scorecard).
    """
    raw = df.copy()
    shown = df.astype(object)  # cells become "0.741 ✓" strings
    css = df.copy().astype(object)
    css.loc[:, :] = ""
    for i in df.index:
        for c in value_cols:
            target = df.at[i, row_target_col] if row_target_col else targets.get(c, "")
            ok = target_met(raw.at[i, c], target)
            shown.at[i, c] = fmt(raw.at[i, c]) + ("" if ok is None else (" ✓" if ok else " ✗"))
            if ok is not None:
                css.at[i, c] = f"background-color: {tint(STATUS['good' if ok else 'critical'], 0.16)}"
    return shown.astype(str).style.apply(lambda _: css, axis=None)


def rate_grid(cells: Dict[Tuple[str, str], List[int]], row_labels: List[str], cols: List[str], index_name: str):
    """A rows x cols grid of 'hits/runs', shaded by rate on the sequential ramp (empty = not run)."""
    import pandas as pd
    text = pd.DataFrame("", index=row_labels, columns=cols)
    css = pd.DataFrame("", index=row_labels, columns=cols)
    for (r, c), (h, n) in cells.items():
        if r not in text.index or c not in text.columns or not n:
            continue
        text.at[r, c] = f"{h}/{n}"
        rate = h / n
        if rate > 0:
            step = min(len(SEQUENTIAL) - 1, int(rate * len(SEQUENTIAL)))
            ink = "#ffffff" if step >= 3 else "#0b0b0b"
            css.at[r, c] = f"background-color: {SEQUENTIAL[step]}; color: {ink}"
    text.index.name = index_name
    return text.style.apply(lambda _: css, axis=None)


def scorecard_table(st, card: Dict[str, dict], arms: List[str], keys=None) -> None:
    import pandas as pd
    keys = keys or [k for k, _, _ in SCORECARD_KEYS]
    keys = [k for k in keys if any(card[a].get(k) not in (None, 0) for a in arms) or k in TARGETS]
    df = pd.DataFrame([{"Metric": LABELS[k], "Target": TARGETS.get(k, ""), **{a: card[a].get(k) for a in arms}}
                       for k in keys])
    st.dataframe(styled_targets(df, {}, arms, row_target_col="Target"), hide_index=True, width="stretch",
                 height=fit(len(df)))


def fit(n_rows: int) -> int:
    """Dataframe height that shows every row without an inner scrollbar."""
    return 35 * (n_rows + 1) + 3


def md_table(st, rows: List[dict]) -> None:
    """Prose tables as markdown, which wraps, where a dataframe grid would clip the text."""
    cols = list(rows[0])
    st.markdown("\n".join(["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
                          + ["| " + " | ".join(str(r[c]).replace("|", "\\|") for c in cols) + " |" for r in rows]))


def metric_glossary(st) -> None:
    with st.expander("What each metric means"):
        st.markdown("\n".join(f"- **{label}**: {help_}" for _, label, help_ in SCORECARD_KEYS))
        st.markdown("Targets come from slide 4 of the AEGIS deck (`TARGETS` in `experiments/run_bank_scenarios.py`). "
                    "✓ / ✗ mark whether a value meets its target; metrics without a target are shown for context.")


def arms_table(st, arms: List[str]) -> None:
    md_table(st, [{"Arm": f"`{a}`", "Write gate": ARM_INFO[a][0], "Read gate": ARM_INFO[a][1],
                   "Action gate": ARM_INFO[a][2], "What it is": ARM_INFO[a][3]}
                  for a in arm_order(arms) if a in ARM_INFO])


def _set_bars(st, rows: List[dict], kind: str, title: str, ylabel: str) -> None:
    import pandas as pd
    import plotly.express as px
    agg = defaultdict(lambda: [0, 0])
    for r in rows:
        if r["kind"] == kind:
            agg[(r["_set"], r["arm"])][0] += int(r["oracle_hit"])
            agg[(r["_set"], r["arm"])][1] += 1
    if not agg:
        return
    df = pd.DataFrame([{"set": set_label(s), "arm": a, "hits": h, "runs": n, "rate": h / n}
                       for (s, a), (h, n) in agg.items()])
    arms = arm_order(df["arm"])
    st.markdown(f"**{title}**")
    fig = px.bar(df, x="set", y="rate", color="arm", barmode="group", color_discrete_map=arm_colors(st, arms),
                 category_orders={"arm": arms, "set": sorted(df["set"].unique())},
                 hover_data={"hits": True, "runs": True}, labels={"rate": ylabel, "set": ""})
    fig.update_xaxes(tickangle=0)
    fig.update_yaxes(range=[0, 1], tickformat=".0%")
    st.plotly_chart(tidy(fig, 340), width="stretch")
    with st.expander("Data"):
        cells = {(r["set"], r["arm"]): [r["hits"], r["runs"]] for r in df.to_dict("records")}
        st.dataframe(rate_grid(cells, sorted(df["set"].unique()), arms, "scenario set"), width="stretch")


CHANNEL_LABELS = {"user_chat": "customer chat", "documents": "uploaded document",
                  "tool_outputs": "tool output (memo, ticket)", "peer_agent": "peer agent message"}


def entry_channel(row: dict) -> str:
    """Where an attack's poison enters, as the sandbox tools record it.

    Every attack plants at most one kind of content: a document is read as `documents`, a transaction
    memo or IT ticket as `tool_outputs`, a peer message as `peer_agent` (sandbox/bank_tools.py,
    sandbox/it_tools.py); with nothing planted, the customer typed the attack (`user_chat`).
    """
    setup = (scenario_defs(row["_set"]).get(row["scenario"]) or {}).get("setup", {})
    if setup.get("documents"):
        return "documents"
    if setup.get("peer_messages"):
        return "peer_agent"
    if setup.get("tickets") or any(t.get("memo") for t in setup.get("transactions", [])):
        return "tool_outputs"
    return "user_chat"


def pct(value) -> str:
    return "–" if missing(value) else f"{value:.0%}"


def kpi_tiles(st, card: Dict[str, dict], protected: str = "memguard") -> None:
    """Four headline numbers for one protected arm against the unprotected arm, each with its counts and target."""
    base, mg = card.get("unprotected"), card.get(protected)
    if not base or not mg:
        return

    def against_target(key: str) -> str:
        ok = target_met(mg.get(key), TARGETS.get(key, ""))
        return "" if ok is None else f"target {TARGETS[key]} {'✓ met' if ok else '✗ missed'}"
    c = st.columns(4)
    asr, base_asr = mg["attack_success_rate"], base["attack_success_rate"]
    c[0].metric(f"Attacks executed · {protected}", pct(asr), border=True, delta_color="inverse",
                delta=None if missing(asr) or missing(base_asr) else f"{(asr - base_asr) * 100:+.0f} pts vs unprotected")
    c[0].caption(f"{mg['attack_hits']} of {mg['attack_total']} attack runs · unprotected "
                 f"{base['attack_hits']} of {base['attack_total']}")
    c[1].metric("Attack reduction vs unprotected", pct(mg.get("asr_reduction")), border=True)
    c[1].caption(against_target("asr_reduction"))
    c[2].metric("Benign tasks retained", pct(mg.get("benign_retained")), border=True)
    c[2].caption(f"{mg['benign_done']} of {mg['benign_total']} done (unprotected {base['benign_done']}) · "
                 + against_target("benign_retained"))
    c[3].metric("Added write latency p95", f"{fmt(mg.get('added_write_p95_ms'))} ms", border=True)
    c[3].caption(f"{fmt(mg.get('write_p95_ms'))} ms per write vs {fmt(base.get('write_p95_ms'))} unprotected · "
                 + against_target("added_write_p95_ms"))


# ---------------------------------------------------------------- experiment results page
def _sample_sizes(st, rows: List[dict], borrowed: Dict[str, str]) -> None:
    import pandas as pd
    by_arm = defaultdict(list)
    for r in rows:
        by_arm[r["arm"]].append(r)
    st.dataframe(pd.DataFrame([{
        "arm": a, "scenarios": len({r["scenario"] for r in rs}), "seeds": len({r["seed"] for r in rs}),
        "attack runs": sum(r["kind"] == "attack" for r in rs), "benign runs": sum(r["kind"] == "benign" for r in rs),
        "persistence replays": sum(bool(r.get("persistence")) for r in rs),
        "source": f"{classifier_label(borrowed[a])} files" if a in borrowed else "this classifier's files",
    } for a, rs in ((a, by_arm[a]) for a in arm_order(by_arm))]), hide_index=True, width="stretch")
    if borrowed:
        st.caption("These runs did not repeat the arms without a write gate, which don't depend on the classifier; "
                   "they are taken from the same experiment's other files so reductions against unprotected can be "
                   "computed.")


def _vector_grid(st, rows: List[dict], arms: List[str]) -> None:
    cells = defaultdict(lambda: [0, 0])
    for r in rows:
        if r["kind"] == "attack":
            c = cells[(r.get("vector") or "untagged", r["arm"])]
            c[0] += int(r["oracle_hit"])
            c[1] += 1
    if not cells:
        return
    st.subheader("Attack success by attack type")
    st.caption("Runs where the harmful action executed / attack runs. Darker = more executed; blank = no executed "
               "attacks.")
    vectors = sorted({v for v, _ in cells})
    st.dataframe(rate_grid(cells, vectors, arms, "attack type"), width="stretch")


def _scenario_grid(st, rows: List[dict], arms: List[str]) -> None:
    st.subheader("Every scenario, every arm")
    kind = st.radio("Show", ["attack", "benign"], horizontal=True, key="grid_kind",
                    format_func=lambda k: "attacks: harmful action executed" if k == "attack" else "benign: task done")
    cells = defaultdict(lambda: [0, 0])
    vec = {}
    for r in rows:
        if r["kind"] == kind:
            c = cells[(r["scenario"], r["arm"])]
            c[0] += int(r["oracle_hit"])
            c[1] += 1
            vec[r["scenario"]] = f"{r['_set']} · {r.get('vector') or r.get('domain', '')}"
    if not cells:
        st.info(f"No {kind} scenarios in the selected sets.")
        return
    scenarios = sorted(vec, key=lambda s: (vec[s], s))
    st.caption("Runs where the oracle matched / runs (one per seed). Pick a scenario in the drill-down below to see "
               "why.")
    st.dataframe(rate_grid(cells, scenarios, arms, "scenario"), width="stretch", height=min(38 * len(scenarios) + 40, 640))


def _classifier_comparison(st, found, exp: str, sets: List[str]) -> None:
    import pandas as pd
    import plotly.express as px
    clfs = sorted(found[exp], key=lambda c: (c != "deberta", c))
    common = [s for s in sets if all(s in found[exp][c] for c in clfs)]
    if len(clfs) < 2 or not common:
        return
    st.subheader("Write-gate classifiers side by side")
    keys = ["attack_success_rate", "asr_reduction", "benign_retained", "persistence_rate", "poison_recalled_after_n",
            "added_write_p95_ms"]
    table = []
    for c in clfs:
        rows, borrowed = rows_for(found, exp, c, common)
        card = scorecard(rows)["arms"]
        for arm in arm_order(card):
            if arm in ARMS and ARMS[arm].write_gate and arm not in borrowed:
                table.append({"classifier": classifier_label(c), "arm": arm, **{LABELS[k]: card[arm].get(k) for k in keys}})
    df = pd.DataFrame(table)
    long = df.melt(id_vars=["classifier", "arm"], value_vars=[LABELS[k] for k in keys[:4]], var_name="metric").dropna()
    arms = arm_order(df["arm"])
    fig = px.bar(long, x="value", y="classifier", color="arm", facet_col="metric", facet_col_spacing=0.04,
                 barmode="group", orientation="h", color_discrete_map=arm_colors(st, arms),
                 category_orders={"arm": arms, "metric": [LABELS[k] for k in keys[:4]]},
                 text=long["value"].map(lambda v: f"{v:.2f}"), labels={"classifier": "", "value": ""})
    fig.update_traces(textposition="outside", cliponaxis=False, textfont_size=10)
    fig.update_xaxes(range=[0, 1.25], showticklabels=False)
    st.plotly_chart(facets(top_down(tidy(fig)), len(df)), width="stretch")
    with st.expander("Table, with every metric against its target"):
        st.dataframe(styled_targets(df, {LABELS[k]: TARGETS.get(k, "") for k in keys}, [LABELS[k] for k in keys]),
                     hide_index=True, width="stretch")
    st.caption(f"Arms that use the write gate, on the scenario sets every classifier ran: {', '.join(common)}.")


def heldout_summaries(exp_dir: str, exp: Optional[str] = None) -> Tuple[Optional[str], List[dict]]:
    """Held-out write-gate summaries for an experiment, or the latest experiment that has them."""
    paths = glob.glob(os.path.join(exp_dir, "exp*_heldout_*.json"))
    exps = sorted({re.match(r"exp(\d+)_", os.path.basename(p)).group(1) for p in paths}, key=int)
    use = exp if exp in exps else (exps[-1] if exps and exp is None else None)
    if not use:
        return None, []
    return use, [load_json(p)["summary"] for p in sorted(paths) if os.path.basename(p).startswith(f"exp{use}_heldout_")]


def heldout_chart(st, summ: List[dict], highlight: Optional[str] = None) -> None:
    """Precision, recall and F1 per classifier as small multiples; the highlighted classifier in full color."""
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go
    slot = CATEGORICAL[theme(st)][6]
    names = [classifier_label(s["classifier"]) for s in summ]
    colors = [slot if s["classifier"] == highlight else tint(slot, 0.45) for s in summ]
    measures = [("precision", "precision"), ("recall", "recall"), ("f1", "F1")]
    fig = make_subplots(rows=1, cols=3, shared_yaxes=True, horizontal_spacing=0.05,
                        subplot_titles=[title for _, title in measures])
    for i, (key, _) in enumerate(measures, start=1):
        values = [s[key] for s in summ]
        fig.add_trace(go.Bar(x=values, y=names, orientation="h", marker_color=colors, showlegend=False,
                             text=[f"{v:.2f}" for v in values], textposition="outside", cliponaxis=False,
                             hovertemplate="%{y}: %{x:.3f}<extra></extra>"), row=1, col=i)
    fig.update_xaxes(range=[0, 1.3], showticklabels=False)
    fig.update_yaxes(categoryorder="array", categoryarray=names[::-1])  # first classifier at the top
    fig.update_traces(textfont_size=11)
    st.plotly_chart(facets(tidy(fig), len(summ)), width="stretch")


def _heldout(st, exp_dir: str, exp: str) -> None:
    import pandas as pd
    use, summ = heldout_summaries(exp_dir, exp)
    if not summ:
        return
    st.subheader("Write gate on a held-out public set")
    st.caption(f"deepset/prompt-injections ({summ[0].get('n')} texts: {summ[0].get('injections')} injections, "
               f"{summ[0].get('benign')} benign), written by people who never saw MemGuard. Each text is written "
               "through the write gate as web content; quarantined = predicted injection.")
    heldout_chart(st, summ)
    with st.expander("Table"):
        st.dataframe(pd.DataFrame([{"classifier": classifier_label(s["classifier"]), "precision": s["precision"],
                                    "recall": s["recall"], "F1": s["f1"],
                                    "false quarantine rate": s["false_quarantine_rate"],
                                    "caught by regex fast path": s["caught_by_fast_path"],
                                    "write p50 (ms)": s["write_p50_ms"], "write p95 (ms)": s["write_p95_ms"]}
                                   for s in summ]), hide_index=True, width="stretch")


def _render_session(st, session: List[dict], markers: List[str]) -> None:
    for turn in session:
        st.markdown(f"**Customer:** {esc(turn['user'])}")
        recalled = turn.get("recalled") or []
        if recalled:
            poisoned = sum(is_poison(m["text"], markers) for m in recalled)
            with st.expander(f"Recalled {len(recalled)} memories" + (f" · {poisoned} poisoned" if poisoned else "")):
                for m in recalled:
                    flag = " · **poisoned**" if is_poison(m["text"], markers) else ""
                    st.markdown(f"`{m.get('tier') or '?'}` · {m.get('channel') or '?'}{flag}  \n{esc(m['text'])}")
        for call in turn.get("tool_calls", []):
            icon = {"executed": "▶", "held": "⏸"}.get(call["outcome"], "•")
            line = f"{icon} `{call['tool']}` `{json.dumps(call['args'], ensure_ascii=False)}` → **{call['outcome']}**"
            if call.get("reasons"):
                line += f"  \n  {'held because' if call['outcome'] == 'held' else 'action gate would hold'}: " \
                        f"{esc('; '.join(call['reasons']))}"
            st.markdown(line)
        st.markdown(f"**Agent:** {esc_reply(turn.get('reply', ''))}")


def _scenario_definition(st, sc: dict) -> None:
    setup = sc.get("setup", {})
    planted = ([("document", d.get("title", d["doc_id"]), d["text"]) for d in setup.get("documents", [])]
               + [("transaction memo", t["txn_id"], t["memo"]) for t in setup.get("transactions", []) if t.get("memo")]
               + [("peer message", m.get("sender", "peer"), m["text"]) for m in setup.get("peer_messages", [])]
               + [("ticket", t.get("title", t["ticket_id"]), t["body"]) for t in setup.get("tickets", [])])
    with st.expander("Scenario definition", expanded=True):
        oracle = sc["oracle"] if isinstance(sc["oracle"], list) else [sc["oracle"]]
        what = "harmful action" if sc["kind"] == "attack" else "task done"
        st.markdown(f"**{sc['kind']}**" + (f" · {sc['vector']}" if sc.get("vector") else "")
                    + f" · {sc.get('domain', 'bank')} sandbox  \n**Counts as {what} if** an executed action matches "
                    + " or ".join(f"`{o['tool']}` with {json.dumps(o.get('args_contains', {}), ensure_ascii=False)}"
                                  for o in oracle))
        for i, msgs in enumerate(sc["sessions"]):
            st.markdown(f"Session {i + 1}: " + " / ".join(f"“{esc(m)}”" for m in msgs))
        if sc.get("trigger"):
            st.markdown("Replayed trigger: " + " / ".join(f"“{esc(m)}”" for m in sc["trigger"]))
        for kind, name, text in planted:
            st.markdown(f"Planted {kind} **{esc(name)}**:")
            st.code(text, language=None, wrap_lines=True)


def _drilldown(st, rows: List[dict]) -> None:
    import pandas as pd
    st.subheader("Scenario drill-down")
    by_scenario = defaultdict(list)
    for r in rows:
        by_scenario[r["scenario"]].append(r)
    sid = st.selectbox("Scenario", sorted(by_scenario, key=lambda s: (by_scenario[s][0]["kind"], s)),
                       format_func=lambda s: f"{s} ({by_scenario[s][0]['kind']}"
                                             + (f", {by_scenario[s][0]['vector']})" if by_scenario[s][0].get("vector") else ")"))
    runs = sorted(by_scenario[sid], key=lambda r: (arm_order([x["arm"] for x in by_scenario[sid]]).index(r["arm"]), r["seed"]))
    first = runs[0]
    sc = scenario_defs(first["_set"]).get(sid)
    if sc:
        _scenario_definition(st, sc)
        st.caption(f"Definition as it is now in experiments/{first['_set']}.json.")
    attack = first["kind"] == "attack"

    def yn(v) -> str:
        return "–" if v is None else ("yes" if v else "no")

    def blast_count(r, key: str) -> str:
        return "–" if not r.get("blast") else str(len(r["blast"][key]))
    st.dataframe(pd.DataFrame([{
        "arm": r["arm"], "seed": r["seed"],
        ("harmful action executed" if attack else "task done"): yn(r["oracle_hit"]),
        **({"in a later session": yn(r.get("delayed_hit")),
            "fired again after deletion": yn((r.get("persistence") or {}).get("replay_hit")),
            "poisoned memories quarantined": "–" if not r.get("blast") else
                f"{r['blast']['poison_quarantined']} of {r['blast']['poison_memories']}",
            "harmful actions": blast_count(r, "truth_actions"),
            "traced by lineage": "–" if not r.get("blast") else
                str(len(set(r["blast"]["truth_actions"]) & set(r["blast"]["found_actions"]))),
            "undone by rollback": blast_count(r, "rollback_undone")} if attack else {}),
        "held calls": sum(a["outcome"] == "held" for a in r["actions"]),
        "rails blocked": len(r.get("rail_blocks", [])),
    } for r in runs]), hide_index=True, width="stretch", height=fit(min(len(runs), 18)))
    st.caption("“–” = not measured for this run: persistence is replayed only for attacks with a later trigger "
               "session, and lineage numbers need planted content to trace.")

    run = runs[st.selectbox("Run", range(len(runs)), format_func=lambda i: f"{runs[i]['arm']} / seed {runs[i]['seed']}"
                            + (" · harmful action executed" if attack and runs[i]["oracle_hit"] else ""))]
    markers = poison_markers(sc) if sc and attack else []
    for i, session in enumerate(run["transcript"]):
        st.markdown(f"##### Session {i + 1}")
        _render_session(st, session, markers)
    pers = run.get("persistence")
    if pers:
        st.markdown(f"##### Replay after {pers['sessions_between']} unrelated sessions (source deleted)")
        st.caption(f"Fired again: {pers['replay_hit']} · poisoned memory recalled: {pers['poison_recalled']}")
        _render_session(st, pers["transcript"], markers)
    if run.get("rail_blocks"):
        with st.expander(f"NeMo rails blocked {len(run['rail_blocks'])} texts"):
            for b in run["rail_blocks"]:
                st.markdown(f"`{b['where']}` · {b['rail']}  \n{esc(b['text'])}")
    with st.expander("Action log (every sensitive call, executed or held)"):
        st.dataframe(pd.DataFrame([{"tool": a["tool"], "args": json.dumps(a["args"], ensure_ascii=False),
                                    "outcome": a["outcome"], "action gate": "; ".join(a.get("reasons") or [])}
                                   for a in run["actions"]]), hide_index=True, width="stretch")


def render(st, exp_dir: str = EXP_DIR) -> None:
    st.header("Experiment results")
    st.caption("Saved results from experiments/run_bank_scenarios.py. Each run is one scenario, one arm, one seed; "
               "an attack counts only if its harmful action executed in the sandbox's action log.")
    found = find_results(exp_dir)
    if not found:
        st.info("No experiment results found in experiments/.")
        return

    c1, c2, c3 = st.columns([1, 1, 2])
    exp = c1.selectbox("Experiment", sorted(found, key=int, reverse=True), format_func=lambda e: f"Experiment {e}")
    clfs = sorted(found[exp], key=lambda c: (c != "deberta", c))
    clf = c2.selectbox("Write-gate classifier", clfs, format_func=classifier_label)
    all_sets = sorted(found[exp][clf])
    sets = c3.multiselect("Scenario sets", all_sets, default=all_sets)
    if not sets:
        st.info("Pick at least one scenario set.")
        return
    notes = [n for n in {load_json(found[exp][clf][s]).get("merged") for s in sets} if n]
    st.caption(f"**Experiment {exp}.** {EXPERIMENT_NOTES.get(exp, '')} {AGENT_NOTE}")
    if notes:
        with st.expander("Notes stored in the results files"):
            st.markdown("\n".join(f"- {esc(n)}" for n in notes))

    rows, borrowed = rows_for(found, exp, clf, sets)
    card = scorecard(rows)["arms"]
    arms = arm_order(card)

    kpi_tiles(st, card)
    left, right = st.columns(2)
    with left:
        _set_bars(st, rows, "attack", "Attacks that executed, by scenario set", "attack success rate")
    with right:
        _set_bars(st, rows, "benign", "Benign tasks done, by scenario set", "tasks done")
    _vector_grid(st, rows, arms)
    _scenario_grid(st, rows, arms)

    st.subheader("Scorecard against the deck's targets")
    scorecard_table(st, card, arms)
    metric_glossary(st)
    with st.expander(f"What was run: {len(rows)} runs, {len({r['scenario'] for r in rows})} scenarios"):
        _sample_sizes(st, rows, borrowed)
        arms_table(st, arms)
    _classifier_comparison(st, found, exp, sets)
    _heldout(st, exp_dir, exp)
    _drilldown(st, rows)


def render_all(st) -> None:
    """Both results pages as tabs (standalone viewer)."""
    from dashboard.executive import render_executive
    t1, t2 = st.tabs(["Executive scorecard", "Experiment results"])
    with t1:
        render_executive(st)
    with t2:
        render(st)
