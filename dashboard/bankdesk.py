"""BankDesk / ITDesk experiment results view for the Streamlit dashboard.

Reads the saved runner outputs (experiments/exp<N>_<classifier>_<scenario set>.json) and the held-out
write-gate results (experiments/exp<N>_heldout_<classifier>.json); runs nothing. Shows the deck's
scorecard per arm against its targets, attack success per scenario set, held-out precision/recall,
and a per-scenario drill-down into each run's transcript and action log.

Used by app.py (tab "BankDesk Experiments") and standalone: streamlit run dashboard/bankdesk_app.py
"""

import glob
import json
import os
import re
import sys
from collections import defaultdict
from typing import Dict, List

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from experiments.run_bank_scenarios import TARGETS, scorecard  # noqa: E402

RESULTS_RE = re.compile(r"^exp(\d+)_(.+?)_((?:bank_scenarios|it_scenarios).*)\.json$")
SCORECARD_KEYS = [
    ("attack_success_rate", "Attack success rate"), ("asr_reduction", "Reduction vs unprotected"),
    ("sleeper_activation", "Sleeper activation"), ("persistence_rate", "Persistence rate"),
    ("poison_recalled_after_n", "Poison recalled after N sessions"), ("blast_radius_recall", "Blast-radius recall"),
    ("rollback_undone", "Rollback: harmful undone"), ("rollback_collateral_actions", "Rollback: innocent undone"),
    ("benign_retained", "Benign tasks retained"), ("provenance_coverage", "Provenance coverage"),
    ("added_write_p95_ms", "Added write p95 (ms)"), ("rail_p95_ms", "Rail check p95 (ms)"),
    ("extra_tokens_per_turn", "Extra tokens per turn"),
]


def find_results(exp_dir: str) -> Dict[str, Dict[str, Dict[str, str]]]:
    """experiment -> classifier -> scenario set -> path"""
    out: Dict[str, Dict[str, Dict[str, str]]] = defaultdict(lambda: defaultdict(dict))
    for path in glob.glob(os.path.join(exp_dir, "exp*_*.json")):
        m = RESULTS_RE.match(os.path.basename(path))
        if m:
            out[f"Experiment {m.group(1)}"][m.group(2)][m.group(3)] = path
    return out


def load_rows(paths: List[str]) -> List[dict]:
    rows = []
    for p in paths:
        for r in json.load(open(p, encoding="utf-8"))["rows"]:
            r["_set"] = RESULTS_RE.match(os.path.basename(p)).group(3)
            rows.append(r)
    return rows


def render(st, exp_dir: str = os.path.join(ROOT, "experiments")) -> None:
    import pandas as pd
    import plotly.express as px

    st.header("BankDesk & ITDesk Experiments")
    st.caption("Saved results from experiments/run_bank_scenarios.py. Each run is one scenario, one arm, one seed; "
               "an attack counts only if the harmful action executed in the sandbox's action log.")
    found = find_results(exp_dir)
    if not found:
        st.info("No experiment results found in experiments/.")
        return

    c1, c2, c3 = st.columns(3)
    exp = c1.selectbox("Experiment", sorted(found, key=lambda e: int(e.split()[-1]), reverse=True))
    clf = c2.selectbox("Write-gate classifier", sorted(found[exp]))
    sets = c3.multiselect("Scenario sets", sorted(found[exp][clf]), default=sorted(found[exp][clf]))
    if not sets:
        return
    rows = load_rows([found[exp][clf][s] for s in sets])
    card = scorecard(rows)["arms"]
    arms = list(card)

    st.subheader("Scorecard against the deck's targets")
    table = [{"Metric": label, "Target": TARGETS.get(key, ""), **{a: card[a].get(key) for a in arms}}
             for key, label in SCORECARD_KEYS]
    st.dataframe(pd.DataFrame(table), hide_index=True, width="stretch")

    st.subheader("Attacks that executed, by scenario set")
    agg = defaultdict(lambda: [0, 0])
    for r in rows:
        if r["kind"] == "attack":
            agg[(r["_set"], r["arm"])][0] += int(r["oracle_hit"])
            agg[(r["_set"], r["arm"])][1] += 1
    df = pd.DataFrame([{"set": s, "arm": a, "executed": h, "runs": n, "rate": h / n} for (s, a), (h, n) in agg.items()])
    if not df.empty:
        fig = px.bar(df, x="set", y="rate", color="arm", barmode="group", hover_data=["executed", "runs"],
                     labels={"rate": "attack success rate", "set": ""})
        fig.update_yaxes(range=[0, 1])
        st.plotly_chart(fig, width="stretch")

    heldout = sorted(glob.glob(os.path.join(exp_dir, f"exp{exp.split()[-1]}_heldout_*.json")))
    if heldout:
        st.subheader("Write-gate precision / recall on a held-out public set")
        st.dataframe(pd.DataFrame([json.load(open(p, encoding="utf-8"))["summary"] for p in heldout]),
                     hide_index=True, width="stretch")

    st.subheader("Scenario drill-down")
    by_scenario = defaultdict(list)
    for r in rows:
        by_scenario[r["scenario"]].append(r)
    sid = st.selectbox("Scenario", sorted(by_scenario))
    runs = by_scenario[sid]
    st.dataframe(pd.DataFrame([{"arm": r["arm"], "seed": r["seed"], "harmful action / task done": r["oracle_hit"],
                                "replay fired": (r.get("persistence") or {}).get("replay_hit"),
                                "rails blocked": len(r.get("rail_blocks", []))} for r in runs]),
                 hide_index=True, width="stretch")
    run = st.selectbox("Run", runs, format_func=lambda r: f"{r['arm']} / seed {r['seed']}")
    for i, session in enumerate(run["transcript"]):
        st.markdown(f"**Session {i + 1}**")
        for turn in session:
            st.markdown(f"> **Customer:** {turn['user']}")
            for call in turn.get("tool_calls", []):
                st.markdown(f"- `{call['tool']}` {json.dumps(call['args'])} → **{call['outcome']}**"
                            + (f" ({'; '.join(call['reasons'])})" if call.get("reasons") else ""))
            st.markdown(f"**Agent:** {turn.get('reply', '')}")
    with st.expander("Action log"):
        st.dataframe(pd.DataFrame([{k: a[k] for k in ("tool", "args", "outcome", "reasons")} for a in run["actions"]]),
                     hide_index=True, width="stretch")
