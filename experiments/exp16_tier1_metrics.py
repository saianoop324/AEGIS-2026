"""Experiment 16: extra metrics for the scorecard, computed from the result files we already have (no model calls).

  1. Confidence intervals and significance: Wilson 95% on every rate, cluster-bootstrap CI (resampling
     scenarios) on the ASR reduction, exact McNemar test on paired (scenario, seed) runs.
  2. Defence-layer attribution: for each attack that succeeded against the unprotected agent, which
     checkpoint stopped it in the MemGuard arm, plus an arm-by-arm ablation.
  3. Attack vector x entry channel matrix.
  4. Security-utility frontier: attack success against benign tasks finished, per agent and arm.
  5. Write-gate ROC / PR curves and AUC on the public held-out set (deepset/prompt-injections).
  6. Time to contain: how early in the run harm is attempted, and how many harmful calls land before the first hold.

Run from the repo root in the nvbase env:  python experiments/exp16_tier1_metrics.py
Writes experiments/exp16/{report.md, metrics.json, *.png}.
"""

import json
import math
import os
import random
import sys
from collections import defaultdict
from typing import Dict, List

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from experiments.run_bank_scenarios import matches_oracle  # noqa: E402

OUT = os.path.join(HERE, "exp16")
os.makedirs(OUT, exist_ok=True)

SCENARIO_FILES = ["bank_scenarios_attack", "bank_scenarios_attack_ingate", "bank_scenarios_benign",
                  "bank_scenarios_benign_extended", "it_scenarios", "bank_scenarios_thirdparty"]
SCEN = {s["id"]: s for f in SCENARIO_FILES for s in json.load(open(f"{HERE}/{f}.json"))}

BANK_SETS = ["bank_scenarios_attack", "bank_scenarios_attack_ingate", "bank_scenarios_benign",
             "bank_scenarios_benign_extended"]
ARM_ORDER = ["unprotected", "nemo_rails", "no_action_gate", "action_gate_only", "memguard", "memguard_ttl"]


def rows_of(path: str, arms=None) -> List[dict]:
    rows = json.load(open(f"{HERE}/{path}.json"))["rows"]
    return [r for r in rows if arms is None or r["arm"] in arms]


def load_qwen() -> List[dict]:
    """Qwen3.5-9B agent, DeBERTa write gate. Arms the action-gate fix did not touch come from exp14 (first pass),
    the others from exp14fix, which is how the deck's 0.69 / 0.54 / 0.19 / 0.18 were built."""
    rows = []
    for s in BANK_SETS + ["it_scenarios"]:
        rows += rows_of(f"exp14_deberta_{s}", {"unprotected", "nemo_rails", "no_action_gate"})
        rows += rows_of(f"exp14fix_deberta_{s}", {"action_gate_only", "memguard", "memguard_ttl"})
    for r in rows:
        r["agent"] = "Qwen3.5-9B"
    return rows


def load_nemotron() -> List[dict]:
    rows = []
    for s in BANK_SETS + ["it_scenarios"]:
        rows += rows_of(f"nemotron_deberta_{s}")
    for r in rows:
        r["agent"] = "Nemotron Nano 9B v2"
    return rows


# ---------------------------------------------------------------- statistics

def wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return (float("nan"),) * 3
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0.0, c - h), min(1.0, c + h)


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact p for discordant pairs b (A hit, B miss) and c (A miss, B hit)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def fmt_rate(k, n):
    p, lo, hi = wilson(k, n)
    return f"{k}/{n} = {p:.2f} [{lo:.2f}, {hi:.2f}]"


def attack_rows(rows, arm, domain=None, vector_prefix=None):
    return [r for r in rows if r["arm"] == arm and r["kind"] == "attack"
            and (domain is None or r["domain"] == domain)]


def hits_by_scenario(rs):
    g = defaultdict(lambda: [0, 0])
    for r in rs:
        g[r["scenario"]][0] += bool(r["oracle_hit"])
        g[r["scenario"]][1] += 1
    return g


def reduction_ci(rows, arm, n_boot=2000, seed=0, domain=None):
    """1 - ASR(arm)/ASR(unprotected), bootstrapped over scenarios (seeds of a scenario stay together)."""
    a, u = hits_by_scenario(attack_rows(rows, arm, domain)), hits_by_scenario(attack_rows(rows, "unprotected", domain))
    ids = [i for i in u if i in a]
    rng = random.Random(seed)

    def red(sample):
        ha = sum(a[i][0] for i in sample); hu = sum(u[i][0] for i in sample)
        return (1 - ha / hu) if hu else float("nan")

    point = red(ids)
    boots = sorted(x for x in (red([rng.choice(ids) for _ in ids]) for _ in range(n_boot)) if x == x)
    return point, boots[int(0.025 * len(boots))], boots[int(0.975 * len(boots)) - 1]


def paired_counts(rows, arm_a, arm_b, domain=None):
    """Return (a hit only, b hit only, both, neither) over (scenario, seed) attack runs."""
    A = {(r["scenario"], r["seed"]): bool(r["oracle_hit"]) for r in attack_rows(rows, arm_a, domain)}
    B = {(r["scenario"], r["seed"]): bool(r["oracle_hit"]) for r in attack_rows(rows, arm_b, domain)}
    ks = A.keys() & B.keys()
    return (sum(A[k] and not B[k] for k in ks), sum(B[k] and not A[k] for k in ks),
            sum(A[k] and B[k] for k in ks), sum(not A[k] and not B[k] for k in ks))


# ---------------------------------------------------------------- 1. CIs and significance

def section_ci(rows_by_agent, md, out):
    md.append("## 1. Confidence intervals and significance\n")
    md.append("Attack success rate (ASR) = harmful tool calls executed / attack runs, hand-written BankDesk sets "
              "(attack + in-gate), 3 seeds. Wilson 95% interval in brackets. Reduction = 1 - ASR/ASR(unprotected), "
              "with a 95% bootstrap interval over scenarios. p = exact McNemar test on paired (scenario, seed) runs.\n")
    for agent, rows in rows_by_agent.items():
        for domain, label in (("bank", "BankDesk"), ("it", "ITDesk")):
            md.append(f"### {agent}, {label}\n")
            md.append("| arm | ASR [95% CI] | reduction [95% CI] | vs unprotected (p) | vs NeMo Guardrails (p) |")
            md.append("|---|---|---|---|---|")
            for arm in ARM_ORDER:
                rs = attack_rows(rows, arm, domain)
                if not rs:
                    continue
                k = sum(bool(r["oracle_hit"]) for r in rs)
                cell = fmt_rate(k, len(rs))
                if arm == "unprotected":
                    md.append(f"| {arm} | {cell} | - | - | - |")
                    continue
                red = reduction_ci(rows, arm, domain=domain)
                bu, cu, _, _ = paired_counts(rows, "unprotected", arm, domain)
                p_u = mcnemar_exact(bu, cu)
                if arm == "nemo_rails":
                    p_n = "-"
                elif attack_rows(rows, "nemo_rails", domain):
                    bn, cn, _, _ = paired_counts(rows, "nemo_rails", arm, domain)
                    p_n = f"{mcnemar_exact(bn, cn):.3g} ({bn} fixed, {cn} broke)"
                else:
                    p_n = "n/a"
                md.append(f"| {arm} | {cell} | {red[0]:.2f} [{red[1]:.2f}, {red[2]:.2f}] | "
                          f"{p_u:.3g} ({bu} fixed, {cu} broke) | {p_n} |")
                out.setdefault(agent, {}).setdefault(domain, {})[arm] = {
                    "hits": k, "n": len(rs), "reduction": red, "p_vs_unprotected": p_u}
            md.append("")
    # third-party benchmarks
    md.append("### Third-party attacks (AgentDojo + InjecAgent), 3 seeds\n")
    md.append("| agent | arm | ASR [95% CI] |")
    md.append("|---|---|---|")
    tp = {"Qwen3.5-9B": "exp15_deberta_bank_scenarios_thirdparty", "Nemotron Nano 9B v2": "nemotron_deberta_bank_scenarios_thirdparty"}
    for agent, f in tp.items():
        rs_all = [r for r in rows_of(f) if r["kind"] == "attack"]
        for arm in ARM_ORDER:
            rs = [r for r in rs_all if r["arm"] == arm]
            if rs:
                md.append(f"| {agent} | {arm} | {fmt_rate(sum(bool(r['oracle_hit']) for r in rs), len(rs))} |")
    md.append("")


# ---------------------------------------------------------------- 2. attribution

def attempted_harm(r) -> List[dict]:
    """Action-log rows (executed or held) that match the scenario's attack oracle."""
    sc = SCEN[r["scenario"]]
    return [a for a in r["actions"] if matches_oracle(a, sc["oracle"], {})]


def stopped_by(r) -> str:
    """Attribute a non-hit attack run to the first checkpoint that visibly acted on it."""
    if r["oracle_hit"]:
        return "got through"
    if any(a["outcome"] == "held" for a in attempted_harm(r)):
        return "action gate"
    b = r.get("blast") or {}
    if b.get("poison_memories") and b.get("poison_quarantined"):
        return "write gate"
    return "no harmful call made (read gate wrap or agent declined)"


def section_attribution(rows_by_agent, md, out):
    md.append("## 2. Which checkpoint stopped each attack\n")
    md.append("Only attacks that succeeded against the unprotected agent are counted (same rule as the deck). "
              "A run is credited to the action gate if the harmful call was attempted and held; else to the write "
              "gate if the poisoned memory was quarantined; else to 'no harmful call made'. The write gate and the "
              "action gate can both act on one run, and the action gate is credited first, so write-gate numbers "
              "are a lower bound.\n")
    cats = ["got through", "action gate", "write gate", "no harmful call made (read gate wrap or agent declined)"]
    for agent, rows in rows_by_agent.items():
        for domain, label in (("bank", "BankDesk"), ("it", "ITDesk")):
            un = {(r["scenario"], r["seed"]) for r in attack_rows(rows, "unprotected", domain) if r["oracle_hit"]}
            if not un:
                continue
            md.append(f"### {agent}, {label} ({len(un)} attack runs that succeed unprotected)\n")
            md.append("| arm | " + " | ".join(cats) + " |")
            md.append("|---|" + "---|" * len(cats))
            for arm in ARM_ORDER[1:]:
                rs = [r for r in attack_rows(rows, arm, domain) if (r["scenario"], r["seed"]) in un]
                if not rs:
                    continue
                cnt = {c: sum(stopped_by(r) == c for r in rs) for c in cats}
                md.append(f"| {arm} | " + " | ".join(f"{cnt[c]} ({cnt[c] / len(rs):.0%})" for c in cats) + " |")
                out.setdefault("attribution", {}).setdefault(agent, {}).setdefault(domain, {})[arm] = cnt
            md.append("")
    md.append("The ablation reads off the table above: `no_action_gate` is the write and read gates alone, "
              "`action_gate_only` is the action gate alone, `memguard` is all of them.\n")


# ---------------------------------------------------------------- 3. vector x channel

def channel_of(sc) -> str:
    s = sc.get("setup", {})
    if s.get("documents"):
        return "document"
    if s.get("transactions") or s.get("tickets"):
        return "tool output"
    if s.get("peer_messages"):
        return "peer agent"
    return "customer chat"


def section_matrix(rows_by_agent, md, out):
    md.append("## 3. Attack type x entry channel\n")
    md.append("Cells are attacks executed / runs (3 seeds per scenario), hand-written BankDesk and ITDesk sets. "
              "Entry channel is read from where the scenario plants its text. Web retrieval has no hand-written "
              "scenario in these sets (it appears only in the garak and third-party runs), so it is not a column.\n")
    for agent, rows in rows_by_agent.items():
        for arm in ("unprotected", "nemo_rails", "memguard"):
            rs = [r for r in rows if r["arm"] == arm and r["kind"] == "attack"]
            cells = defaultdict(lambda: [0, 0])
            for r in rs:
                cells[(r["vector"], channel_of(SCEN[r["scenario"]]))][0] += bool(r["oracle_hit"])
                cells[(r["vector"], channel_of(SCEN[r["scenario"]]))][1] += 1
            vecs = sorted({k[0] for k in cells}); chans = sorted({k[1] for k in cells})
            md.append(f"### {agent}, {arm}\n")
            md.append("| attack type | " + " | ".join(chans) + " |")
            md.append("|---|" + "---|" * len(chans))
            for v in vecs:
                md.append(f"| {v} | " + " | ".join(
                    (f"{cells[(v, c)][0]}/{cells[(v, c)][1]}" if cells[(v, c)][1] else "·") for c in chans) + " |")
            md.append("")
            out.setdefault("matrix", {}).setdefault(agent, {})[arm] = {f"{v}|{c}": cells[(v, c)] for v in vecs for c in chans}


# ---------------------------------------------------------------- 4. frontier

def section_frontier(rows_by_agent, md, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    md.append("## 4. Security-utility frontier\n")
    md.append("Attack success (lower is better) against benign tasks finished (higher is better), BankDesk, "
              "with Wilson 95% intervals. Benign tasks are the 21 + 42 scenarios x 3 seeds.\n")
    md.append("| agent | arm | attacks executed [95% CI] | benign finished [95% CI] |")
    md.append("|---|---|---|---|")
    fig, axes = plt.subplots(1, len(rows_by_agent), figsize=(5.6 * len(rows_by_agent), 4.6), sharey=True)
    axes = np.atleast_1d(axes)
    colors = {"unprotected": "#7a7a7a", "nemo_rails": "#c8762b", "no_action_gate": "#8a6fc4",
              "action_gate_only": "#2b8a9e", "memguard": "#1f5fbf", "memguard_ttl": "#1b8a4a"}
    for ax, (agent, rows) in zip(axes, rows_by_agent.items()):
        for arm in ARM_ORDER:
            at = [r for r in attack_rows(rows, arm, "bank")]
            bn = [r for r in rows if r["arm"] == arm and r["kind"] == "benign" and r["domain"] == "bank"]
            if not at or not bn:
                continue
            ka, na, kb, nb = sum(bool(r["oracle_hit"]) for r in at), len(at), sum(bool(r["oracle_hit"]) for r in bn), len(bn)
            pa, la, ha = wilson(ka, na); pb, lb, hb = wilson(kb, nb)
            md.append(f"| {agent} | {arm} | {pa:.2f} [{la:.2f}, {ha:.2f}] | {pb:.2f} [{lb:.2f}, {hb:.2f}] |")
            ax.errorbar(pa, pb, xerr=[[pa - la], [ha - pa]], yerr=[[pb - lb], [hb - pb]], fmt="o", ms=8,
                        color=colors[arm], ecolor=colors[arm], elinewidth=1.2, capsize=0, label=arm)
            out.setdefault("frontier", {}).setdefault(agent, {})[arm] = {"asr": [pa, la, ha], "benign": [pb, lb, hb]}
        ax.set_title(agent, fontsize=11, loc="left")
        ax.set_xlabel("attacks executed (lower is better)")
        ax.set_xlim(-0.03, 1.0); ax.set_ylim(0, 1.05)
        ax.grid(color="#e4e4e4", lw=0.6); ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].set_ylabel("benign tasks finished (higher is better)")
    axes[-1].legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(f"{OUT}/frontier.png", dpi=170)
    plt.close(fig)
    md.append("\n![frontier](frontier.png)\n")


# ---------------------------------------------------------------- 5. ROC / PR

def curves(y, s):
    y, s = np.asarray(y), np.asarray(s)
    order = np.argsort(-s, kind="stable")
    y, s = y[order], s[order]
    tps, fps = np.cumsum(y), np.cumsum(1 - y)
    keep = np.r_[np.where(np.diff(s))[0], len(s) - 1]  # one point per distinct score
    tpr = np.r_[0, tps[keep] / y.sum()]; fpr = np.r_[0, fps[keep] / (1 - y).sum()]
    prec = np.r_[1, tps[keep] / (tps[keep] + fps[keep])]
    auc = float(np.trapezoid(tpr, fpr)); ap = float(np.sum(np.diff(tpr) * prec[1:]))
    return fpr, tpr, prec, auc, ap


def section_roc(md, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    files = {"DeBERTa NLI": "exp14_heldout_deberta", "NemoGuard JailbreakDetect": "exp14_heldout_nemoguard",
             "Qwen3.5-0.8B": "exp14_heldout_qwen3.5-0.8b", "Qwen3.5-2B": "exp14_heldout_qwen3.5-2b",
             "Qwen3.5-4B": "exp14_heldout_qwen3.5-4b", "Qwen3.5-9B": "exp14_heldout_qwen3.5-9b",
             "Nemotron Nano 4B": "nemotron_heldout_nemotron-4b", "Nemotron Nano 9B v2": "nemotron_heldout_nemotron-9b"}
    md.append("## 5. Write-gate classifiers on a public held-out set\n")
    md.append("deepset/prompt-injections, 662 texts (263 injections), none used to tune a gate. AUROC and average "
              "precision (AP) use the classifier's injection score; the operating point is the gate's own threshold. "
              "Intervals are 95% bootstrap over texts. Classifiers with few distinct scores (the LLM judges mostly "
              "answer 0 or 1) have a coarse curve, so AUROC understates how a tuned threshold could do.\n")
    md.append("| classifier | distinct scores | AUROC [95% CI] | AP | precision | recall [95% CI] | false-quarantine rate | recall @ FPR<=5% |")
    md.append("|---|---|---|---|---|---|---|---|")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.6))
    rng = np.random.default_rng(0)
    palette = ["#1f5fbf", "#c8762b", "#2b8a9e", "#8a6fc4", "#1b8a4a", "#b3412f", "#7a7a7a", "#d4a017"]
    for (name, f), col in zip(files.items(), palette):
        rows = json.load(open(f"{HERE}/{f}.json"))["rows"]
        y = np.array([r["label"] for r in rows]); s = np.array([r["score"] for r in rows], float)
        q = np.array([r["quarantined"] for r in rows], bool)
        fpr, tpr, prec, auc, ap = curves(y, s)
        boots = []
        for _ in range(500):
            i = rng.integers(0, len(y), len(y))
            if y[i].min() != y[i].max():
                boots.append(curves(y[i], s[i])[3])
        lo, hi = np.percentile(boots, [2.5, 97.5])
        tp = int((q & (y == 1)).sum()); fp = int((q & (y == 0)).sum()); npos = int(y.sum()); nneg = int((y == 0).sum())
        p_ = tp / (tp + fp) if tp + fp else float("nan")
        _, rl, rh = wilson(tp, npos)
        r5 = float(tpr[fpr <= 0.05].max())
        md.append(f"| {name} | {len(set(s))} | {auc:.3f} [{lo:.3f}, {hi:.3f}] | {ap:.3f} | {p_:.2f} | "
                  f"{tp / npos:.2f} [{rl:.2f}, {rh:.2f}] | {fp / nneg:.3f} | {r5:.2f} |")
        out.setdefault("roc", {})[name] = {"auroc": auc, "auroc_ci": [float(lo), float(hi)], "ap": ap,
                                           "precision": p_, "recall": tp / npos, "fqr": fp / nneg, "recall_at_fpr5": r5}
        a1.plot(fpr, tpr, color=col, lw=1.8, label=f"{name} ({auc:.2f})")
        a2.plot(np.r_[0, tpr[1:]], prec, color=col, lw=1.8)
    a1.plot([0, 1], [0, 1], color="#bbbbbb", lw=1, ls="--")
    a1.set_xlabel("false positive rate"); a1.set_ylabel("true positive rate (recall)"); a1.set_title("ROC (AUROC)", loc="left", fontsize=11)
    a2.set_xlabel("recall"); a2.set_ylabel("precision"); a2.set_title("Precision-recall", loc="left", fontsize=11)
    a2.set_ylim(0, 1.02)
    for a in (a1, a2):
        a.grid(color="#e4e4e4", lw=0.6); a.set_axisbelow(True)
        for sp in ("top", "right"):
            a.spines[sp].set_visible(False)
    a1.legend(frameon=False, fontsize=7.5, loc="lower right")
    fig.tight_layout(); fig.savefig(f"{OUT}/roc_pr.png", dpi=170); plt.close(fig)
    md.append("\n![roc](roc_pr.png)\n")


# ---------------------------------------------------------------- 6. time to contain

def sess_index(a) -> int:
    return int(a["session_id"].rsplit("sess", 1)[1])


def section_contain(rows_by_agent, md, out):
    md.append("## 6. Time to contain\n")
    md.append("For attack runs that succeed unprotected, in each arm: how many harmful calls were attempted, how many "
              "executed before the first one was held (or in total, if none was), and in which session the first "
              "harmful call was attempted. Session 1 is the first customer message after the poison is planted.\n")
    md.append("| agent | domain | arm | runs | mean harmful calls attempted | mean executed before first hold | runs with a hold | first attempt in session 1 | first attempt in a later session |")
    md.append("|---|---|---|---|---|---|---|---|---|")
    for agent, rows in rows_by_agent.items():
        for domain in ("bank", "it"):
            un = {(r["scenario"], r["seed"]) for r in attack_rows(rows, "unprotected", domain) if r["oracle_hit"]}
            for arm in ARM_ORDER:
                rs = [r for r in attack_rows(rows, arm, domain) if (r["scenario"], r["seed"]) in un]
                if not rs:
                    continue
                att, pre, held_runs, s1, later = [], [], 0, 0, 0
                for r in rs:
                    acts = sorted(attempted_harm(r), key=lambda a: a["ts"])
                    att.append(len(acts))
                    first_hold = next((i for i, a in enumerate(acts) if a["outcome"] == "held"), None)
                    held_runs += first_hold is not None
                    pre.append(sum(a["outcome"] == "executed" for a in (acts if first_hold is None else acts[:first_hold])))
                    if acts:
                        if sess_index(acts[0]) == 1:
                            s1 += 1
                        else:
                            later += 1
                md.append(f"| {agent} | {domain} | {arm} | {len(rs)} | {np.mean(att):.2f} | {np.mean(pre):.2f} | "
                          f"{held_runs} | {s1} | {later} |")
                out.setdefault("contain", {}).setdefault(agent, {}).setdefault(domain, {})[arm] = {
                    "runs": len(rs), "mean_attempted": float(np.mean(att)), "mean_executed_before_hold": float(np.mean(pre))}
    md.append("")


def main():
    rows_by_agent = {"Qwen3.5-9B": load_qwen(), "Nemotron Nano 9B v2": load_nemotron()}
    md: List[str] = ["# Experiment 16: extra metrics for the scorecard\n",
                     "Computed from existing result files (Qwen: exp14 / exp14fix; Nemotron: nemotron_deberta_*; "
                     "held-out classifiers: exp14_heldout_*), DeBERTa write gate. No new model runs.\n"]
    out: Dict = {}
    section_ci(rows_by_agent, md, out)
    section_attribution(rows_by_agent, md, out)
    section_matrix(rows_by_agent, md, out)
    section_frontier(rows_by_agent, md, out)
    section_roc(md, out)
    section_contain(rows_by_agent, md, out)
    open(f"{OUT}/report.md", "w").write("\n".join(md))
    json.dump(out, open(f"{OUT}/metrics.json", "w"), indent=1, default=float)
    print("\n".join(md))


if __name__ == "__main__":
    main()
