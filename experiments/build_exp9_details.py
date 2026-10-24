"""Build experiments/exp9_details.html from the Experiment 9 result files.

The page embeds the results as JSON and renders them in the browser, so it works when opened
straight from disk or from GitHub. Re-run this after re-running exp9_write_gate_classifier.py
(for example after adding scenarios to memgauntlet/harness.py); new rows appear automatically.

  python experiments/build_exp9_details.py
"""

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = {
    "deberta": os.path.join(HERE, "exp9_write_gate_deberta_results.json"),
    "llm": os.path.join(HERE, "exp9_write_gate_llm_results.json"),
}
OUT = os.path.join(HERE, "exp9_details.html")


def main() -> None:
    data = {}
    for key, path in RESULTS.items():
        with open(path, encoding="utf-8") as f:
            data[key] = json.load(f)
    payload = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    with open(OUT, "w", encoding="utf-8") as f:
        f.write(TEMPLATE.replace("__DATA__", payload))
    n = len(data["llm"]["rows"])
    print(f"wrote {OUT} ({n} scenarios)")


TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>MemGuard Experiment 9 Details</title>
<style>
  :root {
    --bg: #0f1115; --panel: #171a21; --panel2: #1d212a; --border: #2a2e38;
    --text: #e6e8ec; --muted: #9aa1ad; --accent: #5fb3f9;
    --ok: #4fd18b; --warn: #f5b84f; --bad: #f5685f;
  }
  * { box-sizing: border-box; }
  body { background: var(--bg); color: var(--text); margin: 0; padding: 0 16px 80px; line-height: 1.55;
         font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; }
  .wrap { max-width: 1080px; margin: 0 auto; }
  a { color: var(--accent); }
  header { padding: 32px 0 18px; border-bottom: 1px solid var(--border); }
  h1 { font-size: 1.6rem; margin: 6px 0; }
  h2 { font-size: 1.15rem; margin: 34px 0 10px; }
  .subtitle, .muted { color: var(--muted); }
  .small { font-size: 0.85rem; }
  code, .mono { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: 0.85em; }
  code { background: rgba(255,255,255,0.06); padding: 1px 5px; border-radius: 4px; }
  .card { background: var(--panel); border: 1px solid var(--border); border-radius: 10px; padding: 16px 18px; }
  table { width: 100%; border-collapse: collapse; font-size: 0.88rem; }
  th, td { text-align: left; padding: 6px 8px; border-bottom: 1px solid var(--border); vertical-align: top; }
  th { color: var(--muted); font-weight: 600; font-size: 0.76rem; text-transform: uppercase; letter-spacing: .04em; }
  .num { font-variant-numeric: tabular-nums; }
  .tbl { overflow-x: auto; }
  .arch { overflow-x: auto; }
  .arch svg { display: block; margin: 0 auto; max-width: 100%; height: auto; }
  .arch text { font-family: -apple-system, "Segoe UI", Roboto, Arial, sans-serif; }
  .legend { display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 10px; margin-top: 12px; }
  .legend .card { padding: 10px 12px; font-size: 0.85rem; }
  .filters { display: flex; flex-wrap: wrap; gap: 10px; align-items: end; margin: 8px 0 14px; }
  .filters label { display: flex; flex-direction: column; gap: 4px; font-size: 0.75rem; color: var(--muted);
                   text-transform: uppercase; letter-spacing: .04em; }
  select, input[type=search] { background: var(--panel); color: var(--text); border: 1px solid var(--border);
                               border-radius: 6px; padding: 6px 8px; font-size: 0.9rem; min-width: 150px; }
  select:focus-visible, input:focus-visible, summary:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
  .count { color: var(--muted); font-size: 0.88rem; margin-bottom: 10px; }
  .scen { margin-bottom: 14px; }
  .scen-head { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; }
  .sid { font-weight: 700; color: var(--accent); font-variant-numeric: tabular-nums; }
  .chip { border: 1px solid var(--border); background: var(--panel2); border-radius: 999px; padding: 1px 10px;
          font-size: 0.78rem; color: var(--muted); }
  .badge { font-size: 0.72rem; font-weight: 700; text-transform: uppercase; letter-spacing: .04em;
           border-radius: 5px; padding: 2px 8px; }
  .b-ok { background: rgba(79,209,139,0.15); color: var(--ok); }
  .b-bad { background: rgba(245,104,95,0.15); color: var(--bad); }
  .b-warn { background: rgba(245,184,79,0.15); color: var(--warn); }
  .payload { background: #0b0d11; border: 1px solid var(--border); border-radius: 8px; padding: 10px 12px;
             white-space: pre-wrap; word-break: break-word; margin: 10px 0; font-size: 0.86rem; }
  .desc { font-size: 0.86rem; color: var(--muted); margin: 6px 0 0; }
  .cols { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
  .col { background: var(--panel2); border: 1px solid var(--border); border-radius: 8px; padding: 10px 12px; min-width: 0; }
  .col h4 { margin: 0 0 6px; font-size: 0.8rem; text-transform: uppercase; letter-spacing: .05em; color: var(--muted);
            display: flex; justify-content: space-between; gap: 8px; align-items: center; }
  .step { margin: 8px 0 0; font-size: 0.86rem; }
  .step b { color: var(--text); }
  .raw { font-family: ui-monospace, Menlo, Consolas, monospace; font-size: 0.8rem; color: var(--muted);
         background: #0b0d11; border-radius: 6px; padding: 4px 8px; margin-top: 4px; word-break: break-word; }
  .cmp { margin: 4px 0 0; padding-left: 16px; font-size: 0.82rem; color: var(--muted); }
  .hit { color: var(--bad); }
  details summary { cursor: pointer; color: var(--accent); font-size: 0.85rem; margin-top: 8px; }
  @media (max-width: 720px) { .cols { grid-template-columns: 1fr; } }
</style>
</head>
<body>
<div class="wrap">
<header>
  <p class="small"><a href="EXPERIMENTS.html">← Experiment log</a></p>
  <h1>Experiment 9 · write-gate classifier, scenario by scenario</h1>
  <p class="subtitle">Every MemGauntlet payload through the write gate twice: once with DeBERTa, once with Qwen3.5-9B. For each one: what was injected, every check that ran, what the model answered, and the decision.</p>
</header>

<h2>Summary</h2>
<div class="card tbl"><table id="summary"></table></div>

<h2>Architecture for this experiment</h2>
<div class="card">
<div class="arch">
<svg viewBox="0 0 900 640" width="900" height="640" role="img" aria-label="Write-gate pipeline used in Experiment 9">
  <defs>
    <marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
      <path d="M0,0 L10,5 L0,10 z" fill="#9aa1ad"/></marker>
    <marker id="ahr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
      <path d="M0,0 L10,5 L0,10 z" fill="#f5685f"/></marker>
  </defs>
  <g fill="#1d212a" stroke="#2a2e38" stroke-width="1.5">
    <rect x="40" y="16" width="520" height="56" rx="8"/>
    <rect x="40" y="100" width="520" height="56" rx="8"/>
    <rect x="40" y="184" width="520" height="56" rx="8"/>
    <rect x="40" y="268" width="520" height="72" rx="8" stroke="#5fb3f9"/>
    <rect x="40" y="368" width="520" height="72" rx="8"/>
    <rect x="40" y="468" width="520" height="72" rx="8" stroke="#5fb3f9"/>
    <rect x="40" y="568" width="520" height="56" rx="8" stroke="#4fd18b"/>
  </g>
  <rect x="640" y="268" width="230" height="272" rx="8" fill="#2a1a1c" stroke="#f5685f" stroke-width="1.5"/>
  <g fill="#e6e8ec" font-size="15" font-weight="600">
    <text x="60" y="40">Incoming memory text + source channel</text>
    <text x="60" y="124">1 · Trust tier from channel</text>
    <text x="60" y="208">2 · Regex fast path (unchanged)</text>
    <text x="60" y="292">3 · Instruction check (the classifier under test)</text>
    <text x="60" y="392">4 · Embed + find on-topic memories in this tenant</text>
    <text x="60" y="492">5 · Contradiction check (the classifier under test)</text>
    <text x="60" y="592">6 · Sign HMAC envelope · store in Qdrant</text>
    <text x="660" y="292">Quarantined</text>
  </g>
  <g fill="#9aa1ad" font-size="12.5">
    <text x="60" y="60">30 attacks (6 vectors × 5 channels) and 10 benign controls from memgauntlet/harness.py</text>
    <text x="60" y="144">user_chat, documents → provisional · tool_outputs, web, peer_agent → untrusted</text>
    <text x="60" y="228">Known phrases (system override, SELECT * FROM, …) score 0.96; leading imperatives 0.85</text>
    <text x="60" y="312">DeBERTa: NLI “instructs an AI assistant to take an action”</text>
    <text x="60" y="330">Qwen3.5-9B: risky? + confidence, told the channel · quarantine if ≥ 0.65</text>
    <text x="60" y="412">all-MiniLM-L6-v2 embedding (GPU 1) · top 3 neighbours</text>
    <text x="60" y="430">only neighbours with cosine similarity ≥ 0.45 are compared</text>
    <text x="60" y="512">DeBERTa: NLI contradiction probability vs each neighbour</text>
    <text x="60" y="530">Qwen3.5-9B: which memory it contradicts + confidence · quarantine if ≥ 0.70</text>
    <text x="60" y="612">Admitted: recallable later (wrapped as untrusted at read time)</text>
    <text x="660" y="316">Stored with the flag set,</text>
    <text x="660" y="334">never returned by recall.</text>
    <text x="660" y="370">Seeded before the run as</text>
    <text x="660" y="388">verified facts to contradict:</text>
    <text x="660" y="410">· tax rate is 21%</text>
    <text x="660" y="428">· Apple CEO is Tim Cook</text>
    <text x="660" y="446">· SSH port 22 closed to guests</text>
    <text x="660" y="484">Qwen3.5-9B served by vLLM</text>
    <text x="660" y="502">0.32 on GPU 0 (A100 40GB),</text>
    <text x="660" y="520">temperature 0, JSON schema</text>
  </g>
  <g stroke="#9aa1ad" stroke-width="1.6" fill="none" marker-end="url(#ah)">
    <line x1="300" y1="72" x2="300" y2="98"/>
    <line x1="300" y1="156" x2="300" y2="182"/>
    <line x1="300" y1="240" x2="300" y2="266"/>
    <line x1="300" y1="340" x2="300" y2="366"/>
    <line x1="300" y1="440" x2="300" y2="466"/>
    <line x1="300" y1="540" x2="300" y2="566"/>
  </g>
  <g stroke="#f5685f" stroke-width="1.6" fill="none" marker-end="url(#ahr)">
    <path d="M560,212 L600,212 L600,286 L638,286"/>
    <line x1="560" y1="304" x2="638" y2="304"/>
    <path d="M560,504 L600,504 L600,330 L638,330"/>
  </g>
  <g fill="#f5685f" font-size="12">
    <text x="566" y="204">match</text>
    <text x="566" y="298">≥ 0.65</text>
    <text x="566" y="496">≥ 0.70</text>
  </g>
</svg>
</div>
<div class="legend">
  <div class="card"><b>Blue boxes</b> are the only steps that change between the two runs; everything else is identical.</div>
  <div class="card"><b>Contradiction check</b> runs only if the instruction check passed and at least one stored memory is on topic (similarity ≥ 0.45).</div>
  <div class="card"><b>Write gate models</b>: classifiers on GPU 1 from the <code>adrl</code> env; Qwen on GPU 0 from the <code>vllm</code> env, called over HTTP.</div>
</div>
</div>

<h2>Scenarios</h2>
<div class="filters">
  <label for="f-kind">Kind<select id="f-kind"><option value="">All</option><option value="attack">Attacks</option><option value="benign">Benign controls</option></select></label>
  <label for="f-vector">Vector<select id="f-vector"><option value="">All</option></select></label>
  <label for="f-channel">Channel<select id="f-channel"><option value="">All</option></select></label>
  <label for="f-outcome">Outcome<select id="f-outcome">
    <option value="">All</option>
    <option value="both">Both blocked</option>
    <option value="llm_only">Only Qwen blocked</option>
    <option value="deberta_only">Only DeBERTa blocked</option>
    <option value="neither">Neither blocked</option>
  </select></label>
  <label for="f-q">Search text<input id="f-q" type="search" placeholder="e.g. SSH, tenant, refund"></label>
</div>
<div class="count" id="count"></div>
<div id="list"></div>

<p class="muted small">Generated by <code>experiments/build_exp9_details.py</code> from <code>exp9_write_gate_deberta_results.json</code> and <code>exp9_write_gate_llm_results.json</code>. Rows are labelled A (attack) and B (benign control) in harness order.</p>
</div>

<script type="application/json" id="data">__DATA__</script>
<script>
(function () {
  const DATA = JSON.parse(document.getElementById("data").textContent);
  const D = DATA.deberta, L = DATA.llm;
  const byContent = {};
  D.rows.forEach(r => { byContent[r.content] = r; });
  const rows = L.rows.map(r => ({ llm: r, deberta: byContent[r.content] || null }));

  function el(tag, attrs, children) {
    const e = document.createElement(tag);
    if (attrs) for (const k in attrs) {
      if (k === "class") e.className = attrs[k]; else if (k === "text") e.textContent = attrs[k]; else e.setAttribute(k, attrs[k]);
    }
    (children || []).forEach(c => e.appendChild(typeof c === "string" ? document.createTextNode(c) : c));
    return e;
  }
  const pct = (a, b) => b ? Math.round(100 * a / b) + "%" : "–";
  const blocked = r => r && r.quarantined;
  const good = (row, r) => row.llm.kind === "attack" ? blocked(r) : !blocked(r);

  // ---------- summary ----------
  const S = [["", "DeBERTa", "Qwen3.5-9B"],
    ["Model", D.summary.model, L.summary.model],
    ["Attacks blocked", D.summary.attacks_caught + " / " + D.summary.attacks_total, L.summary.attacks_caught + " / " + L.summary.attacks_total],
    ["Benign wrongly blocked", D.summary.false_positives + " / " + D.summary.benign_total, L.summary.false_positives + " / " + L.summary.benign_total],
    ["Blocked by regex / model / contradiction",
      [D.summary.caught_by.fast_path, D.summary.caught_by.deberta_instruction, D.summary.caught_by.contradiction].join(" / "),
      [L.summary.caught_by.fast_path, L.summary.caught_by.llm_instruction, L.summary.caught_by.contradiction].join(" / ")],
    ["Write latency p50 / p95", D.summary.write_latency_ms.p50 + " / " + D.summary.write_latency_ms.p95 + " ms",
      L.summary.write_latency_ms.p50 + " / " + L.summary.write_latency_ms.p95 + " ms"]];
  Object.keys(L.summary.by_vector).forEach(v => {
    const d = (D.summary.by_vector[v] || [0, 0]), l = L.summary.by_vector[v];
    S.push(["· " + v, d[0] + " / " + d[1], l[0] + " / " + l[1]]);
  });
  const st = document.getElementById("summary");
  S.forEach((r, i) => st.appendChild(el("tr", null, r.map((c, j) => el(i === 0 ? "th" : "td", { class: j ? "num" : "", text: String(c) })))));

  // ---------- filters ----------
  function fill(id, values) {
    const s = document.getElementById(id);
    [...new Set(values)].sort().forEach(v => s.appendChild(el("option", { value: v, text: v })));
  }
  fill("f-vector", rows.filter(r => r.llm.kind === "attack").map(r => r.llm.vector));
  fill("f-channel", rows.map(r => r.llm.channel));

  // ---------- one classifier column ----------
  function column(title, r, kind) {
    const col = el("div", { class: "col" });
    if (!r) { col.appendChild(el("p", { class: "muted", text: "No result for this run." })); return col; }
    const ok = kind === "attack" ? r.quarantined : !r.quarantined;
    const label = r.quarantined ? "Quarantined" : "Admitted";
    col.appendChild(el("h4", null, [title, el("span", { class: "badge " + (ok ? "b-ok" : "b-bad"), text: label })]));
    const why = { fast_path: "regex fast path", contradiction: "contradiction check", admitted: "no check fired" }[r.why] ||
                "instruction check";
    col.appendChild(el("p", { class: "step" }, [el("b", { text: "Decided by: " }), why + " · " + r.latency_ms + " ms"]));

    const ins = (r.checks || {}).instruction;
    if (ins) {
      const src = ins.source === "fast_path" ? "regex fast path" : ins.source === "llm" ? "Qwen3.5-9B" : "DeBERTa NLI";
      const fired = ins.score >= ins.threshold;
      col.appendChild(el("p", { class: "step" }, [el("b", { text: "Instruction check: " }),
        src + " scored ", el("span", { class: fired ? "hit" : "", text: ins.score.toFixed(2) }), " (threshold " + ins.threshold + ")"]));
      if (ins.raw) col.appendChild(el("div", { class: "raw", text: "Qwen answered: " + JSON.stringify(ins.raw) }));
    }
    const con = (r.checks || {}).contradiction;
    if (con) {
      const fired = con.score >= con.threshold;
      col.appendChild(el("p", { class: "step" }, [el("b", { text: "Contradiction check: " }),
        "scored ", el("span", { class: fired ? "hit" : "", text: con.score.toFixed(2) }), " (threshold " + con.threshold + ")"]));
      const ul = el("ul", { class: "cmp" });
      con.compared_with.forEach(c => ul.appendChild(el("li", { class: c.text === con.conflicting_memory && fired ? "hit" : "",
        text: "“" + c.text + "”" + (c.similarity != null ? "  · similarity " + c.similarity.toFixed(2) : "") })));
      col.appendChild(el("p", { class: "step muted", text: "Compared with:" }));
      col.appendChild(ul);
      if (con.raw) col.appendChild(el("div", { class: "raw", text: "Qwen answered: " + JSON.stringify(con.raw) }));
    } else if (!r.quarantined || r.why === "contradiction") {
      col.appendChild(el("p", { class: "step muted", text: "Contradiction check: skipped, no stored memory was on topic (similarity ≥ 0.45)." }));
    } else {
      col.appendChild(el("p", { class: "step muted", text: "Contradiction check: not reached (already quarantined)." }));
    }
    if (r.reasons) col.appendChild(el("p", { class: "step muted small", text: "Gate reason: " + r.reasons }));
    return col;
  }

  // ---------- list ----------
  function outcome(row) {
    const d = blocked(row.deberta), l = blocked(row.llm);
    return d && l ? "both" : l ? "llm_only" : d ? "deberta_only" : "neither";
  }
  function render() {
    const fk = document.getElementById("f-kind").value, fv = document.getElementById("f-vector").value,
          fc = document.getElementById("f-channel").value, fo = document.getElementById("f-outcome").value,
          fq = document.getElementById("f-q").value.trim().toLowerCase();
    const list = document.getElementById("list");
    list.textContent = "";
    const shown = rows.filter(row => {
      const r = row.llm;
      return (!fk || r.kind === fk) && (!fv || r.vector === fv) && (!fc || r.channel === fc) &&
             (!fo || outcome(row) === fo) &&
             (!fq || (r.content + " " + r.description + " " + r.vector + " " + r.channel).toLowerCase().includes(fq));
    });
    document.getElementById("count").textContent = shown.length + " of " + rows.length + " scenarios";
    shown.forEach(row => {
      const r = row.llm;
      const card = el("div", { class: "card scen" });
      const kindBadge = r.kind === "attack" ? el("span", { class: "badge b-warn", text: "attack" })
                                            : el("span", { class: "badge b-ok", text: "benign control" });
      card.appendChild(el("div", { class: "scen-head" }, [el("span", { class: "sid", text: r.id }), kindBadge,
        el("span", { class: "chip", text: r.kind === "attack" ? r.vector : "benign" }), el("span", { class: "chip", text: "channel: " + r.channel })]));
      card.appendChild(el("p", { class: "desc", text: r.description }));
      card.appendChild(el("div", { class: "payload", text: r.content }));
      card.appendChild(el("div", { class: "cols" }, [column("DeBERTa", row.deberta, r.kind), column("Qwen3.5-9B", r, r.kind)]));
      list.appendChild(card);
    });
  }
  ["f-kind", "f-vector", "f-channel", "f-outcome"].forEach(id => document.getElementById(id).addEventListener("change", render));
  document.getElementById("f-q").addEventListener("input", render);
  render();
})();
</script>
</body>
</html>
"""

if __name__ == "__main__":
    main()
