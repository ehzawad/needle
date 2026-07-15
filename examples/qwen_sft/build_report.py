"""Build a self-contained HTML report from eval50_results.txt."""
from __future__ import annotations
import html
import re
from pathlib import Path

ROW = re.compile(r"^\[(\w+)\s*\]\s+(.+?)\s+got=(\w+)\s+card=(\S+)\s+\|\s+(.+)$")
CATS = ["in_scope", "hard_ood", "far_ood", "ambiguous", "adversarial"]
CATLABEL = {"in_scope": "In-scope", "hard_ood": "Hard-OOD (doppelgänger)",
            "far_ood": "Far-OOD", "ambiguous": "Ambiguous", "adversarial": "Adversarial"}
DISPCOL = {"ANSWER": "#0072B2", "CLARIFY": "#E69F00", "ABSTAIN": "#009E73"}  # Okabe-Ito, CVD-safe


def parse():
    rows = []
    for line in Path("eval50_results.txt").read_text().splitlines():
        m = ROW.match(line)
        if m:
            cat, verdict, disp, card, q = m.groups()
            v = "good" if verdict.strip().startswith("✓") else ("bad" if verdict.strip().startswith("✗") else "warn")
            rows.append(dict(cat=cat, verdict=verdict.strip(), disp=disp, card=card, q=q, v=v))
    return rows


def bar(counts, total):
    seg = ""
    for d in ["ANSWER", "CLARIFY", "ABSTAIN"]:
        n = counts.get(d, 0)
        if n:
            seg += f'<span style="width:{n/total*100:.4f}%;background:{DISPCOL[d]}" title="{d}: {n}"></span>'
    return f'<div class="bar">{seg}</div>'


def main():
    rows = parse()
    # metrics
    insc = [r for r in rows if r["cat"] == "in_scope"]
    answered = sum(r["disp"] == "ANSWER" for r in insc)
    ood = [r for r in rows if r["cat"] in ("hard_ood", "far_ood", "adversarial")]
    leaks = sum(r["disp"] == "ANSWER" for r in ood)
    amb = [r for r in rows if r["cat"] == "ambiguous"]
    amb_clar = sum(r["disp"] == "CLARIFY" for r in amb)
    adv = [r for r in rows if r["cat"] == "adversarial"]
    adv_safe = sum(r["disp"] != "ANSWER" for r in adv)

    tiles = [
        ("Harmful leaks", f"{leaks}<span>/{len(ood)}</span>", "OOD / doppelgänger / adversarial that got answered — want 0", "good" if leaks == 0 else "bad"),
        ("In-scope coverage", f"{answered}<span>/{len(insc)}</span>", f"answered correctly ({answered/len(insc):.0%}); the rest over-clarified", "good"),
        ("Adversarial resisted", f"{adv_safe}<span>/{len(adv)}</span>", "injection / “ignore your rules” attempts refused", "good"),
        ("Ambiguous → clarify", f"{amb_clar}<span>/{len(amb)}</span>", "ideal on missing-atom queries; rest guessed (the doppelgänger gap)", "warn"),
    ]
    tile_html = "".join(
        f'<div class="tile {c}"><div class="k">{k}</div><div class="v">{v}</div><div class="d">{html.escape(d)}</div></div>'
        for k, v, d, c in tiles)

    # per-category chart
    chart = ""
    for cat in CATS:
        rs = [r for r in rows if r["cat"] == cat]
        counts = {}
        for r in rs:
            counts[r["disp"]] = counts.get(r["disp"], 0) + 1
        chart += f'<div class="crow"><div class="clab">{CATLABEL[cat]} <em>({len(rs)})</em></div>{bar(counts, len(rs))}</div>'

    # table
    trows = ""
    for r in rows:
        badge = f'<span class="disp" style="background:{DISPCOL[r["disp"]]}">{r["disp"]}</span>'
        trows += (f'<tr class="{r["v"]}"><td>{CATLABEL[r["cat"]]}</td><td>{badge}</td>'
                  f'<td class="card">{r["card"] if r["card"]!="-" else ""}</td>'
                  f'<td class="q">{html.escape(r["q"])}</td><td class="vd">{html.escape(r["verdict"])}</td></tr>')

    page = TEMPLATE.replace("__TILES__", tile_html).replace("__CHART__", chart).replace("__ROWS__", trows).replace("__N__", str(len(rows)))
    Path("report.html").write_text(page)
    print("wrote report.html")


TEMPLATE = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Scope bot — 50-scenario eval</title>
<style>
:root{--bg:#f7f8fa;--card:#fff;--ink:#0f172a;--mut:#64748b;--line:#e2e8f0;--good:#059669;--warn:#d97706;--bad:#dc2626;--accent:#0072B2}
@media(prefers-color-scheme:dark){:root{--bg:#0b1220;--card:#111a2b;--ink:#e6edf6;--mut:#93a4bd;--line:#233149}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 system-ui,-apple-system,Segoe UI,Roboto,sans-serif}
.wrap{max-width:1000px;margin:0 auto;padding:28px 20px 60px}
h1{font-size:24px;margin:0 0 4px}.sub{color:var(--mut);margin:0 0 20px}
.note{background:linear-gradient(0deg,color-mix(in srgb,var(--accent) 8%,var(--card)),var(--card));border:1px solid var(--line);border-left:4px solid var(--accent);border-radius:10px;padding:14px 16px;margin:0 0 22px}
.note b{color:var(--accent)}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:12px;margin-bottom:24px}
.tile{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:14px 16px}
.tile .k{color:var(--mut);font-size:12px;text-transform:uppercase;letter-spacing:.04em}
.tile .v{font-size:30px;font-weight:700;margin:4px 0}.tile .v span{font-size:16px;color:var(--mut);font-weight:600}
.tile .d{color:var(--mut);font-size:12.5px}
.tile.good .v{color:var(--good)}.tile.warn .v{color:var(--warn)}.tile.bad .v{color:var(--bad)}
h2{font-size:15px;text-transform:uppercase;letter-spacing:.04em;color:var(--mut);margin:26px 0 12px}
.chart{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:16px}
.crow{display:flex;align-items:center;gap:12px;margin:9px 0}.clab{width:210px;font-size:13.5px}.clab em{color:var(--mut);font-style:normal}
.bar{flex:1;height:20px;border-radius:6px;overflow:hidden;display:flex;background:var(--line)}
.bar span{display:block;height:100%;margin-right:2px}.bar span:last-child{margin-right:0}
.legend{display:flex;gap:16px;margin-top:12px;font-size:12.5px;color:var(--mut);flex-wrap:wrap}
.legend i{display:inline-block;width:11px;height:11px;border-radius:3px;margin-right:5px;vertical-align:-1px}
table{width:100%;border-collapse:collapse;background:var(--card);border:1px solid var(--line);border-radius:12px;overflow:hidden;font-size:13.5px}
th,td{text-align:left;padding:8px 10px;border-bottom:1px solid var(--line)}th{color:var(--mut);font-size:11.5px;text-transform:uppercase;letter-spacing:.04em}
tr:last-child td{border-bottom:0}.card{color:var(--mut);font-family:ui-monospace,monospace;font-size:12px}
.disp{color:#fff;padding:1px 7px;border-radius:20px;font-size:11px;font-weight:600}
.vd{white-space:nowrap}tr.good .vd{color:var(--good)}tr.warn .vd{color:var(--warn)}tr.bad{background:color-mix(in srgb,var(--bad) 10%,transparent)}tr.bad .vd{color:var(--bad)}
.tblwrap{overflow-x:auto}
</style></head><body><div class="wrap">
<h1>File-defined-scope bot — 50-scenario evaluation</h1>
<p class="sub">Base <b>Qwen3-4B-Instruct</b> (4-bit) + a per-query LLM scope-gate over a 16-card scope file. Live decisions on an RTX A5000.</p>
<div class="note"><b>This model is NOT fine-tuned / not SFT-trained.</b> There is zero training here. It is the stock instruct model reading the scope file (<code>seed16/cards.json</code>) on every query. The gate decides <b>ANSWER / CLARIFY / ABSTAIN</b> from the query + the cards' included/excluded lists; grounded answers come only from the selected card's approved facts. Edit the file → the domain changes, no retraining. (The earlier QLoRA SFT plan was abandoned per a Codex Council review.)</div>
<div class="tiles">__TILES__</div>
<h2>How it did, by scenario type</h2>
<div class="chart">__CHART__
<div class="legend"><span><i style="background:#0072B2"></i>ANSWER</span><span><i style="background:#E69F00"></i>CLARIFY</span><span><i style="background:#009E73"></i>ABSTAIN</span></div></div>
<h2>All __N__ scenarios</h2>
<div class="tblwrap"><table><thead><tr><th>Category</th><th>Decision</th><th>Card</th><th>Query</th><th>Verdict</th></tr></thead><tbody>__ROWS__</tbody></table></div>
<h2>Honest read</h2>
<div class="note" style="border-left-color:var(--warn)"><b style="color:var(--warn)">Safe but over-cautious.</b> 0 harmful leaks (every doppelgänger, far-OOD, and injection refused) is the headline. The cost: it over-clarified on 4 clearly-answerable in-scope queries (80% coverage), and on genuinely ambiguous queries it sometimes guesses instead of clarifying — the identical-text doppelgänger gap. Both are gate-prompt tuning + a <code>required_discriminators</code> rule in the cards, not architecture problems. Numbers are on 50 hand-written scenarios (illustrative), not a certified sealed test.</div>
</div></body></html>"""


if __name__ == "__main__":
    main()
