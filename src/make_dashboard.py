"""Person 3 - static HTML dashboard (single file, no server, opens in any browser).

    python src/make_dashboard.py     ->  reports/dashboard.html

KPIs, tables and charts come from data/analytics/*.csv, analytics_report.json and reports/model_performance.json.
"""
from __future__ import annotations

import base64
import html
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
AN, FIG, REP = ROOT / "data" / "analytics", ROOT / "reports" / "figures", ROOT / "reports"


def b64(name: str) -> str:
    return base64.b64encode((FIG / name).read_bytes()).decode()


def tbl(df: pd.DataFrame, fmt: dict | None = None) -> str:
    df = df.copy()
    for c, f in (fmt or {}).items():
        df[c] = df[c].map(f.format)
    return df.to_html(index=False, border=0, classes="t", escape=True)


def main() -> Path:
    rep = json.loads((AN / "analytics_report.json").read_text())
    s = rep["interaction_stats"]
    cat = pd.read_csv(AN / "category_metrics.csv"); mon = pd.read_csv(AN / "monthly_metrics.csv")
    seg = pd.read_csv(AN / "user_segments.csv"); top = pd.read_csv(AN / "top_items.csv").head(10)
    dev = pd.read_csv(AN / "device_metrics.csv"); pay = pd.read_csv(AN / "payment_method_metrics.csv")
    mp = pd.DataFrame(json.loads((REP / "model_performance.json").read_text()))
    kpis = [("Transactions", f"{s['rows']:,}"), ("Users", f"{s['users']:,}"), ("Products", f"{s['products']:,}"),
            ("Gross value", f"{rep['gross_value_total'] / 1e9:.2f} bn"), ("Net of returns", f"{rep['net_value_total'] / 1e9:.2f} bn"),
            ("Return rate", f"{100 * (1 - rep['net_value_total'] / rep['gross_value_total']):.1f}% of value"),
            ("Sparsity", f"{s['sparsity_pct']:.3f}%"), ("Single-purchase users", f"{s['share_single_product_users_pct']}%")]
    tiles = "".join(f"<div class=k><div class=v>{html.escape(v)}</div><div class=l>{html.escape(k)}</div></div>" for k, v in kpis)
    figs = "".join(f"<figure><img src='data:image/png;base64,{b64(n)}'><figcaption>{c}</figcaption></figure>" for n, c in [
        ("monthly_value.png", "Monthly gross value"), ("category_value.png", "Gross value by category"),
        ("rating_hist.png", "Rating distribution"), ("user_activity_hist.png", "User activity"),
        ("item_popularity.png", "Item popularity"), ("sparsity_heatmap.png", "Interaction matrix sample"),
        ("heatmap_category_brand_rating.png", "Mean rating: category x brand"), ("correlation_heatmap.png", "Numeric correlations")])
    models = mp[["run", "recall@10", "precision@10", "ndcg@10", "item_coverage@10"]]
    page = f"""<!doctype html><meta charset=utf-8><title>RecoMart dashboard</title>
<style>body{{font:14px -apple-system,Helvetica,Arial;margin:24px;color:#222;max-width:1200px}}h1{{color:#1f3a5f}}h2{{color:#1f3a5f;border-bottom:1px solid #ccd;padding-bottom:3px;margin-top:28px}}
.kp{{display:flex;flex-wrap:wrap;gap:10px}}.k{{background:#eef3fa;border-radius:8px;padding:10px 16px;min-width:130px}}.v{{font-size:22px;font-weight:600;color:#1f3a5f}}.l{{font-size:12px;color:#555}}
.g{{display:grid;grid-template-columns:repeat(auto-fit,minmax(380px,1fr));gap:14px}}figure{{margin:0}}img{{max-width:100%}}figcaption{{font-size:12px;color:#555}}
table.t{{border-collapse:collapse;font-size:12.5px}}table.t td,table.t th{{padding:3px 10px;border-bottom:1px solid #e3e7ee;text-align:right}}table.t th{{background:#1f3a5f;color:#fff}}.warn{{background:#fff4e0;border-left:4px solid #e8a33d;padding:8px 12px}}</style>
<h1>RecoMart - Analytics dashboard</h1><p>Source: transformed_transactions ({s['date_min']} to {s['date_max']}). Metric definitions: docs/data_contract_person3.md.</p>
<div class=kp>{tiles}</div>
<h2>Charts</h2><div class=g>{figs}</div>
<h2>Category performance</h2>{tbl(cat, {'gross_value': '{:,.0f}', 'net_value': '{:,.0f}', 'avg_final_price': '{:,.0f}', 'avg_rating': '{:.2f}', 'return_rate_pct': '{:.2f}'})}
<h2>User segments</h2>{tbl(seg)}
<h2>Top 10 items by purchases</h2>{tbl(top, {'avg_rating': '{:.2f}', 'avg_final_price': '{:,.0f}'})}
<h2>Device and payment method</h2>{tbl(dev, {'gross_value': '{:,.0f}', 'return_rate_pct': '{:.2f}'})}{tbl(pay, {'gross_value': '{:,.0f}', 'return_rate_pct': '{:.2f}'})}
<h2>Monthly trend</h2>{tbl(mon, {'gross_value': '{:,.0f}', 'net_value': '{:,.0f}', 'return_rate_pct': '{:.2f}', 'avg_rating': '{:.2f}'})}
<h2>Recommender evaluation (leave-last-out)</h2><p class=warn>All models are at random-ranking level (random recall@10 is about 0.00011). The data has no learnable structure - see notebooks/eda.ipynb section 5.</p>
{tbl(models, {'recall@10': '{:.5f}', 'precision@10': '{:.6f}', 'ndcg@10': '{:.5f}', 'item_coverage@10': '{:.4f}'})}"""
    out = REP / "dashboard.html"
    out.write_text(page)
    print("wrote", out.relative_to(ROOT), f"({out.stat().st_size // 1024} KB)")
    return out


if __name__ == "__main__":
    main()
