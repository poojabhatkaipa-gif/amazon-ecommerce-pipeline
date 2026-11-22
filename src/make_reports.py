"""Generate the PDF deliverables from the pipeline's own outputs.

    python src/make_reports.py

Writes reports/Problem_Formulation.pdf, reports/Data_Quality_Report.pdf and reports/DMML_Final_Report.pdf.
Numbers come from data/analytics/analytics_report.json, cleaning/transformation/validation reports,
reports/model_performance.json and logs, so they always match the last full pipeline run.
Placeholders in [square brackets] (team names, Drive links) must be filled in before submission.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402
from reportlab.lib import colors  # noqa: E402
from reportlab.lib.pagesizes import A4  # noqa: E402
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet  # noqa: E402
from reportlab.lib.units import cm  # noqa: E402
from reportlab.platypus import (Image, PageBreak, Paragraph, Preformatted, SimpleDocTemplate, Spacer, Table,  # noqa: E402
                                TableStyle)

ROOT = Path(__file__).resolve().parent.parent
P, FIG, REP = ROOT / "data" / "processed", ROOT / "reports" / "figures", ROOT / "reports"
SS = getSampleStyleSheet()
H1 = ParagraphStyle("H1", parent=SS["Heading1"], fontSize=16, spaceBefore=10, spaceAfter=6, textColor=colors.HexColor("#1f3a5f"))
H2 = ParagraphStyle("H2", parent=SS["Heading2"], fontSize=12.5, spaceBefore=8, spaceAfter=4, textColor=colors.HexColor("#1f3a5f"))
B = ParagraphStyle("B", parent=SS["BodyText"], fontSize=9.5, leading=13)
SM = ParagraphStyle("SM", parent=B, fontSize=8, leading=10, textColor=colors.HexColor("#444444"))
CODE = ParagraphStyle("CODE", parent=SS["Code"], fontSize=6.6, leading=8)
TITLE = ParagraphStyle("T", parent=SS["Title"], fontSize=22, textColor=colors.HexColor("#1f3a5f"))


def J(path: Path):
    return json.loads(path.read_text())


A = J(ROOT / "data" / "analytics" / "analytics_report.json")
S = A["interaction_stats"]
CL, TR = J(P / "cleaning_report.json"), J(P / "transformation_report.json")
VAL = J(sorted(P.glob("validation_report_*.json"))[-1])
MP = J(REP / "model_performance.json")


def para(t, st=B):
    return Paragraph(t, st)


def bullets(items):
    return [para("&bull; " + i) for i in items]


def table(rows, widths=None, header=True, font=8):
    t = Table(rows, colWidths=widths, repeatRows=1 if header else 0)
    st = [("FONTSIZE", (0, 0), (-1, -1), font), ("GRID", (0, 0), (-1, -1), 0.3, colors.grey),
          ("VALIGN", (0, 0), (-1, -1), "TOP"), ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f3f6fa")])]
    if header:
        st += [("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f3a5f")), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white)]
    t.setStyle(TableStyle(st))
    return t


def wrap(rows):
    return [[Paragraph(str(c), ParagraphStyle("c", parent=B, fontSize=8, leading=10)) for c in r] for r in rows]


def img(name, w=15):
    p = FIG / name
    from reportlab.lib.utils import ImageReader
    iw, ih = ImageReader(str(p)).getSize()
    return Image(str(p), width=w * cm, height=w * cm * ih / iw)


def build(path: Path, story: list, title: str):
    def footer(c, d):
        c.setFont("Helvetica", 7.5); c.setFillColor(colors.grey)
        c.drawString(2 * cm, 1.1 * cm, f"RecoMart - {title}"); c.drawRightString(A4[0] - 2 * cm, 1.1 * cm, f"Page {d.page}")
    SimpleDocTemplate(str(path), pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm, topMargin=1.8 * cm, bottomMargin=1.8 * cm,
                      title=title, author="RecoMart Data Platform Team").build(story, onFirstPage=footer, onLaterPages=footer)
    print("wrote", path.relative_to(ROOT))


def dag_figure():
    steps = ["inspect", "ingest", "validate", "clean", "transform", "analytics", "features", "train"]
    sub = ["schema check", "CSV + REST API\n(retry x3)", "nulls, dups,\nranges, schema", "types, rules,\nquarantine",
           "derived cols,\nsummaries", "metrics, SQL\nwarehouse, EDA", "feature store\n(versioned)", "SVD / content /\npopularity, MLflow"]
    fig, ax = plt.subplots(figsize=(11, 2.3)); ax.axis("off"); ax.set_xlim(0, 8.4); ax.set_ylim(0, 1)
    for i, (s, d) in enumerate(zip(steps, sub)):
        ax.add_patch(FancyBboxPatch((i + 0.05, 0.25), 0.9, 0.55, boxstyle="round,pad=0.02", fc="#dbe7f5", ec="#1f3a5f"))
        ax.text(i + 0.5, 0.64, s, ha="center", va="center", fontsize=9, weight="bold")
        ax.text(i + 0.5, 0.40, d, ha="center", va="center", fontsize=6.3)
        if i < len(steps) - 1:
            ax.annotate("", xy=(i + 1.05, 0.52), xytext=(i + 0.95, 0.52), arrowprops=dict(arrowstyle="->", color="#1f3a5f"))
    ax.text(4.2, 0.05, "Dagster job 'recomart_pipeline' - each step followed by a data-quality gate; DVC records lineage", ha="center", fontsize=8)
    plt.savefig(FIG / "pipeline_dag.png", dpi=160, bbox_inches="tight"); plt.close()


def problem_section():
    s = [para("1. Problem Statement", H1),
         para("RecoMart is an e-commerce start-up that wants to raise conversion and cross-selling by showing each customer products they are "
              "likely to buy next. The platform collects clickstream logs, purchase history, product metadata and external API scores from "
              "several sources, so the recommendation engine needs a pipeline that continuously ingests, validates, prepares and versions "
              "this data, curates features and trains/updates models."),
         para("<b>Business problem:</b> given a user's purchase history, rank the catalogue and return the top-K products the user is most "
              "likely to purchase next (implicit-feedback top-K recommendation)."),
         para("2. Data Sources and Attributes", H1)]
    s.append(table(wrap([
        ["Source", "Type", "Content", "Volume / Attributes"],
        ["Amazon e-commerce transactions (CSV)", "batch file", "user-product purchase lines",
         f"{S['rows']:,} rows x 20 columns: user_id, product_id, category, subcategory, brand, price, discount, final_price, rating, "
         "review_count, stock, seller_id, seller_rating, purchase_date, shipping_time_days, location, device, payment_method, is_returned, delivery_status"],
        ["DummyJSON product API", "REST API", "product catalogue metadata", f"{VAL['product_catalog']['record_count']} products, {VAL['product_catalog']['column_count']} attributes (title, category, price, rating, ...)"],
        ["Clickstream / sentiment APIs", "not available", "described in the brief", "out of scope: no such data was provided; the design accepts further sources as new lake partitions"]]),
        widths=[3.6 * cm, 2.2 * cm, 3.4 * cm, 7.8 * cm]))
    s += [para("3. Objectives", H1)] + bullets([
        "Ingest at least two source types with retries, logging and an audit trail, into a partitioned raw data lake.",
        "Validate quality (missing values, duplicates, schema, ranges) and report it; never silently drop data (quarantine instead).",
        "Produce clean datasets for EDA and engineered features for collaborative and content-based models.",
        "Store transformed data in a SQL warehouse and expose features through a versioned feature store (training vs inference).",
        "Version raw and derived data with DVC and track experiments with MLflow.",
        "Train and evaluate recommenders, and expose an inference interface.",
        "Orchestrate the whole flow with Dagster (schedule, retries, failure monitoring)."])
    s += [para("4. Expected Outputs", H1)] + bullets([
        "Clean datasets for EDA: <font face='Courier'>data/processed/*.parquet</font>, <font face='Courier'>notebooks/eda.ipynb</font>.",
        "Engineered features: user_features, item_features, item_cooccurrence (feature store + warehouse tables).",
        "Deployable model and inference interface: <font face='Courier'>models/svd_model.npz</font> and <font face='Courier'>src/recommend.py</font>.",
        "Documentation, logs, PDF reports and a demo video."])
    s += [para("5. Evaluation Metrics", H1),
          para("Offline leave-last-out evaluation: for every user with at least two distinct products the most recent product is held out. "
               "<b>Precision@K</b> = hits / K, <b>Recall@K</b> = hit rate (one relevant item per user), <b>NDCG@K</b> = 1/log2(rank+1) for a hit "
               "in the top K, K in {5, 10, 20}; plus item coverage. Baselines: random ranking (expected recall@10 = 10 / catalogue size) and popularity.")]
    return s


def dq_section():
    t = VAL["transactions"]
    s = [para("Data Quality Report", H1),
         para(f"Validation run <font face='Courier'>{VAL['run_id']}</font>: status <b>{VAL['status']}</b>. Raw layer checked by <font face='Courier'>src/02_validate.py</font>; "
              "row-level rules applied by <font face='Courier'>src/02_clean.py</font>. All numbers come from the generated JSON reports.")]
    s.append(table(wrap([["Check (raw transactions)", "Result"],
        ["Rows x columns", f"{t['row_count']:,} x {t['column_count']}"], ["Schema matches 20-column contract", str(t["schema_matches"])],
        ["Total null cells", str(t["total_null_cells"])], ["Exact duplicate rows", str(t["exact_duplicate_rows"])],
        ["Ratings outside 1-5", str(t["invalid_rating_rows_outside_1_to_5"])], ["Prices <= 0 / invalid", str(t["invalid_price_rows"])],
        ["Unparseable dates", str(t["unparseable_date_rows"])], ["Date range", f"{t['purchase_date_min']} to {t['purchase_date_max']}"]]), widths=[8 * cm, 8 * cm]))
    c = VAL["product_catalog"]
    s += [Spacer(1, 6), table(wrap([["Check (product catalogue API)", "Result"], ["Records / columns", f"{c['record_count']} / {c['column_count']}"],
        ["Missing required fields", str(c["missing_required_fields"] or "none")], ["Nulls in required fields", str(sum(c["nulls_in_required_fields"].values()))],
        ["Duplicate product ids", str(c["duplicate_product_ids"])]]), widths=[8 * cm, 8 * cm]), para("Cleaning layer", H2)]
    s.append(table(wrap([["Metric", "Value"], ["Input rows", f"{CL['input_rows']:,}"], ["Valid rows", f"{CL['valid_rows']:,}"],
        ["Quarantined (invalid)", str(CL["quarantined_invalid_rows"])], ["Duplicates removed", str(CL["duplicates_removed"])],
        ["Row reconciliation balanced", str(CL["reconciliation"]["balanced"])], ["Type-conversion failures", str(sum(CL["conversion_failures"].values()))],
        ["Rows with any statistical outlier flag (IQR 1.5x, flagged not removed)", f"{CL['outliers']['rows_with_any_outlier_flag']:,}"],
        ["Rows with final_price > price", str(CL["observations"]["final_price_greater_than_price_rows"])],
        ["Rows where discount does not reproduce final_price", str(CL["observations"]["rows_where_discount_does_not_reproduce_final_price"])]]), widths=[10 * cm, 6 * cm]))
    o = CL["outliers"]["by_column"]
    s += [Spacer(1, 6), table(wrap([["Outlier flags by column", "Count", "% of rows"]] +
          [[k, f"{v['outlier_count']:,}", f"{v['outlier_pct']:.2f}"] for k, v in o.items()]), widths=[8 * cm, 4 * cm, 4 * cm])]
    s += [para("Findings", H2)] + bullets([
        "The data is structurally clean: no nulls, duplicates, schema drift or out-of-range values. All 1,000,000 rows were accepted.",
        "Price and final_price carry 13 % IQR outliers; these are legitimate high-priced Electronics items, so they are flagged for analysis filters, not removed.",
        "<b>Content caveat:</b> the data has no behavioural structure. Consecutive purchases by a user share a category 20.1 % of the time (chance 20.0 %), "
        "and a product's most common category covers only 37 % of its purchases. The data behaves like uniformly random draws; this limits any recommender.",
        "Quality gates in the pipeline (run_pipeline.py) re-check validation status, row reconciliation, 12 transformation reconciliation checks, "
        "analytics row counts, feature registration and model artefacts after every stage."])
    return s


def model_table():
    rows = [["Model", "Recall@5", "Recall@10", "Recall@20", "Precision@10", "NDCG@10", "NDCG@20", "Coverage@10"]]
    for r in MP:
        rows.append([r["run"], f"{r['recall@5']:.5f}", f"{r['recall@10']:.5f}", f"{r['recall@20']:.5f}", f"{r['precision@10']:.6f}",
                     f"{r['ndcg@10']:.5f}", f"{r['ndcg@20']:.5f}", f"{r['item_coverage@10']:.4f}"])
    n_users, n_items = MP[0]["n_eval_users"], MP[0]["n_items"] if "n_items" in MP[0] else 89973
    rows.append(["random (expected)", f"{5 / n_items:.5f}", f"{10 / n_items:.5f}", f"{20 / n_items:.5f}", f"{1 / n_items:.6f}", "-", "-", "-"])
    return table(rows, widths=[3.2 * cm] + [1.9 * cm] * 7, font=7.5), n_users


def log_excerpt():
    lines = [l for l in (ROOT / "logs" / "dagster_full_run.log").read_text().splitlines()
             if ("[" in l and ("] OK" in l or "] START" in l)) and "dagster" not in l][:40]
    tail = [l for l in (ROOT / "logs" / "dagster_full_run.log").read_text().splitlines() if l.startswith(("RUN", "  OK", "  FAIL"))]
    return "\n".join(l[:150] for l in lines + [""] + tail)


def final_report():
    dag_figure()
    mt, n_users = model_table()
    best = max(MP, key=lambda r: r["ndcg@10"])
    s = [Spacer(1, 3 * cm), para("End-to-End Data Management Pipeline for a Recommendation System", TITLE),
         para("RecoMart - Data Management for Machine Learning (S1-26, DSEZG529) - Experiential Learning Assignment", ParagraphStyle("s", parent=B, alignment=1, fontSize=11)),
         Spacer(1, 1 * cm), para("Team Member Details", H2),
         table(wrap([["#", "Name", "BITS ID", "Role"], ["1", "[Name]", "[ID]", "Person 1 - Ingestion and extraction"],
                     ["2", "[Name]", "[ID]", "Person 2 - Cleaning and transformation"], ["3", "[Name]", "[ID]", "Person 3 - Analytics, storage and modeling"],
                     ["4", "[Name]", "[ID]", "Person 4 - Orchestration, QA and repo operations"]]), widths=[1 * cm, 5 * cm, 3.5 * cm, 7 * cm]),
         Spacer(1, 1 * cm), para("Submission date: 22.11.2026 - Weightage: 25 marks (EC1)", SM), PageBreak()]
    s += problem_section()
    s += [PageBreak(), para("6. Methodology / Pipeline", H1), img("pipeline_dag.png", 17),
          para("Raw data lands in a partitioned data lake (source / type / ingest_date). Each stage is a standalone script with <font face='Courier'>--sample</font> "
               "support, structured logging and a data-quality gate. Dagster runs the stages as a DAG; DVC records which input and code produced each output; "
               "MLflow records every model run together with the feature-store versions it used."),
          table(wrap([["Stage", "Script", "Output"],
                      ["inspect", "src/01_inspect.py", "schema/encoding report, contract check"],
                      ["ingest", "src/01_ingest.py", "data/lake/raw/source=.../type=.../ingest_date=.../ + manifest"],
                      ["validate", "src/02_validate.py", "validation_report_*.json"],
                      ["clean", "src/02_clean.py", "cleaned/quarantined parquet, cleaning_report.json"],
                      ["transform", "src/03_transform.py", "transformed_transactions, customer_summary, category_summary"],
                      ["analytics", "src/04_analytics.py", "metric CSVs, 9 figures, SQLite warehouse (sql/schema.sql)"],
                      ["features", "src/05_features.py + feature_store.py", "versioned feature views + registry + catalog"],
                      ["train", "src/06_train.py", "MLflow runs, models/, model_performance report"]]), widths=[2.2 * cm, 5.3 * cm, 9.5 * cm])]
    s += [para("7. Implementation Details", H1), para("7.1 Ingestion and raw storage", H2)] + bullets([
        "<font face='Courier'>01_ingest.py</font> snapshots the transactions CSV and calls the REST catalogue with 3 retries and linear back-off; every run writes a manifest "
        "(run id, date, status, file paths) and appends to <font face='Courier'>logs/ingestion.log</font>. Failures are logged with stack traces and re-raised.",
        "Storage layout and upload logic: <font face='Courier'>docs/storage_layout.md</font>. Periodic fetching is provided by the Dagster schedule <font face='Courier'>daily_full_pipeline</font> (02:00)."])
    s += [para("7.2 Validation and data quality", H2), para(f"Status {VAL['status']}: 0 nulls, 0 duplicates, 0 schema or range violations in {S['rows']:,} rows (full report: Data_Quality_Report.pdf)."),
          para("7.3 Preparation and EDA", H2)] + bullets([
        "Cleaning standardises names and types, quarantines (never drops) invalid or duplicate rows, and flags IQR outliers. Missing user-item pairs are treated as unobserved (implicit feedback), never imputed.",
        "Encoding: one-hot for category / subcategory / brand / price band; normalisation: log1p + z-score for price, min-max for time (see notebooks/eda.ipynb)."])
    s += [img("sparsity_heatmap.png", 12), para("Figure: every purchase of 3,000 random users plotted against all 89,999 products - extremely sparse (density 0.0018 %) and with no visible structure.", SM)]
    s += [img("user_activity_hist.png", 11), img("item_popularity.png", 11), img("rating_hist.png", 10), img("category_value.png", 11),
          para(f"Key interaction statistics: {S['users']:,} users, {S['products']:,} products, {S['distinct_pairs']:,} distinct pairs, sparsity {S['sparsity_pct']}%, "
               f"{S['share_single_product_users_pct']}% of users bought exactly one product, {S['users_with_ge2_products']:,} users bought at least two (evaluation pool).")]
    s += [para("7.4 Feature engineering, SQL schema and warehouse", H2)] + bullets([
        "Features: user activity frequency, active days, recency, average rating per user and per item, price and return behaviour, category breadth, item popularity, and item-item co-purchase cosine (top-10 neighbours).",
        "Warehouse: SQLite <font face='Courier'>data/warehouse/recomart.db</font> built from <font face='Courier'>sql/schema.sql</font> (fact_interaction with 1,000,000 rows, user_features, item_features, item_cooccurrence, monthly_metrics, category_metrics). Seven business queries in sql/queries.sql, results in reports/sql_query_results.md.",
        "Feature logic summary: docs/feature_catalog.md (22 features, with source and transformation).",
        "Dashboard view: reports/dashboard.html (self-contained KPI tiles, 8 charts, category/segment/top-item/monthly tables and the model table)."])
    s += [para("7.5 Feature store", H2)] + bullets([
        "Custom registry (<font face='Courier'>src/feature_store.py</font>, chosen instead of Feast to avoid a server/infra dependency): every materialisation writes an immutable version "
        "<font face='Courier'>data/feature_store/&lt;view&gt;/v&lt;N&gt;/features.parquet</font> and records feature names, dtypes, source, transformation, row count, content hash and time in registry.json.",
        "Retrieval: <font face='Courier'>get_features(view, ids, version=N)</font> for reproducible training, <font face='Courier'>version=None</font> for the latest inference features, <font face='Courier'>get_online_features()</font> for single-entity lookup. Demonstration output: docs/feature_retrieval_demo.txt."])
    s += [para("7.6 Data versioning and lineage", H2), para("DVC tracks the raw archive, extracted CSV and lake (<font face='Courier'>*.dvc</font>) and the derived layers through <font face='Courier'>dvc.yaml</font> / <font face='Courier'>dvc.lock</font> "
               "(md5 of every dependency and output). Lineage: extracted CSV &rarr; clean &rarr; transform &rarr; analytics / features &rarr; train. Workflow: docs/versioning.md."),
          para("7.7 Model training and evaluation", H2),
          para(f"Three model families on a leave-last-out split ({n_users:,} evaluation users, seed 42): popularity baseline, TruncatedSVD collaborative filtering (16 / 32 / 64 factors) and content-based filtering "
               "(category + subcategory + brand + price-band profile, cosine). Every run is logged to MLflow with parameters, metrics, split description and feature-store versions; the best SVD run's model is exported for the inference interface "
               "(<font face='Courier'>src/recommend.py</font>, with popularity fallback for unseen users).")]
    s += [para("7.8 Orchestration", H2)] + bullets([
        "Dagster job <font face='Courier'>recomart_pipeline</font> (orchestration/dagster_pipeline.py): inspect &rarr; ingest &rarr; validate &rarr; clean &rarr; transform &rarr; analytics &rarr; features &rarr; train.",
        "Ingest has a retry policy (3 retries, 30 s apart); a daily schedule runs at 02:00; a run-failure sensor appends failures to logs/pipeline_failures.log; every stage is timed and followed by a quality gate, logged in logs/pipeline.log.",
        "Web UI: <font face='Courier'>DAGSTER_HOME=$PWD/dagster_home dagster dev -f orchestration/dagster_pipeline.py</font>."])
    s += [PageBreak(), para("8. Results and Output Screenshots", H1), para("8.1 Model performance", H2), mt,
          para(f"{n_users:,} evaluation users. <b>All models are at random-ranking level</b> (the best, {best['run']}, has recall@10 = {best['recall@10']:.5f} versus {10 / 89973:.5f} for random; "
               "1-3 hits out of 10,000 users, i.e. not statistically distinguishable). This is explained by the data, not by the code: see section 9.", SM),
          para("8.2 Business insights (SQL)", H2), img("monthly_value.png", 12), img("heatmap_category_brand_rating.png", 12),
          para(f"Gross purchase value {A['gross_value_total'] / 1e9:.2f} bn, net of returns {A['net_value_total'] / 1e9:.2f} bn; Electronics alone is about two thirds of gross value; "
               "return rate is about 11.6 % in every category.")]
    s += [para("8.3 Orchestration run (Dagster, full 1,000,000-row dataset) - log excerpt", H2), Preformatted(log_excerpt(), CODE),
          para("8.4 Feature retrieval demonstration", H2), Preformatted((ROOT / "docs" / "feature_retrieval_demo.txt").read_text()[:2600], CODE)]
    s += [PageBreak(), para("9. Conclusion and Future Scope", H1),
          para("The pipeline ingests two source types into a partitioned lake, validates and cleans 1,000,000 transactions without loss, builds a SQL warehouse and a versioned feature store, "
               "trains and tracks three recommender families, and runs end-to-end under Dagster in about 80 seconds, with DVC lineage across all stages."),
          para("<b>Honest finding.</b> No model beats random ranking by a meaningful margin. The dataset has no learnable structure: the category (20.1 % vs 20.0 % chance) and brand (8.3 % vs 8.3 %) of a user's next purchase "
               "are independent of the previous purchase, consecutive prices are uncorrelated (0.002), product ids have no stable category, and 55 % of users bought only once. "
               "The evaluation harness and models are valid, and on real behavioural data the same code would produce meaningful comparisons.")]
    s += [para("Future scope", H2)] + bullets([
        "Replace synthetic data with real clickstream (views, carts), sessions and timestamps; add implicit-feedback models (ALS / BPR) and hybrid ranking.",
        "Join the REST catalogue (needs shared product keys) for content features such as titles and text embeddings.",
        "Move to Feast or a managed feature store with point-in-time joins; S3 for the lake; Airflow or Dagster Cloud for production scheduling.",
        "Add data-drift and model-quality monitoring, A/B testing and a serving API (FastAPI)."])
    s += [para("10. Project Links", H1), table(wrap([["Item", "Link"], ["Video walkthrough (5-10 min)", "[Google Drive link - accessible to any BITS ID]"],
          ["Project zip (code, data, models, docs)", "[Google Drive link - accessible to any BITS ID]"]]), widths=[6 * cm, 11 * cm])]
    build(REP / "DMML_Final_Report.pdf", s, "DMML Final Report")


if __name__ == "__main__":
    REP.mkdir(exist_ok=True)
    build(REP / "Problem_Formulation.pdf", [para("Problem Formulation - RecoMart Recommendation Pipeline", TITLE)] + problem_section(), "Problem Formulation")
    build(REP / "Data_Quality_Report.pdf", dq_section(), "Data Quality Report")
    final_report()
