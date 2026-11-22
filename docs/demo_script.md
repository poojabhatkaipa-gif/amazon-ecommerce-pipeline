# Demo video script (target 8 minutes)

Record the screen (QuickTime: File > New Screen Recording). Upload to Google Drive and set sharing to "Anyone at BITS (with the link)".

| Time | Show | Say |
|---|---|---|
| 0:00-0:45 | Report title page, team table | Business problem: RecoMart wants personalised recommendations; pipeline from ingestion to model. |
| 0:45-1:30 | `tree -L 2` / `docs/storage_layout.md` | Lake partitioned by source/type/date; two sources: CSV transactions and REST product API. |
| 1:30-2:30 | `logs/ingestion.log`, `data/processed/validation_report_*.json`, Data_Quality_Report.pdf | Retries, logging, validation checks (nulls, duplicates, ranges), 0 issues. |
| 2:30-3:30 | `reports/eda.html` (plots) | Sparsity 99.998 %, 55 % single-purchase users, popularity curve. |
| 3:30-4:15 | `sqlite3 data/warehouse/recomart.db < sql/queries.sql` | Warehouse schema and business queries. |
| 4:15-5:15 | `docs/feature_catalog.md`, run `python src/feature_demo.py` | Feature store: versions, training (pinned) vs inference (latest) retrieval. |
| 5:15-6:30 | `mlflow ui --backend-store-uri sqlite:///mlflow/mlflow.db` , `reports/model_performance.md` | Models, metrics, run ids. Be upfront: all models are at random level because the data has no signal (EDA section 5). |
| 6:30-7:15 | `python src/recommend.py U356787 --k 5` | Inference interface, cold-start fallback. |
| 7:15-8:30 | `dagster dev -f orchestration/dagster_pipeline.py` > Launchpad > run | DAG, retries, schedule, failure sensor; `dvc dag` for lineage. |
