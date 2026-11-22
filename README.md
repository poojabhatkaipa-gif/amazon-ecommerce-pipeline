# RecoMart - End-to-End Data Management Pipeline for a Recommendation System

DMML Experiential Learning assignment (S1-26, DSEZG529). Amazon e-commerce transactions (1,000,000 rows) + DummyJSON product API
-> raw lake -> validation -> cleaning -> transformation -> analytics / SQL warehouse -> feature store -> models -> Dagster orchestration, versioned with DVC.

## Quick start
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # + dagster dagster-webserver dvc reportlab pymupdf tabulate (see requirements.txt)
export DVC_NO_ANALYTICS=1 MLFLOW_DISABLE_AGENT_HINT=1

python src/run_pipeline.py --sample --skip-ingest     # fast dev run (10,000 rows, ~15 s)
python src/run_pipeline.py --skip-ingest              # full run on the existing lake (~80 s)
python src/run_pipeline.py                            # incl. live ingestion (needs internet for the REST API)

DAGSTER_HOME=$PWD/dagster_home python orchestration/run_job.py --skip-ingest   # same, as a Dagster job (in-process)
DAGSTER_HOME=$PWD/dagster_home dagster dev -f orchestration/dagster_pipeline.py # web UI: DAG, runs, schedule

python src/recommend.py U356787 --k 5                 # inference
mlflow ui --backend-store-uri sqlite:///mlflow/mlflow.db   # experiment tracking UI
python src/make_dashboard.py                          # reports/dashboard.html (open in a browser)
python src/run_queries.py                             # run sql/queries.sql
python src/make_reports.py                            # rebuild PDF reports
pytest tests -q                                       # 51 tests
```

## Layout
```
src/            01_inspect 01_ingest 02_validate 02_clean 03_transform 04_analytics 05_features 06_train
                feature_store.py recommend.py run_pipeline.py run_queries.py feature_demo.py make_reports.py
orchestration/  dagster_pipeline.py (job, schedule, failure sensor), run_job.py
sql/            schema.sql, queries.sql        notebooks/  eda.ipynb (+ execute_notebook.py)
docs/           data_contract_person2.md, data_contract_person3.md, storage_layout.md, versioning.md,
                feature_catalog.md, feature_retrieval_demo.txt, demo_script.md
reports/        Problem_Formulation.pdf, Data_Quality_Report.pdf, DMML_Final_Report.pdf, model_performance.md,
                sql_query_results.md, eda.html, figures/
data/ (DVC)     raw, extracted, lake/raw, processed, analytics, warehouse, feature_store      models/ (DVC)
dvc.yaml dvc.lock  *.dvc    lineage and data versions          mlflow/  experiment store        logs/  audit trails
```

## Team roles
| Person | Scope | Main files |
|---|---|---|
| 1 | ingestion and extraction | 01_inspect.py, 01_ingest.py, data/lake |
| 2 | cleaning and transformation | 02_clean.py, 03_transform.py, docs/data_contract_person2.md |
| 3 | analytics, storage, modeling | 04_analytics.py, 05_features.py, 06_train.py, feature_store.py, sql/, docs/data_contract_person3.md |
| 4 | orchestration, QA, repo | run_pipeline.py, orchestration/, dvc.yaml, README.md, .gitignore |

## Important result caveat
Every model is at random-ranking level because the dataset has no learnable structure (see `notebooks/eda.ipynb` section 5 and `reports/model_performance.md`).

## Notes
* Git: `data/`, `models/` are tracked by DVC (pointer files in git); `logs/`, `mlflow/`, `dagster_home/`, `.venv/` are git-ignored. Remote storage: `../amazon-ecommerce-dvc-remote`.
* Ingestion copies the 129 MB CSV into a new lake partition on every run - use `--skip-ingest` for repeat runs.
