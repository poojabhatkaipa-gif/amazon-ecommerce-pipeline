# Data Contract - Person 3 (Analytics, Storage & Modeling)

Owner: Person 3 · Inputs from Person 2 (`data/processed/transformed_transactions.parquet`) · Consumers: Person 4 (orchestration), final report.

> Counts below come from the full 1,000,000-row run. The generated files (`data/analytics/analytics_report.json`,
> `reports/model_performance.json`, `data/feature_store/registry.json`) are the source of truth for any given run.

## 1. Entry points (each exposes `main()` and a `--sample` flag)

| Step | Script | Reads | Writes |
|---|---|---|---|
| 4 | `src/04_analytics.py` | `transformed_transactions[_sample].parquet` | `data/analytics/*.csv`, `reports/figures/*.png`, `data/warehouse/recomart[_sample].db`, `analytics_report.json`, `logs/analytics.log` |
| 5 | `src/05_features.py` | same | `data/feature_store/<view>/v<N>/features.parquet`, `registry.json`, `docs/feature_catalog.md`, warehouse feature tables, `logs/features.log` |
| 6 | `src/06_train.py` | same + feature-store versions | `models/svd_model.npz`, `models/popularity.npz`, `models/train_matrix.npz`, `mlflow/` tracking store, `reports/model_performance.{md,json}`, `logs/training.log` |
| - | `src/recommend.py <user_id> --k N` | `models/*`, feature store | recommendations on stdout |
| - | `src/run_queries.py` | `sql/queries.sql`, warehouse | `reports/sql_query_results.md` (7 business queries) |
| - | `src/make_dashboard.py` | `data/analytics/*`, `reports/model_performance.json` | `reports/dashboard.html` (self-contained KPI/chart dashboard) |
| - | `src/feature_demo.py` | feature store | training / inference / online retrieval demo |

Order: 04 -> 05 -> 06 (06 records the feature-store versions in each MLflow run, so 05 must run first).
**Warning:** `04_analytics.py` rebuilds the warehouse from scratch, so the `user_features`, `item_features` and `item_cooccurrence` tables are empty until `05_features.py` runs again. Always run 05 after 04 (the pipeline does).
Quality gates inside the steps: non-empty input, no nulls in key columns, rating in 1-5, unique `source_row`,
warehouse row count equals input rows, user/item feature counts reconcile to input rows.
Sample runs write `_sample`-suffixed files / `*_sample` views and never touch full-run outputs.

## 2. Metric definitions

* **gross_value** = Σ `final_price` (returns included); **net_value** = Σ `final_price` of non-returned rows (same definitions as Person 2).
* **return_rate_pct** = 100 × mean(`is_returned`).
* **Interaction** = distinct (user_id, product_id) pair. 1,000,000 rows give 999,993 pairs. No quantity column exists, so there are no repeat-purchase weights.
* **Sparsity** = 1 − pairs / (users × products) = 99.998 % (603,815 users, 89,999 products).
* **User segments**: 1 / 2 / 3 / 4+ purchase lines.
* **Recommendation metrics** (K ∈ {5,10,20}): with one held-out item per user, Recall@K = HitRate@K, Precision@K = Recall@K / K,
  NDCG@K = 1 / log2(rank + 1) if the item is in the top K, else 0. `item_coverage@10` = share of the catalogue that appears in any top-10 list.

## 3. Warehouse (`data/warehouse/recomart.db`, SQLite; DDL in `sql/schema.sql`)

`fact_interaction` (1,000,000 rows, one per purchase line) · `user_features` (603,815) · `item_features` (89,999) ·
`item_cooccurrence` (top-10 neighbours per item, 811,120 pairs) · `monthly_metrics` · `category_metrics`.

## 4. Feature store (`src/feature_store.py`)

Custom registry; each `materialize` call creates an immutable version. `get_features(view, ids, version=None)`
(`None` = latest, for inference; a number pins training data), `get_online_features(view, id)`, `describe()`.
Views: `user_features` (9 features), `item_features` (8), `item_cooccurrence` (5). Names, dtypes, sources and
transformations: `docs/feature_catalog.md`.
Known limitation: the stored feature views are computed on all data. `06_train.py` builds its own item attributes from the training split only, so evaluation does not leak, but a training job that reads these views directly must restrict them to a cut-off.

## 5. Model results (leave-last-out, 10,000 evaluation users, seed 42)

| Model | Recall@10 | NDCG@10 |
|---|---|---|
| Random ranking (expected) | 0.00011 | - |
| Popularity | 0.00030 | 0.00013 |
| SVD k=16 / 32 / 64 | 0.00020 / 0.00010 / 0.00000 | 0.00007 / 0.00004 / 0.00000 |
| Content-based | 0.00010 | 0.00003 |

**All models are indistinguishable from random ranking** (1-3 hits out of 10,000 users). This is a property of the data,
not of the code: next-purchase category equals the previous one 20.1 % of the time (chance 20.0 %), brand 8.3 % (chance 8.3 %),
the price correlation between consecutive purchases is 0.002, and a product's most common category covers only 37 % of its
purchases. See section 5 of `notebooks/eda.ipynb`. The deployed SVD model (`recommend.py`) therefore demonstrates the working
pipeline, not predictive value. It is exported from the best SVD run by NDCG@10 (k=16).

## 6. Other limitations

* The REST catalogue (194 `dummyjson` products) does not share keys with transaction `product_id` (`P12345`), so it is not joined.
* The 55 % of users with one purchase can only be served by the popularity fallback or weak SVD vectors.
* `recommend.py` masks items from the training split only; the held-out last purchase of an evaluation user is not masked.
