# Raw data storage layout

Local data lake (`data/lake/raw/`), partitioned by **source / type / ingest_date**, with the run timestamp in the file name:

```
data/lake/raw/
├── manifest_<run_ts>.json                         one manifest per ingestion run (status + file paths)
├── source=amazon_ecommerce_csv/
│   └── type=transactions/ingest_date=YYYY-MM-DD/transactions_<run_ts>.csv      (1,000,000 rows x 20 cols)
└── source=dummyjson_api/
    └── type=product_catalog/ingest_date=YYYY-MM-DD/products_<run_ts>.json      (194 products, REST API)
```

* Written only by `src/01_ingest.py` (CSV snapshot copy + REST call with 3 retries and back-off); files are never modified afterwards.
* `src/02_validate.py` always validates the latest partition of each source.
* Downstream layers: `data/processed/` (cleaned/transformed parquet), `data/analytics/` (metric CSVs), `data/warehouse/` (SQLite),
  `data/feature_store/` (versioned feature views), `models/` (trained model files).
* The whole lake and every derived layer is versioned with DVC (see `docs/versioning.md`).
* Cloud equivalent: the same keys map one-to-one to S3 (`s3://<bucket>/raw/source=.../type=.../ingest_date=.../`).
