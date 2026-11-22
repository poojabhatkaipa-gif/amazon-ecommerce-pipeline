from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
RAW_LAKE_DIR = ROOT_DIR / "data" / "lake" / "raw"
OUTPUT_DIR = ROOT_DIR / "data" / "processed"
LOG_FILE = ROOT_DIR / "logs" / "validation.log"

TXN_COLUMNS = [
    "user_id",
    "product_id",
    "category",
    "subcategory",
    "brand",
    "price",
    "discount",
    "final_price",
    "rating",
    "review_count",
    "stock",
    "seller_id",
    "seller_rating",
    "purchase_date",
    "shipping_time_days",
    "location",
    "device",
    "payment_method",
    "is_returned",
    "delivery_status",
]


def get_logger() -> logging.Logger:
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    log = logging.getLogger("data_validation")
    log.setLevel(logging.INFO)
    if not log.handlers:
        formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
        file_h = logging.FileHandler(LOG_FILE, encoding="utf-8")
        file_h.setFormatter(formatter)
        stream_h = logging.StreamHandler()
        stream_h.setFormatter(formatter)
        log.addHandler(file_h)
        log.addHandler(stream_h)
    return log


logger = get_logger()


def pick_latest(glob_pattern: str) -> Path:
    files = sorted(RAW_LAKE_DIR.glob(glob_pattern))
    if not files:
        raise FileNotFoundError(f"Nothing matched pattern: {glob_pattern}")
    return files[-1]


def check_transactions(csv_file: Path) -> dict:
    logger.info(f"Checking transactions file: {csv_file.relative_to(ROOT_DIR)}")

    rows_seen = 0
    null_counts = {col: 0 for col in TXN_COLUMNS}
    dup_count = 0
    bad_ratings = 0
    bad_prices = 0
    bad_dates = 0
    seen_row_hashes = set()
    earliest_date = None
    latest_date = None

    first_row = pd.read_csv(csv_file, nrows=1)
    found_cols = list(first_row.columns)
    missing = [c for c in TXN_COLUMNS if c not in found_cols]
    extra = [c for c in found_cols if c not in TXN_COLUMNS]

    for batch in pd.read_csv(csv_file, chunksize=200_000, low_memory=False):
        rows_seen += len(batch)

        for col in TXN_COLUMNS:
            if col in batch.columns:
                null_counts[col] += int(batch[col].isna().sum())

        for row_hash in pd.util.hash_pandas_object(batch, index=False):
            if row_hash in seen_row_hashes:
                dup_count += 1
            else:
                seen_row_hashes.add(row_hash)

        if "rating" in batch.columns:
            ratings = pd.to_numeric(batch["rating"], errors="coerce")
            bad_ratings += int(((ratings < 1.0) | (ratings > 5.0) | ratings.isna()).sum())

        if "price" in batch.columns and "final_price" in batch.columns:
            base_price = pd.to_numeric(batch["price"], errors="coerce")
            net_price = pd.to_numeric(batch["final_price"], errors="coerce")
            bad_prices += int(
                ((base_price <= 0) | (net_price < 0) | base_price.isna() | net_price.isna()).sum()
            )

        if "purchase_date" in batch.columns:
            parsed_dates = pd.to_datetime(batch["purchase_date"], errors="coerce")
            bad_dates += int(parsed_dates.isna().sum())
            clean_dates = parsed_dates.dropna()
            if not clean_dates.empty:
                b_min, b_max = clean_dates.min(), clean_dates.max()
                earliest_date = b_min if earliest_date is None or b_min < earliest_date else earliest_date
                latest_date = b_max if latest_date is None or b_max > latest_date else latest_date

    return {
        "file": str(csv_file.relative_to(ROOT_DIR)),
        "row_count": rows_seen,
        "column_count": len(found_cols),
        "schema_matches": not missing and not extra,
        "missing_columns": missing,
        "extra_columns": extra,
        "total_null_cells": sum(null_counts.values()),
        "nulls_by_column": null_counts,
        "exact_duplicate_rows": dup_count,
        "invalid_rating_rows_outside_1_to_5": bad_ratings,
        "invalid_price_rows": bad_prices,
        "unparseable_date_rows": bad_dates,
        "purchase_date_min": earliest_date.strftime("%Y-%m-%d") if earliest_date is not None else None,
        "purchase_date_max": latest_date.strftime("%Y-%m-%d") if latest_date is not None else None,
    }


def check_catalog(json_file: Path) -> dict:
    logger.info(f"Checking catalog JSON: {json_file.relative_to(ROOT_DIR)}")
    raw_items = json.loads(json_file.read_text(encoding="utf-8"))
    catalog_df = pd.json_normalize(raw_items)

    key_fields = ["id", "title", "category", "price", "rating"]
    present_keys = [k for k in key_fields if k in catalog_df.columns]
    missing_keys = [k for k in key_fields if k not in catalog_df.columns]
    key_nulls = {k: int(catalog_df[k].isna().sum()) for k in present_keys}
    dup_ids = int(catalog_df["id"].duplicated().sum()) if "id" in catalog_df.columns else 0

    return {
        "file": str(json_file.relative_to(ROOT_DIR)),
        "record_count": len(catalog_df),
        "column_count": len(catalog_df.columns),
        "missing_required_fields": missing_keys,
        "nulls_in_required_fields": key_nulls,
        "duplicate_product_ids": dup_ids,
    }


def main() -> None:
    run_stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    logger.info(f"Running validation pass {run_stamp}")

    latest_txns = pick_latest("source=amazon_ecommerce_csv/type=transactions/ingest_date=*/transactions_*.csv")
    latest_catalog = pick_latest("source=dummyjson_api/type=product_catalog/ingest_date=*/products_*.json")

    txn_summary = check_transactions(latest_txns)
    catalog_summary = check_catalog(latest_catalog)

    all_good = (
    txn_summary["schema_matches"]
    and txn_summary["row_count"] > 0
    and txn_summary["total_null_cells"] == 0
    and txn_summary["nulls_by_column"]["user_id"] == 0
    and txn_summary["nulls_by_column"]["product_id"] == 0
    and txn_summary["invalid_rating_rows_outside_1_to_5"] == 0
    and txn_summary["invalid_price_rows"] == 0
    and txn_summary["unparseable_date_rows"] == 0
    and catalog_summary["record_count"] > 0
    and not catalog_summary["missing_required_fields"]
    and all(
        v == 0
        for v in catalog_summary["nulls_in_required_fields"].values()
    )
    and catalog_summary["duplicate_product_ids"] == 0
)


    report = {
        "run_id": run_stamp,
        "status": "PASSED" if all_good else "WARNINGS_FOUND",
        "transactions": txn_summary,
        "product_catalog": catalog_summary,
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = OUTPUT_DIR / f"validation_report_{run_stamp}.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info(f"Saved validation report to {report_path.relative_to(ROOT_DIR)} [{report['status']}]")


if __name__ == "__main__":
    main()