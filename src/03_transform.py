"""Person 2 - Step 2: transformation layer.

Consumes ``data/processed/cleaned_transactions[_sample].parquet`` (output of
02_clean.py) and produces:

    transformed_transactions[_sample].parquet   (transaction grain + features)
    customer_summary[_sample].parquet           (one row per user_id)
    category_summary[_sample].parquet           (one row per category+subcategory)
    transformation_report[_sample].json
    logs/transformation.log

Metric definitions (also in docs/data_contract_person2.md):
    * Purchase value of a transaction  = final_price (one row = one unit).
    * ``total_*`` values INCLUDE returned transactions (gross value).
    * ``*_excluding_returns`` values exclude rows where is_returned is True.
    * discount_amount / effective_discount_pct are derived from price and
      final_price only; the source ``discount`` column is preserved untouched.

Usage:
    python src/03_transform.py            # full dataset
    python src/03_transform.py --sample   # reads/writes the *_sample files
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DIR = PROJECT_ROOT / "data" / "processed"
LOG_PATH = PROJECT_ROOT / "logs" / "transformation.log"

logger = logging.getLogger("transform")

REQUIRED_INPUT_COLUMNS = [
    "user_id", "product_id", "category", "subcategory", "price", "discount",
    "final_price", "rating", "purchase_date", "is_returned",
]
NEW_PRICING_COLS = ["discount_amount", "effective_discount_pct"]
NEW_DATE_COLS = [
    "purchase_year", "purchase_month", "purchase_month_name",
    "purchase_quarter", "purchase_year_month",
]
CUSTOMER_COLUMNS = [
    "user_id", "transaction_count", "total_purchase_value",
    "average_purchase_value", "returned_transaction_count", "return_rate_pct",
    "first_purchase_date", "last_purchase_date",
    "non_returned_transaction_count", "returned_purchase_value",
    "total_purchase_value_excluding_returns",
]
CATEGORY_COLUMNS = [
    "category", "subcategory", "transaction_count", "total_purchase_value",
    "average_price", "average_final_price", "average_rating",
    "returned_transaction_count", "return_rate_pct",
    "returned_purchase_value", "total_purchase_value_excluding_returns",
]


# --------------------------------------------------------------------------- #
# Infrastructure
# --------------------------------------------------------------------------- #
def configure_logging(log_path: Path) -> None:
    """Attach file + console handlers (idempotent)."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger.setLevel(logging.INFO)
    if logger.handlers:
        return
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)


def read_parquet(path: Path) -> pd.DataFrame:
    """Read a Parquet file with a clear error if pyarrow is unavailable."""
    try:
        import pyarrow  # noqa: F401
    except ImportError as err:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "pyarrow is required to read Parquet files. Install it with "
            "`pip install pyarrow`."
        ) from err
    return pd.read_parquet(path, engine="pyarrow")


def write_parquet(df: pd.DataFrame, path: Path) -> None:
    """Write a DataFrame to Parquet (pyarrow engine) via a temp file."""
    try:
        import pyarrow  # noqa: F401
    except ImportError as err:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "pyarrow is required to write Parquet files. Install it with "
            "`pip install pyarrow`."
        ) from err
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    df.to_parquet(tmp, engine="pyarrow", index=False, compression="snappy")
    tmp.replace(path)


def write_json(obj: dict, path: Path) -> None:
    """Write a JSON report (numpy scalars handled)."""

    def _default(o: Any) -> Any:
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.bool_):
            return bool(o)
        if isinstance(o, (pd.Timestamp, datetime)):
            return o.isoformat()
        raise TypeError(f"Not JSON serialisable: {type(o)}")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=_default), encoding="utf-8")


# --------------------------------------------------------------------------- #
# Feature engineering
# --------------------------------------------------------------------------- #
def validate_input(df: pd.DataFrame) -> None:
    """Fail fast if the cleaned dataset is not what this step expects."""
    missing = [c for c in REQUIRED_INPUT_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Cleaned dataset is missing columns: {missing}")
    if df.empty:
        raise ValueError("Cleaned dataset is empty")
    if (df["price"] <= 0).any():
        raise ValueError("Cleaned dataset contains price <= 0")


def add_pricing_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add discount_amount and effective_discount_pct.

    Derived purely from price and final_price. The original ``discount`` and
    ``final_price`` columns are never modified. Negative values are possible
    (final_price > price) and are preserved, not clipped.
    """
    out = df.copy()
    out["discount_amount"] = out["price"] - out["final_price"]
    out["effective_discount_pct"] = out["discount_amount"] / out["price"] * 100.0
    return out


def add_date_dimensions(df: pd.DataFrame) -> pd.DataFrame:
    """Add year / month / month name / quarter / year-month from purchase_date."""
    out = df.copy()
    dt = pd.to_datetime(out["purchase_date"])
    out["purchase_year"] = dt.dt.year.astype("int64")
    out["purchase_month"] = dt.dt.month.astype("int64")
    out["purchase_month_name"] = dt.dt.month_name()  # English, locale independent
    out["purchase_quarter"] = dt.dt.quarter.astype("int64")
    out["purchase_year_month"] = dt.dt.strftime("%Y-%m")
    return out


def _with_return_split(df: pd.DataFrame) -> pd.DataFrame:
    """Helper columns used for aggregation (not written to outputs)."""
    tmp = df.copy()
    returned = tmp["is_returned"].astype(bool)
    tmp["_returned_int"] = returned.astype("int64")
    tmp["_returned_value"] = tmp["final_price"].where(returned, 0.0)
    tmp["_kept_value"] = tmp["final_price"].where(~returned, 0.0)
    return tmp


def build_customer_summary(df: pd.DataFrame) -> pd.DataFrame:
    """One row per user_id.

    total_purchase_value                   = sum(final_price), all transactions
    average_purchase_value                 = mean(final_price), all transactions
    return_rate_pct                        = returned / transaction_count * 100
    total_purchase_value_excluding_returns = sum(final_price) of non-returned rows
    """
    tmp = _with_return_split(df)
    grouped = tmp.groupby("user_id", sort=True)
    out = grouped.agg(
        transaction_count=("user_id", "size"),
        total_purchase_value=("final_price", "sum"),
        average_purchase_value=("final_price", "mean"),
        returned_transaction_count=("_returned_int", "sum"),
        first_purchase_date=("purchase_date", "min"),
        last_purchase_date=("purchase_date", "max"),
        returned_purchase_value=("_returned_value", "sum"),
        total_purchase_value_excluding_returns=("_kept_value", "sum"),
    ).reset_index()
    out["transaction_count"] = out["transaction_count"].astype("int64")
    out["returned_transaction_count"] = out["returned_transaction_count"].astype("int64")
    out["non_returned_transaction_count"] = (
        out["transaction_count"] - out["returned_transaction_count"]
    )
    out["return_rate_pct"] = (
        out["returned_transaction_count"] / out["transaction_count"] * 100.0
    )
    return out[CUSTOMER_COLUMNS]


def build_category_summary(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (category, subcategory).

    Returned transactions are INCLUDED in transaction_count, total_purchase_value,
    average_price, average_final_price and average_rating (these describe all
    sales activity). Returns are quantified separately through
    returned_transaction_count, return_rate_pct, returned_purchase_value and
    total_purchase_value_excluding_returns.
    """
    tmp = _with_return_split(df)
    out = (
        tmp.groupby(["category", "subcategory"], sort=True, dropna=False)
        .agg(
            transaction_count=("category", "size"),
            total_purchase_value=("final_price", "sum"),
            average_price=("price", "mean"),
            average_final_price=("final_price", "mean"),
            average_rating=("rating", "mean"),
            returned_transaction_count=("_returned_int", "sum"),
            returned_purchase_value=("_returned_value", "sum"),
            total_purchase_value_excluding_returns=("_kept_value", "sum"),
        )
        .reset_index()
    )
    out["transaction_count"] = out["transaction_count"].astype("int64")
    out["returned_transaction_count"] = out["returned_transaction_count"].astype("int64")
    out["return_rate_pct"] = (
        out["returned_transaction_count"] / out["transaction_count"] * 100.0
    )
    return out[CATEGORY_COLUMNS]


def transform_dataframe(cleaned: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Run every transformation on an in-memory cleaned frame."""
    validate_input(cleaned)
    transformed = add_date_dimensions(add_pricing_features(cleaned))
    return {
        "transformed": transformed.reset_index(drop=True),
        "customer": build_customer_summary(cleaned),
        "category": build_category_summary(cleaned),
    }


# --------------------------------------------------------------------------- #
# Reconciliation / reporting
# --------------------------------------------------------------------------- #
def reconcile(
    cleaned: pd.DataFrame, frames: dict[str, pd.DataFrame]
) -> dict[str, Any]:
    """Cross-check aggregates against the transaction-level data."""
    n = len(cleaned)
    total = float(cleaned["final_price"].sum())
    n_ret = int(cleaned["is_returned"].sum())
    cust, cat, tr = frames["customer"], frames["category"], frames["transformed"]

    def close(a: float, b: float) -> bool:
        return math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-6)

    checks = {
        "transformed_rows_equal_cleaned_rows": len(tr) == n,
        "customer_transaction_count_sums_to_rows": int(cust["transaction_count"].sum()) == n,
        "category_transaction_count_sums_to_rows": int(cat["transaction_count"].sum()) == n,
        "customer_total_value_matches": close(float(cust["total_purchase_value"].sum()), total),
        "category_total_value_matches": close(float(cat["total_purchase_value"].sum()), total),
        "customer_returned_count_matches": int(cust["returned_transaction_count"].sum()) == n_ret,
        "category_returned_count_matches": int(cat["returned_transaction_count"].sum()) == n_ret,
        "customer_user_id_unique": bool(cust["user_id"].is_unique),
        "category_keys_unique": not cat.duplicated(["category", "subcategory"]).any(),
        "no_nulls_in_new_features": int(
            tr[NEW_PRICING_COLS + NEW_DATE_COLS].isna().sum().sum()
        ) == 0,
        "original_final_price_preserved": bool(
            tr["final_price"].equals(cleaned.reset_index(drop=True)["final_price"])
        ),
        "original_discount_preserved": bool(
            tr["discount"].equals(cleaned.reset_index(drop=True)["discount"])
        ),
    }
    return {"all_passed": all(checks.values()), "checks": checks}


def run_transformation(
    input_dir: Path = DEFAULT_DIR,
    output_dir: Path = DEFAULT_DIR,
    sample: bool = False,
) -> dict[str, Any]:
    """Read cleaned Parquet -> transform -> write outputs. Returns the report."""
    started = time.time()
    suffix = "_sample" if sample else ""
    mode = "SAMPLE" if sample else "FULL"
    in_path = input_dir / f"cleaned_transactions{suffix}.parquet"
    if not in_path.exists():
        raise FileNotFoundError(
            f"{in_path} not found. Run `python src/02_clean.py"
            f"{' --sample-rows 10000' if sample else ''}` first."
        )
    logger.info("Starting transformation run | mode=%s | input=%s", mode, in_path)
    cleaned = read_parquet(in_path)
    logger.info("Loaded %d cleaned rows x %d columns", len(cleaned), cleaned.shape[1])

    frames = transform_dataframe(cleaned)
    checks = reconcile(cleaned, frames)
    for name, ok in checks["checks"].items():
        (logger.info if ok else logger.error)("check %s: %s", name, "OK" if ok else "FAILED")

    outputs = {
        "transformed": output_dir / f"transformed_transactions{suffix}.parquet",
        "customer": output_dir / f"customer_summary{suffix}.parquet",
        "category": output_dir / f"category_summary{suffix}.parquet",
        "report": output_dir / f"transformation_report{suffix}.json",
    }
    for key in ("transformed", "customer", "category"):
        write_parquet(frames[key], outputs[key])
        logger.info("Wrote %d rows -> %s", len(frames[key]), outputs[key])

    tr = frames["transformed"]
    report = {
        "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "mode": mode,
        "input_file": str(in_path),
        "runtime_seconds": round(time.time() - started, 2),
        "input_rows": len(cleaned),
        "output_rows": {k: len(v) for k, v in frames.items()},
        "unique_customers": len(frames["customer"]),
        "category_subcategory_groups": len(frames["category"]),
        "reconciliation": checks,
        "pricing_feature_stats": {
            "discount_amount": tr["discount_amount"].describe().to_dict(),
            "effective_discount_pct": tr["effective_discount_pct"].describe().to_dict(),
            "rows_with_negative_discount_amount": int((tr["discount_amount"] < 0).sum()),
            "rows_where_source_discount_differs_from_effective_by_over_0_5pp": int(
                ((tr["discount"] - tr["effective_discount_pct"]).abs() > 0.5).sum()
            ),
        },
        "totals": {
            "total_purchase_value": float(tr["final_price"].sum()),
            "returned_purchase_value": float(tr.loc[tr["is_returned"], "final_price"].sum()),
            "returned_transactions": int(tr["is_returned"].sum()),
        },
        "date_range": {
            "min": tr["purchase_date"].min(),
            "max": tr["purchase_date"].max(),
        },
        "output_files": {k: str(v) for k, v in outputs.items()},
        "dtypes": {
            k: {c: str(t) for c, t in frames[k].dtypes.items()} for k in frames
        },
    }
    write_json(report, outputs["report"])
    if not checks["all_passed"]:
        raise RuntimeError("Transformation reconciliation checks failed; see report")
    logger.info("Transformation finished in %.2fs", time.time() - started)
    return report


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Transform cleaned Amazon transactions.")
    p.add_argument("--sample", action="store_true",
                   help="Use the *_sample cleaned file and write *_sample outputs.")
    p.add_argument("--input-dir", type=Path, default=DEFAULT_DIR)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_DIR)
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    configure_logging(LOG_PATH)
    try:
        run_transformation(args.input_dir, args.output_dir, args.sample)
    except Exception:
        logger.exception("Transformation run failed")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
