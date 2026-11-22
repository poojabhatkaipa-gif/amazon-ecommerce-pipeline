"""Person 2 - Step 1: data cleaning layer.

Reads the extracted Amazon e-commerce CSV, standardises and type-checks it,
validates business rules, flags (never deletes) statistical outliers and writes:

    data/processed/cleaned_transactions[_sample].parquet
    data/processed/quarantined_transactions[_sample].parquet
    data/processed/cleaning_report[_sample].json
    logs/cleaning.log

Every input row ends up in exactly one of three buckets:
    * valid        -> cleaned_transactions
    * invalid      -> quarantined_transactions (with ``rejection_reasons``)
    * duplicate    -> quarantined_transactions (reason ``exact_duplicate``)

Usage:
    python src/02_clean.py                      # full dataset
    python src/02_clean.py --sample-rows 10000  # first 10,000 data rows
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = PROJECT_ROOT / "data" / "extracted" / "amazon_ecommerce_1M.csv"
FALLBACK_INPUT = PROJECT_ROOT / "data" / "raw" / "amazon_ecommerce_1M.csv"
# Development dataset: pass explicitly with --input; never used by default.
DEV_INPUT = PROJECT_ROOT / "data" / "extracted" / "amazon_ecommerce_sample_50000.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
LOG_PATH = PROJECT_ROOT / "logs" / "cleaning.log"

logger = logging.getLogger("clean")

# --------------------------------------------------------------------------- #
# Schema definition
# --------------------------------------------------------------------------- #
EXPECTED_COLUMNS: list[str] = [
    "user_id", "product_id", "category", "subcategory", "brand",
    "price", "discount", "final_price", "rating", "review_count", "stock",
    "seller_id", "seller_rating", "purchase_date", "is_returned",
    "shipping_time_days", "delivery_status", "location", "device",
    "payment_method",
]
FLOAT_COLS = ["price", "discount", "final_price", "rating", "seller_rating"]
INT_COLS = ["review_count", "stock", "shipping_time_days"]
STRING_COLS = [
    "user_id", "product_id", "category", "subcategory", "brand", "seller_id",
    "delivery_status", "location", "device", "payment_method",
]
# Columns whose spelling variants (case / whitespace) are harmonised.
CATEGORICAL_COLS = [
    "category", "subcategory", "brand", "delivery_status", "location",
    "device", "payment_method",
]
# A null in any of these sends the row to quarantine. Other columns
# (brand, subcategory, seller_id, location, device, payment_method,
# delivery_status) may stay null: we never invent replacement values.
REQUIRED_COLS = [
    "user_id", "product_id", "category", *FLOAT_COLS, *INT_COLS,
    "purchase_date", "is_returned",
]
# Business validation rules: column -> predicate that is True for VALID values.
RANGE_RULES: dict[str, tuple[str, Callable[[pd.Series], pd.Series]]] = {
    "price": ("price > 0", lambda s: s > 0),
    "discount": ("0 <= discount <= 100", lambda s: (s >= 0) & (s <= 100)),
    "final_price": ("final_price >= 0", lambda s: s >= 0),
    "rating": ("1 <= rating <= 5", lambda s: (s >= 1) & (s <= 5)),
    "seller_rating": ("1 <= seller_rating <= 5", lambda s: (s >= 1) & (s <= 5)),
    "stock": ("stock >= 0", lambda s: s >= 0),
    "review_count": ("review_count >= 0", lambda s: s >= 0),
    "shipping_time_days": ("shipping_time_days >= 0", lambda s: s >= 0),
}
OUTLIER_COLS = ["price", "final_price", "review_count", "stock", "shipping_time_days"]
MISSING_TOKENS = {"", "na", "n/a", "nan", "null", "none", "nat"}
BOOL_MAP = {
    "true": True, "t": True, "yes": True, "y": True, "1": True, "1.0": True,
    "false": False, "f": False, "no": False, "n": False, "0": False, "0.0": False,
}


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
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.bool_,)):
            return bool(o)
        if isinstance(o, (pd.Timestamp, datetime)):
            return o.isoformat()
        raise TypeError(f"Not JSON serialisable: {type(o)}")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=_default), encoding="utf-8")


# --------------------------------------------------------------------------- #
# Loading and standardisation
# --------------------------------------------------------------------------- #
def resolve_input(path: Optional[Path]) -> Path:
    """Return the CSV to read (explicit path, else extracted/, else raw/)."""
    if path is not None:
        if not path.exists():
            raise FileNotFoundError(f"Input CSV not found: {path}")
        return path
    for candidate in (DEFAULT_INPUT, FALLBACK_INPUT):
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "amazon_ecommerce_1M.csv not found in data/extracted or data/raw"
    )


def load_raw(csv_path: Path, sample_rows: Optional[int]) -> pd.DataFrame:
    """Read the CSV as strings so no value is altered or coerced on load."""
    logger.info("Reading %s (sample_rows=%s)", csv_path, sample_rows)
    df = pd.read_csv(
        csv_path, dtype=str, keep_default_na=False, nrows=sample_rows
    )
    logger.info("Loaded %d rows x %d columns", len(df), df.shape[1])
    return df


def standardize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Lower-case snake_case column names (``Final Price`` -> ``final_price``)."""
    renamed = {
        c: re.sub(r"[^0-9a-zA-Z]+", "_", str(c).strip()).strip("_").lower()
        for c in df.columns
    }
    return df.rename(columns=renamed)


def check_schema(df: pd.DataFrame) -> list[str]:
    """Raise if expected columns are missing; return unexpected extra columns."""
    missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Input is missing expected columns: {missing}")
    extra = [c for c in df.columns if c not in EXPECTED_COLUMNS]
    if extra:
        logger.warning("Ignoring unexpected extra columns: %s", extra)
    return extra


def normalize_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Trim whitespace and turn empty / NA-like tokens into real nulls."""
    out = df.copy()
    for col in out.columns:
        s = out[col].astype("string").str.strip()
        out[col] = s.mask(s.isna() | s.str.lower().isin(MISSING_TOKENS))
    return out


# --------------------------------------------------------------------------- #
# Type conversion
# --------------------------------------------------------------------------- #
def parse_boolean(series: pd.Series) -> pd.Series:
    """Map common true/false spellings to a nullable boolean (unknown -> NA)."""
    lowered = series.astype("string").str.lower()
    values = [BOOL_MAP.get(v, None) if isinstance(v, str) else None for v in lowered]
    return pd.Series(pd.array(values, dtype="boolean"), index=series.index)


def parse_dates(series: pd.Series) -> pd.Series:
    """Parse dates; retry unparsed non-null values with mixed-format parsing."""
    parsed = pd.to_datetime(series, errors="coerce")
    retry = parsed.isna() & series.notna()
    if retry.any():
        parsed.loc[retry] = pd.to_datetime(
            series[retry], errors="coerce", format="mixed"
        )
    return parsed


def coerce_types(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Convert raw string columns to their target types.

    Returns the typed frame (failed conversions become null) and a dict of
    conversion-failure counts: values that were present but unparseable.
    """
    typed = pd.DataFrame(index=raw.index)
    failures: dict[str, int] = {}
    for col in STRING_COLS:
        typed[col] = raw[col]
    for col in FLOAT_COLS + INT_COLS:
        num = pd.to_numeric(raw[col], errors="coerce").astype("float64")
        num = num.where(np.isfinite(num))  # inf -> null (counted as failure)
        typed[col] = num
        failures[col] = int((num.isna() & raw[col].notna()).sum())
    typed["purchase_date"] = parse_dates(raw["purchase_date"])
    failures["purchase_date"] = int(
        (typed["purchase_date"].isna() & raw["purchase_date"].notna()).sum()
    )
    typed["is_returned"] = parse_boolean(raw["is_returned"])
    failures["is_returned"] = int(
        (typed["is_returned"].isna() & raw["is_returned"].notna()).sum()
    )
    return typed[EXPECTED_COLUMNS], failures


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #
def build_rejection_masks(
    typed: pd.DataFrame, raw: pd.DataFrame
) -> dict[str, pd.Series]:
    """Return boolean masks (True = row violates the rule), keyed by reason."""
    masks: dict[str, pd.Series] = {}
    for col in REQUIRED_COLS:
        was_missing = raw[col].isna()
        masks[f"missing_{col}"] = was_missing
        masks[f"unparseable_{col}"] = typed[col].isna() & ~was_missing
    for col, (_, ok) in RANGE_RULES.items():
        value = typed[col]
        masks[f"out_of_range_{col}"] = value.notna() & ~ok(value)
    for col in INT_COLS:
        value = typed[col]
        masks[f"non_integer_{col}"] = value.notna() & (value != np.floor(value))
    return masks


def combine_reasons(
    masks: dict[str, pd.Series], dup_mask: pd.Series
) -> tuple[pd.Series, pd.Series]:
    """Combine masks into (reject_mask, reasons).

    A duplicate row is labelled only ``exact_duplicate`` so that the three
    buckets (valid / invalid / duplicate) are mutually exclusive.
    """
    frame = pd.DataFrame(masks, index=dup_mask.index).fillna(False).astype(bool)
    frame.loc[dup_mask, :] = False
    frame["exact_duplicate"] = dup_mask.astype(bool)
    reject = frame.any(axis=1)
    reasons = pd.Series("", index=frame.index, dtype="object")
    if reject.any():
        cols = np.array(frame.columns)
        failing = frame.loc[reject].to_numpy()
        reasons.loc[reject] = [";".join(cols[row]) for row in failing]
    return reject, reasons


def find_duplicates(df: pd.DataFrame) -> pd.Series:
    """True for every repeat of a fully identical record (first is kept).

    Uses *all* 20 business columns, never just user_id + product_id: the same
    user may legitimately buy the same product several times.
    """
    return df.duplicated(subset=EXPECTED_COLUMNS, keep="first")


# --------------------------------------------------------------------------- #
# Outliers and categoricals
# --------------------------------------------------------------------------- #
def iqr_bounds(series: pd.Series, multiplier: float = 1.5) -> tuple[float, float]:
    """Tukey fences: [Q1 - k*IQR, Q3 + k*IQR]."""
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    return float(q1 - multiplier * iqr), float(q3 + multiplier * iqr)


def flag_outliers(
    df: pd.DataFrame, columns: list[str], multiplier: float = 1.5
) -> tuple[pd.DataFrame, dict[str, dict[str, float]]]:
    """Add ``outlier_<col>`` boolean flags and ``is_outlier_any``.

    Outliers are only *flagged*; no row is removed.
    """
    out = df.copy()
    stats: dict[str, dict[str, float]] = {}
    any_flag = pd.Series(False, index=out.index)
    for col in columns:
        lo, hi = iqr_bounds(out[col], multiplier)
        flag = (out[col] < lo) | (out[col] > hi)
        out[f"outlier_{col}"] = flag
        any_flag |= flag
        stats[col] = {
            "lower_bound": lo,
            "upper_bound": hi,
            "outlier_count": int(flag.sum()),
            "outlier_pct": round(float(flag.mean() * 100), 4) if len(out) else 0.0,
        }
    out["is_outlier_any"] = any_flag
    return out, stats


def normalize_categoricals(
    df: pd.DataFrame, columns: list[str]
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Collapse whitespace and harmonise case variants of the same label.

    Variants that differ only by case (``Electronics`` / ``electronics``) are
    mapped to the most frequent spelling (ties -> alphabetical, deterministic).
    Returns the frame and the number of values changed per column.
    """
    out = df.copy()
    changed: dict[str, int] = {}
    for col in columns:
        original = out[col].astype("string")
        s = original.str.replace(r"\s+", " ", regex=True).str.strip()
        key = s.str.lower()
        counts = (
            pd.DataFrame({"val": s, "key": key})
            .dropna()
            .groupby(["key", "val"]).size().reset_index(name="n")
            .sort_values(["key", "n", "val"], ascending=[True, False, True])
            .drop_duplicates("key")
        )
        mapping = dict(zip(counts["key"], counts["val"]))
        canon = key.map(mapping).astype("string")
        changed[col] = int((canon.notna() & (canon != original)).sum())
        out[col] = canon
    return out, changed


def cast_final_types(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the final, documented dtypes to rows that passed validation."""
    out = df.copy()
    for col in FLOAT_COLS:
        out[col] = out[col].astype("float64")
    for col in INT_COLS:
        out[col] = out[col].astype("int64")
    out["is_returned"] = out["is_returned"].astype(bool)
    out["purchase_date"] = pd.to_datetime(out["purchase_date"]).astype("datetime64[ns]")
    return out


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def clean_dataframe(
    raw_input: pd.DataFrame, iqr_multiplier: float = 1.5
) -> dict[str, Any]:
    """Run the full cleaning logic on an in-memory frame of raw strings.

    Returns a dict with ``cleaned``, ``quarantined`` and ``stats`` entries.
    """
    df = standardize_column_names(raw_input)
    extra_cols = check_schema(df)
    df = normalize_missing(df[EXPECTED_COLUMNS])
    input_rows = len(df)
    raw_null_counts = {c: int(df[c].isna().sum()) for c in EXPECTED_COLUMNS}

    dup_mask = find_duplicates(df)
    logger.info("Exact duplicate records found: %d", int(dup_mask.sum()))

    typed, failures = coerce_types(df)
    logger.info("Conversion failures: %s", {k: v for k, v in failures.items() if v})

    masks = build_rejection_masks(typed, df)
    reject, reasons = combine_reasons(masks, dup_mask)

    # Reason counts over invalid (non-duplicate) rows only.
    invalid_only = reject & ~dup_mask
    reason_counts = {
        name: int((m & invalid_only).sum()) for name, m in masks.items()
        if int((m & invalid_only).sum()) > 0
    }
    rule_results = {}
    for col, (desc, _) in RANGE_RULES.items():
        violations = int(masks[f"out_of_range_{col}"].sum())
        rule_results[desc] = {
            "violations": violations,
            "status": "PASSED" if violations == 0 else "VIOLATIONS_FOUND",
        }
    rule_results["purchase_date parseable"] = {
        "violations": failures["purchase_date"]
        + int(masks["missing_purchase_date"].sum()),
        "status": "PASSED"
        if failures["purchase_date"] + int(masks["missing_purchase_date"].sum()) == 0
        else "VIOLATIONS_FOUND",
    }

    source_row = pd.Series(np.arange(1, input_rows + 1, dtype="int64"), index=df.index)
    quarantined = df.loc[reject].copy()
    quarantined.insert(0, "source_row", source_row[reject].astype("int64"))
    quarantined["rejection_reasons"] = reasons[reject].astype("string")

    valid_typed = typed.loc[~reject].copy()
    valid_typed, cat_changes = normalize_categoricals(valid_typed, CATEGORICAL_COLS)
    valid_typed = cast_final_types(valid_typed)
    cleaned, outlier_stats = flag_outliers(valid_typed, OUTLIER_COLS, iqr_multiplier)
    cleaned.insert(0, "source_row", source_row[~reject].astype("int64"))
    cleaned = cleaned.reset_index(drop=True)
    quarantined = quarantined.reset_index(drop=True)

    # Informational (non-rejecting) consistency observations.
    # Compare in percentage points (robust to 2-decimal rounding on large prices).
    effective_pct = (cleaned["price"] - cleaned["final_price"]) / cleaned["price"] * 100
    mismatch = (effective_pct - cleaned["discount"]).abs() > 0.05
    sub_multi = (
        cleaned.groupby("subcategory")["category"].nunique().gt(1).sum()
        if len(cleaned) else 0
    )
    stats = {
        "input_rows": input_rows,
        "valid_rows": len(cleaned),
        "quarantined_invalid_rows": int(invalid_only.sum()),
        "duplicates_removed": int(dup_mask.sum()),
        "quarantine_file_rows": len(quarantined),
        "extra_columns_ignored": extra_cols,
        "null_counts_raw": raw_null_counts,
        "null_counts_cleaned": {c: int(cleaned[c].isna().sum()) for c in EXPECTED_COLUMNS},
        "conversion_failures": failures,
        "rejection_reason_counts": reason_counts,
        "validation_results": rule_results,
        "outliers": {
            "method": f"IQR Tukey fences (multiplier={iqr_multiplier}); flagged, not removed",
            "rows_with_any_outlier_flag": int(cleaned["is_outlier_any"].sum()),
            "by_column": outlier_stats,
        },
        "categorical_normalization_changes": cat_changes,
        "categorical_profile": {
            c: int(cleaned[c].nunique()) for c in CATEGORICAL_COLS
        },
        "delivery_status_distribution": cleaned["delivery_status"]
        .value_counts(dropna=False).astype(int).to_dict(),
        "is_returned_distribution": cleaned["is_returned"]
        .value_counts().astype(int).rename(index=str).to_dict(),
        "observations": {
            "final_price_greater_than_price_rows": int(
                (cleaned["final_price"] > cleaned["price"]).sum()
            ),
            "rows_where_discount_does_not_reproduce_final_price": int(mismatch.sum()),
            "subcategories_under_multiple_categories": int(sub_multi),
        },
        "purchase_date_min": cleaned["purchase_date"].min() if len(cleaned) else None,
        "purchase_date_max": cleaned["purchase_date"].max() if len(cleaned) else None,
    }
    stats["reconciliation"] = {
        "input_rows": input_rows,
        "valid_plus_invalid_plus_duplicates": stats["valid_rows"]
        + stats["quarantined_invalid_rows"] + stats["duplicates_removed"],
        "balanced": input_rows
        == stats["valid_rows"] + stats["quarantined_invalid_rows"] + stats["duplicates_removed"],
    }
    return {"cleaned": cleaned, "quarantined": quarantined, "stats": stats}


def run_cleaning(
    input_csv: Optional[Path] = None,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    sample_rows: Optional[int] = None,
    iqr_multiplier: float = 1.5,
) -> dict[str, Any]:
    """Read -> clean -> write outputs. Returns the cleaning report."""
    started = time.time()
    csv_path = resolve_input(input_csv)
    # Exactly ONE input file is processed per run. Any run that is not the
    # default full dataset (a --input override or --sample-rows) writes
    # *_sample outputs so full-dataset outputs can never be overwritten.
    is_default = csv_path.resolve() in {DEFAULT_INPUT.resolve(), FALLBACK_INPUT.resolve()}
    is_dev = bool(sample_rows) or not is_default
    suffix = "_sample" if is_dev else ""
    mode = f"SAMPLE({sample_rows or 'all rows'}; {csv_path.name})" if is_dev else "FULL"
    logger.info("Starting cleaning run | mode=%s", mode)

    raw = load_raw(csv_path, sample_rows)
    result = clean_dataframe(raw, iqr_multiplier)
    stats = result["stats"]

    outputs = {
        "cleaned": output_dir / f"cleaned_transactions{suffix}.parquet",
        "quarantined": output_dir / f"quarantined_transactions{suffix}.parquet",
        "report": output_dir / f"cleaning_report{suffix}.json",
    }
    write_parquet(result["cleaned"], outputs["cleaned"])
    write_parquet(result["quarantined"], outputs["quarantined"])
    logger.info("Wrote %d cleaned rows -> %s", stats["valid_rows"], outputs["cleaned"])
    logger.info(
        "Wrote %d quarantined rows (%d invalid, %d duplicates) -> %s",
        stats["quarantine_file_rows"], stats["quarantined_invalid_rows"],
        stats["duplicates_removed"], outputs["quarantined"],
    )

    report = {
        "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "mode": mode,
        "input_file": str(csv_path),
        "runtime_seconds": round(time.time() - started, 2),
        **stats,
        "output_files": {k: str(v) for k, v in outputs.items()},
        "cleaned_dtypes": {c: str(t) for c, t in result["cleaned"].dtypes.items()},
    }
    write_json(report, outputs["report"])
    if not stats["reconciliation"]["balanced"]:
        raise RuntimeError("Row reconciliation failed: rows were lost or duplicated")
    logger.info(
        "Cleaning finished | input=%d valid=%d invalid=%d duplicates=%d outlier_rows=%d",
        stats["input_rows"], stats["valid_rows"], stats["quarantined_invalid_rows"],
        stats["duplicates_removed"], stats["outliers"]["rows_with_any_outlier_flag"],
    )
    return report


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Clean the Amazon e-commerce transactions.")
    p.add_argument("--sample-rows", type=int, default=None,
                   help="Only process the first N data rows (outputs get a _sample suffix).")
    p.add_argument("--input", type=Path, default=None, help="Input CSV (default: data/extracted/amazon_ecommerce_1M.csv). "
                        "Use data/extracted/amazon_ecommerce_sample_50000.csv for development; "
                        "non-default inputs write *_sample outputs.")
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
                   help="Override output directory.")
    p.add_argument("--iqr-multiplier", type=float, default=1.5)
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    configure_logging(LOG_PATH)
    try:
        run_cleaning(args.input, args.output_dir, args.sample_rows, args.iqr_multiplier)
    except Exception:
        logger.exception("Cleaning run failed")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
