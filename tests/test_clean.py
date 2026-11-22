"""Unit tests for src/02_clean.py using small synthetic datasets."""

import json

import numpy as np
import pandas as pd
import pytest


def make_raw(**overrides) -> pd.DataFrame:
    """Three valid, distinct raw (string) rows; overrides replace whole columns."""
    data = {
        "user_id": ["U1", "U2", "U3"],
        "product_id": ["P1", "P2", "P3"],
        "category": ["Books", "Books", "Toys"],
        "subcategory": ["Fiction", "Fiction", "Games"],
        "brand": ["A", "B", "C"],
        "price": ["100", "200", "50"],
        "discount": ["10", "20", "0"],
        "final_price": ["90", "160", "50"],
        "rating": ["4.5", "3.0", "5"],
        "review_count": ["10", "20", "30"],
        "stock": ["5", "6", "7"],
        "seller_id": ["S1", "S2", "S3"],
        "seller_rating": ["4.0", "4.5", "3.5"],
        "purchase_date": ["2024-03-01", "2024-04-15", "2025-01-20"],
        "is_returned": ["True", "False", "false"],
        "shipping_time_days": ["2", "3", "4"],
        "delivery_status": ["Delivered", "Delivered", "Returned"],
        "location": ["Delhi", "Mumbai", "Delhi"],
        "device": ["Mobile", "Web", "Mobile"],
        "payment_method": ["UPI", "Card", "UPI"],
    }
    data.update(overrides)
    return pd.DataFrame(data)


# ----------------------------- type conversion ----------------------------- #
def test_type_conversion(clean_mod):
    out = clean_mod.clean_dataframe(make_raw())
    cleaned = out["cleaned"]
    assert len(cleaned) == 3
    for col in clean_mod.FLOAT_COLS:
        assert cleaned[col].dtype == "float64", col
    for col in clean_mod.INT_COLS:
        assert cleaned[col].dtype == "int64", col
    assert cleaned["is_returned"].dtype == bool
    assert np.issubdtype(cleaned["purchase_date"].dtype, np.datetime64)


def test_boolean_parsing_variants(clean_mod):
    s = pd.Series(["True", "FALSE", "yes", "0", "1", "maybe", None])
    parsed = clean_mod.parse_boolean(s)
    assert parsed.tolist()[:5] == [True, False, True, False, True]
    assert parsed.isna().tolist()[5:] == [True, True]


def test_unparseable_boolean_is_quarantined(clean_mod):
    out = clean_mod.clean_dataframe(make_raw(is_returned=["True", "maybe", "false"]))
    assert len(out["cleaned"]) == 2
    assert "unparseable_is_returned" in out["quarantined"]["rejection_reasons"].iloc[0]
    assert out["stats"]["conversion_failures"]["is_returned"] == 1


def test_date_parsing_and_failure(clean_mod):
    out = clean_mod.clean_dataframe(
        make_raw(purchase_date=["2024-03-01", "not-a-date", "2025-01-20"])
    )
    assert len(out["cleaned"]) == 2
    assert out["stats"]["conversion_failures"]["purchase_date"] == 1
    assert "unparseable_purchase_date" in out["quarantined"]["rejection_reasons"].iloc[0]


def test_column_name_standardisation(clean_mod):
    raw = make_raw().rename(columns={"final_price": " Final Price ", "user_id": "User-ID"})
    out = clean_mod.clean_dataframe(raw)
    assert len(out["cleaned"]) == 3
    assert "final_price" in out["cleaned"].columns


def test_missing_expected_column_raises(clean_mod):
    with pytest.raises(ValueError, match="missing expected columns"):
        clean_mod.clean_dataframe(make_raw().drop(columns=["price"]))


# --------------------------- missing values -------------------------------- #
def test_required_null_goes_to_quarantine_not_imputed(clean_mod):
    out = clean_mod.clean_dataframe(make_raw(price=["100", "", "50"]))
    assert len(out["cleaned"]) == 2
    q = out["quarantined"]
    assert len(q) == 1
    assert "missing_price" in q["rejection_reasons"].iloc[0]
    assert q["price"].isna().iloc[0]  # original (missing) value preserved, not filled


def test_optional_null_is_kept_as_null(clean_mod):
    out = clean_mod.clean_dataframe(make_raw(brand=["A", "NA", ""]))
    cleaned = out["cleaned"]
    assert len(cleaned) == 3
    assert cleaned["brand"].isna().sum() == 2
    assert out["stats"]["null_counts_cleaned"]["brand"] == 2


# ------------------------------ duplicates --------------------------------- #
def test_exact_duplicates_detected_and_kept_once(clean_mod):
    raw = pd.concat([make_raw(), make_raw().iloc[[0]]], ignore_index=True)
    out = clean_mod.clean_dataframe(raw)
    assert out["stats"]["duplicates_removed"] == 1
    assert len(out["cleaned"]) == 3
    assert out["quarantined"]["rejection_reasons"].tolist() == ["exact_duplicate"]
    assert out["stats"]["reconciliation"]["balanced"]


def test_same_user_and_product_is_not_a_duplicate(clean_mod):
    row = make_raw().iloc[[0]].copy()
    row["purchase_date"] = "2024-09-09"  # same user_id + product_id, different purchase
    out = clean_mod.clean_dataframe(pd.concat([make_raw(), row], ignore_index=True))
    assert out["stats"]["duplicates_removed"] == 0
    assert len(out["cleaned"]) == 4


# ------------------------- invalid numeric values -------------------------- #
@pytest.mark.parametrize(
    "col,bad,reason",
    [
        ("price", "0", "out_of_range_price"),
        ("price", "-5", "out_of_range_price"),
        ("discount", "101", "out_of_range_discount"),
        ("discount", "-1", "out_of_range_discount"),
        ("final_price", "-0.01", "out_of_range_final_price"),
        ("rating", "0.5", "out_of_range_rating"),
        ("rating", "5.1", "out_of_range_rating"),
        ("seller_rating", "6", "out_of_range_seller_rating"),
        ("stock", "-1", "out_of_range_stock"),
        ("review_count", "-3", "out_of_range_review_count"),
        ("shipping_time_days", "-2", "out_of_range_shipping_time_days"),
        ("stock", "2.5", "non_integer_stock"),
        ("price", "abc", "unparseable_price"),
        ("price", "inf", "unparseable_price"),
    ],
)
def test_invalid_numeric_values_quarantined(clean_mod, col, bad, reason):
    values = make_raw()[col].tolist()
    values[1] = bad
    out = clean_mod.clean_dataframe(make_raw(**{col: values}))
    assert len(out["cleaned"]) == 2
    assert reason in out["quarantined"]["rejection_reasons"].iloc[0]
    assert out["quarantined"]["source_row"].iloc[0] == 2
    assert out["stats"]["reconciliation"]["balanced"]


def test_boundary_values_are_valid(clean_mod):
    raw = make_raw(
        discount=["0", "100", "50"], rating=["1", "5", "3"],
        seller_rating=["1", "5", "3"], stock=["0", "0", "0"],
        review_count=["0", "0", "0"], shipping_time_days=["0", "0", "0"],
        final_price=["0", "10", "10"],
    )
    out = clean_mod.clean_dataframe(raw)
    assert len(out["cleaned"]) == 3


def test_multiple_reasons_are_joined(clean_mod):
    raw = make_raw(price=["-1", "200", "50"], rating=["9", "3", "5"])
    reasons = clean_mod.clean_dataframe(raw)["quarantined"]["rejection_reasons"].iloc[0]
    assert "out_of_range_price" in reasons and "out_of_range_rating" in reasons


# ------------------------------- outliers ---------------------------------- #
def test_iqr_bounds(clean_mod):
    # Q1=2.75, Q3=6.25, IQR=3.5 -> fences at 2.75-5.25 and 6.25+5.25
    lo, hi = clean_mod.iqr_bounds(pd.Series([1, 2, 3, 4, 5, 6, 7, 8]))
    assert lo == pytest.approx(-2.5)
    assert hi == pytest.approx(11.5)


def test_outliers_flagged_not_removed(clean_mod):
    prices = [str(x) for x in [10, 11, 12, 13, 14, 15, 16, 17, 18, 10000]]
    n = len(prices)
    raw = pd.DataFrame(make_raw().iloc[[0]].to_dict("list")).loc[np.repeat(0, n)].reset_index(drop=True)
    raw["price"] = prices
    raw["final_price"] = prices
    raw["user_id"] = [f"U{i}" for i in range(n)]  # make rows distinct
    out = clean_mod.clean_dataframe(raw)
    cleaned = out["cleaned"]
    assert len(cleaned) == n  # nothing removed
    assert cleaned["outlier_price"].sum() == 1
    assert bool(cleaned.loc[cleaned["price"] == 10000, "outlier_price"].iloc[0])
    assert out["stats"]["outliers"]["by_column"]["price"]["outlier_count"] == 1
    assert cleaned["is_outlier_any"].sum() >= 1


# --------------------------- categorical consistency ----------------------- #
def test_categorical_case_and_whitespace_harmonised(clean_mod):
    raw = make_raw(category=["Books", "books ", "BOOKS"], location=["Delhi", "delhi", "Delhi"])
    cleaned = clean_mod.clean_dataframe(raw)["cleaned"]
    assert cleaned["category"].nunique() == 1
    assert cleaned["location"].nunique() == 1


def test_original_business_values_preserved(clean_mod):
    cleaned = clean_mod.clean_dataframe(make_raw())["cleaned"]
    assert cleaned["discount"].tolist() == [10.0, 20.0, 0.0]
    assert cleaned["final_price"].tolist() == [90.0, 160.0, 50.0]


# ---------------------- reconciliation / output schema --------------------- #
def test_all_rows_accounted_for(clean_mod):
    raw = pd.concat([make_raw(), make_raw().iloc[[1]]], ignore_index=True)
    raw.loc[2, "price"] = "-9"
    s = clean_mod.clean_dataframe(raw)["stats"]
    assert s["input_rows"] == 4
    assert s["valid_rows"] + s["quarantined_invalid_rows"] + s["duplicates_removed"] == 4
    assert s["reconciliation"]["balanced"]


def test_output_schema(clean_mod):
    out = clean_mod.clean_dataframe(make_raw())
    cols = list(out["cleaned"].columns)
    assert cols[0] == "source_row"
    assert cols[1:21] == clean_mod.EXPECTED_COLUMNS
    assert cols[21:] == [f"outlier_{c}" for c in clean_mod.OUTLIER_COLS] + ["is_outlier_any"]
    assert list(out["quarantined"].columns)[-1] == "rejection_reasons"


def test_run_cleaning_writes_files_and_reconciles(clean_mod, tmp_path, pickle_io):
    csv = tmp_path / "in.csv"
    make_raw().to_csv(csv, index=False)
    out_dir = tmp_path / "out"
    report = clean_mod.run_cleaning(csv, out_dir, sample_rows=2)
    assert (out_dir / "cleaned_transactions_sample.parquet").exists()
    assert (out_dir / "quarantined_transactions_sample.parquet").exists()
    saved = json.loads((out_dir / "cleaning_report_sample.json").read_text())
    assert saved["input_rows"] == 2 == report["input_rows"]
    assert saved["reconciliation"]["balanced"] is True
    assert not (out_dir / "cleaned_transactions.parquet").exists()  # no overwrite of full outputs


def test_parquet_roundtrip_preserves_dtypes(clean_mod, tmp_path):
    pytest.importorskip("pyarrow")
    cleaned = clean_mod.clean_dataframe(make_raw())["cleaned"]
    path = tmp_path / "c.parquet"
    clean_mod.write_parquet(cleaned, path)
    back = pd.read_parquet(path)
    assert back["price"].dtype == "float64"
    assert back["stock"].dtype == "int64"
    assert back["is_returned"].dtype == bool
    assert np.issubdtype(back["purchase_date"].dtype, np.datetime64)
    assert len(back) == len(cleaned)


def test_non_default_input_never_overwrites_full_outputs(clean_mod, tmp_path, pickle_io):
    """--input pointing at a dev file (no --sample-rows) must still write *_sample files."""
    csv = tmp_path / "amazon_ecommerce_sample_50000.csv"
    make_raw().to_csv(csv, index=False)
    out_dir = tmp_path / "out"
    report = clean_mod.run_cleaning(csv, out_dir, sample_rows=None)
    assert (out_dir / "cleaned_transactions_sample.parquet").exists()
    assert not (out_dir / "cleaned_transactions.parquet").exists()
    assert report["mode"].startswith("SAMPLE")
    assert report["input_file"] == str(csv)
