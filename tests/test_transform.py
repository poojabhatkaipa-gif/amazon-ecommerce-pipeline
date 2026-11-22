"""Unit tests for src/03_transform.py using small synthetic datasets."""

import json
import math

import pandas as pd
import pytest


def make_cleaned() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "user_id": ["U1", "U1", "U1", "U2", "U3"],
            "product_id": ["P1", "P2", "P3", "P1", "P4"],
            "category": ["Books", "Books", "Toys", "Books", "Toys"],
            "subcategory": ["Fiction", "Fiction", "Games", "Fiction", "Games"],
            "price": [100.0, 200.0, 50.0, 100.0, 80.0],
            "discount": [10.0, 20.0, 5.0, 10.0, 50.0],  # deliberately inconsistent with final_price
            "final_price": [90.0, 150.0, 50.0, 100.0, 100.0],  # last: final > price
            "rating": [4.0, 5.0, 3.0, 2.0, 4.0],
            "is_returned": [False, True, False, True, False],
            "purchase_date": pd.to_datetime(
                ["2024-01-15", "2024-04-20", "2024-12-31", "2025-07-04", "2026-03-01"]
            ),
        }
    )


# --------------------------- pricing features ------------------------------ #
def test_discount_amount_and_effective_pct(transform_mod):
    out = transform_mod.add_pricing_features(make_cleaned())
    assert out["discount_amount"].tolist() == [10.0, 50.0, 0.0, 0.0, -20.0]
    assert out["effective_discount_pct"].tolist() == pytest.approx([10.0, 25.0, 0.0, 0.0, -25.0])


def test_original_columns_not_overwritten(transform_mod):
    src = make_cleaned()
    out = transform_mod.add_pricing_features(src)
    pd.testing.assert_series_equal(out["discount"], src["discount"])
    pd.testing.assert_series_equal(out["final_price"], src["final_price"])
    # effective pct differs from the source discount column (not assumed consistent)
    assert out.loc[1, "effective_discount_pct"] != out.loc[1, "discount"]


# ----------------------------- date dimensions ----------------------------- #
def test_date_dimensions(transform_mod):
    out = transform_mod.add_date_dimensions(make_cleaned())
    assert out["purchase_year"].tolist() == [2024, 2024, 2024, 2025, 2026]
    assert out["purchase_month"].tolist() == [1, 4, 12, 7, 3]
    assert out["purchase_month_name"].tolist() == ["January", "April", "December", "July", "March"]
    assert out["purchase_quarter"].tolist() == [1, 2, 4, 3, 1]
    assert out["purchase_year_month"].tolist() == ["2024-01", "2024-04", "2024-12", "2025-07", "2026-03"]


# --------------------------- customer aggregation -------------------------- #
def test_customer_summary(transform_mod):
    cust = transform_mod.build_customer_summary(make_cleaned()).set_index("user_id")
    u1 = cust.loc["U1"]
    assert u1["transaction_count"] == 3
    assert u1["total_purchase_value"] == 290.0                  # includes returned
    assert u1["average_purchase_value"] == pytest.approx(290 / 3)
    assert u1["returned_transaction_count"] == 1
    assert u1["return_rate_pct"] == pytest.approx(100 / 3)
    assert u1["returned_purchase_value"] == 150.0
    assert u1["total_purchase_value_excluding_returns"] == 140.0
    assert u1["non_returned_transaction_count"] == 2
    assert u1["first_purchase_date"] == pd.Timestamp("2024-01-15")
    assert u1["last_purchase_date"] == pd.Timestamp("2024-12-31")
    assert cust.loc["U2", "return_rate_pct"] == 100.0
    assert cust.loc["U3", "return_rate_pct"] == 0.0


def test_customer_summary_is_unique_and_sums(transform_mod):
    df = make_cleaned()
    cust = transform_mod.build_customer_summary(df)
    assert cust["user_id"].is_unique
    assert cust["transaction_count"].sum() == len(df)
    assert cust["total_purchase_value"].sum() == df["final_price"].sum()


# --------------------------- category aggregation -------------------------- #
def test_category_summary(transform_mod):
    cat = transform_mod.build_category_summary(make_cleaned()).set_index(["category", "subcategory"])
    books = cat.loc[("Books", "Fiction")]
    assert books["transaction_count"] == 3
    assert books["total_purchase_value"] == 340.0
    assert books["average_price"] == pytest.approx(400 / 3)
    assert books["average_final_price"] == pytest.approx(340 / 3)
    assert books["average_rating"] == pytest.approx(11 / 3)
    assert books["returned_transaction_count"] == 2
    assert books["return_rate_pct"] == pytest.approx(200 / 3)
    assert books["total_purchase_value_excluding_returns"] == 90.0
    toys = cat.loc[("Toys", "Games")]
    assert toys["transaction_count"] == 2 and toys["returned_transaction_count"] == 0


# ------------------------------ output schema ------------------------------ #
def test_schemas_and_reconciliation(transform_mod):
    df = make_cleaned()
    frames = transform_mod.transform_dataframe(df)
    tr = frames["transformed"]
    assert list(df.columns) == list(tr.columns[: len(df.columns)])
    for col in transform_mod.NEW_PRICING_COLS + transform_mod.NEW_DATE_COLS:
        assert col in tr.columns
    assert list(frames["customer"].columns) == transform_mod.CUSTOMER_COLUMNS
    assert list(frames["category"].columns) == transform_mod.CATEGORY_COLUMNS
    assert frames["customer"]["transaction_count"].dtype == "int64"
    assert frames["customer"]["return_rate_pct"].dtype == "float64"
    assert frames["category"]["returned_transaction_count"].dtype == "int64"
    rec = transform_mod.reconcile(df, frames)
    assert rec["all_passed"], rec


def test_validate_input_rejects_bad_data(transform_mod):
    with pytest.raises(ValueError, match="missing columns"):
        transform_mod.validate_input(make_cleaned().drop(columns=["price"]))
    bad = make_cleaned()
    bad.loc[0, "price"] = 0
    with pytest.raises(ValueError, match="price <= 0"):
        transform_mod.validate_input(bad)


def test_missing_cleaned_input_gives_clear_error(transform_mod, tmp_path):
    with pytest.raises(FileNotFoundError, match="02_clean.py"):
        transform_mod.run_transformation(tmp_path, tmp_path, sample=True)


# ------------------------- end to end (clean -> transform) ----------------- #
def test_end_to_end_sample_pipeline(clean_mod, transform_mod, tmp_path, pickle_io):
    from test_clean import make_raw  # reuse the synthetic raw builder

    csv = tmp_path / "in.csv"
    pd.concat([make_raw(), make_raw().iloc[[0]]], ignore_index=True).to_csv(csv, index=False)
    out_dir = tmp_path / "processed"
    clean_report = clean_mod.run_cleaning(csv, out_dir, sample_rows=10)
    assert clean_report["duplicates_removed"] == 1

    report = transform_mod.run_transformation(out_dir, out_dir, sample=True)
    for name in ["transformed_transactions", "customer_summary", "category_summary"]:
        assert (out_dir / f"{name}_sample.parquet").exists()
    saved = json.loads((out_dir / "transformation_report_sample.json").read_text())
    assert saved["reconciliation"]["all_passed"] is True
    assert report["input_rows"] == clean_report["valid_rows"] == 3
    assert math.isclose(report["totals"]["total_purchase_value"], 300.0)


def test_parquet_roundtrip_transform_outputs(transform_mod, tmp_path):
    pytest.importorskip("pyarrow")
    frames = transform_mod.transform_dataframe(make_cleaned())
    for name, df in frames.items():
        path = tmp_path / f"{name}.parquet"
        transform_mod.write_parquet(df, path)
        back = transform_mod.read_parquet(path)
        assert list(back.columns) == list(df.columns)
        assert len(back) == len(df)
