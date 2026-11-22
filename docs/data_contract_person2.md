# Data Contract – Person 2 (Cleaning & Transformation)

Owner: Person 2 · Consumers: Person 3 (analytics/storage), Person 4 (orchestration)

> **Status note:** this contract describes the behaviour implemented in `src/02_clean.py` and
> `src/03_transform.py`. Counts and distributions are *not* quoted here because they come from
> the generated `cleaning_report.json` / `transformation_report.json`; treat those reports as the
> source of truth for any given run.

## 1. Pipeline overview

```
data/extracted/amazon_ecommerce_1M.csv          (Person 1, read-only)
        │  src/02_clean.py
        ▼
data/processed/cleaned_transactions.parquet     valid rows (+ outlier flags)
data/processed/quarantined_transactions.parquet invalid + duplicate rows (+ rejection_reasons)
data/processed/cleaning_report.json
        │  src/03_transform.py
        ▼
data/processed/transformed_transactions.parquet transaction grain + derived features
data/processed/customer_summary.parquet         1 row per user_id
data/processed/category_summary.parquet         1 row per (category, subcategory)
data/processed/transformation_report.json
```

Sample runs write the same names with a `_sample` suffix (e.g. `cleaned_transactions_sample.parquet`)
and never touch full-run outputs. Logs: `logs/cleaning.log`, `logs/transformation.log` (appended per run).

## 2. Input

`data/extracted/amazon_ecommerce_1M.csv` (fallback `data/raw/amazon_ecommerce_1M.csv`), 1,000,000
rows × 20 columns, snake_case. The file is read **as strings** so nothing is altered on load.

| Column | Target type | Required? | Notes |
|---|---|---|---|
| user_id | string | yes | customer key |
| product_id | string | yes | |
| category | string | yes | case/whitespace harmonised |
| subcategory | string | no | |
| brand | string | no | |
| price | float64 | yes | must be > 0 |
| discount | float64 | yes | 0–100, preserved as-is |
| final_price | float64 | yes | must be ≥ 0, preserved as-is |
| rating | float64 | yes | 1–5 |
| review_count | int64 | yes | ≥ 0, integral |
| stock | int64 | yes | ≥ 0, integral |
| seller_id | string | no | |
| seller_rating | float64 | yes | 1–5 |
| purchase_date | datetime64[ns] | yes | must parse |
| is_returned | bool | yes | True/False/yes/no/1/0 variants accepted |
| shipping_time_days | int64 | yes | ≥ 0, integral |
| delivery_status | string | no | |
| location | string | no | |
| device | string | no | |
| payment_method | string | no | |

"Required" = a missing/unparseable value sends the row to quarantine. Optional columns may stay
null; **no value is ever imputed**.

## 3. Cleaning operations (`02_clean.py`)

1. Standardise column names (trim, lower-case, non-alphanumerics → `_`). Missing expected column → hard failure; extra columns are ignored with a warning.
2. Trim whitespace; treat `""`, `NA`, `N/A`, `NaN`, `null`, `none`, `NaT` (case-insensitive) as null.
3. **Duplicate detection:** a row identical to an earlier row across **all 20 columns** is a duplicate (first occurrence kept). `user_id + product_id` alone is never used – the same user can buy the same product repeatedly.
4. Type conversion; failed conversions are counted per column (`conversion_failures`).
5. Rule validation (a failing row is quarantined, never silently dropped):

| Reason code | Meaning |
|---|---|
| `missing_<col>` / `unparseable_<col>` | required value absent / present but not convertible (incl. ±inf) |
| `out_of_range_price` | price ≤ 0 |
| `out_of_range_discount` | discount < 0 or > 100 |
| `out_of_range_final_price` | final_price < 0 |
| `out_of_range_rating`, `out_of_range_seller_rating` | outside 1–5 |
| `out_of_range_stock`, `_review_count`, `_shipping_time_days` | < 0 |
| `non_integer_<col>` | stock / review_count / shipping_time_days not a whole number |
| `exact_duplicate` | repeat of a fully identical row |

   Multiple reasons are joined with `;`. A duplicate carries **only** `exact_duplicate`, so the three buckets are mutually exclusive.
6. Categorical harmonisation on valid rows: collapse whitespace; spelling variants differing only by case map to the most frequent variant (ties alphabetical).
7. **Outlier flagging (IQR, 1.5×)** on `price, final_price, review_count, stock, shipping_time_days`, computed on valid rows. Adds `outlier_<col>` booleans and `is_outlier_any`. **No outlier is removed.**
8. Row reconciliation: `input_rows = valid_rows + quarantined_invalid_rows + duplicates_removed`. The script raises if this fails.

### Output: `cleaned_transactions.parquet`
`source_row` (int64, 1-based data-row number in the input CSV) + the 20 columns above with the target types + `outlier_price`, `outlier_final_price`, `outlier_review_count`, `outlier_stock`, `outlier_shipping_time_days`, `is_outlier_any` (bool).

### Output: `quarantined_transactions.parquet`
`source_row` (int64) + the 20 columns as **original trimmed strings** (not converted, so the bad value is visible) + `rejection_reasons` (string). Contains both invalid rows and duplicates.

### Output: `cleaning_report.json`
`input_rows, valid_rows, quarantined_invalid_rows, duplicates_removed, quarantine_file_rows, reconciliation, null_counts_raw, null_counts_cleaned, conversion_failures, rejection_reason_counts, validation_results, outliers (bounds + counts per column), categorical_normalization_changes, categorical_profile, delivery_status_distribution, is_returned_distribution, observations, purchase_date_min/max, cleaned_dtypes, runtime_seconds`.

## 4. Transformation operations (`03_transform.py`)

Input: `cleaned_transactions[_sample].parquet`. All source columns (including `discount` and `final_price`) are preserved unchanged.

| Added column | Definition |
|---|---|
| `discount_amount` | `price - final_price` |
| `effective_discount_pct` | `(price - final_price) / price * 100` |
| `purchase_year` / `purchase_month` / `purchase_quarter` | int64 from `purchase_date` |
| `purchase_month_name` | English month name (`January`…) |
| `purchase_year_month` | `YYYY-MM` string |

**Important:** the source `discount` column is *not* assumed to reproduce `final_price`. Use
`effective_discount_pct` for any discount analysis. Negative values (final_price > price) are kept, not clipped.

## 5. Business metric definitions

* **Purchase value** of a transaction = `final_price` (each row is one purchase line; no quantity column exists).
* **Returned transaction** = `is_returned == True`.
* `total_*` metrics are **gross: returned transactions are included**.
* `*_excluding_returns` metrics exclude returned transactions; `returned_purchase_value` is the complement (`total = excluding_returns + returned`).
* `return_rate_pct` = `returned_transaction_count / transaction_count * 100`.

### `customer_summary.parquet` (one row per `user_id`, sorted by user_id)

| Column | Type | Definition |
|---|---|---|
| user_id | string | |
| transaction_count | int64 | all transactions |
| total_purchase_value | float64 | Σ final_price, all transactions |
| average_purchase_value | float64 | mean final_price, all transactions |
| returned_transaction_count | int64 | |
| return_rate_pct | float64 | 0–100 |
| first_purchase_date / last_purchase_date | datetime64[ns] | min / max purchase_date |
| non_returned_transaction_count | int64 | transaction_count − returned |
| returned_purchase_value | float64 | Σ final_price of returned rows |
| total_purchase_value_excluding_returns | float64 | Σ final_price of non-returned rows |

### `category_summary.parquet` (one row per category + subcategory, sorted)

| Column | Type | Definition |
|---|---|---|
| category, subcategory | string | |
| transaction_count | int64 | includes returned |
| total_purchase_value | float64 | Σ final_price, includes returned |
| average_price | float64 | mean price, includes returned |
| average_final_price | float64 | mean final_price, includes returned |
| average_rating | float64 | mean of per-transaction rating, includes returned |
| returned_transaction_count | int64 | |
| return_rate_pct | float64 | |
| returned_purchase_value | float64 | |
| total_purchase_value_excluding_returns | float64 | |

### `transformed_transactions.parquet`
Cleaned columns (incl. `source_row` and outlier flags) + the 7 derived columns in section 4. Same row count as `cleaned_transactions`.

### `transformation_report.json`
Row counts per output, 12 reconciliation checks (`all_passed`), discount-feature statistics, totals, date range, dtypes. The script raises if any check fails.

## 6. Known limitations / things to watch

* **Source `discount` vs `final_price`:** the project brief says the discount does not reproduce `final_price`. On the 50,000-row development file it did, to within 0.01 percentage points (rounding). It is not assumed consistent anyway: it is re-measured every run (`cleaning_report.json → observations.rows_where_discount_does_not_reproduce_final_price`, tolerance 0.05 pp) and `effective_discount_pct` should be used for analysis. Check the full-dataset run before relying on either view.
* `final_price > price` rows are kept (negative `discount_amount`); the count is in the reports.
* No quantity or product-price-history columns: purchase value = final_price per row.
* Customer and category summaries are **all-time** aggregates; no time-window versions are produced (filter `transformed_transactions` by `purchase_year_month` for that).
* `--sample-rows N` takes the **first N rows**, not a random sample. If the CSV is ordered, a sample may be unrepresentative.
* Duplicate detection is on trimmed text values: `10` and `10.0` are different strings and would not be treated as duplicates.
* Outlier flags are statistical (IQR on skewed price data flags many legitimate high-priced items); use them for analysis filters, not as an error indicator.
* `average_rating` is the mean over transactions, not de-duplicated per product.
* Categorical harmonisation only merges case/whitespace variants – it does not fix spelling differences.

## 7. How Person 3 should consume the data

```python
import pandas as pd
cust = pd.read_parquet("data/processed/customer_summary.parquet")
cat  = pd.read_parquet("data/processed/category_summary.parquet")
tx   = pd.read_parquet("data/processed/transformed_transactions.parquet",
                       columns=["user_id", "category", "final_price", "purchase_year_month", "is_returned"])
```

* Requires `pandas` and `pyarrow`. Types are stored in the file – no re-parsing needed.
* Net revenue: use `total_purchase_value_excluding_returns`; gross: `total_purchase_value`.
* Read only the columns you need from `transformed_transactions` (large).
* Quarantined rows are for audit; do not join them into analytics.
* Always check `cleaning_report.json → reconciliation.balanced` and `transformation_report.json → reconciliation.all_passed` before trusting a run.

## 8. How to run

```bash
pip install pandas pyarrow pytest

# sample (writes *_sample outputs)
python src/02_clean.py --sample-rows 10000
python src/03_transform.py --sample

# full
python src/02_clean.py
python src/03_transform.py

# tests
python -m pytest tests -q
```

Both scripts return exit code 0 on success and 1 on failure (details in the log), so an orchestrator
can branch on the return code. `02_clean.py` must complete before `03_transform.py`.
Optional flags: `02_clean.py --input PATH --output-dir DIR --iqr-multiplier 1.5`; `03_transform.py --input-dir DIR --output-dir DIR`.

## 9. Input file configuration

| Run | Command | Outputs |
|---|---|---|
| Full (default) | `python src/02_clean.py` → reads `data/extracted/amazon_ecommerce_1M.csv` | `cleaned_transactions.parquet`, … |
| Development | `python src/02_clean.py --input data/extracted/amazon_ecommerce_sample_50000.csv` | `*_sample.parquet` / `*_sample.json` |
| First N rows | `python src/02_clean.py --sample-rows 10000` | `*_sample` |

* Exactly **one** input file is processed per run; datasets are never combined.
* Any non-default `--input`, or any `--sample-rows`, writes `_sample` outputs, so full-dataset outputs cannot be overwritten. Run `03_transform.py --sample` afterwards.
* `data/lake/raw/.../transactions_*.csv` is an identical ingestion snapshot of the 1M CSV, **not** an additional dataset; it is never read by these scripts.
* The DummyJSON product catalogue (`data/lake/raw/source=dummyjson_api/...`) is **not used** and not joined: no product-ID mapping to `product_id` exists. Introduce it only once an explicit mapping is agreed.
