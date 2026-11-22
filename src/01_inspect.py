"""Person 1 - schema and sample inspection of the extracted dataset.

    python src/01_inspect.py [--input path.csv] [--rows 5]

Prints file size, encoding/delimiter, column names, dtypes, null counts and sample rows, and checks
the expected 20-column handoff contract (data/extracted/amazon_ecommerce_1M.csv). Exit code 1 if the
file is missing or columns differ from the contract.
"""
from __future__ import annotations

import argparse
import csv
import logging
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DEFAULT = ROOT / "data" / "extracted" / "amazon_ecommerce_1M.csv"
FALLBACK = ROOT / "data" / "raw" / "amazon_ecommerce_1M.csv"
EXPECTED = ["user_id", "product_id", "category", "subcategory", "brand", "price", "discount", "final_price",
            "rating", "review_count", "stock", "seller_id", "seller_rating", "purchase_date", "shipping_time_days",
            "location", "device", "payment_method", "is_returned", "delivery_status"]


def main(path: Path | None, rows: int) -> int:
    (ROOT / "logs").mkdir(exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s",
                        handlers=[logging.FileHandler(ROOT / "logs" / "inspect.log"), logging.StreamHandler(sys.stdout)])
    log = logging.getLogger("inspect")
    f = path or (DEFAULT if DEFAULT.exists() else FALLBACK)
    if not f.exists():
        log.error("input not found: %s", f); return 1
    raw = f.read_bytes()[:65536]
    try:
        raw.decode("utf-8"); enc = "utf-8"
    except UnicodeDecodeError:
        enc = "not utf-8 (check manually)"
    delim = csv.Sniffer().sniff(raw.decode("utf-8", "ignore").splitlines()[0] + "\n").delimiter
    log.info("file=%s size=%.1f MB encoding=%s delimiter=%r", f.relative_to(ROOT) if f.is_relative_to(ROOT) else f,
             f.stat().st_size / 2**20, enc, delim)
    df = pd.read_csv(f, dtype=str, nrows=200_000)
    log.info("columns (%d): %s", len(df.columns), list(df.columns))
    print(df.dtypes.to_string()); print(df.isna().sum().to_string()); print(df.head(rows).to_string())
    missing, extra = [c for c in EXPECTED if c not in df.columns], [c for c in df.columns if c not in EXPECTED]
    if missing or extra:
        log.error("schema mismatch: missing=%s extra=%s", missing, extra); return 1
    log.info("schema matches the 20-column handoff contract")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--input", type=Path); ap.add_argument("--rows", type=int, default=5)
    a = ap.parse_args(); sys.exit(main(a.input, a.rows))
