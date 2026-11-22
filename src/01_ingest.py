from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import shutil
import time
import urllib.error
import urllib.request
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOG_PATH = PROJECT_ROOT / "logs" / "ingestion.log"
RAW_LAKE_ROOT = PROJECT_ROOT / "data" / "lake" / "raw"

CSV_PATHS = [
    PROJECT_ROOT / "data" / "extracted" / "amazon_ecommerce_1M.csv",
    PROJECT_ROOT / "data" / "raw" / "amazon_ecommerce_1M.csv",
]
CATALOG_ENDPOINT = "https://dummyjson.com/products?limit=200"


def setup_logger() -> logging.Logger:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    log = logging.getLogger("ingest")
    log.setLevel(logging.INFO)
    if not log.handlers:
        formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
        file_handler = logging.FileHandler(LOG_PATH, encoding="utf-8")
        file_handler.setFormatter(formatter)
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        log.addHandler(file_handler)
        log.addHandler(console_handler)
    return log


logger = setup_logger()


def find_transactions_csv() -> Path:
    for candidate in CSV_PATHS:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("amazon_ecommerce_1M.csv was not found in data/extracted or data/raw")


def call_api_with_retries(url: str, retries: int = 3, delay: float = 1.5) -> dict:
    for attempt in range(1, retries + 1):
        try:
            logger.info(f"Requesting catalog API ({attempt}/{retries}): {url}")
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=15) as res:
                return json.loads(res.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as err:
            logger.warning(f"Attempt {attempt} failed for {url}: {err}")
            if attempt == retries:
                raise RuntimeError(f"Gave up fetching {url} after {retries} attempts") from err
            time.sleep(delay * attempt)
    raise RuntimeError("Unexpected retry loop exit")


def save_transactions_partition(run_date: str, run_tag: str) -> str:
    source_csv = find_transactions_csv()
    out_dir = (
        RAW_LAKE_ROOT
        / "source=amazon_ecommerce_csv"
        / "type=transactions"
        / f"ingest_date={run_date}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    out_file = out_dir / f"transactions_{run_tag}.csv"
    shutil.copy2(source_csv, out_file)

    size_mb = out_file.stat().st_size / (1024 * 1024)
    rel_path = str(out_file.relative_to(PROJECT_ROOT))
    logger.info(f"Saved transactions snapshot to {rel_path} ({size_mb:.2f} MB)")
    return rel_path


def save_catalog_partition(run_date: str, run_tag: str) -> str:
    body = call_api_with_retries(CATALOG_ENDPOINT)
    items = body.get("products") or []
    if not items:
        raise ValueError("Catalog API returned no products")

    # Normalize just to confirm tabular structure before writing raw JSON
    normalized = pd.json_normalize(items)
    out_dir = (
        RAW_LAKE_ROOT
        / "source=dummyjson_api"
        / "type=product_catalog"
        / f"ingest_date={run_date}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    out_file = out_dir / f"products_{run_tag}.json"
    out_file.write_text(json.dumps(items, indent=2), encoding="utf-8")

    rel_path = str(out_file.relative_to(PROJECT_ROOT))
    logger.info(f"Saved {len(normalized)} catalog products to {rel_path}")
    return rel_path


def run_ingestion() -> None:
    timestamp = datetime.now(timezone.utc)
    run_date = timestamp.strftime("%Y-%m-%d")
    run_tag = timestamp.strftime("%Y%m%dT%H%M%SZ")

    logger.info(f"Starting ingestion run {run_tag}")
    try:
        csv_rel = save_transactions_partition(run_date, run_tag)
        api_rel = save_catalog_partition(run_date, run_tag)

        manifest = {
            "run_id": run_tag,
            "ingest_date": run_date,
            "status": "SUCCESS",
            "files": {
                "transactions": csv_rel,
                "product_catalog": api_rel,
            },
        }
        RAW_LAKE_ROOT.mkdir(parents=True, exist_ok=True)
        manifest_path = RAW_LAKE_ROOT / f"manifest_{run_tag}.json"
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        logger.info(f"Completed ingestion run {run_tag}")
    except Exception as err:
        logger.error(f"Ingestion run {run_tag} failed: {err}", exc_info=True)
        raise

if __name__ == "__main__":
    run_ingestion()