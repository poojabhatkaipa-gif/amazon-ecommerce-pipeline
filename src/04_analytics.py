"""Person 3 - Step 4: analytics, EDA figures and SQL warehouse load.

Reads  data/processed/transformed_transactions[_sample].parquet  (Person 2 output)
Writes data/analytics/*.csv            business metric tables
       reports/figures/*.png           EDA plots
       data/warehouse/recomart.db      SQLite warehouse (schema: sql/schema.sql)
       data/analytics/analytics_report.json
       logs/analytics.log

Metric definitions (see docs/data_contract_person3.md):
    gross_value = sum(final_price), returns included
    net_value   = sum(final_price) over non-returned rows
    sparsity    = 1 - distinct(user, product) pairs / (n_users * n_products)

Usage:
    python src/04_analytics.py            # full dataset
    python src/04_analytics.py --sample   # *_sample input, writes recomart_sample.db
"""
from __future__ import annotations

import argparse
import json
import logging
import sqlite3
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed"
ANALYTICS = ROOT / "data" / "analytics"
WAREHOUSE = ROOT / "data" / "warehouse"
FIGURES = ROOT / "reports" / "figures"
SCHEMA = ROOT / "sql" / "schema.sql"
LOG = logging.getLogger("analytics")

COLS = ["source_row", "user_id", "product_id", "category", "subcategory", "brand", "price",
        "final_price", "rating", "purchase_date", "purchase_year_month", "is_returned",
        "device", "location", "payment_method", "shipping_time_days"]


def setup_logging() -> None:
    (ROOT / "logs").mkdir(exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s",
                        handlers=[logging.FileHandler(ROOT / "logs" / "analytics.log"),
                                  logging.StreamHandler(sys.stdout)])


def load(sample: bool) -> pd.DataFrame:
    path = PROCESSED / f"transformed_transactions{'_sample' if sample else ''}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing - run src/03_transform.py first")
    df = pd.read_parquet(path, columns=COLS)
    LOG.info("Loaded %d rows from %s", len(df), path.name)
    return df


def quality_gate(df: pd.DataFrame) -> None:
    """Lightweight checks before analytics (raises on failure)."""
    assert len(df) > 0, "empty input"
    for c in ("user_id", "product_id", "category", "final_price", "rating", "purchase_date"):
        assert df[c].notna().all(), f"nulls in key column {c}"
    assert df["rating"].between(1, 5).all(), "rating outside 1-5"
    assert df["source_row"].is_unique, "source_row not unique"


def metric_tables(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    d = df.assign(net=df["final_price"].where(~df["is_returned"], 0.0), ret=df["is_returned"].astype(int))
    monthly = d.groupby("purchase_year_month").agg(
        transactions=("user_id", "size"), active_users=("user_id", "nunique"),
        gross_value=("final_price", "sum"), net_value=("net", "sum"),
        return_rate_pct=("ret", lambda s: 100 * s.mean()), avg_rating=("rating", "mean")).reset_index()
    category = d.groupby("category").agg(
        transactions=("user_id", "size"), distinct_products=("product_id", "nunique"),
        gross_value=("final_price", "sum"), net_value=("net", "sum"),
        avg_final_price=("final_price", "mean"), avg_rating=("rating", "mean"),
        return_rate_pct=("ret", lambda s: 100 * s.mean())).reset_index().sort_values("gross_value", ascending=False)
    brand = d.groupby("brand").agg(transactions=("user_id", "size"), gross_value=("final_price", "sum"),
                                   avg_rating=("rating", "mean"),
                                   return_rate_pct=("ret", lambda s: 100 * s.mean())).reset_index()
    seg = {}
    for col in ("device", "payment_method", "location"):
        seg[col] = d.groupby(col).agg(transactions=("user_id", "size"), gross_value=("final_price", "sum"),
                                      return_rate_pct=("ret", lambda s: 100 * s.mean())).reset_index()
    top_items = (d.groupby("product_id").agg(purchases=("user_id", "size"), avg_rating=("rating", "mean"),
                                             avg_final_price=("final_price", "mean"))
                 .sort_values("purchases", ascending=False).head(20).reset_index())
    tx_per_user = d.groupby("user_id").size()
    segmentation = pd.cut(tx_per_user, [0, 1, 2, 3, np.inf], labels=["1 purchase", "2 purchases", "3 purchases", "4+ purchases"])
    seg_tbl = segmentation.value_counts().sort_index().rename_axis("segment").reset_index(name="users")
    return {"monthly_metrics": monthly, "category_metrics": category, "brand_metrics": brand,
            "top_items": top_items, "user_segments": seg_tbl, **{f"{k}_metrics": v for k, v in seg.items()}}


def interaction_stats(df: pd.DataFrame) -> dict:
    n_u, n_i = df["user_id"].nunique(), df["product_id"].nunique()
    pairs = df[["user_id", "product_id"]].drop_duplicates().shape[0]
    per_user = df.groupby("user_id")["product_id"].nunique()
    per_item = df.groupby("product_id")["user_id"].nunique()
    return {"rows": len(df), "users": n_u, "products": n_i, "distinct_pairs": pairs,
            "sparsity_pct": round(100 * (1 - pairs / (n_u * n_i)), 5),
            "users_with_ge2_products": int((per_user >= 2).sum()),
            "share_single_product_users_pct": round(100 * float((per_user == 1).mean()), 2),
            "items_per_user": per_user.describe().round(3).to_dict(),
            "buyers_per_item": per_item.describe().round(3).to_dict(),
            "rating_mean": round(float(df["rating"].mean()), 4),
            "date_min": str(df["purchase_date"].min().date()), "date_max": str(df["purchase_date"].max().date())}


def figures(df: pd.DataFrame, tables: dict[str, pd.DataFrame], sample: bool = False) -> list[str]:
    fig_dir = FIGURES.parent / "figures_sample" if sample else FIGURES
    fig_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")
    out = []

    def save(name):
        plt.tight_layout(); plt.savefig(fig_dir / name, dpi=130); plt.close(); out.append(name)

    sns.histplot(df["rating"], bins=30); plt.title("Rating distribution"); save("rating_hist.png")
    per_user = df.groupby("user_id").size()
    vc = per_user.value_counts().sort_index()
    plt.bar(vc.index.astype(str), vc.values); plt.yscale("log"); plt.xlabel("transactions per user"); plt.ylabel("users (log scale)")
    plt.title("User activity: most users bought once or twice"); save("user_activity_hist.png")
    per_item = df.groupby("product_id").size().sort_values(ascending=False)
    plt.plot(np.arange(len(per_item)), per_item.values); plt.xlabel("item rank"); plt.ylabel("purchases")
    plt.title("Item popularity (rank vs purchases)"); save("item_popularity.png")
    sns.histplot(np.log10(df["final_price"].clip(lower=1)), bins=50)
    plt.xlabel("log10(final_price)"); plt.title("Price distribution"); save("price_hist.png")
    m = tables["monthly_metrics"]
    plt.plot(m["purchase_year_month"], m["gross_value"] / 1e6, marker="o"); plt.xticks(rotation=90)
    plt.ylabel("gross value (millions)"); plt.title("Monthly gross purchase value"); save("monthly_value.png")
    c = tables["category_metrics"]
    sns.barplot(data=c, x="category", y="gross_value"); plt.xticks(rotation=30)
    plt.title("Gross value by category"); save("category_value.png")
    pivot = df.pivot_table(index="category", columns="brand", values="rating", aggfunc="mean")
    plt.figure(figsize=(11, 4)); sns.heatmap(pivot, annot=True, fmt=".2f", cmap="viridis")
    plt.title("Mean rating: category x brand"); save("heatmap_category_brand_rating.png")
    # sparsity pattern: every purchase of 3,000 random users (rows) against all items (columns)
    rng = np.random.default_rng(0)
    users = rng.choice(per_user.index.to_numpy(), size=min(3000, len(per_user)), replace=False)
    sub = df[df["user_id"].isin(users)][["user_id", "product_id"]].drop_duplicates()
    ucode = pd.Series(np.arange(len(users)), index=users)[sub["user_id"]].to_numpy()
    icode = pd.Categorical(sub["product_id"], categories=sorted(df["product_id"].unique())).codes
    plt.figure(figsize=(8, 4.5)); plt.scatter(icode, ucode, s=2, color="#1f3a5f")
    plt.xlabel("item index (all %d products)" % df["product_id"].nunique()); plt.ylabel("user (3,000 random users)")
    plt.title("Interaction matrix sample: each dot is one purchase (density %.4f%%)" % (100 * len(df[["user_id", "product_id"]].drop_duplicates()) / (per_user.size * df["product_id"].nunique())))
    save("sparsity_heatmap.png")
    num = df[["price", "final_price", "rating", "shipping_time_days"]].corr()
    plt.figure(figsize=(5, 4)); sns.heatmap(num, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1)
    plt.title("Numeric correlations"); save("correlation_heatmap.png")
    return out


def load_warehouse(df: pd.DataFrame, tables: dict[str, pd.DataFrame], db_path: Path) -> None:
    WAREHOUSE.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(db_path)
    con.executescript(SCHEMA.read_text())
    fact = df[["source_row", "user_id", "product_id", "category", "subcategory", "brand", "price",
               "final_price", "rating", "purchase_date", "purchase_year_month", "is_returned"]].copy()
    fact["purchase_date"] = fact["purchase_date"].dt.strftime("%Y-%m-%d")
    fact["is_returned"] = fact["is_returned"].astype(int)
    fact.to_sql("fact_interaction", con, if_exists="append", index=False, chunksize=100_000)
    for name in ("monthly_metrics", "category_metrics"):
        tables[name].to_sql(name, con, if_exists="append", index=False)
    n = con.execute("SELECT COUNT(*) FROM fact_interaction").fetchone()[0]
    assert n == len(df), f"warehouse rowcount {n} != {len(df)}"
    con.commit(); con.close()
    LOG.info("Warehouse loaded: %s (%d fact rows)", db_path.name, n)


def main(sample: bool = False) -> dict:
    setup_logging()
    df = load(sample)
    quality_gate(df)
    tables = metric_tables(df)
    ANALYTICS.mkdir(parents=True, exist_ok=True)
    sfx = "_sample" if sample else ""
    for k, v in tables.items():
        v.to_csv(ANALYTICS / f"{k}{sfx}.csv", index=False)
    stats = interaction_stats(df)
    figs = figures(df, tables, sample)
    db = WAREHOUSE / f"recomart{sfx}.db"
    if db.exists():
        db.unlink()
    load_warehouse(df, tables, db)
    report = {"interaction_stats": stats, "figures": figs, "tables": list(tables),
              "net_value_total": float(df.loc[~df["is_returned"], "final_price"].sum()),
              "gross_value_total": float(df["final_price"].sum())}
    (ANALYTICS / f"analytics_report{sfx}.json").write_text(json.dumps(report, indent=2, default=str))
    LOG.info("Analytics finished: %s", json.dumps(stats, default=str))
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", action="store_true")
    main(ap.parse_args().sample)
