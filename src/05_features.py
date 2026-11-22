"""Person 3 - Step 5: feature engineering -> feature store + SQL warehouse.

Builds three feature views from transformed_transactions:
    user_features      (entity user_id)     activity / taste features
    item_features      (entity product_id)  popularity / quality features
    item_cooccurrence  (entity product_id)  top-N co-purchase neighbours (item-item cosine)

Each run registers a NEW version in the feature store (src/feature_store.py) and refreshes the
user_features / item_features / item_cooccurrence tables in the SQLite warehouse.
Also writes docs/feature_catalog.md from the registry.

Usage:  python src/05_features.py [--sample] [--top-n 10]
"""
from __future__ import annotations

import argparse
import logging
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parent))
import feature_store as fs  # noqa: E402

ROOT = fs.ROOT
LOG = logging.getLogger("features")


def mode_of(s: pd.Series):
    return s.value_counts().index[0]


def user_features(df: pd.DataFrame, as_of: pd.Timestamp) -> pd.DataFrame:
    g = df.groupby("user_id")
    out = pd.DataFrame({
        "transaction_count": g.size(),
        "distinct_products": g["product_id"].nunique(),
        "active_days": g["purchase_date"].nunique(),
        "recency_days": (as_of - g["purchase_date"].max()).dt.days,
        "avg_rating": g["rating"].mean(),
        "avg_final_price": g["final_price"].mean(),
        "return_rate_pct": 100 * g["is_returned"].mean(),
        "distinct_categories": g["category"].nunique(),
    })
    top = df.groupby(["user_id", "category"]).size().reset_index(name="n").sort_values(
        ["user_id", "n", "category"], ascending=[True, False, True]).drop_duplicates("user_id")
    out["top_category"] = top.set_index("user_id")["category"]
    return out.reset_index()


def item_features(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("product_id")
    out = pd.DataFrame({
        "purchase_count": g.size(), "distinct_buyers": g["user_id"].nunique(),
        "avg_rating": g["rating"].mean(), "avg_final_price": g["final_price"].mean(),
        "return_rate_pct": 100 * g["is_returned"].mean()})
    for col, name in (("category", "top_category"), ("subcategory", "top_subcategory"), ("brand", "top_brand")):
        cnt = df.groupby(["product_id", col], dropna=False).size().reset_index(name="n").sort_values(
            ["product_id", "n", col], ascending=[True, False, True]).drop_duplicates("product_id")
        out[name] = cnt.set_index("product_id")[col]
    return out.reset_index()


def cooccurrence(df: pd.DataFrame, top_n: int) -> pd.DataFrame:
    """Item-item cosine on the binary user x item matrix, top_n neighbours per item."""
    pairs = df[["user_id", "product_id"]].drop_duplicates()
    u = pairs["user_id"].astype("category"); i = pairs["product_id"].astype("category")
    X = sp.csr_matrix((np.ones(len(pairs), dtype=np.float32), (u.cat.codes, i.cat.codes)))
    C = (X.T @ X).tocoo()
    m = C.row != C.col
    r, c, v = C.row[m], C.col[m], C.data[m]
    deg = np.asarray(X.sum(axis=0)).ravel()
    cos = v / np.sqrt(deg[r] * deg[c])
    t = pd.DataFrame({"r": r, "c": c, "co_count": v.astype(int), "cosine": cos})
    t = t.sort_values(["r", "cosine", "co_count"], ascending=[True, False, False])
    t["rank"] = t.groupby("r").cumcount() + 1
    t = t[t["rank"] <= top_n]
    names = i.cat.categories
    return pd.DataFrame({"product_id": names[t["r"]], "neighbour_id": names[t["c"]],
                         "co_count": t["co_count"].values, "cosine": t["cosine"].values,
                         "rank": t["rank"].values})


USER_DOCS = {
    "transaction_count": {"description": "Purchase lines per user (user activity frequency)", "transformation": "count(rows) group by user_id"},
    "distinct_products": {"description": "Distinct products bought", "transformation": "nunique(product_id)"},
    "active_days": {"description": "Distinct purchase dates", "transformation": "nunique(purchase_date)"},
    "recency_days": {"description": "Days since last purchase vs dataset max date", "transformation": "max(purchase_date over data) - max(purchase_date of user)"},
    "avg_rating": {"description": "Mean rating on the user's purchases", "transformation": "mean(rating)"},
    "avg_final_price": {"description": "Mean spend per line", "transformation": "mean(final_price)"},
    "return_rate_pct": {"description": "Share of returned lines, 0-100", "transformation": "100*mean(is_returned)"},
    "distinct_categories": {"description": "Category breadth", "transformation": "nunique(category)"},
    "top_category": {"description": "Most purchased category (ties alphabetical)", "transformation": "mode(category)"},
}
ITEM_DOCS = {
    "purchase_count": {"description": "Purchase lines (popularity)", "transformation": "count(rows) group by product_id"},
    "distinct_buyers": {"description": "Distinct buyers", "transformation": "nunique(user_id)"},
    "avg_rating": {"description": "Mean rating (average rating per item)", "transformation": "mean(rating)"},
    "avg_final_price": {"description": "Mean final price", "transformation": "mean(final_price)"},
    "return_rate_pct": {"description": "Share of returned lines, 0-100", "transformation": "100*mean(is_returned)"},
    "top_category": {"description": "Modal category of the product", "transformation": "mode(category)"},
    "top_subcategory": {"description": "Modal subcategory", "transformation": "mode(subcategory)"},
    "top_brand": {"description": "Modal brand", "transformation": "mode(brand)"},
}
COOC_DOCS = {
    "neighbour_id": {"description": "Co-purchased product", "transformation": "item-item pair from users with >=2 products"},
    "co_count": {"description": "Users who bought both", "transformation": "(X^T X)[i,j] on binary user x item matrix"},
    "cosine": {"description": "Cosine similarity of buyer sets", "transformation": "co_count / sqrt(buyers_i * buyers_j)"},
    "rank": {"description": "Neighbour rank within the item (1 = most similar)", "transformation": "rank by cosine desc, co_count desc"},
}


def write_catalog() -> None:
    d = fs.describe()
    d = d[~d["view"].str.endswith("_sample")]
    latest = d[d["version"] == d.groupby("view")["version"].transform("max")]
    lines = ["# Feature catalog (auto-generated from data/feature_store/registry.json)\n"]
    for view, g in latest.groupby("view"):
        lines.append(f"\n## {view} (v{g['version'].iloc[0]}, source: `{g['source'].iloc[0]}`)\n")
        lines.append("| Feature | dtype | Description | Transformation |\n|---|---|---|---|")
        for _, r in g.iterrows():
            lines.append(f"| {r['feature']} | {r['dtype']} | {r['description']} | `{r['transformation']}` |")
    (ROOT / "docs" / "feature_catalog.md").write_text("\n".join(lines) + "\n")


def main(sample: bool = False, top_n: int = 10) -> dict:
    (ROOT / "logs").mkdir(exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s",
                        handlers=[logging.FileHandler(ROOT / "logs" / "features.log"), logging.StreamHandler(sys.stdout)])
    sfx = "_sample" if sample else ""
    src = f"data/processed/transformed_transactions{sfx}.parquet"
    df = pd.read_parquet(ROOT / src, columns=["user_id", "product_id", "category", "subcategory", "brand",
                                              "rating", "final_price", "purchase_date", "is_returned"])
    as_of = df["purchase_date"].max()
    uf, itf, co = user_features(df, as_of), item_features(df), cooccurrence(df, top_n)
    assert uf["user_id"].is_unique and itf["product_id"].is_unique
    assert uf["transaction_count"].sum() == len(df) == itf["purchase_count"].sum()
    versions = {}
    versions[f"user_features{sfx}"] = fs.materialize(f"user_features{sfx}", uf, "user_id", src, USER_DOCS, tag=sfx or "full")
    versions[f"item_features{sfx}"] = fs.materialize(f"item_features{sfx}", itf, "product_id", src, ITEM_DOCS, tag=sfx or "full")
    co_store = co.assign(pair_key=co["product_id"] + "|" + co["neighbour_id"])
    versions[f"item_cooccurrence{sfx}"] = fs.materialize(f"item_cooccurrence{sfx}", co_store, "pair_key", src, {**COOC_DOCS, "product_id": {"description": "Anchor product", "transformation": "item i"}}, tag=sfx or "full")
    write_catalog()
    db = ROOT / "data" / "warehouse" / f"recomart{sfx}.db"
    if db.exists():
        con = sqlite3.connect(db)
        for t, frame in (("user_features", uf), ("item_features", itf), ("item_cooccurrence", co)):
            con.execute(f"DELETE FROM {t}")
            frame.assign(**({"recency_days": frame["recency_days"]} if t == "user_features" else {})).to_sql(
                t, con, if_exists="append", index=False, chunksize=100_000)
        con.commit(); con.close()
    LOG.info("Feature versions: %s | users=%d items=%d cooc_pairs=%d", versions, len(uf), len(itf), len(co))
    return versions


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", action="store_true"); ap.add_argument("--top-n", type=int, default=10)
    a = ap.parse_args()
    main(a.sample, a.top_n)
