"""Inference interface for the trained recommender (Person 3).

    python src/recommend.py U356787 --k 10
    from recommend import Recommender; Recommender().recommend("U356787", 10)

Known users get SVD scores with already-purchased items removed. Unknown users (cold start) fall
back to global popularity. Item metadata is enriched from the feature store (item_features).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parent))
import feature_store as fs  # noqa: E402

ROOT = fs.ROOT


class Recommender:
    def __init__(self, sample: bool = False):
        sfx = "_sample" if sample else ""
        m = np.load(ROOT / "models" / f"svd_model{sfx}.npz", allow_pickle=False)
        self.users, self.items = m["users"], m["items"]
        self.uf, self.itf, self.pop = m["user_factors"], m["item_factors"], m["popularity"]
        self.run_id = str(m["run_id"]); self.n_factors = int(m["n_factors"])
        self.X = sp.load_npz(ROOT / "models" / f"train_matrix{sfx}.npz").tocsr()
        self.user_idx = {u: i for i, u in enumerate(self.users)}
        self.item_view = f"item_features{sfx}"

    def recommend(self, user_id: str, k: int = 10) -> pd.DataFrame:
        if user_id in self.user_idx:
            u = self.user_idx[user_id]
            score = self.itf @ self.uf[u]
            score[self.X[u].indices] = -np.inf
            source = "svd"
        else:
            score, source = self.pop.astype(np.float32).copy(), "popularity_cold_start"
        top = np.argpartition(-score, k)[:k]
        top = top[np.argsort(-score[top])]
        ids = self.items[top].tolist()
        out = pd.DataFrame({"rank": range(1, k + 1), "product_id": ids, "score": score[top], "source": source})
        meta = fs.get_features(self.item_view, ids, columns=["top_category", "top_brand", "avg_rating", "purchase_count"])
        return out.merge(meta, on="product_id", how="left")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("user_id"); ap.add_argument("--k", type=int, default=10); ap.add_argument("--sample", action="store_true")
    a = ap.parse_args()
    r = Recommender(a.sample)
    print(f"model run_id={r.run_id} factors={r.n_factors}")
    print(r.recommend(a.user_id, a.k).to_string(index=False))
