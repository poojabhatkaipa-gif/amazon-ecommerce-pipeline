"""Person 3 - Step 6: recommendation models, offline evaluation and MLflow tracking.

Task      : top-K product recommendation from purchase history (implicit feedback).
Split     : leave-last-out. For every user with >=2 distinct products, the most recent product
            (purchase_date, then source_row) is the test item; everything else is training.
            Users with 1 product stay in training only. Evaluation uses a fixed random sample of
            eligible users (--eval-users, seed 42).
Models    : popularity baseline, TruncatedSVD collaborative filtering (grid over factors),
            content-based (category/subcategory/brand/price-band profile, cosine).
Metrics   : Precision@K, Recall@K (= HitRate@K, one test item per user), NDCG@K, K in {5,10,20},
            plus item coverage. Test items never seen in training cannot be recommended; their share is
            reported as `unreachable_test_share` (upper bound on recall).
Tracking  : MLflow (sqlite backend in mlflow/), params + metrics + feature-store versions + artifacts.
Output    : models/svd_model.npz, models/popularity.npz, reports/model_performance.md/json

Usage:  python src/06_train.py [--sample] [--eval-users 10000] [--factors 16 32 64]
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import OneHotEncoder, normalize

sys.path.insert(0, str(Path(__file__).resolve().parent))
import feature_store as fs  # noqa: E402

ROOT = fs.ROOT
LOG = logging.getLogger("train")
KS = (5, 10, 20)
SEED = 42


# --------------------------------------------------------------------------- data
def load_interactions(sample: bool) -> pd.DataFrame:
    p = ROOT / "data" / "processed" / f"transformed_transactions{'_sample' if sample else ''}.parquet"
    df = pd.read_parquet(p, columns=["source_row", "user_id", "product_id", "category", "subcategory",
                                     "brand", "final_price", "purchase_date"])
    # one row per (user, product): keep the latest purchase
    df = df.sort_values(["purchase_date", "source_row"]).drop_duplicates(["user_id", "product_id"], keep="last")
    return df.reset_index(drop=True)


def split(df: pd.DataFrame, n_eval: int):
    df = df.copy()
    df["order"] = df.groupby("user_id").cumcount(ascending=False)  # 0 = latest
    cnt = df.groupby("user_id")["product_id"].transform("size")
    is_test = (df["order"] == 0) & (cnt >= 2)
    test, train = df[is_test], df[~is_test]
    rng = np.random.default_rng(SEED)
    if len(test) > n_eval:
        test = test.iloc[np.sort(rng.choice(len(test), n_eval, replace=False))]
    return train.reset_index(drop=True), test.reset_index(drop=True)


def build_matrix(train: pd.DataFrame, test: pd.DataFrame):
    users = pd.Index(sorted(train["user_id"].unique()))
    items = pd.Index(sorted(train["product_id"].unique()))
    ui = users.get_indexer(train["user_id"]); ii = items.get_indexer(train["product_id"])
    X = sp.csr_matrix((np.ones(len(train), dtype=np.float32), (ui, ii)), shape=(len(users), len(items)))
    tu = users.get_indexer(test["user_id"]); ti = items.get_indexer(test["product_id"])
    return users, items, X, tu, ti


# --------------------------------------------------------------------------- evaluation
def evaluate(score_fn, X: sp.csr_matrix, test_users: np.ndarray, test_items: np.ndarray, batch: int = 500) -> dict:
    """score_fn(user_idx_array) -> dense (len, n_items) float32. Training items are masked."""
    maxk = max(KS)
    hits = {k: 0.0 for k in KS}; ndcg = {k: 0.0 for k in KS}
    recommended = set()
    reachable = test_items >= 0
    for s in range(0, len(test_users), batch):
        u = test_users[s:s + batch]; t = test_items[s:s + batch]
        sc = score_fn(u).astype(np.float32, copy=False)
        rows = X[u]
        sc[np.repeat(np.arange(len(u)), np.diff(rows.indptr)), rows.indices] = -np.inf
        top = np.argpartition(-sc, maxk, axis=1)[:, :maxk]
        order = np.argsort(-np.take_along_axis(sc, top, axis=1), axis=1)
        top = np.take_along_axis(top, order, axis=1)
        recommended.update(top[:, :10].ravel().tolist())
        match = top == t[:, None]
        for k in KS:
            m = match[:, :k]
            hits[k] += m.any(axis=1).sum()
            pos = m.argmax(axis=1)
            ndcg[k] += (m.any(axis=1) / np.log2(pos + 2)).sum()
    n = len(test_users)
    out = {"n_eval_users": n, "unreachable_test_share": float(1 - reachable.mean()),
           "item_coverage@10": len(recommended) / X.shape[1]}
    for k in KS:
        out[f"recall@{k}"] = hits[k] / n
        out[f"precision@{k}"] = hits[k] / (n * k)
        out[f"ndcg@{k}"] = ndcg[k] / n
    return out


# --------------------------------------------------------------------------- models
def popularity_scorer(X):
    pop = np.asarray(X.sum(axis=0)).ravel().astype(np.float32)
    return (lambda u: np.tile(pop, (len(u), 1))), pop


def svd_fit(X, k):
    svd = TruncatedSVD(n_components=k, algorithm="randomized", n_iter=5, random_state=SEED)
    uf = svd.fit_transform(X).astype(np.float32)          # U * S
    itf = svd.components_.T.astype(np.float32)             # V
    return (lambda u: uf[u] @ itf.T), uf, itf, float(svd.explained_variance_ratio_.sum())


def content_fit(train, items, X, price_bins=5):
    g = train.groupby("product_id")
    attrs = pd.DataFrame({
        "category": g["category"].agg(lambda s: s.mode().iat[0]),
        "subcategory": g["subcategory"].agg(lambda s: s.mode().iat[0] if s.notna().any() else "NA"),
        "brand": g["brand"].agg(lambda s: s.mode().iat[0] if s.notna().any() else "NA"),
        "price": g["final_price"].mean()}).reindex(items)
    attrs["price_band"] = pd.qcut(attrs["price"], price_bins, labels=False, duplicates="drop").astype(str)
    F = OneHotEncoder(handle_unknown="ignore").fit_transform(attrs[["category", "subcategory", "brand", "price_band"]])
    F = normalize(F.tocsr().astype(np.float32))
    pop = np.log1p(np.asarray(X.sum(axis=0)).ravel()).astype(np.float32) * 1e-4   # tie-breaker only

    def score(u):
        prof = normalize(X[u] @ F)                          # mean item profile of user history
        return (prof @ F.T).toarray() + pop
    return score, F.shape[1]


# --------------------------------------------------------------------------- main
def main(sample: bool = False, n_eval: int = 10000, factors=(16, 32, 64)) -> dict:
    (ROOT / "logs").mkdir(exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s",
                        handlers=[logging.FileHandler(ROOT / "logs" / "training.log"), logging.StreamHandler(sys.stdout)])
    sfx = "_sample" if sample else ""
    df = load_interactions(sample)
    train, test = split(df, n_eval)
    users, items, X, tu, ti = build_matrix(train, test)
    keep = tu >= 0
    tu, ti = tu[keep], ti[keep]
    LOG.info("train pairs=%d users=%d items=%d | eval users=%d", len(train), len(users), len(items), len(tu))

    (ROOT / "mlflow").mkdir(exist_ok=True)
    mlflow.set_tracking_uri(f"sqlite:///{ROOT / 'mlflow' / 'mlflow.db'}")
    exp = f"recomart_recommender{sfx}"
    if mlflow.get_experiment_by_name(exp) is None:
        mlflow.create_experiment(exp, artifact_location=(ROOT / "mlflow" / "artifacts" / exp).as_uri())
    mlflow.set_experiment(exp)
    try:
        fv = {v: fs.latest_version(f"{v}{sfx}") for v in ("user_features", "item_features", "item_cooccurrence")}
    except KeyError:
        fv = {}
    common = {"split": "leave_last_out", "seed": SEED, "train_pairs": len(train), "n_users": len(users),
              "n_items": len(items), "n_eval_users": len(tu), "data_source": f"transformed_transactions{sfx}.parquet",
              **{f"feature_store.{k}_version": v for k, v in fv.items()}}
    results, best_svd = [], None

    def run(name, params, fn, extra_metrics=None, artifacts=None):
        nonlocal best_svd
        with mlflow.start_run(run_name=name) as r:
            mlflow.log_params({**common, **params}); t0 = time.time()
            m = evaluate(fn, X, tu, ti)
            m["eval_seconds"] = time.time() - t0
            m.update(extra_metrics or {})
            mlflow.log_metrics({k.replace("@", "_at_"): float(v) for k, v in m.items()})
            for a in artifacts or []:
                mlflow.log_artifact(str(a))
            results.append({"run": name, "run_id": r.info.run_id, **params, **m})
            LOG.info("%s | recall@10=%.5f ndcg@10=%.5f", name, m["recall@10"], m["ndcg@10"])

    fn, pop = popularity_scorer(X)
    (ROOT / "models").mkdir(exist_ok=True)
    np.savez(ROOT / "models" / f"popularity{sfx}.npz", items=np.array(items.tolist(), dtype=str), pop=pop)
    run("popularity", {"model": "popularity"}, fn, artifacts=[ROOT / "models" / f"popularity{sfx}.npz"])

    for k in factors:
        k = min(k, min(X.shape) - 1)
        fn, uf, itf, ev = svd_fit(X, k)
        run(f"svd_k{k}", {"model": "svd", "n_factors": k, "n_iter": 5}, fn, {"explained_variance": ev})
        ndcg = results[-1]["ndcg@10"]
        if best_svd is None or ndcg > best_svd[0]:
            best_svd = (ndcg, k, uf, itf, results[-1]["run_id"])

    fn, nf = content_fit(train, items, X)
    run("content_based", {"model": "content", "profile": "category+subcategory+brand+price_band", "n_features": nf}, fn)

    # export deployable SVD model + serving data (training history)
    _, k, uf, itf, rid = best_svd
    mp = ROOT / "models" / f"svd_model{sfx}.npz"
    np.savez_compressed(mp, users=np.array(users.tolist(), dtype=str), items=np.array(items.tolist(), dtype=str), user_factors=uf,
                        item_factors=itf, popularity=pop, n_factors=k, run_id=rid)
    sp.save_npz(ROOT / "models" / f"train_matrix{sfx}.npz", X)
    with mlflow.start_run(run_id=rid):
        mlflow.log_artifact(str(mp))

    res = pd.DataFrame(results)
    (ROOT / "reports").mkdir(exist_ok=True)
    res.to_json(ROOT / "reports" / f"model_performance{sfx}.json", orient="records", indent=2)
    cols = ["run", "recall@5", "recall@10", "recall@20", "precision@10", "ndcg@10", "ndcg@20", "item_coverage@10", "run_id"]
    md = ["# Model performance (leave-last-out)\n",
          f"Eval users: {len(tu)} | train pairs: {len(train)} | catalogue: {len(items)} items | "
          f"unreachable test items: {res['unreachable_test_share'].iloc[0]:.3%}\n",
          "Random-ranking recall@10 would be about %.5f.\n" % (10 / len(items)),
          res[cols].round(5).to_markdown(index=False) if hasattr(res, "to_markdown") else res[cols].round(5).to_string(index=False)]
    (ROOT / "reports" / f"model_performance{sfx}.md").write_text("\n".join(md) + "\n")
    return {"results": results, "best_svd_k": k, "best_svd_run": rid}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", action="store_true"); ap.add_argument("--eval-users", type=int, default=10000)
    ap.add_argument("--factors", type=int, nargs="+", default=[16, 32, 64])
    a = ap.parse_args()
    main(a.sample, a.eval_users, tuple(a.factors))
