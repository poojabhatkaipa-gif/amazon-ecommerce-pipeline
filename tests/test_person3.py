"""Tests for Person 3 modules (metrics, co-occurrence, feature-store versioning)."""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC))
import feature_store as fs  # noqa: E402


def _load(name, file):
    spec = importlib.util.spec_from_file_location(name, SRC / file)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


train_mod = _load("train06", "06_train.py")
feat_mod = _load("feat05", "05_features.py")


def test_evaluate_perfect_and_miss():
    X = sp.csr_matrix(np.zeros((2, 30), dtype=np.float32))
    scores = np.zeros((2, 30), dtype=np.float32); scores[0, 7] = 5; scores[1, 3] = 5
    res = train_mod.evaluate(lambda u: scores[u].copy(), X, np.array([0, 1]), np.array([7, 9]))
    assert res["recall@5"] == 0.5 and res["precision@5"] == 0.1
    assert res["ndcg@5"] == pytest.approx(0.5)            # hit at rank 1 -> 1/log2(2)=1, averaged over 2 users


def test_evaluate_masks_training_items():
    X = sp.csr_matrix(np.eye(1, 30, 7, dtype=np.float32))   # user 0 already bought item 7
    scores = np.zeros((1, 30), dtype=np.float32); scores[0, 7] = 9; scores[0, 2] = 8
    res = train_mod.evaluate(lambda u: scores[u].copy(), X, np.array([0]), np.array([7]))
    assert res["recall@20"] == 0.0


def test_split_leaves_last_item_out():
    df = pd.DataFrame({"source_row": [1, 2, 3, 4], "user_id": ["a", "a", "b", "c"], "product_id": ["p1", "p2", "p1", "p3"],
                       "purchase_date": pd.to_datetime(["2025-01-01", "2025-02-01", "2025-01-01", "2025-01-01"])})
    train, test = train_mod.split(df, 100)
    assert test[["user_id", "product_id"]].values.tolist() == [["a", "p2"]]
    assert len(train) == 3                                   # single-purchase users stay in training


def test_cooccurrence_cosine():
    df = pd.DataFrame({"user_id": ["u1", "u1", "u2", "u2", "u3"], "product_id": ["A", "B", "A", "B", "A"]})
    co = feat_mod.cooccurrence(df, 5)
    ab = co[(co.product_id == "A") & (co.neighbour_id == "B")].iloc[0]
    assert ab.co_count == 2 and ab.cosine == pytest.approx(2 / np.sqrt(3 * 2))


def test_feature_store_versions(tmp_path, monkeypatch):
    monkeypatch.setattr(fs, "STORE", tmp_path / "fs"); monkeypatch.setattr(fs, "REGISTRY", tmp_path / "fs" / "registry.json")
    monkeypatch.setattr(fs, "ROOT", tmp_path)
    docs = {"x": {"description": "d", "transformation": "t"}}
    v1 = fs.materialize("v", pd.DataFrame({"id": ["a", "b"], "x": [1, 2]}), "id", "src", docs)
    v2 = fs.materialize("v", pd.DataFrame({"id": ["a", "b"], "x": [10, 20]}), "id", "src", docs)
    assert (v1, v2) == (1, 2)
    assert fs.get_features("v", ["a"], version=1)["x"].iloc[0] == 1      # pinned (training)
    assert fs.get_features("v", ["a"])["x"].iloc[0] == 10                # latest (inference)
    assert fs.get_online_features("v", "zzz") == {}
    with pytest.raises(ValueError):
        fs.materialize("v", pd.DataFrame({"id": ["a"], "y": [1]}), "id", "src", docs)   # undocumented feature
