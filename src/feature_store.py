"""Minimal file-based feature store with a metadata registry (Person 3).

Layout
    data/feature_store/registry.json
    data/feature_store/<view>/v<N>/features.parquet

Every call to ``materialize`` creates a new immutable version of a feature view and records
its feature names, dtypes, source, transformation text, row count, content hash and time in the
registry. ``get_features`` retrieves a pinned version for training (reproducible) or the latest
version for inference.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
STORE = ROOT / "data" / "feature_store"
REGISTRY = STORE / "registry.json"


def _load_registry() -> dict:
    if REGISTRY.exists():
        return json.loads(REGISTRY.read_text())
    return {"views": {}}


def _save_registry(reg: dict) -> None:
    STORE.mkdir(parents=True, exist_ok=True)
    REGISTRY.write_text(json.dumps(reg, indent=2))


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def materialize(view: str, df: pd.DataFrame, entity_key: str, source: str,
                feature_docs: dict[str, dict], tag: str = "") -> int:
    """Write ``df`` as the next version of ``view`` and register it. Returns the version number."""
    reg = _load_registry()
    entry = reg["views"].setdefault(view, {"entity_key": entity_key, "versions": {}})
    version = max((int(v) for v in entry["versions"]), default=0) + 1
    missing = [c for c in df.columns if c != entity_key and c not in feature_docs]
    if missing:
        raise ValueError(f"undocumented features in {view}: {missing}")
    if not df[entity_key].is_unique:
        raise ValueError(f"{entity_key} not unique in {view}")
    out = STORE / view / f"v{version}" / "features.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    entry["versions"][str(version)] = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "rows": int(len(df)), "path": str(out.relative_to(ROOT)), "sha256_16": _sha(out),
        "source": source, "tag": tag,
        "features": [{"name": c, "dtype": str(df[c].dtype), **feature_docs[c]}
                     for c in df.columns if c != entity_key]}
    _save_registry(reg)
    return version


def latest_version(view: str) -> int:
    reg = _load_registry()
    return max(int(v) for v in reg["views"][view]["versions"])


def get_features(view: str, ids: list | None = None, version: int | None = None,
                 columns: list[str] | None = None) -> pd.DataFrame:
    """Batch retrieval. version=None -> latest (inference); pass a version to pin (training)."""
    reg = _load_registry()
    v = version or latest_version(view)
    meta = reg["views"][view]["versions"][str(v)]
    key = reg["views"][view]["entity_key"]
    df = pd.read_parquet(ROOT / meta["path"], columns=None if columns is None else [key, *columns])
    if ids is not None:
        df = df[df[key].isin(set(ids))]
    return df.reset_index(drop=True)


def get_online_features(view: str, entity_id: str, version: int | None = None) -> dict:
    """Single-entity lookup used by the inference interface."""
    df = get_features(view, [entity_id], version)
    return {} if df.empty else df.iloc[0].to_dict()


def describe() -> pd.DataFrame:
    """Flat table of every registered feature (name, view, version, source, transformation)."""
    rows = []
    for view, e in _load_registry()["views"].items():
        for ver, m in e["versions"].items():
            for f in m["features"]:
                rows.append({"view": view, "version": int(ver), "feature": f["name"], "dtype": f["dtype"],
                             "description": f.get("description", ""), "transformation": f.get("transformation", ""),
                             "source": m["source"], "created_utc": m["created_utc"]})
    return pd.DataFrame(rows)
