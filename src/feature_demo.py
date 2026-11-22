"""Sample feature retrieval: training (pinned version) vs inference (latest) vs online lookup."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import feature_store as fs  # noqa: E402

pd_opts = {"display.width": 160, "display.max_columns": 20}
import pandas as pd  # noqa: E402
pd.set_option("display.width", 160); pd.set_option("display.max_columns", 20)

reg = fs._load_registry()
print("== Registered views ==")
for v, e in reg["views"].items():
    if v.endswith("_sample"):
        continue
    for ver, m in e["versions"].items():
        print(f"{v:18s} v{ver} rows={m['rows']:>8} features={len(m['features']):>2} created={m['created_utc']} sha={m['sha256_16']}")

ids = ["U356787", "U198246", "U100001"]
print("\n== TRAINING: pinned version (reproducible) ==")
print(fs.get_features("user_features", ids, version=1, columns=["transaction_count", "avg_final_price", "return_rate_pct", "top_category"]))
print("\n== INFERENCE: latest version ==")
print(f"latest user_features version = v{fs.latest_version('user_features')}")
print(fs.get_features("user_features", ids, columns=["transaction_count", "avg_final_price", "top_category"]))
print("\n== ONLINE single-entity lookup (item) ==")
print(fs.get_online_features("item_features", "P19370"))
print("\n== Item neighbours (co-purchase) ==")
co = fs.get_features("item_cooccurrence")
print(co.sort_values(["co_count", "cosine"], ascending=False).head(5)[["product_id", "neighbour_id", "co_count", "cosine", "rank"]])
