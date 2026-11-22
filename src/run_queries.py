"""Run every query in sql/queries.sql against the warehouse and save results to reports/sql_query_results.md."""
import re
import sqlite3
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
db = ROOT / "data" / "warehouse" / ("recomart_sample.db" if "--sample" in sys.argv else "recomart.db")
text = (ROOT / "sql" / "queries.sql").read_text()
blocks = [b.strip() for b in re.split(r"\n\s*\n", text) if "SELECT" in b.upper() or "WITH" in b.upper()]
con = sqlite3.connect(db)
out = [f"# SQL query results ({db.name})\n"]
for b in blocks:
    title = next((l[3:].strip() for l in b.splitlines() if l.startswith("-- Q")), "query")
    sql = "\n".join(l for l in b.splitlines() if not l.startswith("--"))
    df = pd.read_sql_query(sql, con)
    out += [f"\n## {title}\n", "```sql\n" + sql + "\n```\n", df.to_markdown(index=False), ""]
    print(f"{title}: {len(df)} rows")
(ROOT / "reports" / "sql_query_results.md").write_text("\n".join(out))
