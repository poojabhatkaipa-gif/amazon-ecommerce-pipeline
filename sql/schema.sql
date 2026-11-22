-- Person 3: analytical warehouse schema (SQLite). Loaded by src/04_analytics.py
DROP TABLE IF EXISTS fact_interaction;
DROP TABLE IF EXISTS user_features;
DROP TABLE IF EXISTS item_features;
DROP TABLE IF EXISTS item_cooccurrence;
DROP TABLE IF EXISTS monthly_metrics;
DROP TABLE IF EXISTS category_metrics;

-- One row per purchase line (transaction grain, from transformed_transactions.parquet)
CREATE TABLE fact_interaction (
    source_row      INTEGER PRIMARY KEY,
    user_id         TEXT NOT NULL,
    product_id      TEXT NOT NULL,
    category        TEXT NOT NULL,
    subcategory     TEXT,
    brand           TEXT,
    price           REAL NOT NULL,
    final_price     REAL NOT NULL,
    rating          REAL NOT NULL CHECK (rating BETWEEN 1 AND 5),
    purchase_date   TEXT NOT NULL,
    purchase_year_month TEXT NOT NULL,
    is_returned     INTEGER NOT NULL CHECK (is_returned IN (0,1))
);
CREATE INDEX ix_fact_user    ON fact_interaction(user_id);
CREATE INDEX ix_fact_product ON fact_interaction(product_id);
CREATE INDEX ix_fact_month   ON fact_interaction(purchase_year_month);

CREATE TABLE user_features (
    user_id TEXT PRIMARY KEY, transaction_count INTEGER, distinct_products INTEGER,
    active_days INTEGER, recency_days INTEGER, avg_rating REAL, avg_final_price REAL,
    return_rate_pct REAL, distinct_categories INTEGER, top_category TEXT
);
CREATE TABLE item_features (
    product_id TEXT PRIMARY KEY, purchase_count INTEGER, distinct_buyers INTEGER,
    avg_rating REAL, avg_final_price REAL, return_rate_pct REAL,
    top_category TEXT, top_subcategory TEXT, top_brand TEXT
);
CREATE TABLE item_cooccurrence (
    product_id TEXT NOT NULL, neighbour_id TEXT NOT NULL, co_count INTEGER NOT NULL,
    cosine REAL NOT NULL, rank INTEGER NOT NULL, PRIMARY KEY (product_id, neighbour_id)
);
CREATE TABLE monthly_metrics (
    purchase_year_month TEXT PRIMARY KEY, transactions INTEGER, active_users INTEGER,
    gross_value REAL, net_value REAL, return_rate_pct REAL, avg_rating REAL
);
CREATE TABLE category_metrics (
    category TEXT PRIMARY KEY, transactions INTEGER, distinct_products INTEGER, gross_value REAL,
    net_value REAL, avg_final_price REAL, avg_rating REAL, return_rate_pct REAL
);
