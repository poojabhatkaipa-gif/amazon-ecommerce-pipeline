-- Person 3: example analytical queries on data/warehouse/recomart.db
-- Run:  sqlite3 data/warehouse/recomart.db < sql/queries.sql      (or via src/run_queries.py)

-- Q1. Revenue and return rate by category (gross vs net)
SELECT category, transactions, ROUND(gross_value) AS gross_value, ROUND(net_value) AS net_value,
       ROUND(return_rate_pct, 2) AS return_rate_pct, ROUND(avg_rating, 2) AS avg_rating
FROM category_metrics ORDER BY gross_value DESC;

-- Q2. Monthly trend: transactions, active users and net value
SELECT purchase_year_month, transactions, active_users, ROUND(net_value) AS net_value
FROM monthly_metrics ORDER BY purchase_year_month;

-- Q3. Customer segmentation by number of purchase lines
SELECT CASE WHEN transaction_count >= 4 THEN '4+' ELSE CAST(transaction_count AS TEXT) END AS purchases,
       COUNT(*) AS users, ROUND(AVG(avg_final_price)) AS avg_spend, ROUND(AVG(return_rate_pct), 2) AS avg_return_rate_pct
FROM user_features GROUP BY 1 ORDER BY 1;

-- Q4. Top 10 most popular items with their dominant category / brand
SELECT product_id, purchase_count, distinct_buyers, ROUND(avg_rating, 2) AS avg_rating, top_category, top_brand
FROM item_features ORDER BY purchase_count DESC, product_id LIMIT 10;

-- Q5. Pricing vs rating pattern: average rating by price quartile of the transaction
WITH q AS (SELECT rating, final_price, NTILE(4) OVER (ORDER BY final_price) AS price_quartile FROM fact_interaction)
SELECT price_quartile, COUNT(*) AS transactions, ROUND(MIN(final_price)) AS min_price, ROUND(MAX(final_price)) AS max_price,
       ROUND(AVG(rating), 3) AS avg_rating FROM q GROUP BY price_quartile ORDER BY price_quartile;

-- Q6. Cross-sell candidates: strongest co-purchased pairs (cosine over buyer sets)
SELECT c.product_id, c.neighbour_id, c.co_count, ROUND(c.cosine, 3) AS cosine,
       a.top_category AS cat_a, b.top_category AS cat_b
FROM item_cooccurrence c JOIN item_features a ON a.product_id = c.product_id JOIN item_features b ON b.product_id = c.neighbour_id
WHERE c.rank = 1 ORDER BY c.co_count DESC, c.cosine DESC LIMIT 10;

-- Q7. Users who bought in more than one category (cross-selling base)
SELECT distinct_categories, COUNT(*) AS users FROM user_features GROUP BY distinct_categories ORDER BY distinct_categories;
