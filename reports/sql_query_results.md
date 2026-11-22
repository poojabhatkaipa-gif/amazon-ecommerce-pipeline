# SQL query results (recomart.db)


## Q1. Revenue and return rate by category (gross vs net)

```sql
SELECT category, transactions, ROUND(gross_value) AS gross_value, ROUND(net_value) AS net_value,
       ROUND(return_rate_pct, 2) AS return_rate_pct, ROUND(avg_rating, 2) AS avg_rating
FROM category_metrics ORDER BY gross_value DESC;
```

| category    |   transactions |   gross_value |   net_value |   return_rate_pct |   avg_rating |
|:------------|---------------:|--------------:|------------:|------------------:|-------------:|
| Electronics |         200038 |   6.57905e+09 | 5.84501e+09 |             11.27 |         4.21 |
| Home        |         200922 |   1.52998e+09 | 1.35328e+09 |             11.58 |         3.9  |
| Sports      |         199889 |   1.12485e+09 | 9.9434e+08  |             11.65 |         3.88 |
| Beauty      |         199327 |   3.75952e+08 | 3.31886e+08 |             11.76 |         3.82 |
| Clothing    |         199824 |   3.2905e+08  | 2.90509e+08 |             11.74 |         3.83 |


## Q2. Monthly trend: transactions, active users and net value

```sql
SELECT purchase_year_month, transactions, active_users, ROUND(net_value) AS net_value
FROM monthly_metrics ORDER BY purchase_year_month;
```

| purchase_year_month   |   transactions |   active_users |   net_value |
|:----------------------|---------------:|---------------:|------------:|
| 2024-03               |           1400 |           1399 | 1.2419e+07  |
| 2024-04               |          41103 |          40165 | 3.63206e+08 |
| 2024-05               |          42342 |          41395 | 3.72859e+08 |
| 2024-06               |          41166 |          40200 | 3.59097e+08 |
| 2024-07               |          42267 |          41284 | 3.74569e+08 |
| 2024-08               |          42414 |          41406 | 3.72558e+08 |
| 2024-09               |          40873 |          39981 | 3.58526e+08 |
| 2024-10               |          42492 |          41493 | 3.76347e+08 |
| 2024-11               |          41073 |          40152 | 3.64751e+08 |
| 2024-12               |          42273 |          41285 | 3.73334e+08 |
| 2025-01               |          42842 |          41847 | 3.77626e+08 |
| 2025-02               |          38035 |          37269 | 3.35036e+08 |
| 2025-03               |          42641 |          41696 | 3.73063e+08 |
| 2025-04               |          40728 |          39783 | 3.57863e+08 |
| 2025-05               |          42771 |          41760 | 3.73042e+08 |
| 2025-06               |          41012 |          40112 | 3.57065e+08 |
| 2025-07               |          42368 |          41379 | 3.74168e+08 |
| 2025-08               |          42601 |          41589 | 3.79616e+08 |
| 2025-09               |          40440 |          39578 | 3.52725e+08 |
| 2025-10               |          42296 |          41325 | 3.76098e+08 |
| 2025-11               |          41014 |          40089 | 3.60237e+08 |
| 2025-12               |          42579 |          41620 | 3.80446e+08 |
| 2026-01               |          42326 |          41340 | 3.75178e+08 |
| 2026-02               |          38329 |          37571 | 3.38572e+08 |
| 2026-03               |          42615 |          41603 | 3.76617e+08 |


## Q3. Customer segmentation by number of purchase lines

```sql
SELECT CASE WHEN transaction_count >= 4 THEN '4+' ELSE CAST(transaction_count AS TEXT) END AS purchases,
       COUNT(*) AS users, ROUND(AVG(avg_final_price)) AS avg_spend, ROUND(AVG(return_rate_pct), 2) AS avg_return_rate_pct
FROM user_features GROUP BY 1 ORDER BY 1;
```

| purchases   |   users |   avg_spend |   avg_return_rate_pct |
|:------------|--------:|------------:|----------------------:|
| 1           |  329745 |        9907 |                 11.6  |
| 2           |  182263 |        9958 |                 11.55 |
| 3           |   67717 |        9984 |                 11.66 |
| 4+          |   24090 |        9884 |                 11.64 |


## Q4. Top 10 most popular items with their dominant category / brand

```sql
SELECT product_id, purchase_count, distinct_buyers, ROUND(avg_rating, 2) AS avg_rating, top_category, top_brand
FROM item_features ORDER BY purchase_count DESC, product_id LIMIT 10;
```

| product_id   |   purchase_count |   distinct_buyers |   avg_rating | top_category   | top_brand   |
|:-------------|-----------------:|------------------:|-------------:|:---------------|:------------|
| P96731       |               28 |                28 |         4.04 | Electronics    | HP          |
| P11002       |               27 |                27 |         3.65 | Clothing       | LG          |
| P13897       |               27 |                27 |         3.89 | Beauty         | Adidas      |
| P43630       |               27 |                27 |         3.91 | Beauty         | Samsung     |
| P50882       |               27 |                27 |         3.96 | Electronics    | Adidas      |
| P12723       |               26 |                26 |         3.73 | Beauty         | HP          |
| P26688       |               26 |                26 |         3.93 | Sports         | Puma        |
| P51710       |               26 |                26 |         3.96 | Home           | H&M         |
| P62266       |               26 |                26 |         3.93 | Sports         | H&M         |
| P75227       |               26 |                26 |         3.9  | Home           | H&M         |


## Q5. Pricing vs rating pattern: average rating by price quartile of the transaction

```sql
WITH q AS (SELECT rating, final_price, NTILE(4) OVER (ORDER BY final_price) AS price_quartile FROM fact_interaction)
SELECT price_quartile, COUNT(*) AS transactions, ROUND(MIN(final_price)) AS min_price, ROUND(MAX(final_price)) AS max_price,
       ROUND(AVG(rating), 3) AS avg_rating FROM q GROUP BY price_quartile ORDER BY price_quartile;
```

|   price_quartile |   transactions |   min_price |   max_price |   avg_rating |
|-----------------:|---------------:|------------:|------------:|-------------:|
|                1 |         250000 |         101 |        1706 |        3.815 |
|                2 |         250000 |        1706 |        3791 |        3.838 |
|                3 |         250000 |        3791 |       10384 |        3.894 |
|                4 |         250000 |       10384 |       75831 |        4.162 |


## Q6. Cross-sell candidates: strongest co-purchased pairs (cosine over buyer sets)

```sql
SELECT c.product_id, c.neighbour_id, c.co_count, ROUND(c.cosine, 3) AS cosine,
       a.top_category AS cat_a, b.top_category AS cat_b
FROM item_cooccurrence c JOIN item_features a ON a.product_id = c.product_id JOIN item_features b ON b.product_id = c.neighbour_id
WHERE c.rank = 1 ORDER BY c.co_count DESC, c.cosine DESC LIMIT 10;
```

| product_id   | neighbour_id   |   co_count |   cosine | cat_a    | cat_b    |
|:-------------|:---------------|-----------:|---------:|:---------|:---------|
| P18887       | P67423         |          2 |    0.224 | Beauty   | Sports   |
| P67423       | P18887         |          2 |    0.224 | Sports   | Beauty   |
| P73434       | P79678         |          2 |    0.201 | Clothing | Home     |
| P79678       | P73434         |          2 |    0.201 | Home     | Clothing |
| P67190       | P92550         |          2 |    0.192 | Beauty   | Clothing |
| P92550       | P67190         |          2 |    0.192 | Clothing | Beauty   |
| P47339       | P90059         |          2 |    0.191 | Clothing | Sports   |
| P90059       | P47339         |          2 |    0.191 | Sports   | Clothing |
| P45620       | P64386         |          2 |    0.185 | Beauty   | Beauty   |
| P64386       | P45620         |          2 |    0.185 | Beauty   | Beauty   |


## Q7. Users who bought in more than one category (cross-selling base)

```sql
SELECT distinct_categories, COUNT(*) AS users FROM user_features GROUP BY distinct_categories ORDER BY distinct_categories;
```

|   distinct_categories |   users |
|----------------------:|--------:|
|                     1 |  369183 |
|                     2 |  182950 |
|                     3 |   45678 |
|                     4 |    5721 |
|                     5 |     283 |
