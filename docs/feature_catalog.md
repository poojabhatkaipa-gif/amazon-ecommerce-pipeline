# Feature catalog (auto-generated from data/feature_store/registry.json)


## item_cooccurrence (v1, source: `data/processed/transformed_transactions.parquet`)

| Feature | dtype | Description | Transformation |
|---|---|---|---|
| product_id | string | Anchor product | `item i` |
| neighbour_id | string | Co-purchased product | `item-item pair from users with >=2 products` |
| co_count | int64 | Users who bought both | `(X^T X)[i,j] on binary user x item matrix` |
| cosine | float32 | Cosine similarity of buyer sets | `co_count / sqrt(buyers_i * buyers_j)` |
| rank | int64 | Neighbour rank within the item (1 = most similar) | `rank by cosine desc, co_count desc` |

## item_features (v1, source: `data/processed/transformed_transactions.parquet`)

| Feature | dtype | Description | Transformation |
|---|---|---|---|
| purchase_count | int64 | Purchase lines (popularity) | `count(rows) group by product_id` |
| distinct_buyers | int64 | Distinct buyers | `nunique(user_id)` |
| avg_rating | float64 | Mean rating (average rating per item) | `mean(rating)` |
| avg_final_price | float64 | Mean final price | `mean(final_price)` |
| return_rate_pct | float64 | Share of returned lines, 0-100 | `100*mean(is_returned)` |
| top_category | string | Modal category of the product | `mode(category)` |
| top_subcategory | string | Modal subcategory | `mode(subcategory)` |
| top_brand | string | Modal brand | `mode(brand)` |

## user_features (v1, source: `data/processed/transformed_transactions.parquet`)

| Feature | dtype | Description | Transformation |
|---|---|---|---|
| transaction_count | int64 | Purchase lines per user (user activity frequency) | `count(rows) group by user_id` |
| distinct_products | int64 | Distinct products bought | `nunique(product_id)` |
| active_days | int64 | Distinct purchase dates | `nunique(purchase_date)` |
| recency_days | int64 | Days since last purchase vs dataset max date | `max(purchase_date over data) - max(purchase_date of user)` |
| avg_rating | float64 | Mean rating on the user's purchases | `mean(rating)` |
| avg_final_price | float64 | Mean spend per line | `mean(final_price)` |
| return_rate_pct | float64 | Share of returned lines, 0-100 | `100*mean(is_returned)` |
| distinct_categories | int64 | Category breadth | `nunique(category)` |
| top_category | string | Most purchased category (ties alphabetical) | `mode(category)` |
