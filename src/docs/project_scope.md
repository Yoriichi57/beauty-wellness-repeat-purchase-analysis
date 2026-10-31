# Project scope and hypotheses

## Business problem
Beauty and wellness retailers want to encourage customers to return after an initial purchase. Without understanding observed repeat-purchase behaviour, marketing teams may spend broadly rather than tailoring retention activity to customer groups and product categories.

## Main research question
What customer and order patterns are associated with a second purchase in the beauty and wellness product categories within the available observation period?

## Supporting questions
1. What share of identified customers placed at least two distinct orders containing the selected product categories?
2. How does the distribution of order frequency vary across customers?
3. Are purchase frequency and spend different across product subcategories?
4. Do review scores and delivery experience show descriptive differences between single-purchase and repeat-purchase customers?
5. Which customer-level features could be used in a future time-based prediction model?

## Hypotheses to investigate
- H1: The observed repeat-purchase rate differs between the health_beauty and perfumery categories.
- H2: Customers with a higher first-order value may have a different observed repeat-purchase rate from customers with a lower first-order value.
- H3: Customers with better review scores may show a different repeat-purchase rate from customers with lower review scores.
- H4: Longer delivery delays may be associated with a lower repeat-purchase rate.
- H5: Repeat-purchase behaviour varies over the observation period.

These are hypotheses, not findings. They must be tested with the downloaded data.

## Target definition
For the initial descriptive analysis:
`repeat_purchase = 1` when a customer has at least two distinct delivered orders containing `health_beauty` or `perfumery` products during the dataset period; otherwise `0`.

This is an observed-period definition, not proof of churn or permanent customer loyalty.

## Candidate variables
- Customer identifier: `customer_unique_id` (anonymized)
- Order identifiers and timestamps: `order_id`, `order_purchase_timestamp`, `order_status`
- Product/category: `product_id`, `product_category_name_english`
- Item/order amount: `price`, `freight_value`, `order_item_id`
- Customer geography: `customer_state`, `customer_city` (aggregate responsibly)
- Experience fields, if joined: `review_score`, delivered and estimated delivery dates

## Future modelling design
A predictive model requires a clear time boundary. For example, use customer activity up to a chosen cutoff date as the observation window and define a second purchase in the following fixed horizon as the target. Exclude customers without enough follow-up time, split train/test chronologically, and compare a simple baseline against logistic regression or a tree-based classifier. All preprocessing must be fitted on training data only.
