from pathlib import Path
import sys

import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

REQUIRED = [
    "olist_customers_dataset.csv",
    "olist_orders_dataset.csv",
    "olist_order_items_dataset.csv",
    "olist_products_dataset.csv",
    "product_category_name_translation.csv",
]

def require_files():
    missing = [name for name in REQUIRED if not (RAW / name).exists()]
    if missing:
        print("Missing dataset files in data/raw/:")
        for name in missing:
            print(f"  - {name}")
        print("\nDownload the Olist dataset (see README.md), extract it, and copy the CSVs into data/raw/.")
        sys.exit(1)

def main():
    require_files()

    customers = pd.read_csv(RAW / "olist_customers_dataset.csv")
    orders = pd.read_csv(RAW / "olist_orders_dataset.csv")
    items = pd.read_csv(RAW / "olist_order_items_dataset.csv")
    products = pd.read_csv(RAW / "olist_products_dataset.csv")
    translations = pd.read_csv(RAW / "product_category_name_translation.csv")

    # Basic data-quality checks.
    for name, frame in {
        "customers": customers, "orders": orders, "items": items,
        "products": products, "translations": translations
    }.items():
        print(f"{name}: {frame.shape[0]:,} rows x {frame.shape[1]} columns")

    orders["order_purchase_timestamp"] = pd.to_datetime(
        orders["order_purchase_timestamp"], errors="coerce"
    )

    # Keep completed orders only for this purchase-behaviour analysis.
    delivered = orders.loc[
        orders["order_status"].eq("delivered"),
        ["order_id", "customer_id", "order_purchase_timestamp"]
    ].dropna(subset=["order_purchase_timestamp"])

    # Translate category names; filter the beauty-related category proxy.
    product_data = products.merge(
        translations, on="product_category_name", how="left"
    )
    product_data["category_english"] = product_data[
        "product_category_name_english"
    ].fillna(product_data["product_category_name"])

    beauty_categories = {"health_beauty", "perfumery"}
    beauty_products = product_data.loc[
        product_data["category_english"].isin(beauty_categories),
        ["product_id", "category_english"]
    ]

    # Join order lines to delivered orders, selected products and stable customer IDs.
    beauty_lines = (
        items.merge(delivered, on="order_id", how="inner")
        .merge(beauty_products, on="product_id", how="inner")
        .merge(
            customers[["customer_id", "customer_unique_id"]],
            on="customer_id", how="left"
        )
    )
    beauty_lines = beauty_lines.dropna(subset=["customer_unique_id"])
    if beauty_lines.empty:
        raise ValueError(
            "No matching beauty-category delivered order lines found. "
            "Check that all CSVs are from the same Olist dataset version."
        )

    # One row per customer-order pair prevents multiple items in one order
    # from being counted as multiple purchases.
    customer_orders = (
        beauty_lines.groupby(
            ["customer_unique_id", "order_id", "order_purchase_timestamp"],
            as_index=False
        )
        .agg(
            order_value=("price", "sum"),
            item_count=("order_item_id", "count"),
            categories=("category_english", lambda x: ", ".join(sorted(set(x))))
        )
    )

    customer_summary = (
        customer_orders.groupby("customer_unique_id", as_index=False)
        .agg(
            beauty_order_count=("order_id", "nunique"),
            first_purchase=("order_purchase_timestamp", "min"),
            last_purchase=("order_purchase_timestamp", "max"),
            total_product_spend=("order_value", "sum"),
            total_items=("item_count", "sum"),
            average_order_value=("order_value", "mean"),
        )
    )
    customer_summary["repeat_purchase"] = (
        customer_summary["beauty_order_count"] >= 2
    ).astype(int)
    customer_summary["days_between_first_and_last"] = (
        customer_summary["last_purchase"] - customer_summary["first_purchase"]
    ).dt.days

    # Save customer-level analysis table and a compact KPI summary.
    customer_summary.to_csv(OUT / "customer_repeat_purchase_summary.csv", index=False)

    total_customers = len(customer_summary)
    repeat_customers = int(customer_summary["repeat_purchase"].sum())
    repeat_rate = repeat_customers / total_customers if total_customers else 0.0
    kpis = pd.DataFrame([{
        "beauty_customers_observed": total_customers,
        "repeat_customers_observed": repeat_customers,
        "single_purchase_customers_observed": total_customers - repeat_customers,
        "observed_repeat_purchase_rate": repeat_rate,
        "category_definition": "health_beauty OR perfumery",
        "order_status_included": "delivered",
    }])
    kpis.to_csv(OUT / "repeat_purchase_kpis.csv", index=False)

    # Monthly count of beauty-category orders.
    monthly = (
        customer_orders.set_index("order_purchase_timestamp")
        .resample("MS")["order_id"].nunique()
        .rename("beauty_orders")
        .reset_index()
    )
    monthly.to_csv(OUT / "monthly_beauty_orders.csv", index=False)

    # Chart 1: observed repeat vs single-purchase customer counts.
    counts = customer_summary["repeat_purchase"].value_counts().reindex([0, 1], fill_value=0)
    plt.figure(figsize=(7, 4))
    plt.bar(["One observed order", "2+ observed orders"], counts.values)
    plt.title("Observed beauty-category purchase frequency")
    plt.ylabel("Unique customers")
    plt.tight_layout()
    plt.savefig(OUT / "repeat_purchase_counts.png", dpi=160)
    plt.close()

    # Chart 2: monthly order volume.
    plt.figure(figsize=(9, 4))
    plt.plot(monthly["order_purchase_timestamp"], monthly["beauty_orders"], marker="o")
    plt.title("Monthly delivered orders containing beauty-category products")
    plt.xlabel("Purchase month")
    plt.ylabel("Number of orders")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(OUT / "monthly_beauty_orders.png", dpi=160)
    plt.close()

    print("\nAnalysis complete.")
    print(f"Customers observed: {total_customers:,}")
    print(f"Repeat customers observed: {repeat_customers:,}")
    print(f"Observed repeat-purchase rate: {repeat_rate:.1%}")
    print(f"Files saved in: {OUT}")

    print(
        "\nInterpretation reminder: a customer with one order in this dataset "
        "is not necessarily churned. The label only describes observed orders "
        "within the dataset period."
    )

if __name__ == "__main__":
    main()
