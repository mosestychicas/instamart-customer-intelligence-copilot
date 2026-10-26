
import os
import sqlite3
import pandas as pd
import numpy as np

DB_PATH = "data/instamart.db"
OUTPUT_DIR = "outputs"
CUTOFF = pd.Timestamp("2025-09-01")
LABEL_END = pd.Timestamp("2025-10-01")

os.makedirs(OUTPUT_DIR, exist_ok=True)

with sqlite3.connect(DB_PATH) as conn:
    orders = pd.read_sql_query("SELECT * FROM orders", conn)
    customers = pd.read_sql_query("SELECT * FROM customers", conn)
    products = pd.read_sql_query("SELECT * FROM products", conn)

orders["order_ts"] = pd.to_datetime(orders["order_ts"], errors="coerce")
customers["signup_date"] = pd.to_datetime(
    customers["signup_date"], errors="coerce"
)

# Feature history includes only completed/cancelled orders before cutoff.
history = orders[orders["order_ts"] < CUTOFF].copy()
completed = history[history["status"] == "completed"].copy()
cancelled = history[history["status"] == "cancelled"].copy()

if completed.empty:
    raise ValueError("No completed orders exist before the cutoff date.")

# Cohort: customers with at least one completed order before cutoff.
cohort_ids = completed["customer_id"].unique()
cohort = customers[customers["customer_id"].isin(cohort_ids)].copy()

# Build customer-level features using pre-cutoff history only.
last_order = completed.groupby("customer_id")["order_ts"].max()
frequency = completed.groupby("customer_id")["order_id"].nunique()
monetary = completed.groupby("customer_id")["order_amount"].sum()
avg_discount = completed.groupby("customer_id")["discount_pct"].mean()
avg_delivery = completed.groupby("customer_id")["delivery_minutes"].mean()

fav_category = (
    completed.merge(products[["product_id", "category"]], on="product_id")
    .groupby(["customer_id", "category"])
    .size()
    .reset_index(name="category_orders")
    .sort_values(
        ["customer_id", "category_orders", "category"],
        ascending=[True, False, True],
    )
    .drop_duplicates("customer_id")
    .set_index("customer_id")["category"]
)

# Cancellation rate uses all final-status orders before cutoff.
all_status_counts = history.groupby("customer_id")["order_id"].nunique()
cancel_counts = cancelled.groupby("customer_id")["order_id"].nunique()
cancellation_rate = (
    cancel_counts.reindex(all_status_counts.index, fill_value=0)
    / all_status_counts
)

features = cohort.set_index("customer_id").copy()
features["recency"] = (
    CUTOFF - last_order.reindex(features.index)
).dt.days
features["frequency"] = frequency.reindex(features.index)
features["monetary"] = monetary.reindex(features.index)
features["avg_discount_pct"] = avg_discount.reindex(features.index)
features["cancellation_rate"] = cancellation_rate.reindex(
    features.index, fill_value=0
)
features["avg_delivery_minutes"] = avg_delivery.reindex(features.index)
features["favourite_category"] = fav_category.reindex(features.index)

features["tenure_days"] = (
    CUTOFF - features["signup_date"]
).dt.days

# Label from completed orders during September 2025.
label_window = orders[
    (orders["order_ts"] >= CUTOFF)
    & (orders["order_ts"] < LABEL_END)
    & (orders["status"] == "completed")
]
active_in_window = set(label_window["customer_id"])
features["lapsed"] = (
    ~features.index.isin(active_in_window)
).astype(int)

# Keep useful customer attributes as categorical/model features.
features = features.reset_index()
features = features[
    [
        "customer_id",
        "recency",
        "frequency",
        "monetary",
        "avg_discount_pct",
        "cancellation_rate",
        "avg_delivery_minutes",
        "favourite_category",
        "city",
        "age",
        "gender",
        "tenure_days",
        "lapsed",
    ]
]

# Fail early if feature construction produced missing values.
if features.isna().any().any():
    print("Missing values by column:")
    print(features.isna().sum()[features.isna().sum() > 0])
    raise ValueError(
        "Missing values found. Resolve them explicitly before modelling."
    )

features["lapsed"] = features["lapsed"].astype(int)
output_path = os.path.join(OUTPUT_DIR, "lapse_dataset.csv")
features.to_csv(output_path, index=False)

print("=" * 70)
print("LAPSE DATASET CHECKPOINTS")
print("=" * 70)
print("Cohort rows:", len(features))
print("Lapsed customers:", int(features["lapsed"].sum()))
print("Lapse rate (%):", round(features["lapsed"].mean() * 100, 2))
print("Feature columns:", list(features.columns))
print("Missing values:", int(features.isna().sum().sum()))
print("Saved:", os.path.abspath(output_path))

print("\nLAPSE DATASET BUILD COMPLETE")
