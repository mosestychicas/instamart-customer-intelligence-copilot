
from pathlib import Path
import sqlite3
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "data" / "instamart.db"

print("=" * 70)
print("INSTAMART PANDAS CROSS-CHECK AND EXPLORATORY DATA ANALYSIS")
print("=" * 70)

with sqlite3.connect(DB_PATH) as conn:
    orders = pd.read_sql_query("SELECT * FROM orders", conn)
    customers = pd.read_sql_query("SELECT * FROM customers", conn)
    products = pd.read_sql_query("SELECT * FROM products", conn)

# Merge completed orders with customer city and product category.
completed = (
    orders.loc[orders["status"] == "completed"]
    .merge(
        customers[["customer_id", "city"]],
        on="customer_id",
        how="left",
        validate="many_to_one",
    )
    .merge(
        products[["product_id", "category"]],
        on="product_id",
        how="left",
        validate="many_to_one",
    )
)

for column in [
    "order_amount",
    "discount_pct",
    "quantity",
    "delivery_minutes",
]:
    completed[column] = pd.to_numeric(
        completed[column], errors="coerce"
    )

# ------------------------------------------------------------
# 1. Pivot-table cross-check of SQL Q2
# ------------------------------------------------------------
print("\n1. REVENUE PIVOT TABLE — CITY x CATEGORY")

pivot = pd.pivot_table(
    completed,
    index="city",
    columns="category",
    values="order_amount",
    aggfunc="sum",
    fill_value=0,
    margins=True,
    margins_name="Grand Total",
)

print(pivot.round(2).to_string())

sql_total = completed["order_amount"].sum()
pivot_total = pivot.loc["Grand Total", "Grand Total"]

print(f"\nSQL revenue total:   INR {sql_total:,.2f}")
print(f"Pivot grand total:   INR {pivot_total:,.2f}")
print(
    "Pivot matches SQL:",
    bool(abs(sql_total - pivot_total) < 0.01),
)

# ------------------------------------------------------------
# 2. City-level sum, mean and count
# ------------------------------------------------------------
print("\n2. COMPLETED ORDER AMOUNT BY CITY")

city_stats = (
    completed.groupby("city")
    .agg(
        revenue_sum=("order_amount", "sum"),
        average_order_amount=("order_amount", "mean"),
        order_count=("order_amount", "count"),
    )
    .sort_index()
)

print(city_stats.round(2).to_string())

# ------------------------------------------------------------
# 3. Distribution statistics and IQR outliers
# ------------------------------------------------------------
print("\n3. COMPLETED ORDER AMOUNT DISTRIBUTION")

amount = completed["order_amount"].dropna()

mean_value = amount.mean()
median_value = amount.median()
skew_value = amount.skew()
q1 = amount.quantile(0.25)
q2 = amount.quantile(0.50)
q3 = amount.quantile(0.75)
iqr = q3 - q1
lower_fence = q1 - 1.5 * iqr
upper_fence = q3 + 1.5 * iqr
below_count = int((amount < lower_fence).sum())
above_count = int((amount > upper_fence).sum())

distribution = pd.Series({
    "Count": amount.count(),
    "Mean": mean_value,
    "Median": median_value,
    "Skewness": skew_value,
    "Q1 (25%)": q1,
    "Q2 (50%)": q2,
    "Q3 (75%)": q3,
    "IQR": iqr,
    "Lower fence": lower_fence,
    "Upper fence": upper_fence,
    "Outliers below lower fence": below_count,
    "Outliers above upper fence": above_count,
})

print(distribution.to_string())

# ------------------------------------------------------------
# 4. Correlation matrix
# ------------------------------------------------------------
print("\n4. CORRELATION MATRIX")

numeric_columns = [
    "order_amount",
    "discount_pct",
    "quantity",
    "delivery_minutes",
]

correlation = completed[numeric_columns].corr()
print(correlation.round(4).to_string())

pairs = []
for i, col_a in enumerate(numeric_columns):
    for col_b in numeric_columns[i + 1:]:
        value = correlation.loc[col_a, col_b]
        if pd.notna(value):
            pairs.append((col_a, col_b, float(value)))

strongest_pair = max(pairs, key=lambda item: abs(item[2]))

print(
    "\nLargest absolute correlation: "
    f"{strongest_pair[0]} vs {strongest_pair[1]} "
    f"= {strongest_pair[2]:.4f}"
)

print("\nEDA ANALYSIS COMPLETE")
