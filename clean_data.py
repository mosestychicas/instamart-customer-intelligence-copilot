
import pandas as pd
from pathlib import Path

# ============================================================
# Instamart Capstone - Part A: Data Cleaning
# ============================================================

RAW = Path("data/raw")
CLEAN = Path("data/clean")
CLEAN.mkdir(parents=True, exist_ok=True)

# ============================================================
# 1. LOAD RAW DATA
# ============================================================

customers = pd.read_csv(RAW / "customers.csv", keep_default_na=False)
products = pd.read_csv(RAW / "products.csv", keep_default_na=False)
orders = pd.read_csv(RAW / "orders.csv", keep_default_na=False)

print("=" * 65)
print("INSTAMART DATA CLEANING")
print("=" * 65)

print("\nRaw row counts:")
print("Customers:", len(customers))
print("Products:", len(products))
print("Orders:", len(orders))

# ============================================================
# 2. CLEAN CUSTOMERS
# ============================================================

# Standardize city names: remove spaces and use title case.
customers["city"] = (
    customers["city"]
    .astype(str)
    .str.strip()
    .str.title()
)

# Standardize gender values.
# Blank or unrecognized values become "unknown".
customers["gender"] = (
    customers["gender"]
    .astype(str)
    .str.strip()
    .str.lower()
    .map({
        "m": "male",
        "male": "male",
        "f": "female",
        "female": "female",
    })
    .fillna("unknown")
)

# Convert age to numeric.
customers["age"] = pd.to_numeric(
    customers["age"],
    errors="coerce"
)

# Treat ages outside 10-100 as missing.
customers.loc[
    ~customers["age"].between(10, 100),
    "age"
] = float("nan")

# Fill missing ages with the median age of the customer's city.
city_median_age = customers.groupby("city")["age"].transform("median")
customers["age"] = customers["age"].fillna(city_median_age)

# Convert signup_date to datetime.
customers["signup_date"] = pd.to_datetime(
    customers["signup_date"],
    errors="coerce"
)

# ============================================================
# 3. CLEAN PRODUCTS
# ============================================================

for column in ["product_id", "product_name", "category"]:
    products[column] = products[column].astype(str).str.strip()

products["price_inr"] = pd.to_numeric(
    products["price_inr"],
    errors="coerce"
)

# ============================================================
# 4. CLEAN ORDERS
# ============================================================

# Step 1: Convert order timestamps to datetime.
orders["order_ts"] = pd.to_datetime(
    orders["order_ts"],
    errors="coerce"
)

# Step 2: Remove exact duplicate rows.
orders = orders.drop_duplicates().copy()

# Step 3: Keep the latest record for each order_id.
orders = (
    orders.sort_values("order_ts", kind="stable")
    .drop_duplicates(subset="order_id", keep="last")
    .copy()
)

# Step 4: Convert numeric fields.
orders["order_amount"] = pd.to_numeric(
    orders["order_amount"],
    errors="coerce"
)

orders["quantity"] = pd.to_numeric(
    orders["quantity"],
    errors="coerce"
)

orders["discount_pct"] = pd.to_numeric(
    orders["discount_pct"],
    errors="coerce"
)

orders["delivery_minutes"] = pd.to_numeric(
    orders["delivery_minutes"],
    errors="coerce"
)

# Fill missing order amounts using the product catalogue price.
price_lookup = products.set_index("product_id")["price_inr"]
orders["price_inr"] = orders["product_id"].map(price_lookup)

calculated_amount = (
    orders["price_inr"]
    * orders["quantity"]
    * (1 - orders["discount_pct"] / 100)
).round(2)

orders["order_amount"] = orders["order_amount"].fillna(
    calculated_amount
)

# Step 5: Set negative delivery times to missing.
orders.loc[
    orders["delivery_minutes"] < 0,
    "delivery_minutes"
] = float("nan")

# Step 6: Impute missing delivery times for completed orders only.
# Use the median delivery time for the product category,
# calculated from completed orders.
products_by_id = products.set_index("product_id")
orders["category"] = orders["product_id"].map(
    products_by_id["category"]
)

completed_mask = (
    orders["status"].astype(str).str.strip().str.lower()
    == "completed"
)

category_medians = (
    orders.loc[completed_mask]
    .groupby("category")["delivery_minutes"]
    .transform("median")
)

orders.loc[completed_mask, "delivery_minutes"] = (
    orders.loc[completed_mask, "delivery_minutes"]
    .fillna(category_medians)
)

# Cancelled orders were not delivered, so their delivery times
# must remain missing.
cancelled_mask = (
    orders["status"].astype(str).str.strip().str.lower()
    == "cancelled"
)

orders.loc[cancelled_mask, "delivery_minutes"] = float("nan")

# Remove temporary helper columns.
orders = orders.drop(columns=["price_inr", "category"])

# Store order_ts as YYYY-MM-DD HH:MM:SS text.
orders["order_ts"] = orders["order_ts"].dt.strftime(
    "%Y-%m-%d %H:%M:%S"
)

# ============================================================
# 5. SAVE CLEAN CSV FILES
# ============================================================

customers.to_csv(CLEAN / "customers.csv", index=False)
products.to_csv(CLEAN / "products.csv", index=False)
orders.to_csv(CLEAN / "orders.csv", index=False)

# ============================================================
# 6. REPORT CLEANING RESULTS
# ============================================================

print("\n" + "=" * 65)
print("CUSTOMER CLEANING RESULTS")
print("=" * 65)

print("Customer rows:", len(customers))
print("\nCity counts:")
print(customers["city"].value_counts().sort_index().to_string())

print("\nGender counts:")
print(customers["gender"].value_counts().sort_index().to_string())

print("\nMissing ages:", customers["age"].isna().sum())
print("Minimum age:", customers["age"].min())
print("Maximum age:", customers["age"].max())
print("Missing signup dates:", customers["signup_date"].isna().sum())

print("\n" + "=" * 65)
print("ORDER CLEANING RESULTS")
print("=" * 65)

print("Order rows:", len(orders))
print("\nOrder status counts:")
print(orders["status"].value_counts().sort_index().to_string())

completed_mask = (
    orders["status"].astype(str).str.strip().str.lower()
    == "completed"
)

cancelled_mask = (
    orders["status"].astype(str).str.strip().str.lower()
    == "cancelled"
)

pending_mask = (
    orders["status"].astype(str).str.strip().str.lower()
    == "pending"
)

print("\nMissing order amounts:", orders["order_amount"].isna().sum())

print(
    "Missing delivery times among completed orders:",
    orders.loc[completed_mask, "delivery_minutes"].isna().sum()
)

print(
    "Missing delivery times among cancelled orders:",
    orders.loc[cancelled_mask, "delivery_minutes"].isna().sum()
)

completed_revenue = orders.loc[
    completed_mask, "order_amount"
].sum()

print("Completed-order revenue: INR", f"{completed_revenue:,.2f}")

# ============================================================
# 7. CHECK REFERENCE CHECKPOINTS
# ============================================================

city_counts = customers["city"].value_counts().to_dict()
gender_counts = customers["gender"].value_counts().to_dict()

checks = {
    "1,200 customers": len(customers) == 1200,
    "40 products": len(products) == 40,
    "12,309 unique orders": len(orders) == 12309,
    "No duplicate order IDs": not orders["order_id"].duplicated().any(),
    "Exactly six cities": customers["city"].nunique() == 6,
    "Female count is 610": gender_counts.get("female", 0) == 610,
    "Male count is 560": gender_counts.get("male", 0) == 560,
    "Unknown gender count is 30": gender_counts.get("unknown", 0) == 30,
    "No missing customer ages": customers["age"].isna().sum() == 0,
    "All ages between 10 and 100": customers["age"].between(10, 100).all(),
    "No pending orders": pending_mask.sum() == 0,
    "11,110 completed orders": completed_mask.sum() == 11110,
    "1,199 cancelled orders": cancelled_mask.sum() == 1199,
    "No missing order amounts": orders["order_amount"].isna().sum() == 0,
    "Completed delivery times all filled":
        orders.loc[completed_mask, "delivery_minutes"].isna().sum() == 0,
    "Cancelled delivery times all missing":
        orders.loc[cancelled_mask, "delivery_minutes"].isna().sum() == 1199,
    "Completed revenue matches reference":
        round(completed_revenue, 2) == 2998668.80,
}

print("\n" + "=" * 65)
print("VALIDATION CHECKPOINTS")
print("=" * 65)

all_passed = True

for description, passed in checks.items():
    print(f"{'PASS' if passed else 'FAIL'}: {description}")
    if not passed:
        all_passed = False

if all_passed:
    print("\nALL CHECKPOINTS PASSED")
else:
    print("\nSome checkpoints failed. Review the results above.")

print("\nClean CSV files saved in:", CLEAN.resolve())
