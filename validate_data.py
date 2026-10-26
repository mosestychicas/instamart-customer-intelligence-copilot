
from pathlib import Path
import sqlite3
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
CLEAN_DIR = BASE_DIR / "data" / "clean"
DB_PATH = BASE_DIR / "data" / "instamart.db"

def check(condition, description):
    assert condition, f"FAIL: {description}"
    print(f"PASS: {description}")

print("=" * 65)
print("INSTAMART DATA VALIDATION")
print("=" * 65)

# Load cleaned CSV files.
customers = pd.read_csv(CLEAN_DIR / "customers.csv")
products = pd.read_csv(CLEAN_DIR / "products.csv")
orders = pd.read_csv(CLEAN_DIR / "orders.csv")

# Check cleaned customer data.
check(len(customers) == 1200, "1,200 customers")
check(len(products) == 40, "40 products")
check(len(orders) == 12309, "12,309 orders")
check(customers["customer_id"].is_unique, "Unique customer IDs")
check(orders["order_id"].is_unique, "Unique order IDs")

check(
    customers["city"].nunique() == 6,
    "Exactly six standardised cities",
)

gender_counts = customers["gender"].value_counts().to_dict()
check(gender_counts.get("female", 0) == 610, "610 female customers")
check(gender_counts.get("male", 0) == 560, "560 male customers")
check(gender_counts.get("unknown", 0) == 30, "30 unknown genders")

ages = pd.to_numeric(customers["age"], errors="coerce")
check(ages.notna().all(), "No missing customer ages")
check(
    ages.between(10, 100).all(),
    "All customer ages between 10 and 100",
)

# Check cleaned orders.
check(
    not orders["status"].eq("pending").any(),
    "No pending orders",
)
check(
    int(orders["status"].eq("completed").sum()) == 11110,
    "11,110 completed orders",
)
check(
    int(orders["status"].eq("cancelled").sum()) == 1199,
    "1,199 cancelled orders",
)

amounts = pd.to_numeric(orders["order_amount"], errors="coerce")
delivery = pd.to_numeric(
    orders["delivery_minutes"], errors="coerce"
)

check(amounts.notna().all(), "No missing order amounts")
check(
    delivery[orders["status"].eq("completed")].notna().all(),
    "Completed orders have delivery times",
)
check(
    delivery[orders["status"].eq("cancelled")].isna().all(),
    "Cancelled orders have missing delivery times",
)

completed_revenue = amounts[orders["status"].eq("completed")].sum()
check(
    abs(completed_revenue - 2998668.80) < 0.01,
    "Completed revenue is INR 2,998,668.80",
)

# Validate SQLite database tables and row counts.
with sqlite3.connect(DB_PATH) as conn:
    for table, expected in {
        "customers": 1200,
        "products": 40,
        "orders": 12309,
    }.items():
        actual = pd.read_sql_query(
            f"SELECT COUNT(*) AS n FROM {table}",
            conn,
        ).iloc[0]["n"]

        check(
            actual == expected,
            f"SQLite table {table} has {expected:,} rows",
        )

print("\nALL CHECKPOINTS PASSED")
