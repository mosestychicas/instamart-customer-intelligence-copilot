
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
CLEAN_DIR = BASE_DIR / "data" / "clean"
OUTPUT_DIR = BASE_DIR / "outputs"

SNAPSHOT = pd.Timestamp("2025-10-01 00:00:00")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 65)
print("INSTAMART RFM CUSTOMER ANALYSIS")
print("=" * 65)

# Load the cleaned data.
customers = pd.read_csv(CLEAN_DIR / "customers.csv")
orders = pd.read_csv(CLEAN_DIR / "orders.csv")

orders["order_ts"] = pd.to_datetime(orders["order_ts"])
orders["order_amount"] = pd.to_numeric(
    orders["order_amount"], errors="raise"
)

# Only completed orders contribute to customer behaviour.
completed = orders.loc[
    orders["status"].eq("completed")
].copy()

# Calculate RFM values for each customer with completed orders.
rfm = (
    completed.groupby("customer_id")
    .agg(
        last_order_ts=("order_ts", "max"),
        frequency=("order_id", "nunique"),
        monetary=("order_amount", "sum"),
    )
    .reset_index()
)

rfm["recency"] = (SNAPSHOT - rfm["last_order_ts"]).dt.days

# Keep the requested columns and a consistent order.
rfm = rfm[
    ["customer_id", "recency", "frequency", "monetary"]
].copy()

rfm["monetary"] = rfm["monetary"].round(2)

# Validate the reference checkpoints.
print("\nRFM CHECKPOINTS")
print("-" * 65)

print(f"Customers with completed orders: {len(rfm):,}")
print(f"Recency median: {rfm['recency'].median():.0f} days")
print(f"Recency minimum: {rfm['recency'].min()} days")
print(f"Recency maximum: {rfm['recency'].max()} days")
print(f"Frequency mean: {rfm['frequency'].mean():.2f}")
print(f"Frequency maximum: {rfm['frequency'].max()}")
print(f"Monetary mean: INR {rfm['monetary'].mean():,.2f}")
print(f"Monetary maximum: INR {rfm['monetary'].max():,.2f}")

checks = [
    (len(rfm) == 1175, "1,175 customers"),
    (rfm["recency"].median() == 19, "Recency median is 19"),
    (rfm["recency"].min() == 0, "Recency minimum is 0"),
    (rfm["recency"].max() == 180, "Recency maximum is 180"),
    (abs(rfm["frequency"].mean() - 9.46) < 0.02,
     "Frequency mean is approximately 9.46"),
    (rfm["frequency"].max() == 33, "Frequency maximum is 33"),
    (abs(rfm["monetary"].mean() - 2552.06) < 0.02,
     "Monetary mean is approximately INR 2,552.06"),
    (abs(rfm["monetary"].max() - 14766.55) < 0.01,
     "Monetary maximum is INR 14,766.55"),
]

all_passed = True

for passed, description in checks:
    print(f"{'PASS' if passed else 'CHECK'}: {description}")
    all_passed = all_passed and passed

# Save the RFM table for the next steps.
output_path = OUTPUT_DIR / "rfm.csv"
rfm.to_csv(output_path, index=False)

print(f"\nRFM table saved to: {output_path}")

if not all_passed:
    print(
        "\nSome reference checkpoints need investigation. "
        "Do not proceed to clustering until we review them."
    )
else:
    print("\nALL RFM CHECKPOINTS PASSED")
