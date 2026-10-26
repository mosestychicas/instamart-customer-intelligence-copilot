
import pandas as pd
from pathlib import Path

RAW = Path("data/raw")

files = {
    "customers": RAW / "customers.csv",
    "products": RAW / "products.csv",
    "orders": RAW / "orders.csv",
    "tickets": RAW / "tickets.csv",
}

for name, path in files.items():
    print("\n" + "=" * 70)
    print(f"TABLE: {name.upper()}")
    print("=" * 70)

    # Preserve empty ticket text as empty strings.
    if name == "tickets":
        df = pd.read_csv(path, keep_default_na=False)
    else:
        df = pd.read_csv(path)

    print(f"\nShape (rows, columns): {df.shape}")

    print("\nData types:")
    print(df.dtypes.to_string())

    print("\nMissing values by column:")
    missing = pd.DataFrame({
        "missing_count": df.isna().sum(),
        "missing_percent": (df.isna().mean() * 100).round(2),
    })
    print(missing.to_string())

    print("\nFirst 5 rows:")
    print(df.head().to_string(index=False))

print("\n" + "=" * 70)
print("SPECIFIC RAW-DATA CHECKPOINTS")
print("=" * 70)

customers = pd.read_csv(files["customers"], keep_default_na=False)
orders = pd.read_csv(files["orders"], keep_default_na=False)

numeric_amount = pd.to_numeric(orders["order_amount"], errors="coerce")
numeric_delivery = pd.to_numeric(orders["delivery_minutes"], errors="coerce")

print(f"Raw order rows: {len(orders)}")
print(f"Exact duplicate order rows: {orders.duplicated().sum()}")
print(f"Pending order rows: {(orders['status'].str.strip().str.lower() == 'pending').sum()}")
print(f"Distinct order IDs: {orders['order_id'].nunique()}")
print(f"Non-numeric order amounts: {numeric_amount.isna().sum()}")
print(f"Negative delivery times: {(numeric_delivery < 0).sum()}")
print(f"Blank delivery times: {orders['delivery_minutes'].astype(str).str.strip().eq('').sum()}")

age = pd.to_numeric(customers["age"], errors="coerce")
print(f"Blank ages: {customers['age'].astype(str).str.strip().eq('').sum()}")
print(f"Ages outside 10-100: {((age < 10) | (age > 100)).sum()}")
print(f"Distinct raw city spellings: {customers['city'].nunique()}")
print(
    "Distinct non-blank raw gender spellings: "
    f"{customers.loc[customers['gender'].astype(str).str.strip().ne(''), 'gender'].nunique()}"
)
print(f"Blank genders: {customers['gender'].astype(str).str.strip().eq('').sum()}")
