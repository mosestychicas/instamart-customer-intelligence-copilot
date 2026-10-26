
from pathlib import Path
import sqlite3
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
CLEAN_DIR = BASE_DIR / "data" / "clean"
DB_PATH = BASE_DIR / "data" / "instamart.db"

DB_PATH.parent.mkdir(parents=True, exist_ok=True)

tables = {
    "customers": CLEAN_DIR / "customers.csv",
    "products": CLEAN_DIR / "products.csv",
    "orders": CLEAN_DIR / "orders.csv",
}

print("=" * 65)
print("INSTAMART SQLITE DATABASE BUILD")
print("=" * 65)

with sqlite3.connect(DB_PATH) as conn:
    for table_name, csv_path in tables.items():
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing cleaned CSV: {csv_path}")

        df = pd.read_csv(csv_path)
        df.to_sql(table_name, conn, if_exists="replace", index=False)

        row_count = pd.read_sql_query(
            f"SELECT COUNT(*) AS total FROM {table_name}",
            conn
        ).iloc[0]["total"]

        print(f"{table_name}: {row_count:,} rows")

    print("\nDatabase validation:")
    expected_counts = {
        "customers": 1200,
        "products": 40,
        "orders": 12309,
    }

    all_passed = True

    for table_name, expected in expected_counts.items():
        actual = pd.read_sql_query(
            f"SELECT COUNT(*) AS total FROM {table_name}",
            conn
        ).iloc[0]["total"]

        passed = actual == expected
        print(
            f"{'PASS' if passed else 'FAIL'}: "
            f"{table_name} has {actual:,} rows "
            f"(expected {expected:,})"
        )
        all_passed = all_passed and passed

    table_names = {
        row[0]
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    }

    for required_table in tables:
        passed = required_table in table_names
        print(
            f"{'PASS' if passed else 'FAIL'}: "
            f"Table '{required_table}' exists"
        )
        all_passed = all_passed and passed

    if not all_passed:
        raise RuntimeError("Database validation failed.")

print(f"\nDatabase saved to: {DB_PATH}")
print("DATABASE BUILD SUCCESSFUL")
