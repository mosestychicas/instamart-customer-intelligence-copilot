
from pathlib import Path
import sqlite3
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "data" / "instamart.db"
SQL_PATH = BASE_DIR / "sql" / "queries.sql"

sql_text = SQL_PATH.read_text(encoding="utf-8")
queries = [
    query.strip()
    for query in sql_text.split(";")
    if query.strip()
]

if len(queries) != 7:
    raise ValueError(
        f"Expected 7 SQL queries, found {len(queries)}"
    )

expected_rows = {
    "Q1": 30,
    "Q2": 30,
    "Q3": 11,
    "Q4": 25,
    "Q5": 550,
    "Q7": 183,
}

print("=" * 65)
print("INSTAMART SQL ANALYSIS")
print("=" * 65)

with sqlite3.connect(DB_PATH) as conn:
    results = {}

    for number, query in enumerate(queries, start=1):
        name = f"Q{number}"
        df = pd.read_sql_query(query, conn)
        results[name] = df

        print(f"\n{name}: {len(df):,} rows")
        print(df.head(5).to_string(index=False))

        if name in expected_rows:
            expected = expected_rows[name]
            result = "PASS" if len(df) == expected else "CHECK"
            print(
                f"{result}: expected {expected:,} rows; "
                f"received {len(df):,}"
            )

    q2 = results["Q2"]
    q2_total = q2["revenue_inr"].sum()
    largest_cell = q2.loc[q2["revenue_inr"].idxmax()]

    print("\nQ2 reference checks:")
    print(f"Revenue total: INR {q2_total:,.2f}")
    print(
        "Largest city/category cell: "
        f"{largest_cell['city']} / {largest_cell['category']} "
        f"— INR {largest_cell['revenue_inr']:,.2f}"
    )

    q5 = results["Q5"]
    if not q5.empty:
        print("\nQ5 average completed spend:")
        print(f"INR {q5['average_spend'].iloc[0]:,.2f}")

    q6 = results["Q6"]
    top_customers = (
        q6.sort_values(["city", "spend_rank", "customer_id"])
        .groupby("city", as_index=False)
        .first()
    )
    print("\nQ6 top customer per city:")
    print(
        top_customers[
            ["city", "customer_id", "total_spend", "spend_rank"]
        ].to_string(index=False)
    )

    q6_rank_diff = (q6["spend_rank"] != q6["row_num"]).sum()
    print(f"Q6 rows where RANK differs from ROW_NUMBER: {q6_rank_diff}")

    q7 = results["Q7"]
    print("\nQ7 final cumulative revenue:")
    print(f"INR {q7['cumulative_revenue_inr'].iloc[-1]:,.2f}")

print("\nSQL ANALYSIS RUN COMPLETE")
