import os
import pandas as pd

OUTPUT_DIR = "outputs"
TICKETS_PATH = os.path.join(OUTPUT_DIR, "triage_few_shot.csv")
SEGMENTS_PATH = os.path.join(OUTPUT_DIR, "customer_segments.csv")
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "triage_results.csv")

def main():
    tickets = pd.read_csv(TICKETS_PATH, keep_default_na=False)
    segments = pd.read_csv(SEGMENTS_PATH, keep_default_na=False)

    required_ticket_columns = {
        "ticket_id", "customer_id", "status", "category", "tokens_used"
    }
    required_segment_columns = {
        "customer_id", "segment_name", "is_premium"
    }

    if not required_ticket_columns.issubset(tickets.columns):
        raise ValueError(
            f"Ticket file missing columns: "
            f"{required_ticket_columns - set(tickets.columns)}"
        )

    if not required_segment_columns.issubset(segments.columns):
        raise ValueError(
            f"Segment file missing columns: "
            f"{required_segment_columns - set(segments.columns)}"
        )

    segments = segments[
        ["customer_id", "segment_name", "is_premium"]
    ].drop_duplicates("customer_id")

    result = tickets.merge(segments, on="customer_id", how="left")
    result["segment_name"] = result["segment_name"].fillna("")
    result["is_premium"] = (
        pd.to_numeric(result["is_premium"], errors="coerce")
        .fillna(0)
        .astype(int)
    )

    review_statuses = {
        "skipped_empty",
        "flagged_long",
        "flagged_invalid_output",
        "budget_exhausted",
    }

    def route(row):
        if row["status"] in review_statuses:
            return "human_review", "NORMAL"
        if row["category"] == "fraud":
            return "security_team", "HIGH"
        if row["is_premium"] == 1:
            return "premium_desk", "HIGH"
        return "support_queue", "NORMAL"

    routing = result.apply(route, axis=1, result_type="expand")
    routing.columns = ["route", "priority"]
    result[["route", "priority"]] = routing

    # Keep exactly the output columns required by the assignment.
    result = result[
        [
            "ticket_id",
            "customer_id",
            "status",
            "category",
            "segment_name",
            "priority",
            "route",
            "tokens_used",
        ]
    ]

    result.to_csv(OUTPUT_PATH, index=False)

    print("Routing complete.")
    print("Rows:", len(result))
    print("\nRoute counts:")
    print(result["route"].value_counts().to_string())
    print("\nPriority counts:")
    print(result["priority"].value_counts().to_string())
    print("\nStatus counts:")
    print(result["status"].value_counts().to_string())
    print("\nSaved:", os.path.abspath(OUTPUT_PATH))

if __name__ == "__main__":
    main()
