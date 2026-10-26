
import os
import re
import pandas as pd
from mock_llm import est_tokens

INPUT_PATH = "data/raw/tickets.csv"
OUTPUT_PATH = "outputs/tickets_prepared.csv"
MAX_INPUT_TOKENS = 400

# Apply these patterns in the exact required order.
PATTERNS = [
    ("email", re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"), "[EMAIL]"),
    ("phone", re.compile(r"\b[6-9]\d{9}\b"), "[PHONE]"),
    ("order_id", re.compile(r"ORD-\d{6}"), "[ORDER_ID]"),
]

df = pd.read_csv(INPUT_PATH, keep_default_na=False)

required_columns = {
    "ticket_id",
    "customer_id",
    "order_id",
    "text",
    "true_category",
}
missing_columns = required_columns - set(df.columns)

if missing_columns:
    raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

substitution_totals = {name: 0 for name, _, _ in PATTERNS}
prepared_texts = []
statuses = []
token_estimates = []

for text in df["text"].astype(str):
    # Input guardrails are checked against the original ticket text.
    if not text.strip():
        status = "skipped_empty"
        token_count = 0
    elif est_tokens(text) > MAX_INPUT_TOKENS:
        status = "flagged_long"
        token_count = est_tokens(text)
    else:
        status = "ready"
        token_count = est_tokens(text)

    # Redact in the exact order required by the assignment.
    redacted = text
    for name, pattern, placeholder in PATTERNS:
        redacted, count = pattern.subn(placeholder, redacted)
        substitution_totals[name] += count

    prepared_texts.append(redacted)
    statuses.append(status)
    token_estimates.append(token_count)

df["redacted_text"] = prepared_texts
df["status"] = statuses
df["estimated_tokens"] = token_estimates

# Assert that no email, phone number, or order ID remains.
remaining_matches = {}
for name, pattern, _ in PATTERNS:
    remaining_matches[name] = sum(
        len(pattern.findall(text))
        for text in df["redacted_text"]
    )

assert all(count == 0 for count in remaining_matches.values()), (
    f"Redaction verification failed: {remaining_matches}"
)

# Empty and long tickets must never be sent to the model.
assert (df.loc[df["text"].str.strip() == "", "status"] == "skipped_empty").all()
assert (
    df.loc[df["estimated_tokens"] > MAX_INPUT_TOKENS, "status"]
    == "flagged_long"
).all()

os.makedirs("outputs", exist_ok=True)
df.to_csv(OUTPUT_PATH, index=False)

print("=" * 65)
print("PART C: TICKET PREPARATION")
print("=" * 65)
print("Total tickets:", len(df))
print("\nStatus counts:")
print(df["status"].value_counts().to_string())

print("\nRedaction substitution counts:")
for name, count in substitution_totals.items():
    print(f"{name}: {count}")

print("Total substitutions:", sum(substitution_totals.values()))
print("\nRemaining matches after redaction:", remaining_matches)
print("Model calls made by this script: 0")
print("Saved:", os.path.abspath(OUTPUT_PATH))
print("\nTICKET PREPARATION COMPLETE")
