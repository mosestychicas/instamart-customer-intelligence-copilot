# Instamart Customer Intelligence & Support Copilot

## Project Overview

This capstone combines customer analytics, retention modelling, support-ticket
classification, policy retrieval, and a tool-using refund assistant.

The project runs locally using Python, SQLite, scikit-learn, and the supplied
offline `mock_llm.py`. It does not require an API key or external LLM service.

## Environment

- Python 3.12
- pandas
- scikit-learn
- SQLite
- joblib
- TF-IDF and cosine similarity for policy retrieval
- Logistic Regression and Random Forest for lapse prediction
- Offline deterministic mock language model

## Part A — Data Preparation and SQL Analytics

Input data is stored under `data/raw/`, with policy documents under
`data/policies/`. The SQLite database is `data/instamart.db`.

Key data checks and findings:

- 1,200 customers in the customer dataset.
- 40 products.
- 12,309 orders.
- 11,110 completed orders and 1,199 cancelled orders.
- Completed-order revenue: INR 2,998,668.80.
- Customer, order, product, and policy data were used for downstream analytics.

The SQL analysis is stored in `sql/queries.sql`.

## Part B — Customer Segmentation and Retention

### RFM segmentation

Outputs:
- `outputs/rfm.csv`
- `outputs/customer_segments.csv`

The selected K-Means solution uses four customer segments.

| Segment | Customers | Mean Recency | Mean Frequency | Mean Monetary |
|---|---:|---:|---:|---:|
| At Risk (segment 0) | 436 | 28.86 | 3.77 | INR 1,057.62 |
| Premium | 138 | 24.12 | 6.77 | INR 7,129.70 |
| Loyal | 419 | 8.77 | 19.40 | INR 3,410.02 |
| At Risk (segment 3) | 182 | 109.04 | 2.20 | INR 685.99 |

The two At Risk groups have different purchase patterns and should be
distinguished in any future marketing strategy.

### Lapse prediction

Outputs:
- `outputs/lapse_model.joblib`
- `outputs/retention_list.csv`

The lapse dataset contained 1,156 customers, with a lapse rate of 38.32%.

| Model | F1 Score |
|---|---:|
| Logistic Regression | 0.6480 |
| Random Forest | 0.6404 |
| Oversampled model | 0.6972 |

The oversampled model achieved recall of 0.8539 and precision of 0.5891.
These are measured results from this dataset and evaluation setup.

## Part C — Support Ticket Classification

Main scripts:
- `prepare_tickets.py`
- `classify_tickets.py`
- `route_tickets.py`

Outputs:
- `outputs/tickets_prepared.csv`
- `outputs/triage_zero_shot.csv`
- `outputs/triage_few_shot.csv`
- `outputs/triage_budget_demo.csv`
- `outputs/triage_results.csv`
- `outputs/triage_retries.jsonl`

The classifier categorizes tickets into billing, delivery, technical, and fraud.
It uses bounded retries, token accounting, privacy checks, and a token-budget
demonstration.

### Measured classification results

| Metric | Zero-shot | Few-shot |
|---|---:|---:|
| Total tickets | 164 | 164 |
| Classified | 156 | 156 |
| JSON classifications | 144 | 144 |
| Keyword fallback classifications | 12 | 12 |
| Empty tickets skipped | 4 | 4 |
| Long tickets flagged | 4 | 4 |
| Tokens used | 8,951 | 10,271 |
| Reported accuracy | 84.62% | 100% |
| Tokens per classified ticket | 57.38 | 65.84 |

Few-shot prompting improved the measured accuracy in this mock-model evaluation
but increased token use. Results should be interpreted within the limitations
of the supplied deterministic mock model and evaluation data.

The 6,000-token budget demonstration used 6,000 tokens and left 52 tickets in
the `budget_exhausted` status.

### Ticket routing

`outputs/triage_results.csv` contains 164 rows.

| Route | Tickets |
|---|---:|
| support_queue | 106 |
| security_team | 39 |
| premium_desk | 11 |
| human_review | 8 |

Priority counts:
- NORMAL: 114
- HIGH: 50

Fraud tickets are routed to the security team with high priority. Premium
customers receive premium-desk routing unless a higher-priority review status
takes precedence.

## Part D — Policy Retrieval and Refund Agent

### Policy retrieval

Script: `policy_retrieval.py`

Output: `outputs/policy_chunks.json`

The retriever:
- Reads text files from `data/policies/`.
- Uses 40-word chunks with 10-word overlap.
- Preserves the policy source filename.
- Uses TF-IDF vectors and cosine similarity.
- Returns the top-ranked chunks with scores.

The index contains 15 chunks across five policy documents.

### Policy assistant evaluation

Script: `policy_assistant.py`

Output: `outputs/policy_assistant_eval.csv`

Five policy questions were evaluated. Four passed the expected-term checks and
one failed.

| Question | Result |
|---|---|
| Cancelled-order refund | Pass |
| Late-delivery refund after 45 minutes | Pass |
| Approval above INR 1,000 | Pass |
| Seven-day refund-request deadline | Fail |
| Unauthorized transaction escalation | Pass |

The zero-context control abstained on all five questions. The failed deadline
question demonstrates that sentence selection can still return irrelevant
policy content, even when relevant text exists in the retrieved chunks.
This limitation is recorded rather than hidden.

### Refund tools

`refund_agent.py` implements five local tools:

1. `get_order`
2. `search_policy`
3. `calculate_refund`
4. `issue_refund`
5. `escalate_to_human`

The refund logic enforces the following rules:

- Only completed orders can receive refunds.
- A completed order delivered in more than 45 minutes qualifies for a full
  late-delivery refund.
- Delivery within 45 minutes does not qualify.
- Refunds of INR 1,000 or less can be processed automatically.
- Refunds above INR 1,000 are sent for human approval.
- Refund actions are recorded in a JSONL ledger.

The seven-day request window is stated in the policy. The current agent does
not independently enforce that deadline against the order timestamp, so that
remains a known implementation limitation.

### Refund scenario results

Output: `outputs/agent_runs.jsonl`

| Scenario | Actual Status | Refund Amount | Model Calls | Tokens |
|---|---|---:|---:|---:|
| S1 | REFUND | INR 128.25 | 5 | 1,119 |
| S2 | REFUND | INR 135.00 | 5 | 1,119 |
| S3 | ESCALATE | INR 1,999.00 | 5 | 1,114 |
| S4 | DENIED | INR 0.00 | 4 | 762 |
| S5 | DENIED | INR 0.00 | 3 | 438 |
| S6 | NOT_FOUND | INR 0.00 | 2 | 123 |
| S7 | FALLBACK_MAX_ITERATIONS | INR 0.00 | 6 | 3,264 |

All seven scenario statuses and expected refund amounts match the supplied
scenario expectations, treating S7's expected `FALLBACK` as
`FALLBACK_MAX_ITERATIONS`.

### Safety and recovery tests

| Test | Observed Result |
|---|---|
| Token limit of 200 | FALLBACK_TOKEN_LIMIT |
| Time limit of zero seconds | FALLBACK_TIMEOUT |
| Crash after second tool execution | Simulated successfully |
| Resume after crash | REFUND: INR 128.25 issued |
| Resumed tool sequence | get_order, search_policy, calculate_refund, issue_refund |

The token-limit test recorded 425 tokens against a configured limit of 200.
The limit is checked after the model call returns, and the requested tool is
not executed after the limit is detected. This is not a strict pre-call token
cap.

Checkpoint files are stored in `state/`. The refund ledger is stored at
`outputs/refund_ledger.jsonl`.

## Running the Project

Activate the virtual environment first.

```powershell
.\.venv\Scripts\Activate.ps1
```

Run the data-generation and analytics scripts used in your workflow, followed
by the policy retrieval and agent scripts:

```powershell
python .\policy_retrieval.py
python .\policy_assistant.py
python .\refund_agent.py
```

For ticket classification and routing:

```powershell
python .\prepare_tickets.py
python .\classify_tickets.py
python .\route_tickets.py
```

Run only scripts whose input data and dependencies have already been prepared.

## Limitations and Future Improvements

1. Improve policy answer grounding and sentence selection.
2. Enforce the seven-day refund-request window in the refund tools.
3. Implement a strict pre-call token budget.
4. Add automated regression tests for the seven refund scenarios.
5. Evaluate classification on a held-out labelled dataset.
6. Add audit controls for duplicate refund requests and idempotent ledger actions.
7. Expand retention evaluation with time-based validation and calibration.
8. Keep the mock-model results separate from any future production LLM evaluation.

## Conclusion

The capstone demonstrates a local customer-intelligence pipeline combining
segmentation, lapse prediction, support-ticket triage, policy retrieval, and a
tool-using refund assistant. The measured outputs, safety fallbacks, and
remaining limitations are documented to support reproducibility and further
development.
