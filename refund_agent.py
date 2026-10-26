import json
import sqlite3
import time
import uuid
from datetime import datetime
from pathlib import Path

from mock_llm import chat, reset

BASE = Path(".")
DB_PATH = BASE / "data" / "instamart.db"
OUTPUT_DIR = BASE / "outputs"
STATE_DIR = BASE / "state"
OUTPUT_DIR.mkdir(exist_ok=True)
STATE_DIR.mkdir(exist_ok=True)

LEDGER_PATH = OUTPUT_DIR / "refund_ledger.jsonl"
RUNS_PATH = OUTPUT_DIR / "agent_runs.jsonl"

TOOLS_SPEC = [
    {"type": "function", "function": {"name": "get_order", "description": "Look up an order by ID.", "parameters": {"type": "object", "properties": {"order_id": {"type": "string"}}, "required": ["order_id"]}}},
    {"type": "function", "function": {"name": "search_policy", "description": "Search refund policy.", "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}},
    {"type": "function", "function": {"name": "calculate_refund", "description": "Calculate refund eligibility and amount.", "parameters": {"type": "object", "properties": {"order_amount": {"type": "number"}, "delivery_minutes": {"type": "number"}}, "required": ["order_amount", "delivery_minutes"]}}},
    {"type": "function", "function": {"name": "issue_refund", "description": "Issue an approved refund of INR 1000 or less.", "parameters": {"type": "object", "properties": {"order_id": {"type": "string"}, "refund_inr": {"type": "number"}}, "required": ["order_id", "refund_inr"]}}},
    {"type": "function", "function": {"name": "escalate_to_human", "description": "Send a refund above INR 1000 for human approval.", "parameters": {"type": "object", "properties": {"order_id": {"type": "string"}, "refund_inr": {"type": "number"}}, "required": ["order_id", "refund_inr"]}}},
]


def append_jsonl(path, data):
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(data, ensure_ascii=False) + "\n")


def log_action(tool_name, order_id, refund_inr):
    append_jsonl(LEDGER_PATH, {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "tool_name": tool_name,
        "order_id": order_id,
        "refund_inr": round(float(refund_inr), 2),
    })


def get_order(order_id):
    with sqlite3.connect(DB_PATH) as connection:
        connection.row_factory = sqlite3.Row
        row = connection.execute(
            "SELECT order_id, customer_id, status, order_amount, "
            "delivery_minutes, order_ts FROM orders WHERE order_id = ?",
            (order_id,),
        ).fetchone()

    if row is None:
        return {"error": "order_not_found", "order_id": order_id}

    result = dict(row)
    result["order_amount"] = (
        round(float(result["order_amount"]), 2)
        if result["order_amount"] is not None else None
    )
    result["delivery_minutes"] = (
        float(result["delivery_minutes"])
        if result["delivery_minutes"] is not None else None
    )
    return result


def search_policy(query):
    from policy_retrieval import PolicyRetriever, load_policy_chunks

    retriever = PolicyRetriever(load_policy_chunks())
    results = retriever.retrieve(query, k=3)
    return {
        "query": query,
        "results": [
            {"source": item["source"], "score": item["score"], "text": item["text"]}
            for item in results
        ],
    }


def calculate_refund(order_amount, delivery_minutes):
    amount = round(float(order_amount), 2)
    minutes = float(delivery_minutes)
    refund = amount if minutes > 45 else 0.0
    return {
        "order_amount": amount,
        "delivery_minutes": minutes,
        "refund_inr": round(refund, 2),
        "reason": "late_delivery" if refund else "delivery_within_45_minutes",
    }


def issue_refund(order_id, refund_inr):
    amount = round(float(refund_inr), 2)
    if amount <= 0 or amount > 1000:
        return {"error": "refund_amount_outside_automatic_approval_limit"}
    order = get_order(order_id)
    if "error" in order:
        return order
    if order["status"] != "completed":
        return {"error": "only_completed_orders_are_refundable"}
    if order["delivery_minutes"] <= 45:
        return {"error": "delivery_did_not_exceed_45_minutes"}
    if amount > order["order_amount"]:
        return {"error": "refund_exceeds_order_amount"}

    log_action("issue_refund", order_id, amount)
    return {"success": True, "order_id": order_id, "refund_inr": amount}


def escalate_to_human(order_id, refund_inr):
    amount = round(float(refund_inr), 2)
    if amount <= 1000:
        return {"error": "human_escalation_not_required_for_this_amount"}
    order = get_order(order_id)
    if "error" in order:
        return order
    if order["status"] != "completed":
        return {"error": "only_completed_orders_are_refundable"}
    if order["delivery_minutes"] <= 45:
        return {"error": "delivery_did_not_exceed_45_minutes"}
    if amount > order["order_amount"]:
        return {"error": "refund_exceeds_order_amount"}

    log_action("escalate_to_human", order_id, amount)
    return {
        "success": True,
        "order_id": order_id,
        "refund_inr": amount,
        "message": "Pending human support lead approval.",
    }


TOOL_MAP = {
    "get_order": get_order,
    "search_policy": search_policy,
    "calculate_refund": calculate_refund,
    "issue_refund": issue_refund,
    "escalate_to_human": escalate_to_human,
}


def save_checkpoint(run_id, state):
    path = STATE_DIR / f"{run_id}.json"
    path.write_text(json.dumps(state, indent=2), encoding="utf-8")


def load_checkpoint(run_id):
    path = STATE_DIR / f"{run_id}.json"
    return json.loads(path.read_text(encoding="utf-8"))


def run_agent(
    user_message,
    scenario_id="MANUAL",
    max_iterations=6,
    token_limit=6000,
    time_limit_seconds=10,
    run_id=None,
    simulate_crash_after_tools=None,
):
    run_id = run_id or f"{scenario_id}_{uuid.uuid4().hex[:8]}"
    state = {
        "run_id": run_id,
        "scenario_id": scenario_id,
        "messages": [
            {
                "role": "system",
                "content": (
                    "AGENT: Process refund requests using tools. Only completed orders "
                    "delivered in more than 45 minutes qualify. Refunds above INR 1000 "
                    "require human approval. Do not invent order details."
                ),
            },
            {"role": "user", "content": user_message},
        ],
        "tokens_used": 0,
        "steps": [],
        "tools_executed": [],
        "llm_calls": 0,
    }
    return _continue_agent(
        state, max_iterations, token_limit, time_limit_seconds,
        simulate_crash_after_tools,
    )


def _continue_agent(
    state, max_iterations=6, token_limit=6000, time_limit_seconds=10,
    simulate_crash_after_tools=None,
):
    started = time.monotonic()
    tools_this_run = 0

    def finish(status, answer):
        result = {
            "run_id": state["run_id"],
            "scenario_id": state["scenario_id"],
            "status": status,
            "final_answer": answer,
            "llm_calls": state["llm_calls"],
            "tokens_used": state["tokens_used"],
            "steps": state["steps"],
            "tools_executed": state["tools_executed"],
        }
        append_jsonl(RUNS_PATH, result)
        return result

    for _ in range(max_iterations):
        if time.monotonic() - started >= time_limit_seconds:
            return finish("FALLBACK_TIMEOUT",
                          "FALLBACK_TIMEOUT: handing the case to a human agent. Summary: time limit reached.")

        try:
            response = chat(state["messages"], tools=TOOLS_SPEC)
        except Exception as exc:
            if exc.__class__.__name__ == "TransientLLMError":
                try:
                    response = chat(state["messages"], tools=TOOLS_SPEC)
                except Exception:
                    return finish("FALLBACK_ERROR",
                                  "FALLBACK_ERROR: handing the case to a human agent. Summary: model retries failed.")
            else:
                return finish("FALLBACK_ERROR",
                              f"FALLBACK_ERROR: handing the case to a human agent. Summary: {type(exc).__name__}.")

        state["llm_calls"] += 1
        used = int(response.usage.total_tokens)
        state["tokens_used"] += used

        if state["tokens_used"] > token_limit:
            return finish("FALLBACK_TOKEN_LIMIT",
                          "FALLBACK_TOKEN_LIMIT: handing the case to a human agent. Summary: token budget exceeded.")

        message = response.choices[0].message
        if not message.tool_calls:
            answer = message.content or "FALLBACK_ERROR: empty model response."
            status = answer.split(":", 1)[0].strip()
            state["messages"].append({"role": "assistant", "content": answer})
            return finish(status, answer)

        state["messages"].append({
            "role": "assistant",
            "content": message.content,
            "tool_calls": [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {"name": call.name, "arguments": call.arguments},
                }
                for call in message.tool_calls
            ],
        })

        for call in message.tool_calls:
            name = call.name
            args = call.arguments
            if isinstance(args, str):
                args = json.loads(args)

            if name not in TOOL_MAP:
                observation = {"error": "unknown_tool"}
            else:
                try:
                    observation = TOOL_MAP[name](**args)
                except Exception as exc:
                    observation = {"error": f"tool_error: {type(exc).__name__}"}

            state["tools_executed"].append(name)
            tools_this_run += 1
            state["steps"].append({
                "step": len(state["steps"]) + 1,
                "tool": name,
                "arguments": args,
                "observation": observation,
            })
            state["messages"].append({
                "role": "tool",
                "name": name,
                "tool_call_id": call.id,
                "content": json.dumps(observation),
            })

            # Persist after every tool observation so a resumed run continues safely.
            save_checkpoint(state["run_id"], state)

            if (simulate_crash_after_tools is not None
                    and tools_this_run == simulate_crash_after_tools):
                raise RuntimeError(f"Simulated crash after {tools_this_run} tools.")

    return finish(
        "FALLBACK_MAX_ITERATIONS",
        "FALLBACK_MAX_ITERATIONS: handing the case to a human agent. Summary: maximum reasoning steps reached.",
    )


def resume_agent(run_id, max_iterations=6, token_limit=6000, time_limit_seconds=10):
    state = load_checkpoint(run_id)
    reset()
    return _continue_agent(state, max_iterations, token_limit, time_limit_seconds)


def run_scenarios():
    scenarios = []
    with (BASE / "data" / "raw" / "agent_scenarios.csv").open(
        encoding="utf-8-sig", newline=""
    ) as file:
        import csv
        scenarios = list(csv.DictReader(file))

    results = []
    for scenario in scenarios:
        result = run_agent(
            scenario["message"],
            scenario_id=scenario["scenario_id"],
        )
        expected_amount = float(scenario["expected_refund_inr"])
        actual_amount = 0.0
        for step in result["steps"]:
            if step["tool"] in ("issue_refund", "escalate_to_human"):
                actual_amount = float(step["observation"].get("refund_inr", 0))
        result["expected_outcome"] = scenario["expected_outcome"]
        result["expected_refund_inr"] = expected_amount
        result["actual_refund_inr"] = actual_amount
        result["refund_amount_match"] = abs(actual_amount - expected_amount) <= 0.01
        results.append(result)

    print("\nSCENARIO RESULTS")
    for result in results:
        print(
            result["scenario_id"],
            "|", result["status"],
            "| refund:", result["actual_refund_inr"],
            "| calls:", result["llm_calls"],
            "| tokens:", result["tokens_used"],
            "| expected:", result["expected_outcome"],
            "| amount match:", result["refund_amount_match"],
        )

    print("\nTOKEN-LIMIT TEST")
    reset()
    token_result = run_agent(
        "Refund order ORD-100048 please.",
        scenario_id="TOKEN_LIMIT_TEST",
        token_limit=200,
    )
    print(token_result["status"], token_result["llm_calls"], token_result["tokens_used"])

    print("\nTIMEOUT TEST")
    reset()
    timeout_result = run_agent(
        "Refund order ORD-100048 please.",
        scenario_id="TIMEOUT_TEST",
        time_limit_seconds=0,
    )
    print(timeout_result["status"], timeout_result["llm_calls"], timeout_result["tokens_used"])

    print("\nCRASH-RESUME TEST")
    reset()
    resume_id = f"RESUME_{uuid.uuid4().hex[:8]}"
    try:
        run_agent(
            "Refund order ORD-100048 please.",
            scenario_id="CRASH_RESUME_TEST",
            run_id=resume_id,
            simulate_crash_after_tools=2,
        )
    except RuntimeError as exc:
        print(exc)

    resumed = resume_agent(resume_id)
    print("Resumed status:", resumed["status"])
    print("Tools executed:", resumed["tools_executed"])
    print("Final answer:", resumed["final_answer"])
    print("Checkpoint:", (STATE_DIR / f"{resume_id}.json").resolve())
    print("\nSaved run log:", RUNS_PATH.resolve())
    print("Refund ledger:", LEDGER_PATH.resolve())


if __name__ == "__main__":
    run_scenarios()
