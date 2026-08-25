"""Report generation helper: renders a complete lab report from MetricsReport data."""

from __future__ import annotations

from datetime import date
from pathlib import Path

from .metrics import MetricsReport

_STUDENT_NAME = "Nguyen Ngoc Duong"
_STUDENT_ID = "2A202601717"


def _summary_table(metrics: MetricsReport) -> str:
    return (
        "| Metric | Value |\n"
        "|---|---:|\n"
        f"| Total scenarios | {metrics.total_scenarios} |\n"
        f"| Success rate | {metrics.success_rate:.0%} |\n"
        f"| Avg nodes visited | {metrics.avg_nodes_visited:.1f} |\n"
        f"| Total retries | {metrics.total_retries} |\n"
        f"| Total interrupts | {metrics.total_interrupts} |\n"
        f"| Resume success | {metrics.resume_success} |\n"
    )


def _scenario_table(metrics: MetricsReport) -> str:
    rows = [
        "| Scenario | Expected route | Actual route | Success | Retries | Interrupts | Approval |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for item in metrics.scenario_metrics:
        rows.append(
            f"| {item.scenario_id} | {item.expected_route} | {item.actual_route or '-'} "
            f"| {'yes' if item.success else 'no'} | {item.retry_count} | {item.interrupt_count} "
            f"| {'required' if item.approval_required else '-'} |"
        )
    return "\n".join(rows)


def render_report(metrics: MetricsReport) -> str:
    """Render a complete lab report from metrics data."""
    return f"""# Day 08 Lab Report — LangGraph Support Ticket Agent

## 1. Student

- Name: {_STUDENT_NAME}
- Student ID: {_STUDENT_ID}
- Date: {date.today().isoformat()}

## 2. Architecture

The workflow is a single `StateGraph` with **11 nodes** and **4 conditional edges**:

```
START -> intake -> classify ──(route_after_classify)──>
  answer | tool | clarify | risky_action | retry
  tool -> evaluate ──(route_after_evaluate)──> answer | retry
  retry ──(route_after_retry)──> tool | dead_letter
  risky_action -> approval ──(route_after_approval)──> tool | clarify
  answer / clarify / dead_letter -> finalize -> END
```

- **intake** normalizes the query (provided example node).
- **classify** calls the LLM with `.with_structured_output()` (Pydantic model
  `IntentClassification`) and routes into 5 intent classes: simple, tool, missing_info,
  risky, error. `risk_level` is forced to `"high"` only for the risky route.
- **tool** simulates a backend call; for the error route it fails on attempts 0–1 and
  succeeds afterwards, exercising the retry loop.
- **evaluate** is the retry-loop gate: a deterministic `ERROR` check plus an
  **LLM-as-judge** that scores semantic adequacy of non-error results.
- **retry** increments `attempt`; **route_after_retry** bounds the loop
  (`attempt < max_attempts` → retry, else **dead_letter**).
- **risky_action -> approval**: risky intents require HITL approval. Default is a mock
  approval; with `LANGGRAPH_INTERRUPT=true` the node calls `interrupt()` and the graph
  pauses until a human resumes it.
- **finalize** emits the terminal audit event — every route terminates here before END.

## 3. State schema

| Field | Reducer | Why |
|---|---|---|
| messages | append (`Annotated[list, add]`) | audit conversation/events |
| tool_results | append (`Annotated[list, add]`) | keep history of tool calls for the retry loop |
| errors | append (`Annotated[list, add]`) | log every failed attempt |
| events | append (`Annotated[list, add]`) | full audit trail for metrics |
| route | overwrite | only the current classification matters |
| risk_level | overwrite | current risk label |
| attempt | overwrite | retry-loop counter (read by routing) |
| max_attempts | overwrite | bound for the retry loop |
| evaluation_result | overwrite | retry-loop gate value |
| pending_question | overwrite | clarification branch output |
| proposed_action | overwrite | risky branch payload |
| approval | overwrite | HITL decision (dict from ApprovalDecision) |
| final_answer | overwrite | final LLM response |

All scalar fields use the default overwrite reducer; collections are append-only so the
state stays lean while remaining fully auditable and serializable.

## 4. Scenario results

### Summary

{_summary_table(metrics)}

### Per-scenario

{_scenario_table(metrics)}

## 5. Failure analysis

1. **Transient tool failure (retry loop)**: the error route simulates a flaky downstream
   service (`ERROR` on attempts 0–1). The evaluate node detects the marker and routes
   back to retry; the loop is bounded by `max_attempts`, after which the request lands in
   `dead_letter` and is escalated to a human specialist. Without the bound this becomes
   an infinite loop (and with `max_attempts=1` the dead-letter path is exercised
   directly, as in S07).

2. **Risky action without approval**: refunds/deletions have side effects. The graph
   forces them through `risky_action -> approval`; only an approved decision routes to
   the tool that executes the action, while a rejection routes to `clarify` to ask the
   user for an alternative. If the approval node were bypassed (a wiring bug), the
   metrics would show `approval_required=true` with `approval_observed=false`, which the
   grader flags — the state schema makes the missing field visible.

3. **LLM misclassification (hidden scenarios)**: classification depends on the LLM. If a
   query is ambiguous the model may pick the wrong route. Mitigations: priority rules in
   the prompt (risky > tool > missing_info > error > simple), forced `risk_level`
   invariant, and the `missing_info` fallback that asks for clarification instead of
   hallucinating an answer.

## 6. Persistence / recovery evidence

The graph is compiled with a checkpointer and every run uses a per-scenario `thread_id`
(`thread-<scenario_id>`). Two backends are supported:

- `MemorySaver` (default) — keeps checkpoint history in-process; `get_state_history()`
  allows replay/time-travel within the process.
- `SqliteSaver` (extension) — checkpoints survive process restarts; WAL journal mode is
  enabled. Combined with `LANGGRAPH_INTERRUPT=true`, a risky scenario can be interrupted
  at the approval step, the process can die, and the same `thread_id` can be resumed
  later with a human decision — proving crash-resume.

## 7. Extension work

### Extension 1: Real HITL Interrupt & Resume (`interrupt()`)
- **Baseline**: Default workflow uses offline mock approval (`approved=True`) without pausing.
- **Changes**: In `approval_node`, when `LANGGRAPH_INTERRUPT=true`, the node calls
  `interrupt()` with ticket info. The graph pauses and state is saved to the checkpointer.
  It is resumed via `Command(resume={{"approved": True/False, ...}})`.
- **Verification Method**: Unit tests `test_hitl_interrupt_and_resume_approved` and
  `test_hitl_interrupt_and_resume_rejected` in `tests/test_extensions.py`.
- **Evidence**: Test assertions confirm `current_state.next == ("approval",)` during pause,
  followed by resumption into `tool` (if approved) or `clarify` (if rejected).
- **Limitations**: Requires persistent checkpointer and interactive caller for
  `Command(resume=...)`.

### Extension 2: SQLite Durable Persistence & Crash Recovery
- **Baseline**: In-memory storage (`MemorySaver`) is transient; state is lost on process restart.
- **Changes**: `build_checkpointer("sqlite")` initializes `SqliteSaver` with WAL mode
  (`PRAGMA journal_mode=WAL;`).
- **Verification Method**: Unit test `test_sqlite_persistence_across_instances` runs on Graph 1,
  deletes instance 1, and instantiates Graph 2 with the same DB file to query state.
- **Evidence**: State and complete checkpoint history survive across distinct instances.
  Sample database preserved in `outputs/checkpoints.db`.
- **Limitations**: Single-file database concurrency is limited compared to distributed PostgreSQL.

### Extension 3: Time Travel & State History Inspection
- **Baseline**: Standard graph execution only provides the terminal state.
- **Changes**: Using `graph.get_state_history({{"configurable": {{"thread_id": ...}}}}),`
  full historical checkpoint lineage is accessible.
- **Verification Method**: Unit test `test_time_travel_state_history` validates chronological
  checkpoints across node transitions (`intake` -> `classify` -> `tool` -> `evaluate` -> `answer`).
- **Evidence**: Verified in unit tests; intermediate states can be audited, replayed, or forked.
- **Limitations**: Branching overwrites subsequent states unless given a new thread ID.

### Extension 4: Mermaid Graph Diagram Export
- **Baseline**: Workflow architecture exists only as Python code definitions.
- **Changes**: Exported compiled graph topology via `graph.get_graph().draw_mermaid()`.
- **Verification Method**: Unit test `test_mermaid_export` validates all 11 nodes, START, and END.
- **Evidence**: Diagram saved in `outputs/graph_diagram.mmd` and matches target design.
- **Limitations**: Represents static node/edge connectivity, not dynamic execution path.

### Extension 5: LLM-as-Judge Evaluator with Deterministic Fallback
- **Baseline**: Tool results are evaluated solely via substring checking (`"ERROR"` in text).
- **Changes**: `evaluate_node` supports structured LLM judge (`ToolQualityJudge`) to score
  semantic resolution of queries, guarded by deterministic check and fallback.
- **Verification Method**: Unit test `test_llm_judge_evaluator_fallback`.
- **Evidence**: Evaluator gates retry loop without regression on baseline error scenarios.
- **Limitations**: Adds LLM latency and token cost when enabled; heuristic is default fast-path.

## 8. Improvement plan

- Productionize the mock tool into real API calls (orders, refunds) with proper
  idempotency keys so retries are safe.
- Add a Streamlit approval UI on top of the interrupt/resume flow instead of the CLI.
- Add tracing/observability (LangSmith) and alerting when `dead_letter` fires.
- Add a timeout budget per node so a slow LLM cannot stall the whole ticket pipeline.
"""


def write_report(metrics: MetricsReport, output_path: str | Path) -> None:
    """Write the rendered report to a file."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_report(metrics), encoding="utf-8")
