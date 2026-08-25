# Day 08 Lab Report — LangGraph Support Ticket Agent

## 1. Student

- Name: Nguyen Ngoc Duong
- Student ID: 2A202601717
- Date: 2026-08-25

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

| Metric | Value |
|---|---:|
| Total scenarios | 7 |
| Success rate | 100% |
| Avg nodes visited | 6.4 |
| Total retries | 3 |
| Total interrupts | 2 |
| Resume success | True |


### Per-scenario

| Scenario | Expected route | Actual route | Success | Retries | Interrupts | Approval |
|---|---|---|---:|---:|---:|---:|
| S01_simple | simple | simple | yes | 0 | 0 | - |
| S02_tool | tool | tool | yes | 0 | 0 | - |
| S03_missing | missing_info | missing_info | yes | 0 | 0 | - |
| S04_risky | risky | risky | yes | 0 | 1 | required |
| S05_error | error | error | yes | 2 | 0 | - |
| S06_delete | risky | risky | yes | 0 | 1 | required |
| S07_dead_letter | error | error | yes | 1 | 0 | - |

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

- **SQLite persistence**: `build_checkpointer("sqlite")` uses `SqliteSaver(conn=...)`
  with WAL mode; the verified demo database is `outputs/checkpoints.db` and contains
  checkpoint history for `thread-sqlite_demo`.
- **Real HITL (interrupt/resume)**: `approval_node` calls `interrupt()` when
  `LANGGRAPH_INTERRUPT=true`; the graph pauses and is resumed via `Command(resume=...)`.
- **LLM-as-Judge**: `evaluate_node` uses a second structured-output LLM call to score
  whether a tool result semantically resolves the query (heuristic fallback if the LLM
  is unavailable).
- **Graph diagram**: `graph.get_graph().draw_mermaid()` output is saved to
  `outputs/graph_diagram.mmd`.

## 8. Improvement plan

- Productionize the mock tool into real API calls (orders, refunds) with proper
  idempotency keys so retries are safe.
- Add a Streamlit approval UI on top of the interrupt/resume flow instead of the CLI.
- Add tracing/observability (LangSmith) and alerting when `dead_letter` fires.
- Add a timeout budget per node so a slow LLM cannot stall the whole ticket pipeline.
