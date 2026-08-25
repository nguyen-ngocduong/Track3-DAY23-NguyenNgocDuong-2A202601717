"""Extension tests for high-grade band (90-100).

Covers:
1. Real HITL interrupt/resume (with langgraph Command(resume=...))
2. SQLite durable crash recovery across graph instances
3. Time travel / State history inspection
4. Mermaid diagram export & topology verification
5. LLM-as-judge evaluator fallback
"""

from __future__ import annotations

import sqlite3
import tempfile
from pathlib import Path

import pytest
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command

from langgraph_agent_lab.graph import build_graph
from langgraph_agent_lab.nodes import evaluate_node
from langgraph_agent_lab.state import Route, Scenario, initial_state


# ─── 1. Real HITL Interrupt and Resume ──────────────────────────────────────────
def test_hitl_interrupt_and_resume_approved(monkeypatch: pytest.MonkeyPatch) -> None:
    """Demonstrate real HITL: graph interrupts at approval, resumes with human decision."""
    monkeypatch.setenv("LANGGRAPH_INTERRUPT", "true")

    checkpointer = MemorySaver()
    graph = build_graph(checkpointer=checkpointer)

    scenario = Scenario(
        id="hitl_approve_demo",
        query="Refund this customer and send confirmation email",
        expected_route=Route.RISKY,
        requires_approval=True,
    )
    state = initial_state(scenario)
    thread_config = {"configurable": {"thread_id": "thread-hitl-approve"}}

    # Run 1: Graph should run until approval_node and pause (interrupt)
    graph.invoke(state, config=thread_config)

    # Verify graph paused at approval
    current_state = graph.get_state(thread_config)
    assert len(current_state.tasks) > 0, "Graph should be paused at interrupt task"
    assert current_state.next == ("approval",)

    # Run 2: Human reviewer approves action via Command(resume=...)
    resumed_result = graph.invoke(
        Command(resume={"approved": True, "reviewer": "admin_duong"}),
        config=thread_config,
    )

    # Verify workflow completed through tool -> evaluate -> answer -> finalize
    assert resumed_result["route"] == "risky"
    assert resumed_result.get("approval", {}).get("approved") is True
    assert resumed_result.get("final_answer") is not None

    events = resumed_result.get("events", [])
    node_names = [e["node"] for e in events]
    assert "risky_action" in node_names
    assert "approval" in node_names
    assert "tool" in node_names
    assert "finalize" in node_names


def test_hitl_interrupt_and_resume_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    """Demonstrate real HITL rejection: resumes into clarify branch."""
    monkeypatch.setenv("LANGGRAPH_INTERRUPT", "true")

    checkpointer = MemorySaver()
    graph = build_graph(checkpointer=checkpointer)

    scenario = Scenario(
        id="hitl_reject_demo",
        query="Delete customer account immediately",
        expected_route=Route.RISKY,
        requires_approval=True,
    )
    state = initial_state(scenario)
    thread_config = {"configurable": {"thread_id": "thread-hitl-reject"}}

    # Run 1: Pauses at approval
    graph.invoke(state, config=thread_config)

    # Run 2: Human reviewer rejects action
    resumed_result = graph.invoke(
        Command(resume={"approved": False, "reviewer": "admin_duong"}),
        config=thread_config,
    )

    # Verify workflow redirected to clarify
    assert resumed_result["route"] == "risky"
    assert resumed_result.get("approval", {}).get("approved") is False
    assert resumed_result.get("pending_question") is not None

    events = resumed_result.get("events", [])
    node_names = [e["node"] for e in events]
    assert "clarify" in node_names
    assert "tool" not in node_names


# ─── 2. SQLite Durable Persistence Across Instances (Crash Recovery) ──────────
def test_sqlite_persistence_across_instances() -> None:
    """Demonstrate that state persists across separate Graph and process instances."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = str(Path(tmpdir) / "test_checkpoints.db")
        thread_id = "thread-sqlite-durability"
        thread_config = {"configurable": {"thread_id": thread_id}}

        # Instance 1: Run graph with SqliteSaver
        conn1 = sqlite3.connect(db_path, check_same_thread=False)
        conn1.execute("PRAGMA journal_mode=WAL;")
        checkpointer1 = SqliteSaver(conn=conn1)
        graph1 = build_graph(checkpointer=checkpointer1)

        scenario = Scenario(
            id="sqlite_test", query="How do I reset my password?", expected_route=Route.SIMPLE
        )
        state = initial_state(scenario)
        result1 = graph1.invoke(state, config=thread_config)
        assert result1["route"] == "simple"
        conn1.close()
        del graph1, checkpointer1  # simulate process shutdown

        # Instance 2: Connect from a new instance and verify checkpoint recovery
        conn2 = sqlite3.connect(db_path, check_same_thread=False)
        checkpointer2 = SqliteSaver(conn=conn2)
        graph2 = build_graph(checkpointer=checkpointer2)

        recovered_state = graph2.get_state(thread_config)
        assert recovered_state.values["route"] == "simple"
        assert recovered_state.values["query"] == "How do I reset my password?"
        assert recovered_state.values.get("final_answer") is not None

        history = list(graph2.get_state_history(thread_config))
        assert len(history) >= 4, "State history must record checkpoint snapshots across nodes"
        conn2.close()


# ─── 3. Time Travel / State History Inspection ────────────────────────────────
def test_time_travel_state_history() -> None:
    """Verify time-travel inspection of past checkpoints."""
    checkpointer = MemorySaver()
    graph = build_graph(checkpointer=checkpointer)

    scenario = Scenario(
        id="time_travel",
        query="Please lookup order status for order 12345",
        expected_route=Route.TOOL,
    )
    state = initial_state(scenario)
    thread_config = {"configurable": {"thread_id": "thread-time-travel"}}

    graph.invoke(state, config=thread_config)

    history = list(graph.get_state_history(thread_config))
    assert len(history) >= 5, "Graph must have recorded multiple checkpoints in history"

    # Verify latest checkpoint is at END / finalize
    latest_snapshot = history[0]
    assert latest_snapshot.values["route"] == "tool"
    assert len(latest_snapshot.values["tool_results"]) >= 1

    # Verify an earlier snapshot before tool execution
    earlier_snapshots = [s for s in history if len(s.values.get("tool_results", [])) == 0]
    assert len(earlier_snapshots) > 0, "Must have an earlier checkpoint before tool results"


# ─── 4. Mermaid Export & Topology Verification ─────────────────────────────────
def test_mermaid_export() -> None:
    """Verify that graph exports a valid Mermaid diagram with all 11 nodes."""
    graph = build_graph()
    mermaid_str = graph.get_graph().draw_mermaid()

    assert "intake" in mermaid_str
    assert "classify" in mermaid_str
    assert "tool" in mermaid_str
    assert "evaluate" in mermaid_str
    assert "answer" in mermaid_str
    assert "clarify" in mermaid_str
    assert "risky_action" in mermaid_str
    assert "approval" in mermaid_str
    assert "retry" in mermaid_str
    assert "dead_letter" in mermaid_str
    assert "finalize" in mermaid_str
    assert "__start__" in mermaid_str or "start" in mermaid_str.lower()
    assert "__end__" in mermaid_str or "end" in mermaid_str.lower()


# ─── 5. LLM-as-Judge Evaluator Logic ──────────────────────────────────────────
def test_llm_judge_evaluator_fallback() -> None:
    """Verify evaluate_node handles error markers and maintains fallbacks."""
    # Test error marker detection
    state_with_error = {"tool_results": ["ERROR: connection timed out"], "query": "test query"}
    res_error = evaluate_node(state_with_error)  # type: ignore[arg-type]
    assert res_error["evaluation_result"] == "needs_retry"

    # Test successful tool result
    state_success = {
        "tool_results": ["Order #12345: Shipped successfully"],
        "query": "lookup order 12345",
    }
    res_success = evaluate_node(state_success)  # type: ignore[arg-type]
    assert res_success["evaluation_result"] == "success"
