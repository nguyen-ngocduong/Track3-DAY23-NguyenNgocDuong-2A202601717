"""Routing functions for conditional edges.

Each function takes AgentState and returns a string — the name of the next node.
These strings MUST match node names registered in graph.py.
"""

from __future__ import annotations

from .state import AgentState

# route value -> next node name, used after the LLM classifier
_CLASSIFY_MAP = {
    "simple": "answer",
    "tool": "tool",
    "missing_info": "clarify",
    "risky": "risky_action",
    "error": "retry",
}


def route_after_classify(state: AgentState) -> str:
    """Map the classified route to the next graph node.

    - "simple"       → "answer"
    - "tool"         → "tool"
    - "missing_info" → "clarify"
    - "risky"        → "risky_action"
    - "error"        → "retry"
    - unknown/default → "answer"
    """
    return _CLASSIFY_MAP.get(state.get("route", ""), "answer")


def route_after_evaluate(state: AgentState) -> str:
    """Decide if the tool result is satisfactory or needs retry.

    - evaluation_result == "needs_retry" → "retry"
    - otherwise → "answer"
    """
    return "retry" if state.get("evaluation_result") == "needs_retry" else "answer"


def route_after_retry(state: AgentState) -> str:
    """Decide whether to retry the tool or give up (bounded retry loop).

    - attempt < max_attempts  → "tool" (try again)
    - attempt >= max_attempts → "dead_letter" (give up, escalate)
    """
    attempt = state.get("attempt", 0)
    max_attempts = state.get("max_attempts", 3)
    return "tool" if attempt < max_attempts else "dead_letter"


def route_after_approval(state: AgentState) -> str:
    """Route based on the human approval decision.

    - approved → "tool" (proceed with the risky action)
    - rejected → "clarify" (ask the user for an alternative)
    """
    approval = state.get("approval") or {}
    approved = approval.get("approved", False) if isinstance(approval, dict) else False
    return "tool" if approved else "clarify"
