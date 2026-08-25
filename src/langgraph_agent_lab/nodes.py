"""Node functions for the LangGraph workflow.

Each function receives AgentState and returns a partial state update dict.
Do NOT mutate input state — return new values only.

LLM REQUIREMENT:
- classify_node MUST use a real LLM call (structured output for intent classification)
- answer_node MUST use a real LLM call (grounded response generation)
- evaluate_node SHOULD use LLM-as-judge (bonus points; heuristic acceptable for base score)
"""

from __future__ import annotations

import os
from typing import Any, Literal, cast

from pydantic import BaseModel, Field

from .llm import get_llm
from .state import AgentState, ApprovalDecision, make_event


class IntentClassification(BaseModel):
    """Structured output for the intent router."""

    route: Literal["simple", "tool", "missing_info", "risky", "error"]
    risk_level: Literal["low", "high"] = "low"
    reasoning: str = Field(default="", description="one-sentence justification for the route")


class ToolQualityJudge(BaseModel):
    """Structured output for the LLM-as-judge evaluator (bonus)."""

    satisfactory: bool = Field(description="true if the tool result resolves the user's query")
    reasoning: str = Field(description="one-sentence justification")


# ─── EXAMPLE: working node (provided for reference) ──────────────────
def intake_node(state: AgentState) -> dict[str, Any]:
    """Normalize raw query. This node is provided as a working example."""
    query = state.get("query", "").strip()
    return {
        "query": query,
        "messages": [f"intake:{query[:40]}"],
        "events": [make_event("intake", "completed", "query normalized")],
    }


# ─── classification ───────────────────────────────────────────────────
def classify_node(state: AgentState) -> dict[str, Any]:
    """Classify the query into a route using an LLM with structured output.

    Priority guide baked into the prompt: risky > tool > missing_info > error > simple.
    risk_level is forced from the route ("high" only for risky) so the state invariant
    holds regardless of what the model returns.
    """
    llm = get_llm()
    # json_mode is the most reliable structured-output method on the lab's providers
    # (function_calling is ignored by some Groq models).
    structured = llm.with_structured_output(IntentClassification, method="json_mode")
    query = state.get("query", "")
    prompt = (
        "You are the intent router of a customer-support ticket system.\n"
        "Classify the user's query into exactly one route:\n"
        "- risky: the user asks to EXECUTE an action with side effects "
        "(refund, delete or cancel account/subscription, transfer money, send email, "
        "issue chargeback). If an action would be performed, it is risky.\n"
        "- tool: information lookup needing a backend tool "
        "(order status, tracking, invoice, account details).\n"
        "- error: the user reports a system failure "
        "(timeout, crash, 500 error, service down, cannot recover).\n"
        "- missing_info: vague or incomplete query with no actionable detail "
        "(e.g. 'Can you fix it?', 'I need help').\n"
        "- simple: general how-to questions answerable directly "
        "(e.g. 'How do I reset my password?').\n"
        "Priority if several fit: risky > tool > missing_info > error > simple.\n"
        "Examples:\n"
        "- 'Refund this customer and send confirmation email' -> risky\n"
        "- 'Please lookup order status for order 12345' -> tool\n"
        "- 'Timeout failure while processing request' -> error\n"
        "- 'Can you fix it?' -> missing_info\n"
        "- 'How do I reset my password?' -> simple\n"
        "Reply with JSON only using this schema: "
        '{"route": "<one of the routes>", "risk_level": "low" or "high", '
        '"reasoning": "<short reason>"}.\n'
        f"User query: {query}\n"
        "Classify now."
    )
    result = cast(IntentClassification, structured.invoke(prompt))
    route = result.route
    risk_level = "high" if route == "risky" else "low"
    return {
        "route": route,
        "risk_level": risk_level,
        "events": [
            make_event(
                "classify",
                "completed",
                f"classified as {route} (risk={risk_level})",
                reasoning=result.reasoning,
            )
        ],
    }


# ─── tool execution ───────────────────────────────────────────────────
def tool_node(state: AgentState) -> dict[str, Any]:
    """Execute a mock tool call.

    Transient failures are simulated for error-route scenarios so the retry loop
    is exercised: attempts 0 and 1 fail, later attempts succeed.
    """
    route = state.get("route", "")
    attempt = state.get("attempt", 0)
    if route == "error" and attempt < 2:
        result = "ERROR: downstream service unavailable (transient)"
        event_type = "failed"
    elif route == "risky":
        result = "Approved action executed: refund processed and confirmation email sent"
        event_type = "completed"
    elif route == "tool":
        result = "Order #12345: Shipped (tracking available)"
        event_type = "completed"
    else:
        result = "Request processed successfully"
        event_type = "completed"
    return {
        "tool_results": [result],
        "events": [make_event("tool", event_type, result, attempt=attempt)],
    }


# ─── evaluation (retry-loop gate) ─────────────────────────────────────
def evaluate_node(state: AgentState) -> dict[str, Any]:
    """Evaluate the latest tool result — the retry-loop gate.

    A hard "ERROR" substring check is the deterministic gate; for non-error results
    an LLM-as-judge scores whether the result semantically resolves the query (bonus).
    """
    tool_results = state.get("tool_results") or []
    latest = tool_results[-1] if tool_results else ""
    if "ERROR" in latest:
        evaluation = "needs_retry"
        reasoning = "tool result contains ERROR marker"
    elif os.getenv("LLM_JUDGE", "false").lower() == "true":
        try:
            judge = get_llm().with_structured_output(ToolQualityJudge, method="json_mode")
            judge_result = cast(ToolQualityJudge, judge.invoke(
                "You judge whether a tool result adequately resolved the user's support request.\n"
                "If it did not, the system will retry. "
                "Be strict about missing or contradictory data.\n"
                f"User query: {state.get('query', '')}\n"
                f"Tool result: {latest}\n"
                "Reply with JSON only using this schema: "
                '{"satisfactory": true, "reasoning": "<short reason>"}.\n'
                "Is the tool result satisfactory?"
            ))
            evaluation = "success" if judge_result.satisfactory else "needs_retry"
            reasoning = judge_result.reasoning
        except Exception as exc:  # judge is bonus; heuristic fallback keeps graph working
            evaluation = "success"
            reasoning = f"judge unavailable, fallback to success ({exc})"
    else:
        evaluation = "success"
        reasoning = "tool result has no ERROR marker"
    return {
        "evaluation_result": evaluation,
        "events": [
            make_event(
                "evaluate",
                "completed",
                f"evaluation_result={evaluation}",
                reasoning=reasoning,
            )
        ],
    }


# ─── answer generation ────────────────────────────────────────────────
def answer_node(state: AgentState) -> dict[str, Any]:
    """Generate the final response with an LLM, grounded in available context."""
    llm = get_llm()
    parts = [f"User query: {state.get('query', '')}"]
    if state.get("tool_results"):
        parts.append(f"Tool results: {state['tool_results']}")
    if state.get("proposed_action"):
        parts.append(f"Proposed action: {state['proposed_action']}")
    if state.get("approval"):
        parts.append(f"Approval decision: {state['approval']}")
    prompt = (
        "You are a customer-support agent. Write a concise, professional reply "
        "to the user's ticket using ONLY the context below. Do not invent facts "
        "that are not supported by the context.\n\n"
        + "\n".join(parts)
    )
    response = llm.invoke(prompt)
    text = response.content if hasattr(response, "content") else str(response)
    return {
        "final_answer": text,
        "events": [make_event("answer", "completed", "generated grounded final answer")],
    }


# ─── clarification branch ─────────────────────────────────────────────
def ask_clarification_node(state: AgentState) -> dict[str, Any]:
    """Ask for missing information instead of hallucinating."""
    query = state.get("query", "")
    fallback = (
        f"Your request \"{query}\" is missing some details. Could you tell us what you "
        "need help with, and include any order or account ID if relevant?"
    )
    try:
        response = get_llm().invoke(
            "The user's support request is too vague to act on. Generate ONE specific "
            "clarification question asking for the missing detail (e.g. what exactly to fix, "
            "or which order/account is involved). Keep it under 40 words.\n"
            f"User request: {query}\n"
            "Clarification question:"
        )
        content = getattr(response, "content", None)
        question = content.strip() if isinstance(content, str) else str(response).strip()
    except Exception:
        question = fallback
    return {
        "pending_question": question,
        "final_answer": question,
        "events": [make_event("clarify", "completed", "requested missing information")],
    }


# ─── risky action + HITL approval ─────────────────────────────────────
def risky_action_node(state: AgentState) -> dict[str, Any]:
    """Prepare a risky action for human approval."""
    proposed = (
        f"PROPOSED ACTION: {state.get('query', '')} — this action is destructive or "
        "financially impactful and MUST be approved by a human before execution."
    )
    return {
        "proposed_action": proposed,
        "events": [make_event("risky_action", "proposed", "risky action prepared for approval")],
    }


def approval_node(state: AgentState) -> dict[str, Any]:
    """Human-in-the-loop approval step.

    Default: mock approval (approved=True) so tests and scenario runs work offline.
    Extension: with LANGGRAPH_INTERRUPT=true the node pauses via interrupt() and the
    workflow is resumed with a human decision.
    """
    if os.getenv("LANGGRAPH_INTERRUPT", "").lower() == "true":
        from langgraph.types import interrupt

        decision = interrupt(
            {
                "question": "Approve the following risky action?",
                "proposed_action": state.get("proposed_action", ""),
            }
        )
        decision_value = decision.get("approved", True) if isinstance(decision, dict) else decision
        approved = bool(decision_value)
        approval = ApprovalDecision(approved=approved, reviewer="human", comment=str(decision))
    else:
        approval = ApprovalDecision(
            approved=True, reviewer="mock-reviewer", comment="auto-approved (offline mode)"
        )
    return {
        "approval": approval.model_dump(),
        "events": [
            make_event(
                "approval",
                "approved" if approval.approved else "rejected",
                f"approval approved={approval.approved} by {approval.reviewer}",
            )
        ],
    }


# ─── retry / dead-letter / finalize ───────────────────────────────────
def retry_or_fallback_node(state: AgentState) -> dict[str, Any]:
    """Record a retry attempt: increment the counter and log the failure."""
    attempt = state.get("attempt", 0) + 1
    tool_results = state.get("tool_results") or []
    latest = tool_results[-1] if tool_results else ""
    detail = latest if latest else "transient tool failure"
    error_message = f"attempt {attempt} failed: {detail}"
    return {
        "attempt": attempt,
        "errors": [error_message],
        "events": [make_event("retry", "retrying", error_message, attempt=attempt)],
    }


def dead_letter_node(state: AgentState) -> dict[str, Any]:
    """Handle unresolvable failures after max retries exceeded."""
    answer = (
        "We could not complete your request after several attempts. "
        "It has been escalated to a human specialist who will follow up with you."
    )
    return {
        "final_answer": answer,
        "events": [make_event("dead_letter", "escalated", "max retries exceeded, escalated")],
    }


def finalize_node(state: AgentState) -> dict[str, Any]:
    """Emit a final audit event. All routes must pass through here before END."""
    return {"events": [make_event("finalize", "completed", "workflow finished")]}
