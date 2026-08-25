"""Streamlit UI for LangGraph Support Ticket Agent.

Features:
- Ticket selection & custom query input
- Interactive Human-In-The-Loop (HITL) approval / rejection simulation
- Dynamic Mermaid diagram path highlighting corresponding to scenario execution
- Complete event audit trail & state inspector
- Safe handling of environment credentials without exposing secrets
"""

from __future__ import annotations

import os
import time

import streamlit as st
from dotenv import load_dotenv
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from langgraph_agent_lab.graph import build_graph
from langgraph_agent_lab.scenarios import load_scenarios
from langgraph_agent_lab.state import Route, Scenario, initial_state

load_dotenv()

# Page configuration
st.set_page_config(
    page_title="LangGraph Agentic Orchestrator",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)


def get_masked_key_status() -> str:
    """Check API key presence without revealing secret characters."""
    for key in ["GROQ_API_KEY", "GEMINI_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY"]:
        val = os.getenv(key)
        if val:
            prefix = key.split("_")[0]
            masked = val[:4] + "..." + val[-4:] if len(val) > 8 else "***"
            return f"✅ {prefix} ({masked})"
    return "❌ No API Key found in .env"


def generate_highlighted_mermaid(visited_nodes: list[str]) -> str:
    """Generate Mermaid graph with highlighted visited nodes and path styling."""
    nodes_set = set(visited_nodes)

    mermaid = """graph TD
    START([START]) --> intake[intake]
    intake --> classify[classify]
    
    classify -->|simple| answer[answer]
    classify -->|tool| tool[tool]
    classify -->|missing_info| clarify[clarify]
    classify -->|risky| risky_action[risky_action]
    classify -->|error| retry[retry]
    
    tool --> evaluate[evaluate]
    evaluate -->|success| answer
    evaluate -->|needs_retry| retry
    
    retry -->|attempt < max| tool
    retry -->|attempt >= max| dead_letter[dead_letter]
    
    risky_action --> approval[approval]
    approval -->|approved| tool
    approval -->|rejected| clarify
    
    answer --> finalize[finalize]
    clarify --> finalize
    dead_letter --> finalize
    finalize --> END([END])

    classDef default fill:#1E1E2F,stroke:#4A4A6A,stroke-width:1px,color:#FFFFFF;
    classDef visited fill:#10B981,stroke:#059669,stroke-width:3px,color:#FFFFFF,font-weight:bold;
    classDef active fill:#F59E0B,stroke:#D97706,stroke-width:3px,color:#FFFFFF,font-weight:bold;
    classDef terminal fill:#3B82F6,stroke:#2563EB,stroke-width:2px,color:#FFFFFF;
"""
    valid_nodes = [
        "intake",
        "classify",
        "tool",
        "evaluate",
        "answer",
        "clarify",
        "risky_action",
        "approval",
        "retry",
        "dead_letter",
        "finalize",
    ]
    for node in nodes_set:
        if node in valid_nodes:
            mermaid += f"    class {node} visited;\n"

    if visited_nodes and visited_nodes[-1] in ["answer", "clarify", "dead_letter", "finalize"]:
        mermaid += "    class finalize,END terminal;\n"

    return mermaid


def render_mermaid_html(mermaid_code: str) -> None:
    """Render Mermaid diagram inside an isolated iframe."""
    html_code = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>
        <script>
            mermaid.initialize({{
                startOnLoad: true,
                theme: 'dark',
                themeVariables: {{
                    darkMode: true,
                    background: '#0E1117',
                    primaryColor: '#1E1E2F',
                    primaryTextColor: '#FFFFFF',
                    primaryBorderColor: '#4A4A6A',
                    lineColor: '#9CA3AF',
                    secondaryColor: '#10B981',
                    tertiaryColor: '#F59E0B'
                }}
            }});
        </script>
        <style>
            body {{
                background-color: #0E1117;
                color: #FAFAFA;
                font-family: sans-serif;
                display: flex;
                justify-content: center;
                align-items: center;
                margin: 0;
                padding: 10px;
            }}
            .mermaid {{
                width: 100%;
                text-align: center;
            }}
        </style>
    </head>
    <body>
        <div class="mermaid">
{mermaid_code}
        </div>
    </body>
    </html>
    """
    st.components.v1.html(html_code, height=520, scrolling=True)


# Main App Layout
st.title("🤖 LangGraph Support Ticket Orchestrator")
st.caption("Day 08 Lab — Agentic Orchestration with State Management, Retry Loops & HITL Approval")

# Sidebar
st.sidebar.header("⚙️ Configuration & Scenario")
model_label = os.getenv("LLM_MODEL", "default")
st.sidebar.info(f"**LLM Provider**: {get_masked_key_status()}\n\n**Model**: `{model_label}`")

scenarios_file = "data/sample/scenarios.jsonl"
preset_scenarios = []
if os.path.exists(scenarios_file):
    preset_scenarios = load_scenarios(scenarios_file)

scenario_options = ["(Custom Query)"] + [f"{s.id} — {s.query[:35]}..." for s in preset_scenarios]
selected_option = st.sidebar.selectbox("Select Test Scenario:", scenario_options)

# Initialize Session State
if "graph_state" not in st.session_state:
    st.session_state.graph_state = None
if "visited_nodes" not in st.session_state:
    st.session_state.visited_nodes = []
if "is_interrupted" not in st.session_state:
    st.session_state.is_interrupted = False
if "thread_id" not in st.session_state:
    st.session_state.thread_id = f"thread-ui-{int(time.time())}"
if "checkpointer" not in st.session_state:
    st.session_state.checkpointer = MemorySaver()

# Determine initial query and settings
default_query = "How do I reset my password?"
default_route = Route.SIMPLE
default_req_approval = False
default_max_attempts = 3

if selected_option != "(Custom Query)":
    sc_idx = scenario_options.index(selected_option) - 1
    selected_sc = preset_scenarios[sc_idx]
    default_query = selected_sc.query
    default_route = selected_sc.expected_route
    default_req_approval = selected_sc.requires_approval
    default_max_attempts = selected_sc.max_attempts

col_input, col_meta = st.columns([3, 1])
with col_input:
    user_query = st.text_area("Support Ticket Query:", value=default_query, height=80)
with col_meta:
    enable_hitl_interrupt = st.checkbox("Enable Real HITL Interrupt", value=default_req_approval)
    max_retries = int(
        st.number_input("Max Attempts:", min_value=1, max_value=5, value=default_max_attempts)
    )

# Execution Controls
col_btn1, col_btn2, _ = st.columns([1, 1, 4])
with col_btn1:
    run_clicked = st.button("🚀 Run Workflow", type="primary", use_container_width=True)
with col_btn2:
    reset_clicked = st.button("🔄 Reset State", use_container_width=True)

if reset_clicked:
    st.session_state.graph_state = None
    st.session_state.visited_nodes = []
    st.session_state.is_interrupted = False
    st.session_state.thread_id = f"thread-ui-{int(time.time())}"
    st.session_state.checkpointer = MemorySaver()
    st.rerun()

# Build Graph Instance
os.environ["LANGGRAPH_INTERRUPT"] = "true" if enable_hitl_interrupt else "false"
graph = build_graph(checkpointer=st.session_state.checkpointer)
thread_config = {"configurable": {"thread_id": st.session_state.thread_id}}

if run_clicked and user_query.strip():
    scenario = Scenario(
        id=f"ui_{int(time.time())}",
        query=user_query.strip(),
        expected_route=default_route,
        requires_approval=enable_hitl_interrupt,
        max_attempts=max_retries,
    )
    init_st = initial_state(scenario)

    with st.spinner("Executing LangGraph agent nodes..."):
        try:
            res = graph.invoke(init_st, config=thread_config)
            cur_state = graph.get_state(thread_config)

            if len(cur_state.tasks) > 0 and cur_state.next == ("approval",):
                st.session_state.is_interrupted = True
                st.session_state.graph_state = cur_state.values
                events = cur_state.values.get("events", [])
                st.session_state.visited_nodes = [e["node"] for e in events]
            else:
                st.session_state.is_interrupted = False
                st.session_state.graph_state = res
                events = res.get("events", [])
                st.session_state.visited_nodes = [e["node"] for e in events]
        except Exception as e:
            st.error(f"Execution Error: {e}")

# Resume from Interrupt
if st.session_state.is_interrupted:
    st.warning("⚠️ **Human-in-the-Loop Interrupt Active**: Graph paused at `approval` node.")
    proposed = st.session_state.graph_state.get(
        "proposed_action", "Action requiring human review."
    )
    st.info(f"**Proposed Action**: {proposed}")

    col_app1, col_app2, col_comm = st.columns([1, 1, 2])
    with col_comm:
        reviewer_comment = st.text_input(
            "Reviewer Notes / Comment:", value="Approved via Streamlit UI"
        )
    with col_app1:
        if st.button("✅ Approve Action", type="primary", use_container_width=True):
            with st.spinner("Resuming workflow with approval..."):
                res = graph.invoke(
                    Command(
                        resume={
                            "approved": True,
                            "reviewer": "human_reviewer",
                            "comment": reviewer_comment,
                        }
                    ),
                    config=thread_config,
                )
                st.session_state.is_interrupted = False
                st.session_state.graph_state = res
                events = res.get("events", [])
                st.session_state.visited_nodes = [e["node"] for e in events]
                st.rerun()
    with col_app2:
        if st.button("❌ Reject Action", use_container_width=True):
            with st.spinner("Resuming workflow with rejection..."):
                res = graph.invoke(
                    Command(
                        resume={
                            "approved": False,
                            "reviewer": "human_reviewer",
                            "comment": reviewer_comment,
                        }
                    ),
                    config=thread_config,
                )
                st.session_state.is_interrupted = False
                st.session_state.graph_state = res
                events = res.get("events", [])
                st.session_state.visited_nodes = [e["node"] for e in events]
                st.rerun()

# Layout: Visualization & Output Tabs
st.divider()
col_graph, col_results = st.columns([1, 1])

with col_graph:
    st.subheader("🗺️ Architecture & Active Path Visualization")
    mermaid_code = generate_highlighted_mermaid(st.session_state.visited_nodes)
    render_mermaid_html(mermaid_code)

    if st.session_state.visited_nodes:
        path_str = " ➔ ".join([f"`{n}`" for n in st.session_state.visited_nodes])
        st.markdown(f"**Path Executed:** {path_str}")

with col_results:
    st.subheader("📊 State & Ticket Output")

    if st.session_state.graph_state:
        state = st.session_state.graph_state

        # Route & Risk Level Badges
        r_col1, r_col2, r_col3 = st.columns(3)
        r_col1.metric("Classified Route", state.get("route", "-"))
        r_col2.metric("Risk Level", state.get("risk_level", "-").upper())
        r_col3.metric(
            "Attempts Taken",
            f"{state.get('attempt', 0)} / {state.get('max_attempts', 3)}",
        )

        # Final Answer / Clarification
        if state.get("final_answer"):
            st.success("### 💬 Final Response\n" + str(state["final_answer"]))
        elif state.get("pending_question"):
            st.warning("### ❓ Clarification Requested\n" + str(state["pending_question"]))

        # Tool & Evaluation details
        if state.get("tool_results"):
            with st.expander("🛠️ Tool Results History", expanded=True):
                for i, tr in enumerate(state["tool_results"]):
                    st.code(f"Call #{i+1}: {tr}")

        # Approval details
        if state.get("approval"):
            with st.expander("🛡️ Approval Details"):
                st.json(state["approval"])
    else:
        st.info("👈 Select a scenario or enter a query, then click **Run Workflow** to simulate.")

# Audit Trail Section
st.divider()
st.subheader("📜 Complete Event Audit Trail (`LabEvent`)")
if st.session_state.graph_state and st.session_state.graph_state.get("events"):
    events_data = []
    for evt in st.session_state.graph_state["events"]:
        events_data.append(
            {
                "Node": evt.get("node"),
                "Event Type": evt.get("event_type"),
                "Message": evt.get("message"),
                "Latency (ms)": evt.get("latency_ms", 0),
                "Metadata": str(evt.get("metadata", {})),
            }
        )
    st.dataframe(events_data, use_container_width=True)

    with st.expander("🔍 Raw JSON State Inspector"):
        st.json(st.session_state.graph_state)
else:
    st.caption("No event logs recorded yet.")
