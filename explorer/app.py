"""Streamlit results explorer for conversation2SQL evaluation runs."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from explorer.loader import RunData, load_run, list_runs

RESULTS_ROOT = Path("results")


# ── Cached loader (keyed on path string so Streamlit can hash it) ──────────────

@st.cache_data
def _load_run_cached(path_str: str) -> RunData:
    return load_run(Path(path_str))


# ── Helper functions ───────────────────────────────────────────────────────────

def _question(r: dict) -> str:
    return r.get("amb_user_query") or r.get("not_ambiguos_query", "")


def _render_conversation(record: dict) -> None:
    acc = record.get("execution_accuracy", False)
    badge = "✓ PASS" if acc else "✗ FAIL"
    st.subheader(
        f"Conversation: `{record.get('instance_id', '')}` · "
        f"db: `{record.get('selected_database', '')}` · **{badge}**"
    )

    col1, col2 = st.columns(2)
    with col1:
        with st.expander("Ground-truth SQL"):
            sol = record.get("sol_sql", "")
            if isinstance(sol, list):
                sol = "\n\n".join(sol)
            st.code(sol, language="sql")
    with col2:
        with st.expander("Predicted SQL"):
            st.code(record.get("predicted_sql", "") or "", language="sql")

    for msg in record.get("messages", []):
        role = msg.get("role")

        if role in ("user", "system"):
            with st.expander("System prompt — click to expand"):
                st.text(msg.get("content", ""))

        elif role == "human":
            with st.chat_message("user"):
                st.markdown(msg.get("content", ""))

        elif role == "ai":
            with st.chat_message("assistant"):
                content = msg.get("content", "")
                if content:
                    st.markdown(content)
                st.caption(
                    f"tokens: {msg.get('prompt_tokens', 0)}↑ {msg.get('completion_tokens', 0)}↓"
                    f" | cost: ${msg.get('cost_usd', 0):.5f}"
                    f" | finish: {msg.get('finish_reason', 'unknown')}"
                )
                for tc in msg.get("tool_calls", []):
                    tool_name = tc.get("tool_name", "unknown")
                    args = tc.get("arguments", {})
                    args_str = json.dumps(args, ensure_ascii=False)
                    label = (
                        f"🔧 {tool_name}({args_str[:60]}…)"
                        if len(args_str) > 60
                        else f"🔧 {tool_name}({args_str})"
                    )
                    with st.expander(label):
                        st.json(args)

        elif role == "tool":
            tool_name = msg.get("tool_name", "tool")
            status = msg.get("status", "")
            status_badge = "✓" if status == "success" else "✗"
            with st.chat_message("user"):
                st.markdown(f"**{tool_name}** {status_badge}")
                content = msg.get("content", "")
                with st.expander("Tool output — click to expand"):
                    if isinstance(content, dict):
                        st.json(content)
                    else:
                        st.text(str(content))


# ── App layout ─────────────────────────────────────────────────────────────────

st.set_page_config(page_title="Results Explorer", layout="wide")
st.title("Results Explorer")

# Sidebar: cascading run selector
with st.sidebar:
    st.header("Select Run")
    runs = list_runs(RESULTS_ROOT)
    if not runs:
        st.error(f"No results found in `{RESULTS_ROOT.resolve()}`")
        st.stop()
    baseline = st.selectbox("Baseline", list(runs.keys()))
    dates = list(runs[baseline].keys())
    date = st.selectbox("Date", dates)
    times = runs[baseline][date]
    time_key = st.selectbox("Time", times, index=0)

run_path = RESULTS_ROOT / baseline / date / time_key
run = _load_run_cached(str(run_path))

with st.sidebar:
    predictor = run.config.get("predictor", {})
    if predictor:
        st.caption(
            f"Model: `{predictor.get('model_name', 'unknown')}`  \n"
            f"Temp: `{predictor.get('temperature', '?')}`"
        )

if run.malformed_count:
    st.warning(f"{run.malformed_count} malformed line(s) skipped in {run.source_file}.")

# Stats panel: four metric cards
stats = run.stats
pct = stats.n_passed / max(stats.n_total, 1) * 100
c1, c2, c3, c4 = st.columns(4)
c1.metric("Accuracy", f"{stats.n_passed} / {stats.n_total} ({pct:.1f}%)")
c2.metric("Avg Tokens", f"{stats.avg_tokens:,.0f}")
c3.metric("Avg Cost", f"${stats.avg_cost:.5f}")
c4.metric("Avg Budget Remaining", f"{stats.avg_budget_remaining:.1f}")

# Charts: accuracy by category + tool usage (hidden for no_tool baseline)
has_tools = bool(stats.tool_usage)
chart_cols = st.columns(2 if has_tools else 1)

with chart_cols[0]:
    st.subheader("Accuracy by Category")
    if stats.accuracy_by_category:
        cat_df = pd.DataFrame(
            [{"Category": k, "Pass Rate": v} for k, v in stats.accuracy_by_category.items()]
        ).set_index("Category")
        st.bar_chart(cat_df)
    else:
        st.info("No category data available.")

if has_tools:
    with chart_cols[1]:
        st.subheader("Tool Usage")
        tool_df = pd.DataFrame(
            [{"Tool": k, "Calls": v} for k, v in stats.tool_usage.most_common()]
        ).set_index("Tool")
        st.bar_chart(tool_df)

st.divider()

# Task list with filters
st.subheader("Tasks")
f1, f2, f3 = st.columns([1, 2, 3])
with f1:
    pass_filter = st.radio("Pass / Fail", ["All", "Passed", "Failed"], horizontal=True)
with f2:
    all_cats = sorted({r.get("category", "Unknown") for r in run.records})
    cat_filter = st.multiselect("Category", all_cats, default=all_cats)
with f3:
    search = st.text_input("Search question", placeholder="substring…")

filtered = run.records
if pass_filter == "Passed":
    filtered = [r for r in filtered if r.get("execution_accuracy")]
elif pass_filter == "Failed":
    filtered = [r for r in filtered if not r.get("execution_accuracy")]
if cat_filter:
    filtered = [r for r in filtered if r.get("category", "Unknown") in cat_filter]
if search:
    filtered = [r for r in filtered if search.lower() in _question(r).lower()]

if not filtered:
    st.info("No tasks match the current filters.")
    st.stop()

rows = []
for i, r in enumerate(filtered):
    q = _question(r)
    rows.append({
        "#": i + 1,
        "instance_id": r.get("instance_id", ""),
        "database": r.get("selected_database", ""),
        "Question": (q[:80] + "…") if len(q) > 80 else q,
        "category": r.get("category", ""),
        "Accuracy": "✓" if r.get("execution_accuracy") else "✗",
        "tokens": r.get("total_tokens", 0),
        "cost": r.get("total_cost", 0.0),
    })

df = pd.DataFrame(rows)
event = st.dataframe(
    df,
    use_container_width=True,
    hide_index=True,
    on_select="rerun",
    selection_mode="single-row",
)

selected_indices = event.selection.rows
if not selected_indices:
    st.stop()

selected_record = filtered[selected_indices[0]]
st.divider()
_render_conversation(selected_record)
