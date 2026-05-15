"""Streamlit results explorer for conversation2SQL evaluation runs."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from loader import RunData
from loader import load_run
from loader import list_runs
from render import render_conversation


RESULTS_ROOT = Path("results")

_GEN_PARAM_KEYS = (
    "temperature",
    "top_p",
    "top_k",
    "min_p",
    "presence_penalty",
    "repetition_penalty",
    "max_new_tokens",
    "reasoning_effort",
    "enable_thinking",
)


# ── Cached loader (keyed on path string so Streamlit can hash it) ──────────────


@st.cache_data
def _load_run_cached(path_str: str) -> RunData:
    return load_run(Path(path_str))


# ── Helper functions ───────────────────────────────────────────────────────────


def _question(r: dict) -> str:
    return r.get("amb_user_query") or r.get("not_ambiguos_query", "")


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
    dates = list(runs.keys())
    date = st.selectbox("Date", dates)
    times = runs[date]
    time_key = st.selectbox("Time", times, index=0)

run_path = RESULTS_ROOT / date / time_key
run = _load_run_cached(str(run_path))

with st.sidebar:
    predictor = run.config.get("predictor", {})
    if predictor:
        st.markdown(f"**Model:** `{predictor.get('model_name', 'unknown')}`")
        gen_params = {k: predictor[k] for k in _GEN_PARAM_KEYS if k in predictor}
        with st.expander("Generation params"):
            st.json(gen_params)

if run.malformed_count:
    st.warning(f"{run.malformed_count} malformed line(s) skipped in {run.source_file}.")

# Stats panel: five metric cards
stats = run.stats
pct = stats.n_passed / max(stats.n_total, 1) * 100
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Accuracy", f"{stats.n_passed} / {stats.n_total} ({pct:.1f}%)")
c2.metric("Avg Input Tokens", f"{stats.avg_input_tokens:,.0f}")
c3.metric("Avg Output Tokens", f"{stats.avg_output_tokens:,.0f}")
c4.metric("Avg Cost", f"${stats.avg_cost:.5f}")
c5.metric("Avg Budget Remaining", f"{stats.avg_budget_remaining:.1f}")

# Charts: accuracy by database + error distribution + tool usage (hidden for no_tool baseline)
has_tools = bool(stats.tool_usage)
chart_cols = st.columns(3 if has_tools else 2)

with chart_cols[0]:
    st.subheader("Accuracy by Database")
    db_df = pd.DataFrame(
        [{"Database": k, "Pass Rate": v} for k, v in stats.accuracy_by_database.items()]
    ).set_index("Database")
    st.bar_chart(db_df)

with chart_cols[1]:
    st.subheader("Error Distribution")
    err_df = pd.DataFrame(
        [
            {"Error Class": k, "Count": v}
            for k, v in stats.error_distribution.most_common()
        ]
    ).set_index("Error Class")
    st.bar_chart(err_df)

if has_tools:
    with chart_cols[2]:
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
    all_error_classes = sorted({r.get("_error_class", "Other") for r in run.records})
    error_filter = st.multiselect(
        "Error Class", all_error_classes, default=all_error_classes
    )
with f3:
    search = st.text_input("Search question", placeholder="substring…")

filtered = run.records
if pass_filter == "Passed":
    filtered = [r for r in filtered if r.get("execution_accuracy")]
elif pass_filter == "Failed":
    filtered = [r for r in filtered if not r.get("execution_accuracy")]
if error_filter:
    filtered = [r for r in filtered if r.get("_error_class", "Other") in error_filter]
if search:
    filtered = [r for r in filtered if search.lower() in _question(r).lower()]

if not filtered:
    st.info("No tasks match the current filters.")
    st.stop()

rows = []
for i, r in enumerate(filtered):
    q = _question(r)
    rows.append(
        {
            "#": i + 1,
            "instance_id": r.get("instance_id", ""),
            "database": r.get("selected_database", ""),
            "Question": (q[:80] + "…") if len(q) > 80 else q,
            "error_class": r.get("_error_class", ""),
            "Accuracy": "✓" if r.get("execution_accuracy") else "✗",
            "input_tokens": r.get("mean_prompt_tokens", 0),
            "output_tokens": r.get("mean_completion_tokens", 0),
            "cost": r.get("total_cost", 0.0),
        }
    )

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
render_conversation(selected_record)
