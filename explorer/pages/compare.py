"""Streamlit comparison page for conversation2SQL evaluation runs."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from loader import RunData
from loader import join_runs
from loader import list_runs
from loader import load_run
from render import render_conversation


RESULTS_ROOT = Path("results")


@st.cache_data
def _load_run_cached(path_str: str) -> RunData:
    return load_run(Path(path_str))


# ── Page config ────────────────────────────────────────────────────────────────

st.set_page_config(page_title="Run Comparison", layout="wide")
st.title("Run Comparison")

# ── Sidebar: multi-run selector ────────────────────────────────────────────────

runs_tree = list_runs(RESULTS_ROOT)
if not runs_tree:
    st.error(f"No results found in `{RESULTS_ROOT.resolve()}`")
    st.stop()

all_run_labels: dict[str, Path] = {}
for date, run_names in runs_tree.items():
    for run_key in run_names:
        label = f"{date} / {run_key}"
        all_run_labels[label] = RESULTS_ROOT / date / run_key

with st.sidebar:
    st.header("Select Runs")
    selected_labels: list[str] = st.multiselect(
        "Runs to compare",
        options=list(all_run_labels.keys()),
        default=[],
    )

if len(selected_labels) < 2:
    st.warning("Select at least 2 runs from the sidebar to compare.")
    st.stop()

runs: dict[str, RunData] = {
    label: _load_run_cached(str(all_run_labels[label]))
    for label in selected_labels
}

# ── Aggregate stats panel ──────────────────────────────────────────────────────

stat_cols = st.columns(len(runs))
for col, (label, run) in zip(stat_cols, runs.items()):
    with col:
        st.markdown(f"**{label}**")
        predictor = run.config.get("predictor", {})
        model = predictor.get("model_name", "unknown")
        st.markdown(f"`{model}`")
        stats = run.stats
        pct = stats.n_passed / max(stats.n_total, 1) * 100
        st.metric("Accuracy", f"{stats.n_passed}/{stats.n_total} ({pct:.1f}%)")
        st.metric("Avg Input Tokens", f"{stats.avg_input_tokens:,.0f}")
        st.metric("Avg Output Tokens", f"{stats.avg_output_tokens:,.0f}")
        st.metric("Avg Cost", f"${stats.avg_cost:.5f}")
        st.metric("Avg Budget Remaining", f"{stats.avg_budget_remaining:.1f}")
        if run.n_iterations >= 2 and run.stats.reliability is not None:
            rel = run.stats.reliability
            st.metric("Average P̄", f"{rel.avg_performance * 100:.1f}%")
            st.metric("Aptitude A⁹⁰", f"{rel.aptitude * 100:.1f}%")
            st.metric("Unreliability U₁₀⁹⁰", f"{rel.unreliability * 100:.1f}%")
            st.metric("Reliability R", f"{rel.reliability * 100:.1f}%")
            if rel.passk:
                st.metric("pass@N", f"{rel.passk[max(rel.passk)] * 100:.1f}%")

st.divider()

# ── Charts ─────────────────────────────────────────────────────────────────────

import altair as alt

db_cols = st.columns(len(runs))
for col, (_, run) in zip(db_cols, runs.items()):
    with col:
        st.subheader("Accuracy by Database")
        db_df = pd.DataFrame(
            [{"Database": k, "Pass Rate": v} for k, v in run.stats.accuracy_by_database.items()]
        ).sort_values("Database")
        st.altair_chart(
            alt.Chart(db_df).mark_bar().encode(
                x=alt.X("Database:N", sort=None),
                y=alt.Y("Pass Rate:Q", scale=alt.Scale(domain=[0, 1])),
            ).properties(height=300),
            use_container_width=True,
        )

max_err_count = max(
    (sum(run.stats.error_distribution.values()) for run in runs.values()),
    default=1,
)
err_cols = st.columns(len(runs))
for col, (_, run) in zip(err_cols, runs.items()):
    with col:
        st.subheader("Error Distribution")
        err_df = pd.DataFrame(
            [{"Error Class": k, "Count": v} for k, v in run.stats.error_distribution.items()]
        ).sort_values("Error Class")
        st.altair_chart(
            alt.Chart(err_df).mark_bar().encode(
                x=alt.X("Error Class:N", sort=None),
                y=alt.Y("Count:Q", scale=alt.Scale(domain=[0, max_err_count])),
            ).properties(height=300),
            use_container_width=True,
        )

has_tools = any(bool(run.stats.tool_usage) for run in runs.values())
if has_tools:
    tool_cols = st.columns(len(runs))
    for col, (_, run) in zip(tool_cols, runs.items()):
        with col:
            st.subheader("Tool Usage")
            tool_df = pd.DataFrame(
                [{"Tool": k, "Calls": v} for k, v in run.stats.tool_usage.most_common()]
            ).set_index("Tool")
            st.bar_chart(tool_df)

st.divider()

# ── Per-task table ─────────────────────────────────────────────────────────────

st.subheader("Per-Task Comparison")

run_labels = list(runs.keys())

f1, f2 = st.columns([2, 3])
with f1:
    agreement_filter = st.radio(
        "Filter",
        ["All", "Disagreement only", "All Passed", "All Failed"],
        horizontal=True,
    )
with f2:
    search = st.text_input("Search question", placeholder="substring…")

task_df = join_runs(runs)


def _cell_state(val: str) -> str | None:
    """Map a 'c/n' cell to 'pass' (c==n), 'fail' (c==0), 'partial', or None for '—'."""
    if val == "—":
        return None
    c_str, n_str = val.split("/")
    c, n = int(c_str), int(n_str)
    if c == n:
        return "pass"
    if c == 0:
        return "fail"
    return "partial"


if agreement_filter == "Disagreement only":
    def _disagrees(row: pd.Series) -> bool:
        states = [s for c in run_labels if (s := _cell_state(row[c])) is not None]
        return len(set(states)) > 1 if states else False
    task_df = task_df[task_df.apply(_disagrees, axis=1)]
elif agreement_filter == "All Passed":
    task_df = task_df[task_df[run_labels].apply(
        lambda row: all(_cell_state(v) == "pass" for v in row), axis=1)]
elif agreement_filter == "All Failed":
    task_df = task_df[task_df[run_labels].apply(
        lambda row: all(_cell_state(v) == "fail" for v in row), axis=1)]

if search:
    task_df = task_df[task_df["Question"].str.lower().str.contains(search.lower(), na=False)]

if task_df.empty:
    st.info("No tasks match the current filters.")
    st.stop()

event = st.dataframe(
    task_df,
    use_container_width=True,
    hide_index=True,
    on_select="rerun",
    selection_mode="single-row",
)

selected_indices = event.selection.rows
if not selected_indices:
    st.stop()

selected_instance_id = task_df.iloc[selected_indices[0]]["instance_id"]

st.divider()

# ── Conversation viewer ────────────────────────────────────────────────────────

st.subheader(f"Conversation: `{selected_instance_id}`")

if len(selected_labels) == 2:
    run_a_label = selected_labels[0]
    run_b_label = selected_labels[1]
else:
    sel_cols = st.columns(2)
    with sel_cols[0]:
        run_a_label = st.selectbox("Run A", selected_labels, index=0, key="run_a")
    with sel_cols[1]:
        run_b_label = st.selectbox("Run B", selected_labels, index=min(1, len(selected_labels) - 1), key="run_b")

conv_cols = st.columns(2)
for col, label in zip(conv_cols, [run_a_label, run_b_label]):
    with col:
        st.markdown(f"**{label}**")
        group = runs[label].groups.get(selected_instance_id, [])
        if not group:
            st.info("Task not found in this run.")
            continue
        if len(group) > 1:
            iters = [r.get("iteration", 0) for r in group]
            chosen = st.selectbox(
                "Iteration",
                iters,
                format_func=lambda i: f"iteration {i}",
                key=f"iter_{label}",
            )
            record = next(r for r in group if r.get("iteration", 0) == chosen)
        else:
            record = group[0]
        render_conversation(record)
