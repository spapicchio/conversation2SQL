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
for baseline, dates in runs_tree.items():
    for date, times in dates.items():
        for time_key in times:
            label = f"{baseline} / {date} / {time_key}"
            all_run_labels[label] = RESULTS_ROOT / baseline / date / time_key

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

st.divider()

# ── Charts ─────────────────────────────────────────────────────────────────────

db_cols = st.columns(len(runs))
for col, (_, run) in zip(db_cols, runs.items()):
    with col:
        st.subheader("Accuracy by Database")
        db_df = pd.DataFrame(
            [{"Database": k, "Pass Rate": v} for k, v in run.stats.accuracy_by_database.items()]
        ).set_index("Database")
        st.bar_chart(db_df)

err_cols = st.columns(len(runs))
for col, (_, run) in zip(err_cols, runs.items()):
    with col:
        st.subheader("Error Distribution")
        err_df = pd.DataFrame(
            [{"Error Class": k, "Count": v} for k, v in run.stats.error_distribution.most_common()]
        ).set_index("Error Class")
        st.bar_chart(err_df)

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


def _is_pass(val: str) -> bool | None:
    if val == "✓":
        return True
    if val == "✗":
        return False
    return None


if agreement_filter == "Disagreement only":
    def _disagrees(row: pd.Series) -> bool:
        outcomes = [_is_pass(row[c]) for c in run_labels if _is_pass(row[c]) is not None]
        return len(set(outcomes)) > 1 if outcomes else False
    task_df = task_df[task_df.apply(_disagrees, axis=1)]
elif agreement_filter == "All Passed":
    task_df = task_df[task_df[run_labels].apply(lambda row: all(v == "✓" for v in row), axis=1)]
elif agreement_filter == "All Failed":
    task_df = task_df[task_df[run_labels].apply(lambda row: all(v == "✗" for v in row), axis=1)]

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
        record = next(
            (r for r in runs[label].records if r.get("instance_id") == selected_instance_id),
            None,
        )
        if record is None:
            st.info("Task not found in this run.")
        else:
            render_conversation(record)
