"""Streamlit comparison page for conversation2SQL evaluation runs."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from charts import aptitude_unreliability_box
from charts import reliability_bars
from charts import reliability_explainer
from loader import RunData
from loader import join_runs
from loader import list_runs
from loader import load_run
from render import render_conversation
from patterns import PATTERN_CATALOG


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

# Operational metrics → direction-aware highlighted table (metrics as rows, runs as columns).
# (formatter, higher_is_better | None to skip highlighting)
_OPERATIONAL_ROWS: list[tuple[str, Callable[[object], str], bool | None]] = [
    ("Model", lambda v: str(v), None),
    ("Accuracy (pass@1)", lambda v: f"{v * 100:.1f}%", True),
    ("Avg Input Tokens", lambda v: f"{v:,.0f}", False),
    ("Avg Output Tokens", lambda v: f"{v:,.0f}", False),
    ("Avg Input Tokens / call", lambda v: f"{v:,.0f}", False),
    ("Avg Output Tokens / call", lambda v: f"{v:,.0f}", False),
    ("Avg Model Calls", lambda v: f"{v:.1f}", False),
    ("Avg Cost", lambda v: f"${v:.5f}", False),
    ("Avg Budget Remaining", lambda v: f"{v:.1f}", True),
]


def _operational_value(run: RunData, metric: str):
    stats = run.stats
    return {
        "Model": run.config.get("predictor", {}).get("model_name", "unknown"),
        "Accuracy (pass@1)": stats.pass_at_1,
        "Avg Input Tokens": stats.avg_total_input_tokens,
        "Avg Output Tokens": stats.avg_total_output_tokens,
        "Avg Input Tokens / call": stats.avg_input_tokens,
        "Avg Output Tokens / call": stats.avg_output_tokens,
        "Avg Model Calls": stats.avg_model_calls,
        "Avg Cost": stats.avg_cost,
        "Avg Budget Remaining": stats.avg_budget_remaining,
    }[metric]


raw = pd.DataFrame(
    {label: {m: _operational_value(run, m) for m, _, _ in _OPERATIONAL_ROWS}
     for label, run in runs.items()}
)
display = pd.DataFrame(
    {label: {m: fmt(_operational_value(run, m)) for m, fmt, _ in _OPERATIONAL_ROWS}
     for label, run in runs.items()}
)


def _highlight_best(row: pd.Series) -> list[str]:
    metric = str(row.name)
    direction = {m: hib for m, _, hib in _OPERATIONAL_ROWS}[metric]
    if direction is None or len(runs) < 2:
        return [""] * len(row)
    numeric = raw.loc[metric]
    best = numeric.max() if direction else numeric.min()
    return ["background-color: rgba(0, 200, 0, 0.18)" if numeric[c] == best else ""
            for c in row.index]


st.caption(
    f"Accuracy is pass@1 over n={next(iter(runs.values())).stats.n_instances} "
    "instances (dataset size). Green = best per row."
)
st.dataframe(
    display.style.apply(_highlight_best, axis=1),
    use_container_width=True,
)

# Reliability percentages → grouped barplot (only meaningful with >=2 iterations).
bar_rels = {
    label: run.stats.reliability
    for label, run in runs.items()
    if run.n_iterations >= 2 and run.stats.reliability is not None
}
if bar_rels:
    st.subheader("Reliability metrics")
    st.plotly_chart(reliability_bars(bar_rels), use_container_width=True)

st.divider()

# ── Charts ─────────────────────────────────────────────────────────────────────

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

# ── Anti-pattern frequency (sample-average) ─────────────────────────────────────
st.divider()
st.subheader("Tool-interaction anti-patterns")
st.caption(
    "Sample-average rate among each pattern's *applicable* samples; `n` is applicable "
    "**instances** (the unit the rate is averaged over, not samples). Red = worst (highest) per row."
)

_pat_labels = dict(PATTERN_CATALOG)


def _pat_stat(run, name):
    return run.stats.pattern_stats.get(name)


pat_raw = pd.DataFrame(
    {label: {_pat_labels[name]: (s.rate if (s := _pat_stat(run, name)) else 0.0)
             for name, _ in PATTERN_CATALOG}
     for label, run in runs.items()}
)
pat_display = pd.DataFrame(
    {label: {_pat_labels[name]:
             (f"{s.rate * 100:.1f}% (n={s.applicable_instances})" if (s := _pat_stat(run, name)) else "—")
             for name, _ in PATTERN_CATALOG}
     for label, run in runs.items()}
)


def _highlight_worst(row: pd.Series) -> list[str]:
    if len(runs) < 2:
        return [""] * len(row)
    numeric = pat_raw.loc[str(row.name)]
    worst = numeric.max()  # higher anti-pattern rate is worse
    return ["background-color: rgba(220, 0, 0, 0.18)" if numeric[c] == worst and worst > 0 else ""
            for c in row.index]


st.dataframe(
    pat_display.style.apply(_highlight_worst, axis=1),
    use_container_width=True,
)

# ── Aptitude / Unreliability box plot (one box per run) ─────────────────────────

rels = {
    label: run.stats.reliability
    for label, run in runs.items()
    if run.stats.reliability is not None and run.stats.reliability.percentiles
}
if rels:
    st.subheader("Aptitude & Unreliability")
    reliability_explainer()
    st.plotly_chart(aptitude_unreliability_box(rels), use_container_width=True)

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
