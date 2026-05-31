"""Streamlit results explorer for conversation2SQL evaluation runs."""

from __future__ import annotations

import os
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from charts import aptitude_unreliability_box
from charts import reliability_explainer
from loader import RunData
from loader import load_run
from loader import list_runs
from render import render_conversation


# Reads RESULTS_ROOT env var (exported by evaluate.sh); falls back to ../results
# relative to this file so the app works regardless of the working directory.
RESULTS_ROOT = Path(os.environ.get("RESULTS_ROOT", str(Path(__file__).parent.parent / "results")))

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

# Sidebar: cascading run selector (date → run)
with st.sidebar:
    st.header("Select Run")
    runs_tree = list_runs(RESULTS_ROOT)
    if not runs_tree:
        st.error(f"No results found in `{RESULTS_ROOT.resolve()}`")
        st.stop()
    def _fmt_run(key: str) -> str:
        # "16-23-07/no_tool__Qwen3.5-9B__ddl" → "16:23:07  no_tool__Qwen3.5-9B__ddl"
        # "16_23_07__no_tool__Qwen3.5-9B__ddl" → kept as-is (old layout)
        if "/" in key:
            time_part, slug = key.split("/", 1)
            return f"{time_part.replace('-', ':')}  {slug}"
        return key

    dates = list(runs_tree.keys())
    date = st.selectbox("Date", dates)
    run_names = runs_tree[date]
    run_key = st.selectbox("Run", run_names, index=0, format_func=_fmt_run)

run_path = RESULTS_ROOT / date / run_key
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

tab_results, tab_config = st.tabs(["Results", "Config"])

with tab_config:
    if run.config:
        st.json(run.config)
    else:
        st.info("No config.yaml found for this run.")

with tab_results:
    # Stats panel: five metric cards
    stats = run.stats
    pct = stats.n_passed / max(stats.n_total, 1) * 100
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Accuracy", f"{stats.n_passed} / {stats.n_total} ({pct:.1f}%)")
    c2.metric("Avg Input Tokens", f"{stats.avg_input_tokens:,.0f}")
    c3.metric("Avg Output Tokens", f"{stats.avg_output_tokens:,.0f}")
    c4.metric("Avg Cost", f"${stats.avg_cost:.5f}")
    c5.metric("Avg Budget Remaining", f"{stats.avg_budget_remaining:.1f}")

    # Reliability metrics (multi-iteration runs only)
    if run.n_iterations >= 2 and stats.reliability is not None:
        rel = stats.reliability
        st.subheader(f"Reliability ({run.n_iterations} iterations)")
        r1, r2, r3, r4 = st.columns(4)
        r1.metric("Average P̄", f"{rel.avg_performance * 100:.1f}%")
        r2.metric("Aptitude A⁹⁰", f"{rel.aptitude * 100:.1f}%")
        r3.metric("Unreliability U₁₀⁹⁰", f"{rel.unreliability * 100:.1f}%")
        r4.metric("Reliability R", f"{rel.reliability * 100:.1f}%")
        reliability_explainer()
        if rel.percentiles:
            model_label = predictor.get("model_name", "this run") if predictor else "this run"
            st.plotly_chart(
                aptitude_unreliability_box({model_label: rel}),
                use_container_width=True,
            )
        if rel.passk:
            passk_df = pd.DataFrame(
                [{"k": k, "pass@k": v} for k, v in sorted(rel.passk.items())]
            )
            st.altair_chart(
                alt.Chart(passk_df).mark_line(point=True).encode(
                    x=alt.X("k:Q", scale=alt.Scale(nice=False)),
                    y=alt.Y("pass@k:Q", scale=alt.Scale(domain=[0, 1])),
                ).properties(height=250, title="pass@k"),
                use_container_width=True,
            )
    elif run.n_iterations == 1:
        st.caption("Single-iteration run — reliability metrics not applicable.")

    # Charts: accuracy by database + error distribution + tool usage (hidden for no_tool baseline)
    has_tools = bool(stats.tool_usage)
    chart_cols = st.columns(3 if has_tools else 2)

    with chart_cols[0]:
        st.subheader("Accuracy by Database")
        db_df = pd.DataFrame(
            [{"Database": k, "Pass Rate": v} for k, v in stats.accuracy_by_database.items()]
        ).sort_values("Database")
        st.altair_chart(
            alt.Chart(db_df).mark_bar().encode(
                x=alt.X("Database:N", sort=None),
                y=alt.Y("Pass Rate:Q", scale=alt.Scale(domain=[0, 1])),
            ).properties(height=300),
            use_container_width=True,
        )

    with chart_cols[1]:
        st.subheader("Error Distribution")
        err_df = pd.DataFrame(
            [{"Error Class": k, "Count": v} for k, v in stats.error_distribution.items()]
        ).sort_values("Error Class")
        st.altair_chart(
            alt.Chart(err_df).mark_bar().encode(
                x=alt.X("Error Class:N", sort=None),
                y=alt.Y("Count:Q", scale=alt.Scale(domain=[0, stats.n_total])),
            ).properties(height=300),
            use_container_width=True,
        )

    if has_tools:
        with chart_cols[2]:
            st.subheader("Tool Usage")
            tool_df = pd.DataFrame(
                [{"Tool": k, "Calls": v} for k, v in stats.tool_usage.most_common()]
            ).set_index("Tool")
            st.bar_chart(tool_df)

    st.divider()

    st.subheader("Tasks")

    if run.n_iterations >= 2:
        # ── Per-instance view (multi-iteration) ──────────────────────────────
        f1, f2 = st.columns([1, 3])
        with f1:
            pass_filter = st.radio(
                "Pass / Fail", ["All", "All Passed", "All Failed"], horizontal=True
            )
        with f2:
            search = st.text_input("Search question", placeholder="substring…")

        instance_rows = []
        for iid, group in run.groups.items():
            n = len(group)
            c = sum(1 for r in group if r.get("execution_accuracy"))
            q = _question(group[0])
            instance_rows.append(
                {
                    "instance_id": iid,
                    "database": group[0].get("selected_database", ""),
                    "Question": (q[:80] + "…") if len(q) > 80 else q,
                    "pass_rate": f"{c}/{n}",
                    "_c": c,
                    "_n": n,
                    "avg_cost": sum(r.get("total_cost") or 0 for r in group) / n,
                }
            )

        if pass_filter == "All Passed":
            instance_rows = [r for r in instance_rows if r["_c"] == r["_n"]]
        elif pass_filter == "All Failed":
            instance_rows = [r for r in instance_rows if r["_c"] == 0]
        if search:
            instance_rows = [
                r for r in instance_rows if search.lower() in r["Question"].lower()
            ]

        if not instance_rows:
            st.info("No tasks match the current filters.")
            st.stop()

        df = pd.DataFrame(
            [
                {
                    "#": i + 1,
                    "instance_id": r["instance_id"],
                    "database": r["database"],
                    "Question": r["Question"],
                    "pass_rate": r["pass_rate"],
                    "avg_cost": r["avg_cost"],
                }
                for i, r in enumerate(instance_rows)
            ]
        )
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

        selected_iid = instance_rows[selected_indices[0]]["instance_id"]
        group = run.groups[selected_iid]
        st.divider()
        iters = [r.get("iteration", 0) for r in group]
        chosen = st.selectbox("Iteration", iters, format_func=lambda i: f"iteration {i}")
        selected_record = next(r for r in group if r.get("iteration", 0) == chosen)
        render_conversation(selected_record)
    else:
        # ── Per-record view (single iteration) — unchanged behavior ──────────
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
