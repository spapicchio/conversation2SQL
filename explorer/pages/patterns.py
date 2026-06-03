"""Single-run tool-interaction anti-pattern diagnosis page."""

from __future__ import annotations

import os
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from loader import RunData, list_runs, load_run
from patterns import PATTERN_CATALOG, positional_tool_distribution
from render import render_conversation


RESULTS_ROOT = Path(os.environ.get("RESULTS_ROOT", str(Path(__file__).parent.parent.parent / "results")))

_LABELS = dict(PATTERN_CATALOG)  # name -> human label

# Per-pattern explanation: what it flags and how it is detected from the tool-call trace.
_DESCRIPTIONS: dict[str, str] = {
    "blind_submit": "Agent calls `submit_sql` without ever running `execute_sql` first — it never validated the query against the DB. **Detected:** a `submit_sql` event appears with no preceding `execute_sql`.",
    "repeated_identical_call": "Agent issues the exact same call twice — a sign it isn't tracking what it already did. **Detected:** the same tool name + identical arguments occurs ≥2 times (SQL compared with whitespace collapsed).",
    "submit_after_error": "Agent submits a query that already failed when run. **Detected:** the submitted SQL is identical (whitespace-collapsed) to an `execute_sql` whose result had an error status.",
    "unrecovered_error_loop": "Agent keeps re-running broken SQL without recovering. **Detected:** ≥3 consecutive `execute_sql` errors with no successful run in between (the streak resets on any success).",
    "kb_blind": "External knowledge was available but the agent ignored it. **Detected:** the task has KB entries (`masked_agent_kb`/`gt_knowledge_base`) yet `get_knowledge_definition` is never called.",
    "budget_death": "Agent runs out of patience budget before landing a successful answer. **Detected:** no successful `submit_sql`, and either a tool message reports the budget exhausted or `updated_user_patience` ≤ 0.",
    "no_submission": "Conversation ends without the agent ever answering. **Detected:** no `submit_sql` call appears anywhere in the trace.",
}


@st.cache_data
def _load_run_cached(path_str: str) -> RunData:
    return load_run(Path(path_str))


def _question(r: dict) -> str:
    return r.get("amb_user_query") or r.get("not_ambiguos_query", "")


st.set_page_config(page_title="Tool Anti-Patterns", layout="wide")
st.title("Tool-Interaction Anti-Patterns")

# ── Sidebar: run selector (mirrors app.py) ──────────────────────────────────────
runs_tree = list_runs(RESULTS_ROOT)
if not runs_tree:
    st.error(f"No results found in `{RESULTS_ROOT.resolve()}`")
    st.stop()

with st.sidebar:
    st.header("Select Run")
    date = st.selectbox("Date", list(runs_tree.keys()))
    run_key = st.selectbox("Run", runs_tree[date], index=0)

run = _load_run_cached(str(RESULTS_ROOT / date / run_key))
stats = run.stats

# ── Headline: anti-pattern frequency (sample-average) ───────────────────────────
st.subheader("Anti-pattern frequency (sample-average)")
st.caption(f"Fraction of instances hitting each pattern · clean: {stats.clean_fraction * 100:.1f}%")
with st.expander("What do these anti-patterns mean?"):
    st.caption(
        "Each is a deterministic, per-conversation flag computed from the tool-call trace "
        "(no LLM). Frequency above is the sample-average: the mean over instances of the "
        "fraction of that instance's samples hitting the pattern."
    )
    for name, label in PATTERN_CATALOG:
        st.markdown(f"**{label}** — {_DESCRIPTIONS[name]}")
freq_df = pd.DataFrame(
    [{"Pattern": _LABELS[name], "Frequency": stats.pattern_frequency.get(name, 0.0)}
     for name, _ in PATTERN_CATALOG]
).sort_values("Frequency", ascending=False)
st.altair_chart(
    alt.Chart(freq_df).mark_bar().encode(
        x=alt.X("Frequency:Q", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%")),
        y=alt.Y("Pattern:N", sort="-x"),
    ).properties(height=260),
    use_container_width=True,
)

# ── Positional tool distribution (100%-stacked) ─────────────────────────────────
st.subheader("Tool by call position")
pos_df = positional_tool_distribution(run.groups)
if not pos_df.empty:
    pos_order = [str(p) for p in range(1, 8)] + ["≥8"]
    st.altair_chart(
        alt.Chart(pos_df).mark_bar().encode(
            x=alt.X("position:N", sort=pos_order, title="Call position"),
            y=alt.Y("share:Q", stack="normalize", axis=alt.Axis(format="%"), title="Share"),
            color=alt.Color("tool:N", title="Tool"),
            order=alt.Order("tool:N"),
        ).properties(height=320),
        use_container_width=True,
    )

# ── Drill-down: pick a pattern, list flagged tasks, inspect ─────────────────────
st.divider()
st.subheader("Drill-down")
pattern_label = st.selectbox("Pattern", [lbl for _, lbl in PATTERN_CATALOG])
pattern_name = next(name for name, lbl in PATTERN_CATALOG if lbl == pattern_label)

flagged = [
    r for r in run.records
    if any(h.name == pattern_name for h in r.get("_pattern_hits", []))
]
if not flagged:
    st.info("No records hit this pattern.")
    st.stop()

rows = []
for i, r in enumerate(flagged):
    hit = next(h for h in r["_pattern_hits"] if h.name == pattern_name)
    q = _question(r)
    rows.append({
        "#": i + 1,
        "instance_id": r.get("instance_id", ""),
        "iteration": r.get("iteration", 0),
        "database": r.get("selected_database", ""),
        "Question": (q[:70] + "…") if len(q) > 70 else q,
        "Evidence": hit.detail,
        "Accuracy": "✓" if r.get("execution_accuracy") else "✗",
    })
event = st.dataframe(
    pd.DataFrame(rows), use_container_width=True, hide_index=True,
    on_select="rerun", selection_mode="single-row",
)
selected = event.selection.rows
if not selected:
    st.stop()

record = flagged[selected[0]]
hit = next(h for h in record["_pattern_hits"] if h.name == pattern_name)
st.divider()
st.warning(f"**{pattern_label}** — {hit.detail}  ·  flagged messages: {hit.message_indices}")
render_conversation(record, highlight_indices=set(hit.message_indices))
