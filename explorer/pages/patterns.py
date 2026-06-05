"""Single-run tool-interaction anti-pattern diagnosis page."""

from __future__ import annotations

import os
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from colors import color_scale
from loader import RunData, list_runs, load_run
from patterns import (
    PATTERN_CATALOG,
    first_submit_accuracy_by_quintile,
    per_iteration_pattern_rates,
    positional_tool_distribution,
    repeated_identical_tool_counts,
    tool_position_accuracy,
)
from render import render_conversation
from run_selector import select_run_sidebar


RESULTS_ROOT = Path(os.environ.get("RESULTS_ROOT", str(Path(__file__).parent.parent.parent / "results")))

_LABELS = dict(PATTERN_CATALOG)  # name -> human label

# Per-pattern explanation: what it flags and how it is detected from the tool-call trace.
_DESCRIPTIONS: dict[str, str] = {
    "blind_submit": "Agent calls `submit_sql` without ever running `execute_sql` first — it never validated the query against the DB. **Detected:** a `submit_sql` event appears with no preceding `execute_sql`.",
    "repeated_identical_call": "Agent issues the exact same call twice — a sign it isn't tracking what it already did. **Detected:** the same tool name + identical arguments occurs ≥2 times (SQL compared with whitespace collapsed).",
    "submit_after_error": "Agent submits a query that already failed when run. **Detected:** the submitted SQL is identical (whitespace-collapsed) to an `execute_sql` whose result had an error status.",
    "unrecovered_error_loop": "Agent keeps re-running broken SQL without recovering. **Detected:** ≥3 consecutive `execute_sql` errors with no successful run in between (the streak resets on any success).",
    "kb_blind": "The task's answer needs external knowledge but the agent ignored it. **Detected:** the gold solution depends on ≥1 KB entry (`gt_knowledge_base` non-empty) yet `get_knowledge_definition` is never called. *Only applicable to KB-needing tasks* — its rate is measured over that subset, not all samples.",
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

date, run_key = select_run_sidebar(runs_tree)

run = _load_run_cached(str(RESULTS_ROOT / date / run_key))
stats = run.stats

# ── Headline: anti-pattern rate among applicable samples ────────────────────────
st.subheader("Anti-pattern rate (sample-average, among applicable samples)")
st.caption(f"Rate over the samples each pattern could fire on · clean: {stats.clean_fraction * 100:.1f}%")
with st.expander("What do these anti-patterns mean?"):
    st.caption(
        "Each is a deterministic, per-conversation flag computed from the tool-call trace "
        "(no LLM). The rate is the sample-average — mean over instances of the fraction of "
        "that instance's samples hitting the pattern — but measured over each pattern's "
        "**applicable** samples (the denominator `n`), so patterns with different "
        "denominators (e.g. KB-blind only applies to KB-needing tasks) compare honestly."
    )
    for name, label in PATTERN_CATALOG:
        st.markdown(f"**{label}** — {_DESCRIPTIONS[name]}")

def _count(x: float) -> str:
    """Effective flagged-instance count: integer when whole, else 1 decimal."""
    return str(int(round(x))) if abs(x - round(x)) < 1e-9 else f"{x:.1f}"


# Aggregate (per-instance sample-average) rows. Always computed: it backs the
# default view and defines a stable pattern order for the per-iteration view.
# `n` is applicable *instances* (the rate's denominator), not samples; the
# numerator `flagged/n` is rate × instances — fractional under multiple iterations.
_freq_rows = []
for name, _ in PATTERN_CATALOG:
    s = stats.pattern_stats.get(name)
    rate, inst = (s.rate, s.applicable_instances) if s else (0.0, 0)
    _freq_rows.append({
        "Pattern": _LABELS[name],
        "Rate": rate,
        "n": inst,
        "Label": f"{rate * 100:.0f}%  ({_count(rate * inst)}/{inst})",
    })
freq_df = pd.DataFrame(_freq_rows).sort_values("Rate", ascending=False)
pattern_order = freq_df["Pattern"].tolist()  # shared sort for both views

# Toggle between the aggregate bar and a per-iteration breakdown. Only offered
# when the run has >1 iteration (otherwise the two views are identical).
view = "Aggregate"
if run.n_iterations > 1:
    view = st.radio(
        "View", ["Aggregate", "Per iteration"], horizontal=True, label_visibility="collapsed",
    )

if view == "Per iteration":
    iter_df = per_iteration_pattern_rates(run.records)
    iter_df["Iteration"] = iter_df["iteration"].map(lambda i: f"iter {i}")
    iter_order = [f"iter {i}" for i in sorted(iter_df["iteration"].unique())]
    iter_base = alt.Chart(iter_df).encode(
        y=alt.Y("pattern:N", sort=pattern_order, title=None),
        yOffset=alt.YOffset("Iteration:N", sort=iter_order),
        color=alt.Color("Iteration:N", sort=iter_order, title="Iteration",
                        scale=color_scale(iter_order, kind="iteration")),
        tooltip=[
            alt.Tooltip("pattern:N", title="Pattern"),
            alt.Tooltip("Iteration:N"),
            alt.Tooltip("rate:Q", title="Rate", format=".0%"),
            alt.Tooltip("n:Q", title="n"),
        ],
    )
    st.altair_chart(
        iter_base.mark_bar().encode(
            x=alt.X("rate:Q", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%"), title="Rate"),
        ).properties(height=max(40 * len(pattern_order), 260)),
        use_container_width=True,
    )
    st.caption(
        f"Each iteration's rate over its applicable records ({run.n_iterations} iterations). "
        "Their mean matches the Aggregate view for a complete run; gaps reveal run-to-run variance."
    )
else:
    freq_base = alt.Chart(freq_df).encode(y=alt.Y("Pattern:N", sort="-x"))
    st.altair_chart(
        (
            freq_base.mark_bar().encode(
                x=alt.X("Rate:Q", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%")),
            )
            + freq_base.mark_text(align="left", dx=3, color="#444").encode(
                x=alt.X("Rate:Q"), text="Label:N",
            )
        ).properties(height=260),
        use_container_width=True,
    )
    st.caption(
        "Bar = sample-average rate: per instance, the fraction of its applicable samples "
        "hitting the pattern, averaged over instances. The label reads "
        "**rate (flagged / N)** where **N is applicable instances** (not samples) — the "
        "unit the rate is averaged over. `flagged` = rate × N (the effective number of "
        "flagged instances; whole for single-iteration runs, fractional when iterations > 1)."
    )

# Shared quintile ordering for the positional plot and the two accuracy heatmaps.
_QUINTILE_ORDER = ["0-20%", "20-40%", "40-60%", "60-80%", "80-100%"]
# Reused wherever a quintile (normalized-position) axis appears, as a help tooltip.
_QUINTILE_HELP = (
    "Position is **normalized per conversation**: each tool call's slot is "
    "`call index ÷ total calls`, bucketed into fifths. Every quintile spans **20% "
    "of the conversation** — `0-20%` is the opening fifth of the calls, `80-100%` the "
    "closing fifth. So a tool shows up where it runs *relative to the conversation's "
    "length*, not at a fixed call number."
)

# ── Positional tool distribution (100%-stacked) ─────────────────────────────────
st.subheader("Tool by call position", help=_QUINTILE_HELP)
pos_df = positional_tool_distribution(run.groups)
if not pos_df.empty:
    pos_base = alt.Chart(pos_df).encode(
        x=alt.X("quintile:N", sort=_QUINTILE_ORDER,
                title="Call position (share of conversation)"),
        # Normalize + share the order so bars and labels stack identically.
        y=alt.Y("share:Q", stack="normalize", title="Share"),
        order=alt.Order("tool:N"),
    )
    pos_bars = pos_base.mark_bar().encode(
        y=alt.Y("share:Q", stack="normalize", axis=alt.Axis(format="%"), title="Share"),
        color=alt.Color("tool:N", title="Tool",
                        scale=color_scale(pos_df["tool"], kind="tool")),
        tooltip=[
            alt.Tooltip("quintile:N", title="Call position"),
            alt.Tooltip("tool:N", title="Tool"),
            alt.Tooltip("share:Q", title="Share", format=".0%"),
        ],
    )
    # Annotate each segment with its share, centred via bandPosition. Hide slivers
    # below 5% so the labels stay legible.
    pos_labels = (
        pos_base.transform_filter("datum.share >= 0.05")
        .mark_text(baseline="middle", fontSize=10, color="white")
        .encode(
            y=alt.Y("share:Q", stack="normalize", bandPosition=0.5),
            text=alt.Text("share:Q", format=".0%"),
        )
    )
    st.altair_chart(
        (pos_bars + pos_labels).properties(height=320),
        use_container_width=True,
    )
    st.caption(
        "Share of tool calls in each **fifth of the conversation** (normalized by "
        "`call index ÷ total calls`, so each band = 20% of the calls). Bars sum to "
        "100% per fifth. Normalizing keeps a long run of one tool inside the "
        "quintiles it spans instead of letting it spill into a trailing bucket."
    )
# Text turns white on the darker (high-accuracy) cells so labels stay legible.
_HEAT_TEXT_COLOR = alt.condition("datum.accuracy > 0.5", alt.value("white"), alt.value("black"))


def _acc_label(row: pd.Series) -> str:
    return "—" if pd.isna(row["accuracy"]) else f"{row['accuracy'] * 100:.0f}% ({int(row['n'])})"


# ── Accuracy by first-answer timing ("patience pays off") ───────────────────────
st.subheader("Accuracy by first-answer timing")
fs_df = first_submit_accuracy_by_quintile(run.records)
if fs_df["n"].sum() == 0:
    st.info("No conversation in this run ever submitted an answer.")
else:
    fs_df["model"] = run.config.get("predictor", {}).get("model_name", "unknown")
    fs_df["Label"] = fs_df.apply(_acc_label, axis=1)
    fs_base = alt.Chart(fs_df).encode(
        x=alt.X("quintile:N", sort=_QUINTILE_ORDER,
                title="First submit_sql position (share of conversation)"),
        y=alt.Y("model:N", title=None),
    )
    fs_heat = fs_base.mark_rect(stroke="white").encode(
        color=alt.Color("accuracy:Q", scale=alt.Scale(scheme="blues", domain=[0, 1]),
                        legend=alt.Legend(format="%", title="Accuracy")),
        tooltip=[
            alt.Tooltip("model:N", title="Model"),
            alt.Tooltip("quintile:N", title="First-submit position"),
            alt.Tooltip("accuracy:Q", title="Accuracy", format=".0%"),
            alt.Tooltip("n:Q", title="Conversations"),
        ],
    )
    fs_text = fs_base.mark_text(baseline="middle", fontSize=12).encode(
        text="Label:N", color=_HEAT_TEXT_COLOR,
    )
    st.altair_chart((fs_heat + fs_text).properties(height=80), use_container_width=True)
    st.caption(
        "Each conversation is placed by **where its first `submit_sql` lands** "
        "(call index ÷ total tool calls), then bucketed into fifths. The cell is the "
        "pooled accuracy of conversations answering in that fifth — `acc% (conversations)`; "
        "conversations that never submit are excluded. A left-to-right climb is the "
        "“patience pays off” trend (gathering context before answering helps). Rows are "
        "per model, so a multi-run comparison stacks models here."
    )

# ── Tool importance by call position (accuracy cross-tab) ───────────────────────
st.subheader("Tool importance by call position")
tp_df = tool_position_accuracy(run.records)
if tp_df.empty:
    st.info("No tool calls to position.")
else:
    tp_df["Label"] = tp_df.apply(_acc_label, axis=1)
    # Order rows by how often the tool appears, busiest at the top.
    tool_order = tp_df.groupby("tool")["n"].sum().sort_values(ascending=False).index.tolist()
    tp_base = alt.Chart(tp_df).encode(
        x=alt.X("quintile:N", sort=_QUINTILE_ORDER,
                title="Call position (share of conversation)"),
        y=alt.Y("tool:N", sort=tool_order, title="Tool"),
    )
    tp_heat = tp_base.mark_rect(stroke="white").encode(
        color=alt.Color("accuracy:Q", scale=alt.Scale(scheme="blues", domain=[0, 1]),
                        legend=alt.Legend(format="%", title="Accuracy")),
        tooltip=[
            alt.Tooltip("tool:N", title="Tool"),
            alt.Tooltip("quintile:N", title="Call position"),
            alt.Tooltip("accuracy:Q", title="Accuracy", format=".0%"),
            alt.Tooltip("n:Q", title="Conversations"),
        ],
    )
    tp_text = tp_base.mark_text(baseline="middle", fontSize=11).encode(
        text="Label:N", color=_HEAT_TEXT_COLOR,
    )
    st.altair_chart(
        (tp_heat + tp_text).properties(height=max(36 * len(tool_order), 160)),
        use_container_width=True,
    )
    st.caption(
        "Accuracy cross-tab: a cell is the pooled accuracy of conversations that "
        "called that tool **anywhere in that fifth** of the conversation — `acc% "
        "(conversations)`. A conversation can appear in several cells, so this is "
        "exploratory: a bright cell may reflect the tool, the timing, or a confound. "
        "It is the accuracy-coloured companion to *Tool by call position* above."
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

# For repeated identical calls, summarize which tool gets re-issued most across
# all flagged samples (total surplus repeats: a call made k× counts k-1).
if pattern_name == "repeated_identical_call":
    repeat_df = repeated_identical_tool_counts(flagged)
    if not repeat_df.empty:
        st.markdown("**Most repeated calls by tool** (total re-issues across flagged samples)")
        repeat_base = alt.Chart(repeat_df).encode(y=alt.Y("tool:N", sort="-x", title="Tool"))
        st.altair_chart(
            (
                repeat_base.mark_bar().encode(
                    x=alt.X("repeats:Q", title="Surplus repeats", axis=alt.Axis(format="d")),
                )
                + repeat_base.mark_text(align="left", dx=3, color="#444").encode(
                    x=alt.X("repeats:Q"), text=alt.Text("repeats:Q", format="d"),
                )
            ).properties(height=min(40 * len(repeat_df) + 40, 320)),
            use_container_width=True,
        )

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
