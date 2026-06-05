"""Chart helpers for the Aptitude / Unreliability reliability view.

Shared by the single-run page (`app.py`) and the comparison page
(`pages/compare.py`) so the box plot and its explainer stay in one place.

The box plot follows the paper (arXiv:2505.06120): each box is built from the
*averaged per-task percentiles* A^p (mean over tasks of the p-th percentile of a
task's per-iteration scores), so it matches the metric cards exactly:

    upper whisker = A^90 (Aptitude)
    lower whisker = A^10
    whisker span  = A^90 - A^10 (Unreliability)
    box           = A^25 .. A^75
    median tick   = A^50

This is the one view where Plotly beats Altair: ``go.Box`` accepts the
precomputed quartiles/fences directly (``q1``/``median``/``q3``/``lowerfence``/
``upperfence``), so the box matches these definitions without manually stacking
marks — whereas Altair's ``mark_boxplot`` recomputes quartiles from raw data with
1.5*IQR whiskers. The simpler bar/line charts elsewhere stay in Altair.
"""

from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

from colors import stable_color
from metrics import ReliabilityStats


RELIABILITY_EXPLANATION = """
**Setup.** Each task is run for *N* iterations; every iteration scores **1** if the
submitted SQL passes and **0** otherwise. For a task with scores *g = [s₁ … sₙ]* we take
percentiles of that per-task distribution, then average across all tasks.
(Metrics from [arXiv:2505.06120](https://arxiv.org/abs/2505.06120).)

- **Average P̄** — mean score across tasks: the *typical* pass rate.
- **Aptitude A⁹⁰** — the 90th-percentile score (averaged over tasks): near **best-case**
  capability, i.e. what the model manages on a good run. *Higher is better.*
- **Unreliability U₁₀⁹⁰ = A⁹⁰ − A¹⁰** — the gap between the 90th- and 10th-percentile
  scores: how much the outcome swings between lucky and unlucky runs. *Lower = more
  consistent.*
- **Reliability R = 1 − U** — convenience complement of unreliability.

**Reading the box plot.** Each box summarises the averaged per-task percentile curve:
the **top whisker = A⁹⁰ (Aptitude)**, the **bottom whisker = A¹⁰**, so the **whisker span
is the Unreliability**. The box spans A²⁵–A⁷⁵ and the tick is the median A⁵⁰.

*Caveat:* scores are binary and *N* is small, so per-task percentiles are coarse — read
these as indicative, not precise.
"""


def reliability_explainer() -> None:
    """Render a collapsed explainer for the Aptitude / Unreliability metrics."""
    with st.expander("ℹ️ What do Aptitude & Unreliability mean?"):
        st.markdown(RELIABILITY_EXPLANATION)


def aptitude_unreliability_box(rels: dict[str, ReliabilityStats]) -> go.Figure:
    """Box plot of the averaged per-task percentiles, one box per run.

    Whiskers are A¹⁰ / A⁹⁰ (span = Unreliability), the box is A²⁵ / A⁷⁵ and the
    tick is the median A⁵⁰. Fed to ``go.Box`` as precomputed quartiles/fences so
    it matches the metric cards exactly (no recomputation from raw data).
    """
    fig = go.Figure()
    for label, rel in ((lbl, r) for lbl, r in rels.items() if r.percentiles):
        color = stable_color(label, kind="run")
        p = rel.percentiles
        fig.add_trace(
            go.Box(
                name=label,
                x=[label],
                q1=[p[25]],
                median=[p[50]],
                q3=[p[75]],
                lowerfence=[p[10]],
                upperfence=[p[90]],
                marker_color=color,
                width=0.5,
                hovertext=[
                    f"<b>{label}</b><br>"
                    f"Aptitude A⁹⁰: {p[90]:.1%}<br>"
                    f"Unreliability U₁₀⁹⁰: {p[90] - p[10]:.1%}<br>"
                    f"Median A⁵⁰: {p[50]:.1%}<br>"
                    f"A²⁵–A⁷⁵: {p[25]:.1%}–{p[75]:.1%}"
                ],
                hoverinfo="text",
            )
        )
        # Annotate the headline values just right of each box so runs are easy to
        # compare at a glance: A⁹⁰ (Aptitude), A⁵⁰ (median), A¹⁰ (floor).
        for tag, value in (("A⁹⁰", p[90]), ("A⁵⁰", p[50]), ("A¹⁰", p[10])):
            fig.add_annotation(
                x=label,
                y=value,
                text=f"{tag} {value:.0%}",
                showarrow=False,
                xanchor="left",
                xshift=22,
                font=dict(size=11, color=color),
            )

    fig.update_layout(
        height=360,
        showlegend=False,
        margin=dict(t=10, b=10, l=10, r=70),  # room for the value labels right of each box
    )
    fig.update_yaxes(
        title_text="Per-task score", range=[0, 1], tickformat=".0%"
    )
    return fig


def reliability_bars(rels: dict[str, ReliabilityStats]) -> go.Figure:
    """Grouped bar chart of the headline reliability metrics, one group per metric.

    X-axis is the metric (P̄, A⁹⁰, U₁₀⁹⁰, R, pass@N); bars within a group are the
    runs (``barmode="group"``). All values are 0..1 and share the percentage
    y-axis. Note U₁₀⁹⁰ is *lower = better* while the others are *higher = better*.
    """
    metrics = ["Average P̄", "Aptitude A⁹⁰", "Unreliability U₁₀⁹⁰", "Reliability R", "pass@N"]
    fig = go.Figure()
    for label, rel in rels.items():
        passk = rel.passk[max(rel.passk)] if rel.passk else 0.0
        values = [rel.avg_performance, rel.aptitude, rel.unreliability, rel.reliability, passk]
        fig.add_trace(
            go.Bar(
                name=label,
                x=metrics,
                y=values,
                marker_color=stable_color(label, kind="run"),
                text=[f"{v:.1%}" for v in values],
                textposition="outside",
                hovertemplate="<b>%{fullData.name}</b><br>%{x}: %{y:.1%}<extra></extra>",
            )
        )

    fig.update_layout(
        barmode="group",
        height=360,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(t=40, b=10, l=10, r=10),
    )
    fig.update_yaxes(range=[0, 1], tickformat=".0%")
    return fig
