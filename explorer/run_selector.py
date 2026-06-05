"""Shared sidebar run selector that persists the chosen run across pages.

Streamlit's ``session_state`` is shared across all pages of a session, so the
last-selected (date, run) is stashed there and used as the default on whatever
page renders next. This way picking a run on one page keeps it selected when you
switch to another.
"""

from __future__ import annotations

import streamlit as st


_LABEL_KEY = "explorer_selected_label"


def _fmt_label(date: str, run_key: str) -> str:
    """Full-path label for a run, e.g. ``"2026-06-05 / 16-23-07/slug"``."""
    return f"{date} / {run_key}"


def select_run_sidebar(
    runs_tree: dict[str, list[str]],
    *,
    header: str = "Select Run",
) -> tuple[str, str]:
    """Render a single full-path run selector in the sidebar and return ``(date, run_key)``.

    Mirrors the comparison page: every run is shown as one ``date / run_key``
    option so you can jump between dates without first switching a separate date
    dropdown. The selection is persisted in ``session_state`` so it carries
    across pages; if the previously-selected run is no longer available it falls
    back to the first option.
    """
    # Flatten the date→runs tree into a single ordered label → (date, run) map.
    options: dict[str, tuple[str, str]] = {
        _fmt_label(date, run_key): (date, run_key)
        for date, run_names in runs_tree.items()
        for run_key in run_names
    }
    labels = list(options.keys())

    with st.sidebar:
        st.header(header)
        # Streamlit truncates long selectbox text with an ellipsis; allow the
        # selected value (and the dropdown options) to wrap so full run paths
        # stay readable in the narrow sidebar. BaseWeb sets `white-space: nowrap`
        # with high specificity, so override every descendant of the control.
        st.markdown(
            """
            <style>
            /* Closed control: let the box grow and the value text wrap. */
            div[data-baseweb="select"] > div {
                height: auto !important;
                min-height: 38px;
            }
            div[data-baseweb="select"] * {
                white-space: normal !important;
                overflow: visible !important;
                text-overflow: clip !important;
                overflow-wrap: anywhere !important;
                word-break: break-word !important;
            }
            /* Open dropdown menu (rendered in a portal at the DOM root). */
            ul[role="listbox"] li,
            div[data-baseweb="popover"] li[role="option"] {
                white-space: normal !important;
                overflow-wrap: anywhere !important;
                word-break: break-word !important;
                height: auto !important;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )
        prev_label = st.session_state.get(_LABEL_KEY)
        index = labels.index(prev_label) if prev_label in labels else 0
        label = st.selectbox("Run", labels, index=index)
        st.session_state[_LABEL_KEY] = label

    return options[label]
