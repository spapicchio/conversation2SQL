"""Stable, centralized color assignments for explorer charts.

Every categorical field that drives color (tools, anti-patterns, iterations,
run labels) resolves to a fixed color here, so a given category keeps the same
color across runs, pages, and reloads — independent of selection order or which
categories happen to be present in a particular chart.

- Known finite domains (tools, anti-patterns) get hand-picked distinct colors.
- Open-ended domains (run labels) fall back to a deterministic hash into the
  shared palette, so the *same* label always lands on the *same* color. We use
  ``hashlib`` (not the builtin ``hash``, which is salted per process and would
  change every restart).

Altair and Plotly both consume the same palette so a tool/run looks identical
whichever library renders it. Altair callers use :func:`color_scale` to build an
``alt.Scale`` with an explicit ``domain``/``range``; Plotly callers use
:func:`stable_color` per trace.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable

import altair as alt

from patterns import PATTERN_CATALOG

# Shared 24-color qualitative palette (Plotly/D3 "category20"-style). The first
# ten match Plotly's default so the pinned colors below blend with anything that
# still falls through to the palette by hash.
_PALETTE: list[str] = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    "#aec7e8", "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5",
    "#c49c94", "#f7b6d2", "#c7c7c7", "#dbdb8d", "#9edae5",
    "#393b79", "#637939", "#8c6d31", "#843c39",
]

# ── Pinned domains ───────────────────────────────────────────────────────────

# Agent tool names (must match the keys of ``TOOL_COSTS`` in the eval package).
TOOL_COLORS: dict[str, str] = {
    "execute_sql": "#1f77b4",
    "get_schema": "#ff7f0e",
    "get_column_meaning": "#2ca02c",
    "get_all_column_meanings": "#98df8a",
    "get_all_external_knowledge_names": "#9467bd",
    "get_knowledge_definition": "#8c564b",
    "get_all_knowledge_definitions": "#c49c94",
    "ask_user": "#e377c2",
    "submit_sql": "#d62728",
}

# Anti-patterns, keyed by both internal name and human label (charts use the
# label) so either resolves to the same color.
_PATTERN_NAME_COLORS: dict[str, str] = {
    name: _PALETTE[i]
    for i, (name, _) in enumerate(PATTERN_CATALOG)
}
PATTERN_COLORS: dict[str, str] = {
    **_PATTERN_NAME_COLORS,
    **{label: _PATTERN_NAME_COLORS[name] for name, label in PATTERN_CATALOG},
}

# Per-kind pinned lookups consulted by ``stable_color`` before it hashes.
_PINNED: dict[str, dict[str, str]] = {
    "tool": TOOL_COLORS,
    "pattern": PATTERN_COLORS,
}


def _hashed_color(value: str) -> str:
    """Deterministic palette color for an arbitrary string (process-independent)."""
    digest = hashlib.md5(value.encode("utf-8")).hexdigest()
    return _PALETTE[int(digest, 16) % len(_PALETTE)]


def stable_color(value: object, kind: str = "") -> str:
    """Return the fixed hex color for a category value.

    ``kind`` selects a pinned lookup (``"tool"``/``"pattern"``); ``"iteration"``
    maps by the iteration's integer index; anything unknown (e.g. ``"run"``)
    hashes the string. Unpinned values within a pinned kind also hash, so the
    color is always defined.
    """
    if kind == "iteration":
        idx = _iteration_index(value)
        if idx is not None:
            return _PALETTE[idx % len(_PALETTE)]
        return _hashed_color(str(value))
    pinned = _PINNED.get(kind)
    if pinned and str(value) in pinned:
        return pinned[str(value)]
    return _hashed_color(str(value))


def _iteration_index(value: object) -> int | None:
    """Pull the integer index out of an iteration label like ``"iter 3"`` or ``3``."""
    if isinstance(value, int):
        return value
    text = str(value).strip()
    digits = "".join(ch for ch in text if ch.isdigit())
    return int(digits) if digits else None


def color_scale(values: Iterable[object], kind: str = "") -> alt.Scale:
    """Build an ``alt.Scale`` pinning each value in ``values`` to its stable color.

    De-dupes while preserving order so the domain/range arrays line up. Pass the
    column (or sort order) that feeds an ``alt.Color`` encoding.
    """
    domain = list(dict.fromkeys(str(v) for v in values))
    return alt.Scale(domain=domain, range=[stable_color(v, kind) for v in domain])
