# Run Comparison Page — Design Spec

**Date:** 2026-05-15
**Status:** Approved

## Overview

A new Streamlit page (`explorer/pages/compare.py`) that lets a researcher select an arbitrary number of evaluation runs and compare them side by side: aggregate stats, per-database charts, a per-task pass/fail table (with disagreement filter), and a two-column conversation viewer for drilling into individual tasks.

---

## File Layout

```
explorer/
├── app.py          # small change: imports render_conversation from render.py
├── loader.py       # add join_runs()
├── render.py       # new: render_conversation(), render_ai_content()
├── pages/
│   └── compare.py  # new comparison page
└── README.md       # updated
```

**Dependency:** no new packages — reuses `streamlit`, `pandas`, `pyyaml` already in the project.
**Launch:** `uv run streamlit run explorer/app.py` — Streamlit auto-discovers `pages/`.

Extracting `render.py` is required to share conversation rendering between `app.py` and `compare.py` without duplication.

---

## Section 1: Sidebar — Run Selector

`list_runs(RESULTS_ROOT)` returns `{baseline: {date: [time, ...]}}`. This is flattened into a list of human-readable labels:

```
no_tool / 2026_05_14 / 18_07_13
no_tool / 2026_05_15 / 07_06_40
```

Each label maps to its resolved `Path`. The sidebar renders a `st.multiselect` over all labels. A `st.warning` is shown if fewer than 2 runs are selected and the rest of the page is suppressed with `st.stop()`.

Each selected run is loaded via a `@st.cache_data`-wrapped `load_run` call defined in `compare.py`. Streamlit keys cache entries on function identity, so this cache is independent from `app.py`'s wrapper — a run may be loaded once per page per session, which is acceptable given the small file sizes. The result is a `dict[str, RunData]` (label → RunData) threaded through the rest of the page.

---

## Section 2: Aggregate Stats Panel

`st.columns(N)` — one column per selected run. Each column contains:

- Run label as `st.markdown` header
- Model name read from `run.config.get("predictor", {}).get("model_name")`
- Five `st.metric` cards: Accuracy `X/N (Y%)`, Avg Input Tokens, Avg Output Tokens, Avg Cost, Avg Budget Remaining

Below the per-run metric columns, charts are rendered in rows — each row uses `st.columns(N)` so runs align horizontally:

- **Row 1 — Accuracy by Database:** `st.bar_chart` of `stats.accuracy_by_database` per run
- **Row 2 — Error Distribution:** `st.bar_chart` of `stats.error_distribution` per run
- **Row 3 — Tool Usage:** `st.bar_chart` of `stats.tool_usage` per run — only rendered if at least one run has non-empty tool data

---

## Section 3: Per-Task Table and `join_runs()`

### `join_runs()` in `loader.py`

```python
def join_runs(runs: dict[str, RunData]) -> pd.DataFrame:
```

Builds a merged DataFrame keyed on `instance_id`. The union of all instance IDs across all runs forms the row index. Columns:

| Column | Source |
|---|---|
| `instance_id` | shared key |
| `database` | from the first run that contains this task |
| `Question` | `amb_user_query` or `not_ambiguos_query`, truncated to 80 chars |
| `<run_label>` (one per run) | `✓` if passed, `✗` if failed, `—` if task absent in that run |

Tasks absent in a run get `—` to make coverage gaps visible.

### Filter bar

One row of controls above the table:

| Control | Type | Effect |
|---|---|---|
| Agreement filter | Radio: All / Disagreement only / All Passed / All Failed | Filters rows |
| Search | Text input | Substring match on Question |

"Disagreement only" keeps rows where not all runs share the same outcome — the primary view for understanding why one run outperforms another.

### Table rendering

`st.dataframe(on_select="rerun", selection_mode="single-row")`. Clicking a row triggers the conversation viewer below.

---

## Section 4: Conversation Viewer

Rendered when a task row is selected.

**Run selection:**
- If exactly 2 runs are selected: both shown automatically.
- If N > 2: two `st.selectbox` widgets let the user pick which two runs to compare; pre-populated with the first two run labels. The columns re-render on change.

**Layout:** `st.columns(2)`, one per chosen run. Each column:
- Run label as header
- Full conversation rendered by `render_conversation(record)` imported from `render.py`
- If the task is absent in that run (`—`): a brief `st.info("Task not found in this run.")` shown instead

---

## `render.py` — Shared Rendering Helpers

Extracted from `app.py` with no behavioral change:

```python
def render_ai_content(content) -> None: ...
def render_conversation(record: dict) -> None: ...
```

`app.py` is updated to `from render import render_conversation, render_ai_content` and its local definitions removed.

---

## Data Contract additions to `loader.py`

```python
def join_runs(runs: dict[str, RunData]) -> pd.DataFrame:
    """Merge N RunData objects on instance_id. One pass/fail column per run label."""
```

No changes to existing `RunData`, `RunStats`, `load_run`, or `list_runs`.

---

## Error Handling

- Fewer than 2 runs selected: `st.warning` + `st.stop()`.
- Task absent in a run: rendered as `—` in the table and `st.info` in the conversation column.
- Malformed records in any run: inherit the existing `run.malformed_count` warning from `loader.py`.
- Missing `config.yaml`: model name shown as `unknown`, no crash.

---

## Out of Scope

- Editing or annotating results
- Statistical significance testing between runs
- Exporting the comparison table
- Authentication / access control
