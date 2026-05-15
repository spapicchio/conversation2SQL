# Results Explorer — Design Spec

**Date:** 2026-05-14
**Status:** Approved

## Overview

A single-file Streamlit application (`explorer/app.py`) that lets a researcher load one evaluation run, inspect aggregate stats, filter the task list, and step through individual task conversations in a LangSmith-style viewer. Targets the `results/<baseline>/<date>/<time>/` folder layout produced by `workflow_evaluation_pipeline`.

---

## File Layout

```
explorer/
├── app.py       # Streamlit entrypoint (~300 LOC); all st.* calls live here
└── loader.py    # Pure-Python data loading and stat computation; no Streamlit imports
```

**Dependency:** add `streamlit` via `uv add streamlit`.
**Launch:** `uv run streamlit run explorer/app.py` from the repo root.
**Data root:** `results/` folder resolved relative to the working directory.

Splitting loader from app means `loader.py` is unit-testable without a browser and the frontend can be replaced independently.

---

## Section 1: Sidebar — Run Selector

Three cascading dropdowns:

1. **Baseline** — populated from direct subdirectories of `results/` (e.g. `no_tool`, `bird_full`)
2. **Date** — `YYYY_MM_DD` subdirectories under the selected baseline
3. **Time** — `HH_MM_SS` subdirectories under the selected date; most recent pre-selected

On selection, the app calls `loader.load_run(path)` which reads `results_smaller.jsonl` eagerly (fast projected fields). The full `results.jsonl` is never loaded eagerly — it is only needed if the smaller file is missing.

A metadata line below the selector shows `model_name` and `temperature` read from `config.yaml` in the same run folder.

---

## Section 2: Stats Panel

Four `st.metric` cards in a single row:

| Card | Source field(s) |
|---|---|
| Accuracy `X / N (Y%)` | `execution_accuracy` |
| Avg Tokens | mean of `total_tokens` per record |
| Avg Cost | mean of `total_cost` per record |
| Avg Budget Remaining | mean of `updated_user_patience` per record (terminal states use `-2`; shown as-is) |

Below the cards, two `st.bar_chart` columns:

- **Accuracy by category** — pass rate per unique `category` value
- **Tool usage frequency** — summed counts from `tool_calls_in_order` across all records; hidden when all records have empty tool lists (i.e. `no_tool` baseline)

All stats are derived from the already-loaded `results_smaller.jsonl` records — no additional file I/O.

---

## Section 3: Task List with Filtering

Filter bar (three controls in one row):

| Control | Type | Filters on |
|---|---|---|
| Pass / Fail | Radio: All / Passed / Failed | `execution_accuracy` |
| Category | Multiselect | `category` |
| Question search | Text input | substring match on `amb_user_query` or `not_ambiguos_query` |

Filtered records render as `st.dataframe` with columns:

`#` · `instance_id` · `selected_database` · `Question` (truncated 80 chars) · `category` · `Accuracy` · `total_tokens` · `total_cost`

Row selection uses `st.dataframe(on_select="rerun")` — native Streamlit, no custom JS. Selecting a row sets session state and scrolls to the conversation viewer below.

---

## Section 4: Conversation Viewer

Rendered when a task row is selected. Header line:

```
[instance_id]  db: selected_database  [PASS ✓ / FAIL ✗]
```

Two collapsible blocks before the message thread:

- `st.expander("Ground-truth SQL")` — shows `sol_sql`
- `st.expander("Predicted SQL")` — shows `predicted_sql`

### Message rendering (in order)

**`user` role** (system prompt):
- Wrapped in `st.expander("System prompt — click to expand")`, collapsed by default.
- Content rendered as plain text (can be very long — contains schema + task).

**`ai` role**:
- Always visible via `st.chat_message("assistant")`.
- Content rendered as markdown.
- Metadata line below: `tokens: {prompt_tokens}↑ {completion_tokens}↓ | cost: ${cost_usd:.5f} | finish: {finish_reason}`.
- If `tool_calls` is non-empty, each call renders as `st.expander("🔧 tool_name(args truncated)")` collapsed by default; expanded content shows full JSON arguments.

**`tool` role**:
- Rendered via `st.chat_message("user")` with a label override showing `tool_name`.
- Status badge: green `✓` for success, red `✗` for error.
- Content inside `st.expander("Tool output — click to expand")`, collapsed by default (outputs can be very long).

---

## Data Contract

`loader.load_run(path: Path) -> RunData` returns a dataclass:

```python
@dataclass
class RunData:
    records: list[dict]      # raw dicts from results_smaller.jsonl
    config: dict             # parsed config.yaml
    stats: RunStats          # pre-computed aggregate stats
```

```python
@dataclass
class RunStats:
    n_total: int
    n_passed: int
    avg_tokens: float
    avg_cost: float
    avg_budget_remaining: float
    accuracy_by_category: dict[str, float]   # category -> pass rate
    tool_usage: Counter[str]                  # tool_name -> total calls
```

`loader.py` owns all `json`, `yaml`, `pathlib`, and `collections` imports. `app.py` imports only `RunData`, `RunStats`, and `load_run`.

---

## Error Handling

- If `results_smaller.jsonl` is missing (e.g. error run), fall back to `results.jsonl` with a warning banner.
- If `config.yaml` is missing, skip the metadata line silently.
- If a record is missing an expected field (e.g. `total_cost`), treat it as 0 with no crash.
- Malformed JSONL lines are skipped with a `st.warning` count at the top.

---

## Out of Scope

- Cross-run comparison (future work)
- Authentication / access control
- Editing or annotating results
- Deployment beyond local `localhost`
