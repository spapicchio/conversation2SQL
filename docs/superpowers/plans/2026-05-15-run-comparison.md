# Run Comparison Page Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a Streamlit comparison page (`explorer/pages/compare.py`) that lets the user select N runs, view side-by-side stats/charts, filter a per-task pass/fail table, and step through two conversations at a time.

**Architecture:** Extract shared rendering helpers into `explorer/render.py` (used by both `app.py` and `compare.py`), add `join_runs()` to `loader.py` for the merged per-task DataFrame, and build the UI in `explorer/pages/compare.py` which Streamlit auto-discovers as a second nav page.

**Tech Stack:** Python 3.12, Streamlit, pandas, PyYAML — all already in the project venv. Run everything with `uv run`.

---

## File map

| File | Change |
|---|---|
| `explorer/render.py` | **Create** — `render_ai_content()`, `render_conversation()` extracted from `app.py` |
| `explorer/app.py` | **Modify** — replace local render defs with imports from `render.py` |
| `explorer/loader.py` | **Modify** — add `import pandas as pd` and `join_runs()` |
| `explorer/pages/compare.py` | **Create** — full comparison page UI |
| `tests/explorer/test_loader.py` | **Modify** — append `TestJoinRuns` class |
| `explorer/README.md` | **Modify** — document the compare page |

---

## Task 1: Extract render.py and update app.py

**Files:**
- Create: `explorer/render.py`
- Modify: `explorer/app.py`

- [ ] **Step 1: Create `explorer/render.py`**

```python
"""Shared Streamlit rendering helpers for conversation records."""

from __future__ import annotations

import json

import streamlit as st


def render_ai_content(content) -> None:
    """Render AI message content: str passes through, list-of-blocks separates thinking from text."""
    if isinstance(content, str):
        st.markdown(content)
    elif isinstance(content, list):
        for block in content:
            if not isinstance(block, dict):
                st.markdown(str(block))
                continue
            block_type = block.get("type", "")
            if block_type == "thinking":
                with st.expander("Thinking — click to expand"):
                    st.text(block.get("thinking", ""))
            elif block_type == "text":
                st.markdown(block.get("text", ""))
            else:
                st.json(block)
    else:
        st.text(str(content))


def render_conversation(record: dict) -> None:
    acc = record.get("execution_accuracy", False)
    badge = "✓ PASS" if acc else "✗ FAIL"
    st.subheader(
        f"Conversation: `{record.get('instance_id', '')}` · "
        f"db: `{record.get('selected_database', '')}` · **{badge}**"
    )

    col1, col2 = st.columns(2)
    with col1:
        with st.expander("Ground-truth SQL"):
            sol = record.get("sol_sql", "")
            if isinstance(sol, list):
                sol = "\n\n".join(sol)
            st.code(sol, language="sql")
    with col2:
        with st.expander("Predicted SQL"):
            st.code(record.get("predicted_sql", "") or "", language="sql")

    for msg in record.get("messages", []):
        role = msg.get("role")

        if role in ("user", "system"):
            with st.expander(f"{role.capitalize()} prompt — click to expand"):
                st.text(msg.get("content", ""))

        elif role == "human":
            with st.chat_message("user"):
                st.markdown(msg.get("content", ""))

        elif role == "ai":
            with st.chat_message("assistant"):
                content = msg.get("content", "")
                if content:
                    render_ai_content(content)
                st.caption(
                    f"tokens: {msg.get('prompt_tokens', 0)}↑ {msg.get('completion_tokens', 0)}↓"
                    f" | cost: ${msg.get('cost_usd', 0):.5f}"
                    f" | finish: {msg.get('finish_reason', 'unknown')}"
                )
                for tc in msg.get("tool_calls", []):
                    tool_name = tc.get("tool_name", "unknown")
                    args = tc.get("arguments", {})
                    args_str = json.dumps(args, ensure_ascii=False)
                    label = (
                        f"🔧 {tool_name}({args_str[:60]}…)"
                        if len(args_str) > 60
                        else f"🔧 {tool_name}({args_str})"
                    )
                    with st.expander(label):
                        st.json(args)

        elif role == "tool":
            tool_name = msg.get("tool_name", "tool")
            status = msg.get("status", "")
            status_badge = "✓" if status == "success" else "✗"
            with st.chat_message("user"):
                st.markdown(f"**{tool_name}** {status_badge}")
                content = msg.get("content", "")
                with st.expander("Tool output — click to expand"):
                    if isinstance(content, dict):
                        st.json(content)
                    else:
                        st.text(str(content))
```

- [ ] **Step 2: Update `explorer/app.py` to import from render.py**

Replace the two local function definitions and update all call sites. The diff is:

Remove the `_render_ai_content` and `_render_conversation` function bodies from `app.py` and replace the import block at the top with:

```python
from loader import RunData
from loader import load_run
from loader import list_runs
from render import render_ai_content
from render import render_conversation
```

Then in the body of `app.py`, replace the single call site:
- `_render_conversation(selected_record)` → `render_conversation(selected_record)`

The `_render_ai_content` call inside `_render_conversation` disappears because those functions are now in `render.py`.

- [ ] **Step 3: Verify tests still pass**

```bash
uv run pytest tests/ -x -q
```

Expected: all tests pass (no logic changed, only moved).

- [ ] **Step 4: Commit**

```bash
git add explorer/render.py explorer/app.py
git commit -m "refactor: extract render helpers into render.py for reuse"
```

---

## Task 2: Add `join_runs()` to loader.py

**Files:**
- Modify: `explorer/loader.py`
- Modify: `tests/explorer/test_loader.py`

- [ ] **Step 1: Write failing tests — append to `tests/explorer/test_loader.py`**

Add these imports at the top of the file (after existing imports):

```python
import pandas as pd
from explorer.loader import join_runs
```

Then append this class at the bottom of the file:

```python
# ── join_runs ──────────────────────────────────────────────────────────────────

def _make_run_data(records: list[dict]) -> RunData:
    from collections import Counter
    return RunData(
        records=records,
        config={},
        stats=RunStats(
            n_total=len(records),
            n_passed=sum(1 for r in records if r.get("execution_accuracy")),
            avg_input_tokens=0.0,
            avg_output_tokens=0.0,
            avg_cost=0.0,
            avg_budget_remaining=0.0,
            accuracy_by_database={},
            error_distribution=Counter(),
            tool_usage=Counter(),
        ),
        malformed_count=0,
    )


class TestJoinRuns:
    def test_two_runs_same_task_pass_fail(self):
        r_a = make_record(instance_id="t1", execution_accuracy=True, selected_database="db1")
        r_b = make_record(instance_id="t1", execution_accuracy=False, selected_database="db1")
        runs = {"run_a": _make_run_data([r_a]), "run_b": _make_run_data([r_b])}
        df = join_runs(runs)
        assert len(df) == 1
        assert df.iloc[0]["run_a"] == "✓"
        assert df.iloc[0]["run_b"] == "✗"

    def test_task_absent_in_one_run_shows_dash(self):
        r_a = make_record(instance_id="t1", execution_accuracy=True, selected_database="db1")
        r_b = make_record(instance_id="t2", execution_accuracy=False, selected_database="db1")
        runs = {"run_a": _make_run_data([r_a]), "run_b": _make_run_data([r_b])}
        df = join_runs(runs)
        assert len(df) == 2
        t1 = df[df["instance_id"] == "t1"].iloc[0]
        t2 = df[df["instance_id"] == "t2"].iloc[0]
        assert t1["run_a"] == "✓"
        assert t1["run_b"] == "—"
        assert t2["run_a"] == "—"
        assert t2["run_b"] == "✗"

    def test_question_truncated_at_80_chars(self):
        long_q = "A" * 100
        r = make_record(instance_id="t1", amb_user_query=long_q, selected_database="db1")
        runs = {"run_a": _make_run_data([r])}
        df = join_runs(runs)
        q = df.iloc[0]["Question"]
        assert q.endswith("…")
        assert len(q) == 81  # 80 chars + ellipsis

    def test_question_not_truncated_when_short(self):
        r = make_record(instance_id="t1", amb_user_query="Short", selected_database="db1")
        runs = {"run_a": _make_run_data([r])}
        df = join_runs(runs)
        assert df.iloc[0]["Question"] == "Short"

    def test_not_ambiguous_query_fallback(self):
        r = make_record(instance_id="t1", selected_database="db1")
        r["amb_user_query"] = ""
        r["not_ambiguos_query"] = "Fallback question"
        runs = {"run_a": _make_run_data([r])}
        df = join_runs(runs)
        assert df.iloc[0]["Question"] == "Fallback question"

    def test_database_column_from_first_run(self):
        r = make_record(instance_id="t1", execution_accuracy=True, selected_database="my_db")
        runs = {"run_a": _make_run_data([r])}
        df = join_runs(runs)
        assert df.iloc[0]["database"] == "my_db"

    def test_empty_runs_returns_empty_dataframe(self):
        runs = {"run_a": _make_run_data([]), "run_b": _make_run_data([])}
        df = join_runs(runs)
        assert len(df) == 0
        assert isinstance(df, pd.DataFrame)

    def test_columns_include_all_run_labels(self):
        r = make_record(instance_id="t1", execution_accuracy=True, selected_database="db1")
        runs = {"alpha": _make_run_data([r]), "beta": _make_run_data([r]), "gamma": _make_run_data([r])}
        df = join_runs(runs)
        assert "alpha" in df.columns
        assert "beta" in df.columns
        assert "gamma" in df.columns
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/explorer/test_loader.py::TestJoinRuns -v
```

Expected: `ImportError: cannot import name 'join_runs'`

- [ ] **Step 3: Implement `join_runs()` in `loader.py`**

Add `import pandas as pd` near the top of `loader.py` (after the existing stdlib imports, before `import yaml`):

```python
import pandas as pd
```

Then append this function at the bottom of `loader.py` (after `load_run`):

```python
def join_runs(runs: dict[str, RunData]) -> "pd.DataFrame":
    """Merge N RunData objects on instance_id into a comparison DataFrame.

    Columns: instance_id, database, Question, then one column per run label
    with values ✓ (passed), ✗ (failed), or — (task absent in that run).
    """
    all_ids: dict[str, dict] = {}
    for run_data in runs.values():
        for r in run_data.records:
            iid = r.get("instance_id", "")
            if iid not in all_ids:
                q = r.get("amb_user_query") or r.get("not_ambiguos_query", "")
                all_ids[iid] = {
                    "instance_id": iid,
                    "database": r.get("selected_database", ""),
                    "Question": (q[:80] + "…") if len(q) > 80 else q,
                }

    id_to_records: dict[str, dict[str, dict]] = {}
    for label, run_data in runs.items():
        for r in run_data.records:
            iid = r.get("instance_id", "")
            id_to_records.setdefault(iid, {})[label] = r

    rows = []
    for iid, base in all_ids.items():
        row = dict(base)
        for label in runs:
            record = id_to_records.get(iid, {}).get(label)
            if record is None:
                row[label] = "—"
            elif record.get("execution_accuracy"):
                row[label] = "✓"
            else:
                row[label] = "✗"
        rows.append(row)

    return pd.DataFrame(rows)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/explorer/test_loader.py::TestJoinRuns -v
```

Expected: all 8 tests pass.

- [ ] **Step 5: Run full test suite**

```bash
uv run pytest tests/ -x -q
```

Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add explorer/loader.py tests/explorer/test_loader.py
git commit -m "feat: add join_runs() to loader for per-task run comparison"
```

---

## Task 3: Build compare.py — sidebar, stats, charts

**Files:**
- Create: `explorer/pages/compare.py`

- [ ] **Step 1: Create the `explorer/pages/` directory**

```bash
mkdir -p explorer/pages
```

- [ ] **Step 2: Create `explorer/pages/compare.py`** with sidebar, stats panel, and charts

```python
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

_GEN_PARAM_KEYS = (
    "temperature", "top_p", "top_k", "min_p",
    "presence_penalty", "repetition_penalty", "max_new_tokens",
    "reasoning_effort", "enable_thinking",
)


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
for col, (label, run) in zip(db_cols, runs.items()):
    with col:
        st.subheader("Accuracy by Database")
        db_df = pd.DataFrame(
            [{"Database": k, "Pass Rate": v} for k, v in run.stats.accuracy_by_database.items()]
        ).set_index("Database")
        st.bar_chart(db_df)

err_cols = st.columns(len(runs))
for col, (label, run) in zip(err_cols, runs.items()):
    with col:
        st.subheader("Error Distribution")
        err_df = pd.DataFrame(
            [{"Error Class": k, "Count": v} for k, v in run.stats.error_distribution.most_common()]
        ).set_index("Error Class")
        st.bar_chart(err_df)

has_tools = any(bool(run.stats.tool_usage) for run in runs.values())
if has_tools:
    tool_cols = st.columns(len(runs))
    for col, (label, run) in zip(tool_cols, runs.items()):
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
```

- [ ] **Step 3: Commit**

```bash
git add explorer/pages/compare.py
git commit -m "feat: add run comparison page with sidebar, stats, charts, task table, and conversation viewer"
```

---

## Task 4: Update README.md

**Files:**
- Modify: `explorer/README.md`

- [ ] **Step 1: Add compare page section to README**

Append after the existing Features table:

```markdown
## Compare page

Navigate to **Run Comparison** in the Streamlit sidebar to compare multiple runs side by side.

| Panel | Description |
|---|---|
| **Sidebar** | Multi-select any number of runs (baseline / date / time). Requires ≥ 2. |
| **Stats** | Per-run metric cards: accuracy, avg tokens, avg cost, avg budget remaining. |
| **Charts** | Accuracy by database, error distribution, and tool usage — one chart per run, aligned in columns. |
| **Task table** | One row per `instance_id`, one column per run (✓ / ✗ / —). Filter by agreement, and search by question text. |
| **Conversation viewer** | Click a row to view conversations side by side. With > 2 runs, two dropdowns let you choose which pair to compare. |
```

- [ ] **Step 2: Commit**

```bash
git add explorer/README.md
git commit -m "docs: document run comparison page in explorer README"
```

---

## Self-review

**Spec coverage:**
- ✅ Separate Streamlit page (`pages/compare.py`)
- ✅ Sidebar multi-select for arbitrary N runs
- ✅ Aggregate stats (5 metrics) per run
- ✅ Accuracy by database, error distribution, tool usage charts per run
- ✅ Per-task table with all runs as columns (✓ / ✗ / —)
- ✅ Filter: All / Disagreement only / All Passed / All Failed
- ✅ Conversation viewer with side-by-side columns
- ✅ If N > 2: two selectboxes to choose which two runs to compare
- ✅ Task absent in a run: `st.info("Task not found in this run.")`
- ✅ `render.py` extracted and `app.py` updated
- ✅ `join_runs()` added to `loader.py` with full test coverage

**Placeholder scan:** No TBDs or TODOs present.

**Type consistency:**
- `join_runs(runs: dict[str, RunData]) -> pd.DataFrame` defined in Task 2, called in Task 3 ✅
- `render_conversation(record: dict)` defined in Task 1, called in Task 3 ✅
- `_load_run_cached`, `list_runs`, `load_run` all used consistently ✅
- `RunData`, `RunStats` field names match `loader.py` (`avg_input_tokens`, `avg_output_tokens`, `accuracy_by_database`, `error_distribution`, `tool_usage`) ✅
