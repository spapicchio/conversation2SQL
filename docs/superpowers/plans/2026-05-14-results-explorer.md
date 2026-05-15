# Results Explorer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a local Streamlit app (`explorer/app.py`) for exploring text2SQL evaluation runs stored as JSONL files under `results/<baseline>/<date>/<time>/`.

**Architecture:** Two files — `explorer/loader.py` (pure Python data loading and stat computation, no Streamlit) and `explorer/app.py` (all Streamlit UI calls). The loader is unit-testable without a browser; the app imports from it via `from explorer.loader import ...`.

**Tech Stack:** Python 3.12, Streamlit, pandas, PyYAML (already installed), pytest.

---

## File Map

| Action | Path | Responsibility |
|--------|------|----------------|
| Create | `explorer/__init__.py` | Makes `explorer` an importable package |
| Create | `explorer/loader.py` | `RunStats`, `RunData` dataclasses; `list_runs()`, `_compute_stats()`, `load_run()` |
| Create | `explorer/app.py` | Full Streamlit UI — sidebar, stats panel, task list, conversation viewer |
| Create | `tests/explorer/__init__.py` | Test package marker |
| Create | `tests/explorer/test_loader.py` | Unit tests for all loader functions |

---

## Data structures confirmed from codebase

From `src/conversation2sql/eval_framework/agents/utils.py`:
- `tool_calls` in AI messages: `list[{"tool_name": str, "arguments": dict, "tool_cost": int}]`
- `tool_calls_in_order` in result records: same list, concatenated across all AI messages
- `tool` messages: `{"role": "tool", "tool_name": str, "status": "success"|"error", "content": dict|str}`
- `ai` messages: `{"role": "ai", "content": str, "prompt_tokens": int, "completion_tokens": int, "total_tokens": int, "model_name": str, "finish_reason": str, "cost_usd": float, "tool_calls": [...], "invalid_tool_calls": [...]}`
- `user` messages: `{"role": "user", "content": str}`

Field name quirk: the dataset uses `not_ambiguos_query` (typo, one 'u') — preserved exactly.

---

## Task 1: Add streamlit + scaffold explorer/ package

**Files:**
- Create: `explorer/__init__.py`
- Create: `tests/explorer/__init__.py`

- [ ] **Step 1: Add streamlit via uv**

```bash
uv add streamlit
```

Expected: `pyproject.toml` and `uv.lock` updated.

- [ ] **Step 2: Verify streamlit is importable**

```bash
uv run python -c "import streamlit; print('ok')"
```

Expected output: `ok`

- [ ] **Step 3: Create package markers**

Create `explorer/__init__.py` (empty file):
```python
```

Create `tests/explorer/__init__.py` (empty file):
```python
```

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml uv.lock explorer/__init__.py tests/explorer/__init__.py
git commit -m "feat: add streamlit dep + scaffold explorer package"
```

---

## Task 2: Implement loader.py — data types, list_runs, _compute_stats

**Files:**
- Create: `tests/explorer/test_loader.py`
- Create: `explorer/loader.py`

- [ ] **Step 1: Write failing tests for data types, list_runs, and _compute_stats**

Create `tests/explorer/test_loader.py`:

```python
import json
from collections import Counter
from pathlib import Path

import pytest

from explorer.loader import RunData, RunStats, _compute_stats, list_runs, load_run


# ── helpers ────────────────────────────────────────────────────────────────────

def make_record(
    execution_accuracy=False,
    total_tokens=100,
    total_cost=0.001,
    updated_user_patience=3,
    category="Query",
    tool_calls_in_order=None,
    **extra,
):
    return {
        "execution_accuracy": execution_accuracy,
        "total_tokens": total_tokens,
        "total_cost": total_cost,
        "updated_user_patience": updated_user_patience,
        "category": category,
        "tool_calls_in_order": tool_calls_in_order or [],
        "amb_user_query": "What is the count?",
        "not_ambiguos_query": "What is the count?",
        "instance_id": "test_001",
        "selected_database": "mydb",
        "sol_sql": ["SELECT 1"],
        "predicted_sql": "SELECT 1",
        "messages": [],
        **extra,
    }


# ── _compute_stats ─────────────────────────────────────────────────────────────

class TestComputeStats:
    def test_empty_records(self):
        stats = _compute_stats([])
        assert stats.n_total == 0
        assert stats.n_passed == 0
        assert stats.avg_tokens == 0.0
        assert stats.avg_cost == 0.0
        assert stats.avg_budget_remaining == 0.0
        assert stats.accuracy_by_category == {}
        assert stats.tool_usage == Counter()

    def test_single_passed(self):
        records = [make_record(execution_accuracy=True, total_tokens=200, total_cost=0.01, updated_user_patience=4)]
        stats = _compute_stats(records)
        assert stats.n_total == 1
        assert stats.n_passed == 1
        assert stats.avg_tokens == 200.0
        assert stats.avg_cost == pytest.approx(0.01)
        assert stats.avg_budget_remaining == 4.0

    def test_accuracy_by_category(self):
        records = [
            make_record(execution_accuracy=True, category="Query"),
            make_record(execution_accuracy=False, category="Query"),
            make_record(execution_accuracy=True, category="Management"),
        ]
        stats = _compute_stats(records)
        assert stats.accuracy_by_category["Query"] == pytest.approx(0.5)
        assert stats.accuracy_by_category["Management"] == pytest.approx(1.0)

    def test_tool_usage_dict_format(self):
        records = [
            make_record(tool_calls_in_order=[
                {"tool_name": "execute_sql", "arguments": {}, "tool_cost": 1},
                {"tool_name": "submit_sql", "arguments": {}, "tool_cost": 0},
            ]),
            make_record(tool_calls_in_order=[
                {"tool_name": "execute_sql", "arguments": {}, "tool_cost": 1},
            ]),
        ]
        stats = _compute_stats(records)
        assert stats.tool_usage["execute_sql"] == 2
        assert stats.tool_usage["submit_sql"] == 1

    def test_missing_fields_default_to_zero(self):
        stats = _compute_stats([{"execution_accuracy": False}])
        assert stats.avg_tokens == 0.0
        assert stats.avg_cost == 0.0


# ── list_runs ──────────────────────────────────────────────────────────────────

class TestListRuns:
    def test_empty_root(self, tmp_path):
        assert list_runs(tmp_path) == {}

    def test_nonexistent_root(self, tmp_path):
        assert list_runs(tmp_path / "nonexistent") == {}

    def test_single_run(self, tmp_path):
        (tmp_path / "no_tool" / "2026_05_14" / "09_17_54").mkdir(parents=True)
        assert list_runs(tmp_path) == {"no_tool": {"2026_05_14": ["09_17_54"]}}

    def test_multiple_times_newest_first(self, tmp_path):
        for t in ["09_00_00", "10_00_00", "08_00_00"]:
            (tmp_path / "no_tool" / "2026_05_14" / t).mkdir(parents=True)
        times = list_runs(tmp_path)["no_tool"]["2026_05_14"]
        assert times == ["10_00_00", "09_00_00", "08_00_00"]

    def test_dates_newest_first(self, tmp_path):
        for d in ["2026_05_12", "2026_05_14", "2026_05_13"]:
            (tmp_path / "no_tool" / d / "09_00_00").mkdir(parents=True)
        dates = list(list_runs(tmp_path)["no_tool"].keys())
        assert dates == ["2026_05_14", "2026_05_13", "2026_05_12"]

    def test_skips_files_in_baseline_dir(self, tmp_path):
        (tmp_path / "no_tool" / "2026_05_14" / "09_00_00").mkdir(parents=True)
        (tmp_path / "no_tool" / "some_file.txt").write_text("noise")
        result = list_runs(tmp_path)
        assert "some_file.txt" not in result.get("no_tool", {})
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/explorer/test_loader.py -v 2>&1 | head -30
```

Expected: `ImportError` or `ModuleNotFoundError` (loader.py doesn't exist yet).

- [ ] **Step 3: Implement loader.py data types, _compute_stats, and list_runs**

Create `explorer/loader.py`:

```python
from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class RunStats:
    n_total: int
    n_passed: int
    avg_tokens: float
    avg_cost: float
    avg_budget_remaining: float
    accuracy_by_category: dict[str, float]  # category -> pass rate
    tool_usage: Counter[str]  # tool_name -> total calls


@dataclass
class RunData:
    records: list[dict]
    config: dict
    stats: RunStats
    malformed_count: int = 0
    source_file: str = "results_smaller.jsonl"


def _compute_stats(records: list[dict]) -> RunStats:
    n_total = len(records)
    n_passed = sum(1 for r in records if r.get("execution_accuracy", False))
    avg_tokens = sum(r.get("total_tokens", 0) for r in records) / max(n_total, 1)
    avg_cost = sum(r.get("total_cost", 0) for r in records) / max(n_total, 1)
    avg_budget = sum(r.get("updated_user_patience", 0) for r in records) / max(n_total, 1)

    cat_totals: dict[str, int] = {}
    cat_passed: dict[str, int] = {}
    for r in records:
        cat = r.get("category", "Unknown")
        cat_totals[cat] = cat_totals.get(cat, 0) + 1
        if r.get("execution_accuracy", False):
            cat_passed[cat] = cat_passed.get(cat, 0) + 1
    accuracy_by_category = {
        cat: cat_passed.get(cat, 0) / total
        for cat, total in cat_totals.items()
    }

    tool_usage: Counter[str] = Counter()
    for r in records:
        for tc in r.get("tool_calls_in_order", []):
            if isinstance(tc, dict):
                tool_usage[tc.get("tool_name", "unknown")] += 1
            elif isinstance(tc, str):
                tool_usage[tc] += 1

    return RunStats(
        n_total=n_total,
        n_passed=n_passed,
        avg_tokens=avg_tokens,
        avg_cost=avg_cost,
        avg_budget_remaining=avg_budget,
        accuracy_by_category=accuracy_by_category,
        tool_usage=tool_usage,
    )


def list_runs(results_root: Path) -> dict[str, dict[str, list[str]]]:
    """Return {baseline: {date: [time, ...]}} newest-first within each date."""
    runs: dict[str, dict[str, list[str]]] = {}
    if not results_root.exists():
        return runs
    for baseline_dir in sorted(results_root.iterdir()):
        if not baseline_dir.is_dir():
            continue
        dates: dict[str, list[str]] = {}
        for date_dir in sorted(baseline_dir.iterdir(), reverse=True):
            if not date_dir.is_dir():
                continue
            times = sorted(
                [d.name for d in date_dir.iterdir() if d.is_dir()],
                reverse=True,
            )
            if times:
                dates[date_dir.name] = times
        if dates:
            runs[baseline_dir.name] = dates
    return runs


def load_run(path: Path) -> RunData:
    """Load records and config from results/<baseline>/<date>/<time>/."""
    raise NotImplementedError
```

- [ ] **Step 4: Run tests again — _compute_stats and list_runs should pass, load_run skipped**

```bash
uv run pytest tests/explorer/test_loader.py::TestComputeStats tests/explorer/test_loader.py::TestListRuns -v
```

Expected: all 10 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add explorer/loader.py tests/explorer/test_loader.py tests/explorer/__init__.py
git commit -m "feat: add loader data types, _compute_stats, list_runs with tests"
```

---

## Task 3: Implement loader.py — load_run

**Files:**
- Modify: `tests/explorer/test_loader.py` (append TestLoadRun class)
- Modify: `explorer/loader.py` (replace `raise NotImplementedError` with real implementation)

- [ ] **Step 1: Append TestLoadRun to tests/explorer/test_loader.py**

Add this class at the bottom of `tests/explorer/test_loader.py`:

```python
# ── load_run ───────────────────────────────────────────────────────────────────

class TestLoadRun:
    def _write_jsonl(self, path: Path, records: list[dict]) -> None:
        path.write_text(
            "\n".join(json.dumps(r) for r in records) + "\n",
            encoding="utf-8",
        )

    def test_loads_smaller_jsonl(self, tmp_path):
        self._write_jsonl(tmp_path / "results_smaller.jsonl", [make_record()])
        run = load_run(tmp_path)
        assert len(run.records) == 1
        assert run.source_file == "results_smaller.jsonl"

    def test_falls_back_to_full_jsonl(self, tmp_path):
        self._write_jsonl(tmp_path / "results.jsonl", [make_record()])
        run = load_run(tmp_path)
        assert len(run.records) == 1
        assert run.source_file == "results.jsonl"

    def test_prefers_smaller_over_full(self, tmp_path):
        self._write_jsonl(tmp_path / "results_smaller.jsonl", [make_record(total_tokens=111)])
        self._write_jsonl(tmp_path / "results.jsonl", [make_record(total_tokens=999)])
        run = load_run(tmp_path)
        assert run.records[0]["total_tokens"] == 111

    def test_loads_config_yaml(self, tmp_path):
        self._write_jsonl(tmp_path / "results_smaller.jsonl", [make_record()])
        (tmp_path / "config.yaml").write_text("pipeline:\n  baseline: no_tool\n")
        run = load_run(tmp_path)
        assert run.config["pipeline"]["baseline"] == "no_tool"

    def test_missing_config_yaml_returns_empty_dict(self, tmp_path):
        self._write_jsonl(tmp_path / "results_smaller.jsonl", [make_record()])
        run = load_run(tmp_path)
        assert run.config == {}

    def test_malformed_lines_counted(self, tmp_path):
        content = (
            json.dumps(make_record()) + "\n"
            + "NOT JSON\n"
            + json.dumps(make_record()) + "\n"
        )
        (tmp_path / "results_smaller.jsonl").write_text(content, encoding="utf-8")
        run = load_run(tmp_path)
        assert len(run.records) == 2
        assert run.malformed_count == 1

    def test_empty_run_folder_returns_empty(self, tmp_path):
        run = load_run(tmp_path)
        assert run.records == []
        assert run.malformed_count == 0
        assert run.config == {}

    def test_stats_computed(self, tmp_path):
        records = [make_record(execution_accuracy=True), make_record(execution_accuracy=False)]
        self._write_jsonl(tmp_path / "results_smaller.jsonl", records)
        run = load_run(tmp_path)
        assert run.stats.n_total == 2
        assert run.stats.n_passed == 1
```

- [ ] **Step 2: Run TestLoadRun to verify it fails**

```bash
uv run pytest tests/explorer/test_loader.py::TestLoadRun -v 2>&1 | head -20
```

Expected: all tests FAIL with `NotImplementedError`.

- [ ] **Step 3: Replace the NotImplementedError stub with the real load_run in explorer/loader.py**

Replace the `load_run` function body (keep the signature):

```python
def load_run(path: Path) -> RunData:
    """Load records and config from results/<baseline>/<date>/<time>/."""
    smaller = path / "results_smaller.jsonl"
    full = path / "results.jsonl"
    source = smaller if smaller.exists() else full

    records: list[dict] = []
    malformed = 0
    if source.exists():
        with source.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    malformed += 1

    config: dict = {}
    config_path = path / "config.yaml"
    if config_path.exists():
        with config_path.open(encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}

    return RunData(
        records=records,
        config=config,
        stats=_compute_stats(records),
        malformed_count=malformed,
        source_file=source.name,
    )
```

- [ ] **Step 4: Run the full test suite**

```bash
uv run pytest tests/explorer/test_loader.py -v
```

Expected: all tests PASS (19 total).

- [ ] **Step 5: Run the existing test suite to confirm no regressions**

```bash
uv run pytest tests/ -v --ignore=tests/eval_framework/integration 2>&1 | tail -20
```

Expected: all previously passing tests still PASS.

- [ ] **Step 6: Commit**

```bash
git add explorer/loader.py tests/explorer/test_loader.py
git commit -m "feat: implement load_run with fallback and malformed-line handling"
```

---

## Task 4: Implement explorer/app.py

**Files:**
- Create: `explorer/app.py`

- [ ] **Step 1: Create explorer/app.py with the complete implementation**

```python
"""Streamlit results explorer for conversation2SQL evaluation runs."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from explorer.loader import RunData, load_run, list_runs

RESULTS_ROOT = Path("results")


# ── Cached loader (keyed on path string so Streamlit can hash it) ──────────────

@st.cache_data
def _load_run_cached(path_str: str) -> RunData:
    return load_run(Path(path_str))


# ── Helper functions ───────────────────────────────────────────────────────────

def _question(r: dict) -> str:
    return r.get("amb_user_query") or r.get("not_ambiguos_query", "")


def _render_conversation(record: dict) -> None:
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

        if role == "user":
            with st.expander("System prompt — click to expand"):
                st.text(msg.get("content", ""))

        elif role == "ai":
            with st.chat_message("assistant"):
                content = msg.get("content", "")
                if content:
                    st.markdown(content)
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


# ── App layout ─────────────────────────────────────────────────────────────────

st.set_page_config(page_title="Results Explorer", layout="wide")
st.title("Results Explorer")

# Sidebar: cascading run selector
with st.sidebar:
    st.header("Select Run")
    runs = list_runs(RESULTS_ROOT)
    if not runs:
        st.error(f"No results found in `{RESULTS_ROOT.resolve()}`")
        st.stop()
    baseline = st.selectbox("Baseline", list(runs.keys()))
    dates = list(runs[baseline].keys())
    date = st.selectbox("Date", dates)
    times = runs[baseline][date]
    time_key = st.selectbox("Time", times, index=0)

run_path = RESULTS_ROOT / baseline / date / time_key
run = _load_run_cached(str(run_path))

with st.sidebar:
    predictor = run.config.get("predictor", {})
    if predictor:
        st.caption(
            f"Model: `{predictor.get('model_name', 'unknown')}`  \n"
            f"Temp: `{predictor.get('temperature', '?')}`"
        )

if run.malformed_count:
    st.warning(f"{run.malformed_count} malformed line(s) skipped in {run.source_file}.")

# Stats panel: four metric cards
stats = run.stats
pct = stats.n_passed / max(stats.n_total, 1) * 100
c1, c2, c3, c4 = st.columns(4)
c1.metric("Accuracy", f"{stats.n_passed} / {stats.n_total} ({pct:.1f}%)")
c2.metric("Avg Tokens", f"{stats.avg_tokens:,.0f}")
c3.metric("Avg Cost", f"${stats.avg_cost:.5f}")
c4.metric("Avg Budget Remaining", f"{stats.avg_budget_remaining:.1f}")

# Charts: accuracy by category + tool usage (hidden for no_tool baseline)
has_tools = bool(stats.tool_usage)
chart_cols = st.columns(2 if has_tools else 1)

with chart_cols[0]:
    st.subheader("Accuracy by Category")
    cat_df = pd.DataFrame(
        [{"Category": k, "Pass Rate": v} for k, v in stats.accuracy_by_category.items()]
    ).set_index("Category")
    st.bar_chart(cat_df)

if has_tools:
    with chart_cols[1]:
        st.subheader("Tool Usage")
        tool_df = pd.DataFrame(
            [{"Tool": k, "Calls": v} for k, v in stats.tool_usage.most_common()]
        ).set_index("Tool")
        st.bar_chart(tool_df)

st.divider()

# Task list with filters
st.subheader("Tasks")
f1, f2, f3 = st.columns([1, 2, 3])
with f1:
    pass_filter = st.radio("Pass / Fail", ["All", "Passed", "Failed"], horizontal=True)
with f2:
    all_cats = sorted({r.get("category", "Unknown") for r in run.records})
    cat_filter = st.multiselect("Category", all_cats, default=all_cats)
with f3:
    search = st.text_input("Search question", placeholder="substring…")

filtered = run.records
if pass_filter == "Passed":
    filtered = [r for r in filtered if r.get("execution_accuracy")]
elif pass_filter == "Failed":
    filtered = [r for r in filtered if not r.get("execution_accuracy")]
if cat_filter:
    filtered = [r for r in filtered if r.get("category", "Unknown") in cat_filter]
if search:
    filtered = [r for r in filtered if search.lower() in _question(r).lower()]

if not filtered:
    st.info("No tasks match the current filters.")
    st.stop()

rows = []
for i, r in enumerate(filtered):
    q = _question(r)
    rows.append({
        "#": i + 1,
        "instance_id": r.get("instance_id", ""),
        "database": r.get("selected_database", ""),
        "Question": (q[:80] + "…") if len(q) > 80 else q,
        "category": r.get("category", ""),
        "Accuracy": "✓" if r.get("execution_accuracy") else "✗",
        "tokens": r.get("total_tokens", 0),
        "cost": r.get("total_cost", 0.0),
    })

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
_render_conversation(selected_record)
```

- [ ] **Step 2: Start streamlit and visually verify the app**

```bash
uv run streamlit run explorer/app.py
```

Open the URL shown (usually `http://localhost:8501`). Check:
- Sidebar shows baseline / date / time dropdowns, pre-selects the newest run
- Model name and temperature shown below selectors
- Four metric cards visible (Accuracy, Avg Tokens, Avg Cost, Avg Budget Remaining)
- Accuracy by Category bar chart rendered (Tool Usage chart hidden for no_tool)
- Task list table shows all 10 tasks with filter controls
- Clicking a row renders the conversation viewer below with system prompt expander, AI message, and tool message
- "Ground-truth SQL" and "Predicted SQL" expanders open correctly

Stop the server with Ctrl-C when done.

- [ ] **Step 3: Run full test suite to confirm no regressions**

```bash
uv run pytest tests/ --ignore=tests/eval_framework/integration -v 2>&1 | tail -20
```

Expected: all tests PASS.

- [ ] **Step 4: Commit**

```bash
git add explorer/app.py
git commit -m "feat: add results explorer Streamlit app"
```

---

## Self-review

**Spec coverage:**
- ✓ Run selector (sidebar cascading dropdowns, newest pre-selected)
- ✓ Metadata line (model_name, temperature from config.yaml)
- ✓ Four metric cards
- ✓ Accuracy by category bar chart
- ✓ Tool usage chart (hidden when empty)
- ✓ Pass/fail radio, category multiselect, question text search
- ✓ Dataframe with correct columns and row selection
- ✓ Conversation viewer with header, SQL expanders, role-based messages
- ✓ Collapsible system prompt, tool_calls, tool output
- ✓ AI message metadata caption
- ✓ Fallback to results.jsonl when results_smaller.jsonl missing
- ✓ Malformed line warning
- ✓ Missing config.yaml handled silently
- ✓ Missing record fields default to 0

**Placeholder scan:** None found. All steps contain complete code.

**Type consistency:** `RunData`, `RunStats`, `load_run`, `list_runs`, `_compute_stats` names are consistent across loader.py, test_loader.py, and app.py.
