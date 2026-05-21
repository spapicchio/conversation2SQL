# KB Linearization — Move to Agent Layer

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move KB-to-string linearization out of the dataset reader into a shared `agents/utils_kb_linearize.py` module; replace the per-entry subgraph strategy with a single flat string (Strategy 1 from `docs/kb_linearization.md`).

**Architecture:** A new pure-Python module owns all KB linearization. The reader stores only the raw DAG dict (`masked_agent_kb: dict[str, ExternalKnowledgeEntry]`). The three KB env tools call the new module at tool-call time when `is_kb_linearized=True`; otherwise they fall back to the existing raw-JSON dump. `no_tool_baseline` uses the same helper to build its prompt.

**Tech Stack:** Python 3.12, Pydantic v2, pytest, uv

---

## File map

| Action | File |
|--------|------|
| **Create** | `src/conversation2sql/eval_framework/agents/utils_kb_linearize.py` |
| **Create** | `tests/eval_framework/agents/test_utils_kb_linearize.py` |
| **Modify** | `src/conversation2sql/eval_framework/state.py` |
| **Modify** | `src/conversation2sql/eval_framework/dataset_readers/bird_interact_reader.py` |
| **Delete** | `src/conversation2sql/eval_framework/dataset_readers/utils_kb.py` |
| **Modify** | `src/conversation2sql/eval_framework/agents/bird_baseline/tools/bird_interact_env_tools.py` |
| **Modify** | `src/conversation2sql/eval_framework/agents/no_tool_baseline/baseline_model.py` |
| **Modify** | `tests/eval_framework/test_main_pipe_workflow.py` (drop stale fixture key) |
| **Modify** | `tests/eval_framework/tools/test_bird_interact_env_tools.py` (add linearized-branch tests) |
| **Modify** | `src/conversation2sql/eval_framework/CLAUDE.md` |
| **Modify** | `src/conversation2sql/eval_framework/dataset_readers/CLAUDE.md` |
| **Modify** | `src/conversation2sql/eval_framework/agents/CLAUDE.md` |

---

### Task 1: Create `agents/utils_kb_linearize.py` (test-first)

**Files:**
- Create: `tests/eval_framework/agents/test_utils_kb_linearize.py`
- Create: `src/conversation2sql/eval_framework/agents/utils_kb_linearize.py`

- [ ] **Step 1.1: Write the failing tests**

Create `tests/eval_framework/agents/test_utils_kb_linearize.py`:

```python
"""Tests for the Strategy-1 KB linearizer in agents/utils_kb_linearize.py."""
import pytest

from conversation2sql.eval_framework.agents.utils_kb_linearize import (
    format_entry_line,
    linearize_kb,
)
from conversation2sql.eval_framework.state import ExternalKnowledgeEntry


def _entry(id: int, knowledge: str, definition: str = "", description: str = "",
           children: list[int] | None = None) -> ExternalKnowledgeEntry:
    return ExternalKnowledgeEntry(
        id=id,
        knowledge=knowledge,
        description=description,
        definition=definition,
        type="domain_knowledge",
        children_knowledge=children or [],
    )


# ---------------------------------------------------------------------------
# linearize_kb
# ---------------------------------------------------------------------------

def test_empty_kb_returns_empty_string():
    assert linearize_kb({}) == ""


def test_single_leaf_entry_no_edges_block():
    """A standalone entry has no dependencies → no edges header."""
    kb = {"Active User (AU)": _entry(1, "Active User (AU)",
                                     description="logged in ≤ 30 days",
                                     definition="days_since_login <= 30")}
    result = linearize_kb(kb)
    assert "# Dependency edges" not in result
    assert "# Definitions" in result
    assert "[AU] Active User" in result
    assert "days_since_login <= 30" in result


def test_two_entries_prerequisite_produces_edge():
    """B depends on A → edge (A, prerequisite_of, B) appears."""
    a = _entry(1, "Net Profit (NP)", definition="Revenue - Costs")
    b = _entry(2, "Net Profit Margin (NPM)",
               definition=r"\frac{NP}{REV}", children=[1])
    kb = {"Net Profit (NP)": a, "Net Profit Margin (NPM)": b}
    result = linearize_kb(kb)
    assert "# Dependency edges" in result
    assert "(NP, prerequisite_of, NPM)" in result
    assert "# Definitions" in result
    # A (leaf) must appear before B (dependent) in definitions
    def_section = result.split("# Definitions")[1]
    assert def_section.index("[NP]") < def_section.index("[NPM]")


def test_shared_prereq_appears_once_in_definitions():
    """When A is a prereq of both B and C, A must appear exactly once
    in the definitions block regardless of how many dependents it has."""
    a = _entry(1, "Base (B)")
    b = _entry(2, "Derived1 (D1)", children=[1])
    c = _entry(3, "Derived2 (D2)", children=[1])
    kb = {"Base (B)": a, "Derived1 (D1)": b, "Derived2 (D2)": c}
    result = linearize_kb(kb)
    assert result.count("[B] Base") == 1
    assert "(B, prerequisite_of, D1)" in result
    assert "(B, prerequisite_of, D2)" in result


def test_latex_frac_simplified():
    """\\frac{X}{Y} in definition must appear as (X) / (Y) in output."""
    kb = {"Rate (R)": _entry(1, "Rate (R)", definition=r"\frac{A}{B}")}
    result = linearize_kb(kb)
    assert "(A) / (B)" in result
    assert r"\frac" not in result


def test_sentinel_children_minus_one_treated_as_no_deps():
    """`children_knowledge=[-1]` is the raw dataset sentinel for 'no deps'.
    No edge should be generated for it."""
    kb = {
        "Alpha": _entry(1, "Alpha", children=[-1]),
        "Beta":  _entry(2, "Beta",  children=[-1]),
    }
    result = linearize_kb(kb)
    assert "# Dependency edges" not in result


# ---------------------------------------------------------------------------
# format_entry_line
# ---------------------------------------------------------------------------

def test_format_entry_line_missing_name():
    kb = {"Foo (F)": _entry(1, "Foo (F)")}
    assert format_entry_line("does_not_exist", kb) == "Knowledge not found."


def test_format_entry_line_matches_linearize_kb_line():
    """The line returned by format_entry_line must be byte-for-byte the same
    as the corresponding line in the linearize_kb output."""
    entry = _entry(1, "Active User (AU)",
                   description="logged in recently",
                   definition="days <= 30")
    kb = {"Active User (AU)": entry}
    flat = linearize_kb(kb)
    def_lines = [l for l in flat.splitlines() if l.startswith("[")]
    assert len(def_lines) == 1
    assert format_entry_line("Active User (AU)", kb) == def_lines[0]


def test_format_entry_line_no_subgraph_context():
    """format_entry_line returns ONLY the named entry's line —
    no prerequisites are bundled in."""
    a = _entry(1, "Base (B)", description="the base")
    b = _entry(2, "Derived (D)", description="uses base", children=[1])
    kb = {"Base (B)": a, "Derived (D)": b}
    line = format_entry_line("Derived (D)", kb)
    assert "[B]" not in line
    assert "[D]" in line
```

- [ ] **Step 1.2: Run tests to confirm they fail**

```bash
uv run pytest tests/eval_framework/agents/test_utils_kb_linearize.py -v
```

Expected: `ModuleNotFoundError: No module named 'conversation2sql.eval_framework.agents.utils_kb_linearize'`

- [ ] **Step 1.3: Create `agents/utils_kb_linearize.py`**

Create `src/conversation2sql/eval_framework/agents/utils_kb_linearize.py`:

```python
"""Strategy-1 KB linearizer: single flat string for the entire KB DAG.

Public interface
----------------
linearize_kb(masked_agent_kb)      -> str   (whole KB as one formatted block)
format_entry_line(name, masked_agent_kb) -> str  (one definition line, or not-found sentinel)
"""
from __future__ import annotations

import re

from conversation2sql.eval_framework.state import ExternalKnowledgeEntry

# ---------------------------------------------------------------------------
# LaTeX -> code-like notation
# ---------------------------------------------------------------------------
_SIMPLE_LATEX_REPLACEMENTS: tuple[tuple[str, str], ...] = (
    (r"\times", " * "),
    (r"\cdot", " * "),
    (r"\div", " / "),
    (r"\leq", " <= "),
    (r"\geq", " >= "),
    (r"\neq", " != "),
    (r"\le ", " <= "),
    (r"\ge ", " >= "),
    (r"\ne ", " != "),
    (r"\pm", " +/- "),
    (r"\left", ""),
    (r"\right", ""),
    (r"\,", " "),
    (r"\;", " "),
    (r"\:", " "),
    (r"\!", ""),
    (r"\%", "%"),
    (r"\$", "$"),
)

_UNWRAP_COMMANDS: tuple[str, ...] = (
    r"\text",
    r"\textit",
    r"\textbf",
    r"\textrm",
    r"\mathrm",
    r"\mathbf",
    r"\mathit",
    r"\mathsf",
    r"\operatorname",
)


def _extract_braced(s: str, start: int) -> tuple[str, int]:
    if start >= len(s) or s[start] != "{":
        return "", start
    depth = 0
    for i in range(start, len(s)):
        if s[i] == "{":
            depth += 1
        elif s[i] == "}":
            depth -= 1
            if depth == 0:
                return s[start + 1 : i], i + 1
    return s[start + 1 :], len(s)


def _convert_cases(body: str) -> str:
    rows = re.split(r"\\\\", body)
    out: list[str] = []
    first = True
    for row in rows:
        row = row.strip()
        if not row:
            continue
        parts = row.split("&", 1)
        if len(parts) != 2:
            out.append(row)
            continue
        expr = parts[0].strip()
        cond = re.sub(r"\\text\{([^{}]*)\}", r"\1", parts[1]).strip()
        if "otherwise" in cond.lower() or cond.lower() in {"", "else"}:
            out.append(f"else: {expr}")
            first = False
            continue
        keyword = "if" if first else "elif"
        m = re.match(r"if\s+(.*)", cond, flags=re.IGNORECASE)
        condition = m.group(1).strip() if m else cond
        out.append(f"{keyword} {condition}: {expr}")
        first = False
    return "{ " + "; ".join(out) + " }"


def _process_braced_commands(s: str) -> str:
    candidates: list[tuple[str, int]] = []
    m = re.search(r"\\begin\{cases\}", s)
    if m:
        candidates.append(("cases", m.start()))
    for cmd in (r"\frac", r"\sqrt", *_UNWRAP_COMMANDS):
        idx = s.find(cmd)
        if idx >= 0:
            candidates.append((cmd, idx))
    if not candidates:
        return s
    candidates.sort(key=lambda pair: pair[1])
    cmd, idx = candidates[0]
    if cmd == "cases":
        end_match = re.search(r"\\end\{cases\}", s[idx:])
        if not end_match:
            return s
        body_start = idx + len(r"\begin{cases}")
        body_end = idx + end_match.start()
        return s[:idx] + _convert_cases(s[body_start:body_end]) + s[idx + end_match.end():]
    brace_start = idx + len(cmd)
    if cmd == r"\frac":
        a, after_a = _extract_braced(s, brace_start)
        if after_a == brace_start:
            return s
        b, after_b = _extract_braced(s, after_a)
        if after_b == after_a:
            return s
        return s[:idx] + f"({a}) / ({b})" + s[after_b:]
    if cmd == r"\sqrt":
        x, after = _extract_braced(s, brace_start)
        if after == brace_start:
            return s
        return s[:idx] + f"sqrt({x})" + s[after:]
    x, after = _extract_braced(s, brace_start)
    if after == brace_start:
        return s
    return s[:idx] + x + s[after:]


def simplify_latex(definition: str) -> str:
    """Rewrite LaTeX formula into simple code-like notation (best-effort)."""
    if not definition:
        return definition
    s = definition
    for old, new in _SIMPLE_LATEX_REPLACEMENTS:
        s = s.replace(old, new)
    prev: str | None = None
    while prev != s:
        prev = s
        s = _process_braced_commands(s)
    s = re.sub(r"\^\{([^{}]*)\}", r"**(\1)", s)
    s = re.sub(r"\^(\w)", r"**\1", s)
    s = re.sub(r"\\\\", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


# ---------------------------------------------------------------------------
# Token helpers
# ---------------------------------------------------------------------------
_TOKEN_RE = re.compile(r"\(([A-Za-z][A-Za-z0-9_]{1,15})\)\s*$")


def _extract_token(knowledge: str) -> str:
    """Return the trailing (ACRONYM) if present, else the whole name."""
    m = _TOKEN_RE.search(knowledge)
    return m.group(1) if m else knowledge.strip()


def _strip_token(knowledge: str) -> str:
    return _TOKEN_RE.sub("", knowledge).strip()


# ---------------------------------------------------------------------------
# Topological sort (Kahn's algorithm, stable on id)
# ---------------------------------------------------------------------------
def _topological_sort(
    nodes: list[ExternalKnowledgeEntry],
) -> list[ExternalKnowledgeEntry]:
    """Return nodes leaves-first. IDs within each ready layer are sorted for stability."""
    node_ids = {n.id for n in nodes}
    prereqs: dict[int, set[int]] = {
        n.id: {c for c in (n.children_knowledge or []) if c in node_ids}
        for n in nodes
    }
    remaining = {n.id: n for n in nodes}
    placed: set[int] = set()
    ordered: list[ExternalKnowledgeEntry] = []
    while remaining:
        ready = sorted(nid for nid in remaining if prereqs[nid].issubset(placed))
        if not ready:
            ready = [min(remaining)]
        for nid in ready:
            ordered.append(remaining.pop(nid))
            placed.add(nid)
    return ordered


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------
def _format_line(entry: ExternalKnowledgeEntry, token: str) -> str:
    name = _strip_token(entry.knowledge)
    desc = (entry.description or "").strip().rstrip(".")
    formula = simplify_latex(entry.definition or "")
    suffix = f" - formula: {formula}" if formula else ""
    prefix = f"[{token}] {name}" if name else f"[{token}]"
    return f"{prefix} - {desc}{suffix}" if desc else f"{prefix}{suffix}"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def linearize_kb(masked_agent_kb: dict[str, ExternalKnowledgeEntry]) -> str:
    """Strategy 1: single flat string for the whole (masked) KB.

    Empty KB → ``""``.
    No-edge KB → just the ``# Definitions`` block.
    """
    if not masked_agent_kb:
        return ""

    entries = list(masked_agent_kb.values())
    ordered = _topological_sort(entries)
    in_kb = {n.id for n in ordered}
    token_of = {n.id: _extract_token(n.knowledge) for n in ordered}

    edges = [
        (token_of[child_id], token_of[n.id])
        for n in ordered
        for child_id in (n.children_knowledge or [])
        if child_id in in_kb
    ]

    lines: list[str] = []
    if edges:
        lines.append("# Dependency edges (prerequisite -> dependent)")
        lines.extend(f"({a}, prerequisite_of, {b})" for a, b in edges)
        lines.append("")
    lines.append("# Definitions (topological order: leaves first)")
    for n in ordered:
        lines.append(_format_line(n, token_of[n.id]))
    return "\n".join(lines)


def format_entry_line(
    name: str,
    masked_agent_kb: dict[str, ExternalKnowledgeEntry],
) -> str:
    """Return the single ``[TOKEN] name - desc - formula: …`` line for ``name``.

    Returns ``"Knowledge not found."`` when ``name`` is absent from the KB.
    """
    entry = masked_agent_kb.get(name)
    if entry is None:
        return "Knowledge not found."
    return _format_line(entry, _extract_token(entry.knowledge))
```

- [ ] **Step 1.4: Run tests to verify they pass**

```bash
uv run pytest tests/eval_framework/agents/test_utils_kb_linearize.py -v
```

Expected: all 9 tests **PASS**.

- [ ] **Step 1.5: Commit**

```bash
git add src/conversation2sql/eval_framework/agents/utils_kb_linearize.py \
        tests/eval_framework/agents/test_utils_kb_linearize.py
git commit -m "feat: add agents/utils_kb_linearize.py with Strategy-1 flat KB string"
```

---

### Task 2: Update `state.py` and fix the stale test fixture

**Files:**
- Modify: `src/conversation2sql/eval_framework/state.py:59-64`
- Modify: `tests/eval_framework/test_main_pipe_workflow.py:56`

- [ ] **Step 2.1: Edit `state.py`** — drop `masked_agent_kb_linearized`, tighten type

In `src/conversation2sql/eval_framework/state.py`, replace lines 59–64:

```python
    # Before
    masked_agent_kb: dict[str, ExternalKnowledgeEntry | str] = Field(default_factory=dict)
    # Per-entry linearized subgraph (triples + topologically ordered definitions),
    # keyed by `knowledge` name. See dataset_readers/utils_kb.py.
    masked_agent_kb_linearized: dict[str, str] = Field(default_factory=dict)
    # When True the KB tools expose masked_agent_kb_linearized instead of masked_agent_kb.
    is_kb_linearized: bool = False
```

with:

```python
    # After
    masked_agent_kb: dict[str, ExternalKnowledgeEntry] = Field(default_factory=dict)
    # When True the KB tools linearize output via agents/utils_kb_linearize.py.
    is_kb_linearized: bool = False
```

- [ ] **Step 2.2: Fix the stale fixture in `test_main_pipe_workflow.py`**

In `tests/eval_framework/test_main_pipe_workflow.py` inside `_fake_task()`, remove the `"masked_agent_kb_linearized": {}` key from the dict returned by `t.model_dump.return_value`:

```python
    # Before
    t.model_dump.return_value = {
        "instance_id": "task_1",
        "selected_database": "db1",
        "amb_user_query": "q?",
        "sol_sql": "SELECT 1",
        "sql_query_conditions": {},
        "not_ambiguos_query": "q",
        "gt_knowledge_base": [],
        "category": "easy",
        "ddl_database_schema": "CREATE TABLE t (id INT);",
        "masked_agent_kb_linearized": {},
        "user_query_ambiguity": {},
    }

    # After
    t.model_dump.return_value = {
        "instance_id": "task_1",
        "selected_database": "db1",
        "amb_user_query": "q?",
        "sol_sql": "SELECT 1",
        "sql_query_conditions": {},
        "not_ambiguos_query": "q",
        "gt_knowledge_base": [],
        "category": "easy",
        "ddl_database_schema": "CREATE TABLE t (id INT);",
        "user_query_ambiguity": {},
    }
```

- [ ] **Step 2.3: Run full test suite to verify no regressions**

```bash
uv run pytest tests/ -v --tb=short 2>&1 | tail -30
```

Expected: all existing tests pass. There may be failures in env-tools tests if they still import `masked_agent_kb_linearized` — those are fixed in Task 4.

- [ ] **Step 2.4: Commit**

```bash
git add src/conversation2sql/eval_framework/state.py \
        tests/eval_framework/test_main_pipe_workflow.py
git commit -m "refactor: drop masked_agent_kb_linearized from TaskData, tighten type"
```

---

### Task 3: Clean up the reader and delete `utils_kb.py`

**Files:**
- Modify: `src/conversation2sql/eval_framework/dataset_readers/bird_interact_reader.py`
- Delete: `src/conversation2sql/eval_framework/dataset_readers/utils_kb.py`

- [ ] **Step 3.1: Edit `bird_interact_reader.py`**

**a) Remove the import** (line 20):

```python
# Remove this line entirely:
from conversation2sql.eval_framework.dataset_readers.utils_kb import linearize_kb
```

**b) In `load_bird_interact_as_tasks`**, remove the two-line linearization block and the `masked_agent_kb_linearized` kwarg from the `TaskData(...)` call.

Before (lines ~366–400 in `load_bird_interact_as_tasks`):

```python
            linearized_kb = linearize_kb(masked_agent_kb)
            if is_kb_linearized:
                masked_agent_kb = linearized_kb

            sample = TaskData(
                instance_id=line.pop("instance_id"),
                ...
                masked_agent_kb=masked_agent_kb,
                masked_agent_kb_linearized=linearized_kb,
                ...
                is_kb_linearized=is_kb_linearized,
                **line,
            )
```

After:

```python
            sample = TaskData(
                instance_id=line.pop("instance_id"),
                ...
                masked_agent_kb=masked_agent_kb,
                ...
                is_kb_linearized=is_kb_linearized,
                **line,
            )
```

(Remove only the `linearized_kb = linearize_kb(masked_agent_kb)` line, the `if is_kb_linearized: masked_agent_kb = linearized_kb` block, and the `masked_agent_kb_linearized=linearized_kb,` kwarg. Leave everything else unchanged.)

- [ ] **Step 3.2: Delete `utils_kb.py`**

```bash
git rm src/conversation2sql/eval_framework/dataset_readers/utils_kb.py
```

- [ ] **Step 3.3: Run tests to confirm nothing broke**

```bash
uv run pytest tests/ -v --tb=short 2>&1 | tail -30
```

Expected: all tests that were passing before still pass. No import errors from `utils_kb`.

- [ ] **Step 3.4: Commit**

```bash
git add src/conversation2sql/eval_framework/dataset_readers/bird_interact_reader.py
git commit -m "refactor: reader emits raw DAG dict only; remove utils_kb.py linearization"
```

---

### Task 4: Update `bird_interact_env_tools.py` and add linearized-branch tests

**Files:**
- Modify: `src/conversation2sql/eval_framework/agents/bird_baseline/tools/bird_interact_env_tools.py`
- Modify: `tests/eval_framework/tools/test_bird_interact_env_tools.py`

- [ ] **Step 4.1: Write the failing tests for the linearized branch**

Append to the end of `tests/eval_framework/tools/test_bird_interact_env_tools.py`:

```python
# ---------------------------------------------------------------------------
# Linearized branch — is_kb_linearized=True
# ---------------------------------------------------------------------------
import dataclasses


def _task_data_linearized(masked_agent_kb, task_data):
    """Return a TaskData copy with is_kb_linearized=True."""
    return task_data.model_copy(update={"is_kb_linearized": True})


def test_get_knowledge_definition_linearized_returns_single_line(task_data, masked_agent_kb):
    """With is_kb_linearized=True, the tool returns one formatted line for the entry,
    not a JSON-dumped ExternalKnowledgeEntry."""
    ctx = _task_data_linearized(masked_agent_kb, task_data)
    raw = _invoke_tool(
        env_tools.get_knowledge_definition,
        knowledge_name="active_user",
        runtime=_Runtime(ctx),
    )
    decoded = json.loads(raw)
    assert "knowledge" in decoded
    # The value is a plain string (not a JSON object nested inside it)
    assert isinstance(decoded["knowledge"], str)
    assert "[active_user]" in decoded["knowledge"]
    # No subgraph context — no entry for "revenue" in the active_user line
    assert "revenue" not in decoded["knowledge"].lower()


def test_get_knowledge_definition_linearized_missing_returns_sentinel(task_data):
    """Missing name under is_kb_linearized=True → same not-found sentinel."""
    ctx = _task_data_linearized(task_data.masked_agent_kb, task_data)
    raw = _invoke_tool(
        env_tools.get_knowledge_definition,
        knowledge_name="ghost",
        runtime=_Runtime(ctx),
    )
    assert json.loads(raw) == {"knowledge": "Knowledge not found."}


def test_get_all_knowledge_definitions_linearized_returns_flat_string(task_data, masked_agent_kb):
    """With is_kb_linearized=True, the tool returns a single flat string,
    not a list of per-entry JSON strings."""
    ctx = _task_data_linearized(masked_agent_kb, task_data)
    raw = _invoke_tool(
        env_tools.get_all_knowledge_definitions,
        runtime=_Runtime(ctx),
    )
    decoded = json.loads(raw)
    assert "knowledge" in decoded
    # Value is a string (the flat linearized block), not a list
    assert isinstance(decoded["knowledge"], str)
    assert "# Definitions" in decoded["knowledge"]


def test_get_all_external_knowledge_names_same_regardless_of_linearized_flag(task_data, masked_agent_kb):
    """Names are identical whether linearized=True or False."""
    ctx_false = task_data
    ctx_true = _task_data_linearized(masked_agent_kb, task_data)
    names_false = json.loads(_invoke_tool(env_tools.get_all_external_knowledge_names, runtime=_Runtime(ctx_false)))
    names_true  = json.loads(_invoke_tool(env_tools.get_all_external_knowledge_names, runtime=_Runtime(ctx_true)))
    assert sorted(names_false["names"]) == sorted(names_true["names"])
```

Also add the import at the top of the test file (if not already present):
```python
import dataclasses  # (only if not already imported)
```

- [ ] **Step 4.2: Run the new tests to confirm they fail**

```bash
uv run pytest tests/eval_framework/tools/test_bird_interact_env_tools.py \
  -k "linearized" -v
```

Expected: FAIL — `get_knowledge_definition` still reads `masked_agent_kb_linearized` which no longer exists.

- [ ] **Step 4.3: Update `bird_interact_env_tools.py`**

**a) Add import** at the top (after existing imports):

```python
from conversation2sql.eval_framework.agents.utils_kb_linearize import (
    format_entry_line,
    linearize_kb,
)
```

**b) Replace the `get_all_external_knowledge_names` tool** (remove the `is_kb_linearized` branch):

```python
@tool
def get_all_external_knowledge_names(
        runtime: ToolRuntime[TaskData, CustomAgentState],
) -> str:
    """Get the names of all available external knowledge entries for this database.
    Use this to discover what domain knowledge is available.
    Cost: 0.5 bird-coins.

    Returns:
        JSON list of knowledge entry names.
    """
    return json.dumps(
        get_all_external_knowledge_names_impl(masked_agent_kb=runtime.context.masked_agent_kb),
        indent=2,
    )
```

**c) Replace the `get_knowledge_definition` tool**:

```python
@tool
def get_knowledge_definition(
        knowledge_name: str,
        runtime: ToolRuntime[TaskData, CustomAgentState],
) -> str:
    """Get the definition/details of a specific external knowledge entry.
    Cost: 0.5 bird-coins.

    Args:
        knowledge_name: The name of the knowledge entry to look up.

    Returns:
        JSON string with the knowledge definition.
    """
    if runtime.context.is_kb_linearized:
        line = format_entry_line(knowledge_name, runtime.context.masked_agent_kb)
        return json.dumps({"knowledge": line}, indent=2)
    return json.dumps(
        get_knowledge_definition_impl(
            knowledge_name=knowledge_name,
            masked_agent_kb=runtime.context.masked_agent_kb,
        ),
        indent=2,
    )
```

**d) Replace the `get_all_knowledge_definitions` tool**:

```python
@tool
def get_all_knowledge_definitions(
        runtime: ToolRuntime[TaskData, CustomAgentState],
) -> str:
    """Return all external knowledge with definitions (cost: 1 patience)."""
    if runtime.context.is_kb_linearized:
        flat = linearize_kb(runtime.context.masked_agent_kb)
        return json.dumps({"knowledge": flat}, indent=2)
    return json.dumps(
        get_all_knowledge_definitions_impl(
            masked_agent_kb=runtime.context.masked_agent_kb,
        ),
        indent=2,
    )
```

**e) Update type hints on the three `*_impl` functions** — replace `dict[str, ExternalKnowledgeEntry]` where there was `dict[str, ExternalKnowledgeEntry | str]` (the union is no longer needed):

In `get_all_external_knowledge_names_impl`, `get_knowledge_definition_impl`, and `get_all_knowledge_definitions_impl`, change the parameter type annotation from `dict[str, ExternalKnowledgeEntry]` to `dict[str, ExternalKnowledgeEntry]` — (they already have this type; just remove `| str` if it appears). Check each function signature and remove `| str` if present.

- [ ] **Step 4.4: Run all tests to verify the linearized branch tests pass**

```bash
uv run pytest tests/eval_framework/tools/test_bird_interact_env_tools.py -v
```

Expected: all tests **PASS** (existing non-linearized tests + new linearized tests).

- [ ] **Step 4.5: Commit**

```bash
git add src/conversation2sql/eval_framework/agents/bird_baseline/tools/bird_interact_env_tools.py \
        tests/eval_framework/tools/test_bird_interact_env_tools.py
git commit -m "feat: KB tools call linearize_kb/format_entry_line at tool-call time"
```

---

### Task 5: Update `no_tool_baseline/baseline_model.py`

**Files:**
- Modify: `src/conversation2sql/eval_framework/agents/no_tool_baseline/baseline_model.py`

- [ ] **Step 5.1: Add import and update the `run_baseline_no_tool` function**

In `src/conversation2sql/eval_framework/agents/no_tool_baseline/baseline_model.py`:

**Add import** near the top (after the existing imports):

```python
from conversation2sql.eval_framework.agents.utils_kb_linearize import linearize_kb
```

**Replace the `build_omnisql_prompt` call** so that the `kb` param is the linearized string when the flag is set:

```python
    # Before
    user_messages = build_omnisql_prompt(
        params={
            "schema": single_task.ddl_database_schema,
            "question": single_task.task_question,
            "kb": single_task.masked_agent_kb,
        }
    )

    # After
    kb_for_prompt = (
        linearize_kb(single_task.masked_agent_kb)
        if single_task.is_kb_linearized
        else single_task.masked_agent_kb
    )
    user_messages = build_omnisql_prompt(
        params={
            "schema": single_task.ddl_database_schema,
            "question": single_task.task_question,
            "kb": kb_for_prompt,
        }
    )
```

- [ ] **Step 5.2: Run full test suite**

```bash
uv run pytest tests/ -v --tb=short 2>&1 | tail -30
```

Expected: all tests pass.

- [ ] **Step 5.3: Commit**

```bash
git add src/conversation2sql/eval_framework/agents/no_tool_baseline/baseline_model.py
git commit -m "feat: no_tool_baseline uses linearize_kb for prompt when is_kb_linearized=True"
```

---

### Task 6: Update CLAUDE.md documentation

**Files:**
- Modify: `src/conversation2sql/eval_framework/CLAUDE.md`
- Modify: `src/conversation2sql/eval_framework/dataset_readers/CLAUDE.md`
- Modify: `src/conversation2sql/eval_framework/agents/CLAUDE.md`

- [ ] **Step 6.1: Update `eval_framework/CLAUDE.md`**

Replace the `masked_agent_kb_linearized` bullet in the TaskData section:

```markdown
# Before
- `masked_agent_kb` — KB with ambiguous entries deleted (the agent must ask the user to resolve them)
- `masked_agent_kb_linearized` — pre-linearized KB subgraphs keyed by knowledge name

# After
- `masked_agent_kb` — KB with ambiguous entries deleted; `dict[str, ExternalKnowledgeEntry]` encoding the DAG via `children_knowledge` IDs
- `is_kb_linearized` — when `True` the KB tools call `agents/utils_kb_linearize.py` to produce a flat string at tool-call time; the reader no longer pre-computes any string representation
```

- [ ] **Step 6.2: Update `dataset_readers/CLAUDE.md`**

Replace the `## utils_kb.py` section at the end:

```markdown
# Before
## utils_kb.py

`linearize_kb(masked_agent_kb)` produces a per-entry string representation of the KB subgraph (triples + topologically ordered definitions). Used to populate `masked_agent_kb_linearized` in `TaskData`.

# After
(Remove the section entirely — utils_kb.py has been deleted. The reader
 now stores only the raw `masked_agent_kb: dict[str, ExternalKnowledgeEntry]`.)
```

- [ ] **Step 6.3: Update `agents/CLAUDE.md`**

Add a `## utils_kb_linearize.py` section after `## utils.py`:

```markdown
## utils_kb_linearize.py

Shared KB linearization helpers used by both agents when `is_kb_linearized=True`.

- `linearize_kb(masked_agent_kb)` — Strategy-1 flat string: one `# Dependency edges` block (prereq → dependent triples) followed by all definitions in topological order (leaves first). Returns `""` for an empty KB.
- `format_entry_line(name, masked_agent_kb)` — formats the single `[TOKEN] name - desc - formula: …` line for one entry. Returns `"Knowledge not found."` if the name is absent.

Used by `bird_baseline/tools/bird_interact_env_tools.py` (KB tools when `is_kb_linearized=True`) and `no_tool_baseline/baseline_model.py` (prompt rendering when `is_kb_linearized=True`).
```

- [ ] **Step 6.4: Run the full test suite one final time**

```bash
uv run pytest tests/ -v --tb=short 2>&1 | tail -30
```

Expected: all tests **PASS**.

- [ ] **Step 6.5: Commit**

```bash
git add src/conversation2sql/eval_framework/CLAUDE.md \
        src/conversation2sql/eval_framework/dataset_readers/CLAUDE.md \
        src/conversation2sql/eval_framework/agents/CLAUDE.md
git commit -m "docs: update CLAUDE.md files to reflect KB linearization move to agents/"
```

---

## Self-review

**Spec coverage:**
- ✅ `state.py` — drop `masked_agent_kb_linearized`, tighten type (Task 2)
- ✅ `bird_interact_reader.py` — remove `linearize_kb` call and `masked_agent_kb_linearized` kwarg (Task 3)
- ✅ Delete `dataset_readers/utils_kb.py` (Task 3 step 3.2)
- ✅ New `agents/utils_kb_linearize.py` with `linearize_kb` + `format_entry_line` (Task 1)
- ✅ `get_all_external_knowledge_names` — branch removed (Task 4)
- ✅ `get_knowledge_definition` — linearized returns single line (Task 4)
- ✅ `get_all_knowledge_definitions` — linearized returns flat string (Task 4)
- ✅ `no_tool_baseline` uses linearizer when flag is set (Task 5)
- ✅ Fixture `test_main_pipe_workflow.py` stale key removed (Task 2)
- ✅ New linearized-branch tool tests (Task 4)
- ✅ New `utils_kb_linearize` module tests (Task 1)
- ✅ Three CLAUDE.md files updated (Task 6)

**Placeholder scan:** No TBDs, incomplete steps, or "similar to Task N" references found.

**Type consistency:**
- `linearize_kb` takes `dict[str, ExternalKnowledgeEntry]` everywhere it appears (Tasks 1, 4, 5) ✅
- `format_entry_line(name: str, masked_agent_kb: dict[str, ExternalKnowledgeEntry]) -> str` consistent across Tasks 1 and 4 ✅
- `masked_agent_kb: dict[str, ExternalKnowledgeEntry]` (no `| str`) consistent across Tasks 2, 4 ✅
