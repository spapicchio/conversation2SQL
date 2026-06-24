# Deep Agent Baseline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `deep_agent` evaluation baseline built on LangChain's `deepagents` package, where the agent explores the Postgres DB through a virtual filesystem (`ls`/`read_file`/`grep`/`glob`) and talks to the simulated user via `ask_user`, all under the existing bird-coin patience budget.

**Architecture:** Build the agent with LangChain's `create_agent` (same primitive as `bird_baseline`), composing deepagents' `FilesystemMiddleware` (state-backed virtual filesystem, seeded with DB-info files) with the **reused** patience-budget middleware and the reused `execute_sql`/`ask_user`/`submit_sql` tools. deepagents' default extras (todos, subagents, summarization, FS writes) are **off** by default; each is an `enable_*` ablation flag. We use `create_agent` + composable middleware rather than `create_deep_agent`, because `create_deep_agent` only removes built-in tools via a **globally-registered** `HarnessProfile` keyed by model provider — unworkable for per-run ablations.

**Tech Stack:** Python 3.12, `uv`, LangChain `create_agent`, `deepagents>=0.6.11` (already added to `pyproject.toml`), LangGraph, pytest (`asyncio_mode=auto`), pyrefly.

## Global Constraints

- Always run Python/pytest via `uv run` (project venv with `--system-site-packages`).
- `deepagents>=0.6.11` is the pinned floor (already in `pyproject.toml`).
- New ablation flags default to `False` (minimal agent) and are threaded `ConfigReader` → `TaskData` exactly like `enable_psql_console`.
- The patience-budget invariant must be reused **unchanged** — import the middleware from `bird_baseline/agent_callback.py`; do not reimplement it.
- Reuse `execute_sql`, `submit_sql`, `return_tool_ask_user` verbatim from `bird_baseline/tools`; do not fork them.
- Verification gate for every task: `uv run pytest tests/` and `uv run pyrefly check` pass.

### Pinned deepagents API facts (verified against 0.6.11)

- `from deepagents import FilesystemMiddleware` — `FilesystemMiddleware(*, backend=None, system_prompt=None, ...)`. With `backend=None` it defaults to `deepagents.backends.state.StateBackend` (state-backed virtual FS).
- A constructed `FilesystemMiddleware()` exposes `.tools` as a **mutable list** of tools named: `['ls', 'read_file', 'write_file', 'edit_file', 'glob', 'grep', 'execute']`. Filtering this list restricts the exposed FS tools.
- `.state_schema` is `deepagents.middleware.filesystem.FilesystemState`, which adds `files: dict[str, FileData]` to `AgentState`.
- `FileData` is a `TypedDict`: `{"content": str, "encoding": str}` (`encoding="utf-8"` for text; `created_at`/`modified_at` optional). Seed files by putting a `{path: FileData}` dict under the `files` state key.
- `from deepagents.middleware.filesystem import FilesystemOperation` → `Literal["read", "write"]`. `FilesystemPermission(operations=["write"], paths=["/**"], mode="deny")` denies writes.
- Todos: `from langchain.agents.middleware import TodoListMiddleware`. Summarization: `from langchain.agents.middleware import SummarizationMiddleware`. Subagents: `from deepagents import SubAgentMiddleware` (`SubAgentMiddleware(*, backend, subagents, ...)`).
- `class DeepAgentCustomState(FilesystemState, CustomAgentState): pass` merges cleanly — annotations include `files`, `initial_user_patience`, `updated_user_patience`, `tool_called_patience`, `messages`. (Verified.)
- deepagents middleware are `langchain.agents.middleware.types.AgentMiddleware` subclasses, so `create_agent(..., middleware=[...])` accepts them (verified).

---

## File Structure

- Create: `src/conversation2sql/eval_framework/agents/deep_agent/__init__.py`
- Create: `src/conversation2sql/eval_framework/agents/deep_agent/agent_code_state.py` — merged state schema
- Create: `src/conversation2sql/eval_framework/agents/deep_agent/filesystem_seed.py` — `/db/*` file builder + FS tool costs
- Create: `src/conversation2sql/eval_framework/agents/deep_agent/prompts.py` — Jinja2 system/user prompt
- Create: `src/conversation2sql/eval_framework/agents/deep_agent/agent_code.py` — `run_agent_deep_agent`
- Modify: `src/conversation2sql/config_input.py` — 4 ablation flags + `deep_agent` baseline literal
- Modify: `src/conversation2sql/eval_framework/state.py` — 4 ablation flags on `TaskData`
- Modify: `src/conversation2sql/eval_framework/dataset_readers/bird_interact_reader.py` — thread flags onto `TaskData`
- Modify: `src/conversation2sql/eval_framework/agents/__init__.py` — export `run_agent_deep_agent`
- Modify: `src/conversation2sql/eval_framework/main_pipe_workflow.py` — baseline dispatch
- Modify: `src/conversation2sql/presets.py` — run-dir slug suffixes
- Tests under `tests/eval_framework/agents/deep_agent/`

---

## Task 1: Config + TaskData ablation flags and `deep_agent` baseline

**Files:**
- Modify: `src/conversation2sql/config_input.py`
- Modify: `src/conversation2sql/eval_framework/state.py`
- Modify: `src/conversation2sql/eval_framework/dataset_readers/bird_interact_reader.py`
- Test: `tests/eval_framework/agents/deep_agent/test_config_flags.py`

**Interfaces:**
- Produces: `ConfigReader` fields `deep_enable_todos`, `deep_enable_subagents`, `deep_enable_summarization`, `deep_enable_fs_write` (all `bool`, default `False`); `ConfigPipeline.baseline` accepts `"deep_agent"`; `TaskData` carries the same 4 flags (default `False`).

- [ ] **Step 1: Write the failing test**

Create `tests/eval_framework/agents/deep_agent/__init__.py` (empty) and `tests/eval_framework/agents/deep_agent/test_config_flags.py`:

```python
from conversation2sql.config_input import ConfigPipeline, ConfigReader
from conversation2sql.eval_framework.state import TaskData


def test_config_reader_deep_flags_default_false():
    cfg = ConfigReader()
    assert cfg.deep_enable_todos is False
    assert cfg.deep_enable_subagents is False
    assert cfg.deep_enable_summarization is False
    assert cfg.deep_enable_fs_write is False


def test_pipeline_accepts_deep_agent_baseline():
    cfg = ConfigPipeline(baseline="deep_agent")
    assert cfg.baseline == "deep_agent"


def test_taskdata_carries_deep_flags(make_minimal_task_kwargs):
    task = TaskData(**make_minimal_task_kwargs())
    assert task.deep_enable_todos is False
    assert task.deep_enable_fs_write is False
```

Add a `make_minimal_task_kwargs` fixture to `tests/eval_framework/agents/deep_agent/conftest.py` that returns a callable producing the minimum required `TaskData` kwargs. Inspect `state.py`’s `TaskData` for required fields and mirror an existing fixture (search `tests/` for an existing `task_data` fixture, e.g. in `tests/eval_framework/tools/conftest.py`, and reuse its construction). If a shared `task_data` fixture already exists at a conftest visible here, import its kwargs instead of duplicating.

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/eval_framework/agents/deep_agent/test_config_flags.py -v`
Expected: FAIL — `ConfigReader` has no `deep_enable_todos`; `baseline="deep_agent"` raises a Pydantic `ValidationError`.

- [ ] **Step 3: Implement**

In `config_input.py`, extend the baseline literal (line ~12):

```python
    baseline: Literal['no_tool', 'tools_only', 'tools_user', 'bird_full', 'deep_agent'] = 'bird_full'
```

In `ConfigReader` (after the existing `enable_python_udf` field, ~line 36):

```python
    # --- deep_agent baseline ablations (only meaningful when baseline='deep_agent') ---
    deep_enable_todos: bool = False  # add deepagents planning/write_todos middleware
    deep_enable_subagents: bool = False  # add deepagents subagents (task tool) middleware
    deep_enable_summarization: bool = False  # add deepagents/langchain SummarizationMiddleware
    deep_enable_fs_write: bool = False  # expose write_file/edit_file (default: read-only FS)
```

In `state.py` `TaskData` (after the existing `enable_python_udf` field, ~line 75):

```python
    deep_enable_todos: bool = False
    deep_enable_subagents: bool = False
    deep_enable_summarization: bool = False
    deep_enable_fs_write: bool = False
```

In `bird_interact_reader.py`, find where `enable_python_udf` is copied from the config onto each `TaskData` (grep `enable_python_udf` in that file) and add the four `deep_enable_*` flags alongside it in the same `TaskData(...)` construction.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/eval_framework/agents/deep_agent/test_config_flags.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/conversation2sql/config_input.py src/conversation2sql/eval_framework/state.py src/conversation2sql/eval_framework/dataset_readers/bird_interact_reader.py tests/eval_framework/agents/deep_agent/
git commit -m "feat(deep_agent): add ablation config flags and deep_agent baseline literal

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 2: Filesystem seed — build `/db/*` files from `TaskData`

**Files:**
- Create: `src/conversation2sql/eval_framework/agents/deep_agent/__init__.py` (empty)
- Create: `src/conversation2sql/eval_framework/agents/deep_agent/filesystem_seed.py`
- Test: `tests/eval_framework/agents/deep_agent/test_filesystem_seed.py`

**Interfaces:**
- Consumes: `TaskData` (`.ddl_database_schema`, `.column_meanings`, `.masked_agent_kb`, `.is_kb_linearized`); the reused impls `get_all_column_meanings_impl`, `get_all_knowledge_definitions_impl` from `bird_baseline.tools`.
- Produces: `build_db_filesystem(task: TaskData) -> dict[str, FileData]` returning exactly keys `/db/schema.sql`, `/db/column_meanings.md`, `/db/knowledge_base.md`; `DB_FS_PATHS: tuple[str, str, str]`.

- [ ] **Step 1: Write the failing test**

Create `tests/eval_framework/agents/deep_agent/test_filesystem_seed.py`:

```python
from conversation2sql.eval_framework.agents.deep_agent.filesystem_seed import (
    build_db_filesystem,
    DB_FS_PATHS,
)


def test_builds_exactly_three_db_files(task_data):
    files = build_db_filesystem(task_data)
    assert set(files) == set(DB_FS_PATHS)
    assert set(DB_FS_PATHS) == {
        "/db/schema.sql",
        "/db/column_meanings.md",
        "/db/knowledge_base.md",
    }


def test_schema_file_contains_raw_ddl(task_data):
    files = build_db_filesystem(task_data)
    schema = files["/db/schema.sql"]
    assert schema["encoding"] == "utf-8"
    assert task_data.ddl_database_schema.strip()[:20] in schema["content"]


def test_empty_kb_yields_none_sentinel(task_data):
    task_data.masked_agent_kb = {}
    files = build_db_filesystem(task_data)
    assert files["/db/knowledge_base.md"]["content"].strip() == "(none)"


def test_empty_column_meanings_yields_none_sentinel(task_data):
    task_data.column_meanings = {}
    files = build_db_filesystem(task_data)
    assert files["/db/column_meanings.md"]["content"].strip() == "(none)"
```

Reuse the existing `task_data` fixture. If `tests/eval_framework/agents/deep_agent/conftest.py` does not yet re-expose it, add `from <existing path> import task_data  # noqa` or define a local fixture mirroring the existing one. (Search: `grep -rn "def task_data" tests/`.)

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/eval_framework/agents/deep_agent/test_filesystem_seed.py -v`
Expected: FAIL — module `filesystem_seed` does not exist.

- [ ] **Step 3: Implement**

Create empty `src/conversation2sql/eval_framework/agents/deep_agent/__init__.py`.

Create `src/conversation2sql/eval_framework/agents/deep_agent/filesystem_seed.py`:

```python
"""Build the virtual-filesystem seed exposing the Postgres DB info to the deep agent.

Instead of get_schema-style tools, the deep_agent reads these files via the
deepagents FilesystemMiddleware (ls/read_file/grep/glob). One file per concern.
"""
from __future__ import annotations

from deepagents.middleware.filesystem import FileData

from conversation2sql.eval_framework.agents.bird_baseline.tools import (
    get_all_column_meanings_impl,
    get_all_knowledge_definitions_impl,
)
from conversation2sql.eval_framework.state import TaskData

DB_FS_PATHS: tuple[str, str, str] = (
    "/db/schema.sql",
    "/db/column_meanings.md",
    "/db/knowledge_base.md",
)

_NONE_SENTINEL = "(none)"


def _text_file(content: str) -> FileData:
    body = content if content and content.strip() else _NONE_SENTINEL
    return {"content": body, "encoding": "utf-8"}


def build_db_filesystem(task: TaskData) -> dict[str, FileData]:
    """Render the three DB-info files seeded into the deep agent's filesystem."""
    schema = task.ddl_database_schema or ""
    columns = get_all_column_meanings_impl(task.column_meanings)
    knowledge = get_all_knowledge_definitions_impl(
        task.masked_agent_kb, is_kb_linearized=task.is_kb_linearized
    )
    return {
        "/db/schema.sql": _text_file(schema),
        "/db/column_meanings.md": _text_file(columns),
        "/db/knowledge_base.md": _text_file(knowledge),
    }
```

**Before coding, verify the real signatures** of `get_all_column_meanings_impl` and `get_all_knowledge_definitions_impl` in `bird_baseline/tools/bird_interact_env_tools.py` (grep them). They take the task's column-meanings / KB dict — match their exact parameter names/order. If `get_all_knowledge_definitions_impl` does not accept `is_kb_linearized`, drop that kwarg. If either returns a non-string, coerce to string.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/eval_framework/agents/deep_agent/test_filesystem_seed.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/conversation2sql/eval_framework/agents/deep_agent/__init__.py src/conversation2sql/eval_framework/agents/deep_agent/filesystem_seed.py tests/eval_framework/agents/deep_agent/test_filesystem_seed.py
git commit -m "feat(deep_agent): seed virtual filesystem with DB schema/columns/KB

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 3: Merged state schema + FS tool costs

**Files:**
- Create: `src/conversation2sql/eval_framework/agents/deep_agent/agent_code_state.py`
- Modify: `src/conversation2sql/eval_framework/agents/deep_agent/filesystem_seed.py` (append cost map)
- Test: `tests/eval_framework/agents/deep_agent/test_state_and_costs.py`

**Interfaces:**
- Produces: `DeepAgentCustomState` (merged `FilesystemState` + `CustomAgentState`); `FS_TOOL_COSTS: dict[str, float]` with keys `ls`, `read_file`, `glob`, `grep`, `write_file`, `edit_file`; `deep_tool_costs(*, enable_fs_write: bool) -> dict[str, float]` merging `FS_TOOL_COSTS` (read subset or full) with the reused `execute_sql`/`ask_user`/`submit_sql` costs from `TOOL_COSTS`.

- [ ] **Step 1: Write the failing test**

Create `tests/eval_framework/agents/deep_agent/test_state_and_costs.py`:

```python
from conversation2sql.eval_framework.agents.deep_agent.agent_code_state import (
    DeepAgentCustomState,
)
from conversation2sql.eval_framework.agents.deep_agent.filesystem_seed import (
    FS_TOOL_COSTS,
    deep_tool_costs,
)


def test_merged_state_has_files_and_patience_keys():
    keys = DeepAgentCustomState.__annotations__.keys()
    assert "files" in keys
    assert "updated_user_patience" in keys
    assert "tool_called_patience" in keys


def test_read_only_costs_exclude_write_tools():
    costs = deep_tool_costs(enable_fs_write=False)
    assert "read_file" in costs and "ls" in costs
    assert "write_file" not in costs and "edit_file" not in costs
    # reused tools are accounted for
    assert "execute_sql" in costs and "ask_user" in costs and "submit_sql" in costs


def test_fs_write_costs_include_write_tools():
    costs = deep_tool_costs(enable_fs_write=True)
    assert "write_file" in costs and "edit_file" in costs


def test_fs_tool_costs_are_positive():
    assert all(v > 0 for v in FS_TOOL_COSTS.values())
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/eval_framework/agents/deep_agent/test_state_and_costs.py -v`
Expected: FAIL — `agent_code_state` module / `FS_TOOL_COSTS` missing.

- [ ] **Step 3: Implement**

Create `src/conversation2sql/eval_framework/agents/deep_agent/agent_code_state.py`:

```python
"""Merged agent state: deepagents filesystem (`files`) + bird patience fields."""
from __future__ import annotations

from deepagents.middleware.filesystem import FilesystemState

from conversation2sql.eval_framework.agents.bird_baseline.agent_code_state import (
    CustomAgentState,
)


class DeepAgentCustomState(FilesystemState, CustomAgentState):
    """Both bases are TypedDicts extending AgentState; MRO merges their keys:
    `files` (FilesystemState) + the three patience fields (CustomAgentState)."""
```

Append to `filesystem_seed.py`:

```python
from conversation2sql.eval_framework.agents.bird_baseline.tools import TOOL_COSTS

# Bird-coin costs for the deepagents filesystem read/write tools. Reads are cheap
# (static DB info); writes are only present under the deep_enable_fs_write ablation.
FS_TOOL_COSTS: dict[str, float] = {
    "ls": 0.5,
    "read_file": 0.5,
    "glob": 0.5,
    "grep": 0.5,
    "write_file": 0.5,
    "edit_file": 0.5,
}

_FS_READ_TOOLS = ("ls", "read_file", "glob", "grep")
_FS_WRITE_TOOLS = ("write_file", "edit_file")
# Reused tools the deep_agent always has; pull their costs from the shared table.
_REUSED_TOOLS = ("execute_sql", "ask_user", "submit_sql")


def deep_tool_costs(*, enable_fs_write: bool) -> dict[str, float]:
    """Cost map the patience middleware consults for the deep_agent's tool set."""
    names = list(_FS_READ_TOOLS)
    if enable_fs_write:
        names += list(_FS_WRITE_TOOLS)
    costs = {n: FS_TOOL_COSTS[n] for n in names}
    for n in _REUSED_TOOLS:
        costs[n] = TOOL_COSTS[n]
    return costs
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/eval_framework/agents/deep_agent/test_state_and_costs.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/conversation2sql/eval_framework/agents/deep_agent/agent_code_state.py src/conversation2sql/eval_framework/agents/deep_agent/filesystem_seed.py tests/eval_framework/agents/deep_agent/test_state_and_costs.py
git commit -m "feat(deep_agent): merged state schema and FS tool cost map

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 4: Prompt template

**Files:**
- Create: `src/conversation2sql/eval_framework/agents/deep_agent/prompts.py`
- Test: `tests/eval_framework/agents/deep_agent/test_prompts.py`

**Interfaces:**
- Consumes: `utils_build_messages` from `eval_framework.agents.utils`.
- Produces: `build_deep_agent_messages(params: dict) -> list[dict]`. `params` keys: `total_budget`, `amb_user_query`, `enable_fs_write`, `enable_todos`, `enable_subagents`.

- [ ] **Step 1: Write the failing test**

Create `tests/eval_framework/agents/deep_agent/test_prompts.py`:

```python
from conversation2sql.eval_framework.agents.deep_agent.prompts import (
    build_deep_agent_messages,
)


def _render(**overrides):
    params = {
        "total_budget": 20,
        "amb_user_query": "How many active users?",
        "enable_fs_write": False,
        "enable_todos": False,
        "enable_subagents": False,
    }
    params.update(overrides)
    return build_deep_agent_messages(params)


def test_prompt_mentions_db_filesystem_paths():
    msgs = _render()
    system = msgs[0]["content"]
    assert "/db/schema.sql" in system
    assert "read_file" in system
    # No get_schema-style tools are advertised.
    assert "get_schema" not in system


def test_prompt_includes_user_query_and_budget():
    msgs = _render()
    joined = " ".join(m["content"] for m in msgs)
    assert "How many active users?" in joined
    assert "20" in joined


def test_todos_block_only_when_enabled():
    assert "write_todos" not in _render()[0]["content"]
    assert "write_todos" in _render(enable_todos=True)[0]["content"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/eval_framework/agents/deep_agent/test_prompts.py -v`
Expected: FAIL — module missing.

- [ ] **Step 3: Implement**

Create `src/conversation2sql/eval_framework/agents/deep_agent/prompts.py` (mirror `bird_baseline/prompts.py` structure):

```python
"""Deep-agent prompt templates — inline Jinja2 with typed params."""
from __future__ import annotations

from conversation2sql.eval_framework.agents.utils import utils_build_messages

_DEEP_AGENT_SYSTEM = """
You are a helpful PostgreSQL agent that interacts with a user and a database to solve the user's ambiguous question.

You explore the database NOT through dedicated schema tools, but through a virtual
filesystem under /db. Use the filesystem tools (ls, read_file, grep, glob) to read:
- /db/schema.sql          — the full DDL (tables, columns, foreign keys)
- /db/column_meanings.md  — natural-language meaning of each column
- /db/knowledge_base.md   — external knowledge definitions relevant to the task

You also have:
- execute_sql: run a read-only SELECT/WITH/EXPLAIN against the live database to test a query.
- ask_user: ask the user ONE clarifying question when their intent is ambiguous.
- submit_sql: submit your final SQL for grading (this ends the task).
{% if enable_fs_write %}- write_file / edit_file: scratch space for notes/drafts under /scratch.
{% endif %}{% if enable_todos %}- write_todos: maintain a short task plan.
{% endif %}{% if enable_subagents %}- task: delegate an isolated sub-task to an ephemeral subagent.
{% endif %}
Each action costs bird-coins from a fixed budget; be efficient. The interaction
ends when you submit the correct SQL or the budget runs out.

Strategy:
- Start by reading /db/schema.sql and the column meanings to understand the data.
- grep /db for relevant table/column names instead of reading everything.
- If the user's intent is ambiguous, ask one clarifying question before committing to SQL.
- Test SQL with execute_sql before submit_sql when useful.
- Track your remaining budget and submit before it runs out.
"""

_DEEP_AGENT_USER = """
User's Question:
{{ amb_user_query }}

[SYSTEM NOTE: You have a total action budget of {{ total_budget }} units. Each action consumes budget. If the budget runs out, you must submit.]
"""


def build_deep_agent_messages(params: dict) -> list[dict]:
    return utils_build_messages(_DEEP_AGENT_SYSTEM, _DEEP_AGENT_USER, params)
```

**Before coding:** confirm `utils_build_messages(system, user, params)` argument order by reading `eval_framework/agents/utils.py` (the `bird_baseline` prompt calls it as `utils_build_messages(_SYSTEM, _USER, params)`), and confirm it returns a list whose first element’s `["content"]` is the rendered system text. Adjust the test’s indexing if the returned shape differs (e.g. objects vs dicts).

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/eval_framework/agents/deep_agent/test_prompts.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/conversation2sql/eval_framework/agents/deep_agent/prompts.py tests/eval_framework/agents/deep_agent/test_prompts.py
git commit -m "feat(deep_agent): system/user prompt advertising /db filesystem

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 5: Agent assembly — `run_agent_deep_agent`

**Files:**
- Create: `src/conversation2sql/eval_framework/agents/deep_agent/agent_code.py`
- Test: `tests/eval_framework/agents/deep_agent/test_agent_code.py`

**Interfaces:**
- Consumes: everything above; `create_agent` from `langchain.agents`; `FilesystemMiddleware`, `SubAgentMiddleware` from `deepagents`; `TodoListMiddleware`, `SummarizationMiddleware` from `langchain.agents.middleware`; the patience middleware (`check_budget_limit`, `sanitize_thinking_history`, `wrap_model_append_tool_message`, `tool_wrapper_patience_and_submit`) and `utils_process_agent_response`, `_extract_predicted_sql` reused from `bird_baseline`.
- Produces: `run_agent_deep_agent(single_task, model_agent, model_user_parsing, model_user_generator, *, enable_ask_user=True) -> dict`; helpers `_build_fs_middleware(*, enable_fs_write)`, `_build_deep_middleware(single_task)`, `_build_deep_tools(single_task, model_user_parsing, model_user_generator)`.

The helpers are split out so they can be unit-tested **without invoking a model**.

- [ ] **Step 1: Write the failing test**

Create `tests/eval_framework/agents/deep_agent/test_agent_code.py`:

```python
from conversation2sql.eval_framework.agents.deep_agent import agent_code
from langchain.agents.middleware import TodoListMiddleware, SummarizationMiddleware
from deepagents import FilesystemMiddleware, SubAgentMiddleware


def _task(task_data, **flags):
    for k, v in flags.items():
        setattr(task_data, k, v)
    return task_data


def test_fs_middleware_read_only_by_default(task_data):
    mw = agent_code._build_fs_middleware(enable_fs_write=False)
    names = {t.name for t in mw.tools}
    assert {"ls", "read_file", "glob", "grep"} <= names
    assert "write_file" not in names and "edit_file" not in names
    assert "execute" not in names  # inert on StateBackend; hidden


def test_fs_middleware_adds_writes_when_enabled(task_data):
    mw = agent_code._build_fs_middleware(enable_fs_write=True)
    names = {t.name for t in mw.tools}
    assert "write_file" in names and "edit_file" in names


def test_middleware_minimal_by_default(task_data):
    mws = agent_code._build_deep_middleware(_task(task_data))
    types = {type(m) for m in mws}
    assert FilesystemMiddleware in types
    assert TodoListMiddleware not in types
    assert SummarizationMiddleware not in types
    assert SubAgentMiddleware not in types
    # patience middleware present (match by callable identity)
    from conversation2sql.eval_framework.agents.bird_baseline.agent_callback import (
        tool_wrapper_patience_and_submit,
    )
    assert tool_wrapper_patience_and_submit in mws


def test_middleware_flags_add_components(task_data):
    mws = agent_code._build_deep_middleware(
        _task(
            task_data,
            deep_enable_todos=True,
            deep_enable_summarization=True,
            deep_enable_subagents=True,
        )
    )
    types = {type(m) for m in mws}
    assert TodoListMiddleware in types
    assert SummarizationMiddleware in types
    assert SubAgentMiddleware in types


def test_tools_include_reused_and_ask_user(task_data, make_chat_model):
    tools = agent_code._build_deep_tools(
        _task(task_data),
        model_user_parsing=make_chat_model("<s>x</s>"),
        model_user_generator=make_chat_model("<s>y</s>"),
    )
    names = {t.name for t in tools}
    assert {"execute_sql", "ask_user", "submit_sql"} <= names
```

Reuse `task_data` and `make_chat_model` fixtures (the latter is used in `tests/eval_framework/tools/test_bird_interact_user_tools.py`; re-expose via `conftest.py`).

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/eval_framework/agents/deep_agent/test_agent_code.py -v`
Expected: FAIL — `agent_code` module missing.

- [ ] **Step 3: Implement**

Create `src/conversation2sql/eval_framework/agents/deep_agent/agent_code.py`:

```python
"""deep_agent baseline: deepagents FilesystemMiddleware + bird patience budget."""
from __future__ import annotations

from langchain.agents import create_agent
from langchain.agents.middleware import (
    ModelRetryMiddleware,
    ToolRetryMiddleware,
    SummarizationMiddleware,
    TodoListMiddleware,
)
from langchain_core.language_models import BaseChatModel
from deepagents import FilesystemMiddleware, SubAgentMiddleware
from deepagents.backends.state import StateBackend

from conversation2sql.eval_framework.agents.bird_baseline.agent_callback import (
    check_budget_limit,
    sanitize_thinking_history,
    tool_wrapper_patience_and_submit,
    wrap_model_append_tool_message,
)
from conversation2sql.eval_framework.agents.bird_baseline.agent_code import (
    _extract_predicted_sql,
    utils_process_agent_response,
)
from conversation2sql.eval_framework.agents.bird_baseline.tools import (
    execute_sql,
    submit_sql,
    return_tool_ask_user,
)
from conversation2sql.eval_framework.agents.deep_agent.agent_code_state import (
    DeepAgentCustomState,
)
from conversation2sql.eval_framework.agents.deep_agent.filesystem_seed import (
    build_db_filesystem,
    deep_tool_costs,
)
from conversation2sql.eval_framework.agents.deep_agent.prompts import (
    build_deep_agent_messages,
)
from conversation2sql.eval_framework.state import TaskData
from conversation2sql.logger import get_logger

logger = get_logger(__name__)

_FS_READ_TOOLS = ("ls", "read_file", "glob", "grep")
_FS_WRITE_TOOLS = ("write_file", "edit_file")


def _build_fs_middleware(*, enable_fs_write: bool) -> FilesystemMiddleware:
    """State-backed filesystem, restricted to a read-only subset by default.

    The middleware ships ls/read_file/write_file/edit_file/glob/grep/execute; we
    filter its `.tools` list down to the read subset (plus writes when enabled).
    `execute` is always dropped — it errors on the non-sandbox StateBackend.
    """
    mw = FilesystemMiddleware(backend=StateBackend())
    allowed = set(_FS_READ_TOOLS)
    if enable_fs_write:
        allowed |= set(_FS_WRITE_TOOLS)
    mw.tools = [t for t in mw.tools if t.name in allowed]
    return mw


def _build_deep_tools(
    single_task: TaskData,
    model_user_parsing: BaseChatModel,
    model_user_generator: BaseChatModel,
) -> list:
    tools = [execute_sql, submit_sql,
             return_tool_ask_user(model_user_parsing, model_user_generator)]
    return tools


def _build_deep_middleware(single_task: TaskData) -> list:
    mws: list = [
        ModelRetryMiddleware(max_delay=60.0, on_failure="error"),
        ToolRetryMiddleware(max_delay=60.0, on_failure="error"),
        _build_fs_middleware(enable_fs_write=single_task.deep_enable_fs_write),
    ]
    if single_task.deep_enable_todos:
        mws.append(TodoListMiddleware())
    if single_task.deep_enable_summarization:
        mws.append(SummarizationMiddleware(model=None))  # see note in Step 3b
    if single_task.deep_enable_subagents:
        mws.append(
            SubAgentMiddleware(backend=StateBackend(), subagents=[])
        )
    # Patience budget — appended last, same relative order as bird_baseline.
    mws += [
        check_budget_limit,
        sanitize_thinking_history,
        wrap_model_append_tool_message,
        tool_wrapper_patience_and_submit,
    ]
    return mws


def run_agent_deep_agent(
    single_task: TaskData,
    model_agent: BaseChatModel,
    model_user_parsing: BaseChatModel,
    model_user_generator: BaseChatModel,
    *,
    enable_ask_user: bool = True,
) -> dict:
    assert model_user_parsing is not None and model_user_generator is not None, (
        "deep_agent always uses ask_user; user-sim models must not be None"
    )
    messages = build_deep_agent_messages(
        params={
            "total_budget": single_task.task_budget,
            "amb_user_query": single_task.task_question,
            "enable_fs_write": single_task.deep_enable_fs_write,
            "enable_todos": single_task.deep_enable_todos,
            "enable_subagents": single_task.deep_enable_subagents,
        }
    )
    agent = create_agent(
        model_agent,
        _build_deep_tools(single_task, model_user_parsing, model_user_generator),
        state_schema=DeepAgentCustomState,
        context_schema=TaskData,
        middleware=_build_deep_middleware(single_task),  # pyrefly: ignore
    )
    agent_state = {
        "messages": messages,
        "files": build_db_filesystem(single_task),
        "initial_user_patience": single_task.task_budget,
        "updated_user_patience": single_task.task_budget,
        "tool_called_patience": [],
    }
    response = agent.invoke(agent_state, context=single_task)  # pyrefly: ignore
    predicted_sql = _extract_predicted_sql(response["messages"])
    output = utils_process_agent_response(
        response, tool_costs=deep_tool_costs(enable_fs_write=single_task.deep_enable_fs_write)
    )
    output["predicted_sql"] = predicted_sql
    return output
```

**Step 3b — resolve two API details before finalizing (do NOT leave guesses):**
1. `SummarizationMiddleware(model=...)` — read its signature (`uv run python -c "import inspect; from langchain.agents.middleware import SummarizationMiddleware; print(inspect.signature(SummarizationMiddleware))"`). It likely **requires** a model. Pass `model_agent` through to `_build_deep_middleware` (add it as a parameter) instead of `None`. Update the helper signature and the `test_middleware_*` tests to pass a stub/`make_chat_model` if a model is required.
2. `SubAgentMiddleware(subagents=[])` — confirm an empty list is accepted (`inspect.signature` + a quick construct). If it requires at least one subagent, construct a single general-purpose `SubAgent` reusing `_build_deep_tools`; keep its prompt minimal (detailed subagent design is out of scope per the spec). If construction is non-trivial, gate `deep_enable_subagents` behind a `NotImplementedError` with a clear message and mark the prompt/flag as reserved — but prefer a minimal working subagent.

Adjust the helper signatures/tests to match whatever the real constructors require. The **behavioral contract the tests pin** (read-only FS by default, flags add the right middleware types, patience middleware present, reused tools wired) must hold regardless.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/eval_framework/agents/deep_agent/test_agent_code.py -v`
Expected: PASS. Then `uv run pyrefly check`.

- [ ] **Step 5: Commit**

```bash
git add src/conversation2sql/eval_framework/agents/deep_agent/agent_code.py tests/eval_framework/agents/deep_agent/test_agent_code.py
git commit -m "feat(deep_agent): assemble agent with FS middleware + patience budget

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 6: Pipeline wiring + presets slug

**Files:**
- Modify: `src/conversation2sql/eval_framework/agents/__init__.py`
- Modify: `src/conversation2sql/eval_framework/main_pipe_workflow.py`
- Modify: `src/conversation2sql/presets.py`
- Test: `tests/eval_framework/agents/deep_agent/test_pipeline_wiring.py`
- Test: `tests/scripts/test_generate_catalog.py` or existing presets test — extend if one exists

**Interfaces:**
- Consumes: `run_agent_deep_agent`.
- Produces: `_resolve_baseline_settings("deep_agent")` → `(True, run_agent_deep_agent, True)`; the runner dispatch calls `run_agent_deep_agent` for `baseline == "deep_agent"`.

- [ ] **Step 1: Write the failing test**

Create `tests/eval_framework/agents/deep_agent/test_pipeline_wiring.py`:

```python
from conversation2sql.eval_framework.main_pipe_workflow import (
    _resolve_baseline_settings,
)
from conversation2sql.eval_framework.agents import run_agent_deep_agent


def test_deep_agent_baseline_resolves_to_runner():
    forced_amb, runner, needs_user_sim = _resolve_baseline_settings("deep_agent")
    assert forced_amb is True
    assert needs_user_sim is True
    assert runner is run_agent_deep_agent
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/eval_framework/agents/deep_agent/test_pipeline_wiring.py -v`
Expected: FAIL — `run_agent_deep_agent` not exported / baseline unknown.

- [ ] **Step 3: Implement**

In `agents/__init__.py`:

```python
from conversation2sql.eval_framework.agents.deep_agent.agent_code import run_agent_deep_agent
```
and add `"run_agent_deep_agent"` to `__all__`.

In `main_pipe_workflow.py`:
- Import `run_agent_deep_agent` alongside the existing runner imports (line ~20).
- Add to the `_resolve_baseline_settings` table (line ~44):
  ```python
      "deep_agent": (True, run_agent_deep_agent, True),
  ```
- In the runner-dispatch block (around line 235–244, where `baseline == "no_tool"` is special-cased and otherwise `run_agent_bird_baseline(..., enable_ask_user=...)` is called): add a branch so that when `baseline == "deep_agent"` it calls `run_agent_deep_agent(single_task, model_agent, model_user_parsing, model_user_generator, enable_ask_user=True)`. Match the exact argument names used for `run_agent_bird_baseline` at that call site (read the surrounding code first).

In `presets.py`: find where `enable_psql_console` contributes the `__psql` run-dir slug suffix (grep `__psql`). Add suffixes for the deep flags, e.g. `__todos`, `__subagents`, `__summar`, `__fswrite`, only appended when the corresponding flag is true and `baseline == "deep_agent"`. Follow the exact mechanism already used (string concat into the slug). If the slug logic lives in a bash/justfile path rather than `presets.py`, add the suffixes there instead — grep the repo for `__psql` to locate the single source.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/eval_framework/agents/deep_agent/ -v`
Expected: PASS. Then full suite: `uv run pytest tests/` and `uv run pyrefly check`.

- [ ] **Step 5: Commit**

```bash
git add src/conversation2sql/eval_framework/agents/__init__.py src/conversation2sql/eval_framework/main_pipe_workflow.py src/conversation2sql/presets.py tests/eval_framework/agents/deep_agent/test_pipeline_wiring.py
git commit -m "feat(deep_agent): wire deep_agent baseline into pipeline + run-dir slug

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 7: Documentation

**Files:**
- Create: `src/conversation2sql/eval_framework/agents/deep_agent/CLAUDE.md`
- Modify: `src/conversation2sql/eval_framework/agents/CLAUDE.md` (add deep_agent row to the variants table + a pointer)

- [ ] **Step 1: Write `deep_agent/CLAUDE.md`**

Document: the agent is `create_agent` + deepagents `FilesystemMiddleware` (not `create_deep_agent`, and why — global HarnessProfile doesn't fit per-run ablation); the `/db/*` seed files; read-only FS by default; the four `deep_enable_*` ablations and their slug suffixes; that the patience budget is reused from `bird_baseline`; `deep_tool_costs` is the cost table fed to `utils_process_agent_response`.

- [ ] **Step 2: Update `agents/CLAUDE.md`**

Add a `deep_agent` row to the variants table: `run_agent_deep_agent` | FS (read) | ✅ ask_user | ambiguous query. Note it parallels `bird_full` but swaps schema tools for the filesystem.

- [ ] **Step 3: Verify nothing broke**

Run: `uv run pytest tests/` and `uv run pyrefly check`.
Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add src/conversation2sql/eval_framework/agents/deep_agent/CLAUDE.md src/conversation2sql/eval_framework/agents/CLAUDE.md
git commit -m "docs(deep_agent): document the deepagents-based baseline

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Self-Review notes (coverage map)

- Spec §"Package & wiring" → Tasks 1, 6, 7.
- Spec §"Tools (4 total)" → Tasks 3 (costs), 5 (assembly).
- Spec §"Filesystem seed" → Task 2.
- Spec §"State & budget integration" → Tasks 3, 5.
- Spec §"Ablation flags" → Tasks 1 (config), 5 (middleware/tools), 4 (prompt blocks), 6 (slug).
- Spec §"Prompts" → Task 4.
- Spec §"Testing" → tests in every task.
- Spec §"Open implementation risk" → Task 5 Step 3b (SummarizationMiddleware/SubAgentMiddleware signatures) + the pinned-API block at the top (resolved: FilesystemMiddleware tool filtering, FileData seed shape, merged state, FilesystemOperation).
```
