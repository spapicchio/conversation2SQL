# maintenance_agent Baseline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a new `maintenance_agent` eval baseline that evolves `deep_agent` into the software-maintenance framing — an on-disk `ISSUE.md` comment thread, a `write_query` tool scoped to `queries/answer.sql`, a structural `run_tests` primitive, and a silent (no pass/fail) `submit` — per `docs/superpowers/specs/2026-07-03-maintenance-agent-baseline-design.md` (Ticket A only).

**Architecture:** A new sibling package `agents/maintenance_agent/`, same shape as `agents/deep_agent/`, reusing `deep_agent.catalog_seed.materialize_catalog_dir` for the `docs/` catalog content, `deep_agent.tools.bash_tool` for exploration, and `bird_baseline`'s `ask_user_impl`/`submit_sql_impl` as the engines behind the new tools. A new patience-budget middleware factory (`make_tool_wrapper_patience_and_submit_silent`) is added to `bird_baseline/agent_callback.py` since the existing factory is hardcoded to `submit_sql`'s `passed`-JSON retry logic and cannot express an unconditionally-terminal submit.

**Tech Stack:** Python 3.12, LangGraph/LangChain agents (`create_agent`, `@wrap_tool_call`/`@wrap_model_call`/`@before_model` middleware), Pydantic (`TaskData`), psycopg2, pytest.

## Global Constraints

- Run all Python/pytest through `uv run` (project venv at `.venv`).
- Ticket A only — no Ticket B (repair/follow-up) work in this plan.
- `run_tests` and the visible `tests/test_contract.py` reference must never reveal correctness — structure only (file non-empty, `EXPLAIN` succeeds).
- `submit` is always terminal and gives no pass/fail feedback to the agent — grading happens outside the agent's view, in the harness, after `agent.invoke` returns.
- Follow existing patterns exactly where `deep_agent`/`bird_baseline` already solved the same problem (catalog materialization, tool-cost stamping, patience-budget middleware shape, test fixture reuse via `tests/eval_framework/tools/conftest.py`).
- After all tasks: `uv run pytest tests/` must pass and `uv run pyrefly check` must be clean (per repo `CLAUDE.md`'s "Apply changes" instruction).

---

## File Structure

```
src/conversation2sql/eval_framework/agents/
├── bird_baseline/
│   └── agent_callback.py                    # MODIFY: + make_tool_wrapper_patience_and_submit_silent
├── maintenance_agent/                        # NEW package
│   ├── __init__.py                           # empty (mirrors deep_agent/__init__.py)
│   ├── agent_code.py                         # run_agent_maintenance + _build_maintenance_tools/_middleware
│   ├── agent_code_state.py                   # MaintenanceAgentCustomState
│   ├── catalog_seed.py                       # materialize_maintenance_workspace + maintenance_tool_costs
│   ├── prompts.py                            # build_maintenance_agent_messages
│   └── tools/
│       ├── __init__.py                       # MA_TOOL_SPECS / MA_TOOL_COSTS
│       └── maintenance_tools.py               # write_query, run_tests, comment_on_issue, submit
├── __init__.py                                # MODIFY: export run_agent_maintenance
src/conversation2sql/
├── config_input.py                            # MODIFY: baseline Literal + 'maintenance_agent'
├── presets.py                                 # MODIFY: _TOOL_BASELINES + 'maintenance_agent'
└── eval_framework/main_pipe_workflow.py        # MODIFY: _resolve_baseline_settings + enable_ask_user set

tests/eval_framework/agents/
├── test_maintenance_silent_submit.py           # NEW
└── maintenance_agent/                          # NEW package
    ├── __init__.py
    ├── conftest.py
    ├── test_catalog_seed.py
    ├── test_agent_code.py
    ├── test_prompts.py
    ├── test_pipeline_wiring.py
    └── tools/
        ├── __init__.py
        └── test_maintenance_tools.py
tests/eval_framework/test_main_pipe_workflow.py   # MODIFY: parametrize + dispatch test
tests/test_presets.py                             # MODIFY: baseline_uses_tools assertion
```

Each `maintenance_agent/*.py` file has exactly the same single responsibility as its `deep_agent/*.py` counterpart, so a reviewer already familiar with `deep_agent` can review each file in isolation.

---

### Task 1: Silent-submit patience middleware factory

**Files:**
- Modify: `src/conversation2sql/eval_framework/agents/bird_baseline/agent_callback.py` (append after line 273, the `tool_wrapper_patience_and_submit = make_tool_wrapper_patience_and_submit(TOOL_COSTS)` line)
- Test: `tests/eval_framework/agents/test_maintenance_silent_submit.py`

**Interfaces:**
- Produces: `make_tool_wrapper_patience_and_submit_silent(tool_costs: dict[str, float], submit_tool_name: str = "submit") -> Callable` — a `@wrap_tool_call` middleware. Calling `submit_tool_name` is **always** terminal (`PATIENCE_SUBMIT_PASSED` if `user_patience >= 0` at call time, `PATIENCE_SUBMIT_EXHAUSTED` otherwise) — no `passed`-JSON inspection, no retry branch. All other tools are cost-blocked/recorded exactly like `make_tool_wrapper_patience_and_submit`. Consumed by Task 5's `agent_code.py`.

- [ ] **Step 1: Write the failing tests**

Create `tests/eval_framework/agents/test_maintenance_silent_submit.py`:

```python
"""Tests for make_tool_wrapper_patience_and_submit_silent — the maintenance_agent
variant of the patience/submit wrapper whose submit tool carries no pass/fail
signal (see docs/superpowers/specs/2026-07-03-maintenance-agent-baseline-design.md).
Mirrors tests/eval_framework/agents/test_budget_terminal_state.py's patterns for
the bird_baseline factory.
"""
from types import SimpleNamespace

from langchain_core.messages import ToolMessage
from langgraph.types import Command

from conversation2sql.eval_framework.agents.bird_baseline.agent_callback import (
    PATIENCE_BLOCKED,
    PATIENCE_SUBMIT_EXHAUSTED,
    PATIENCE_SUBMIT_PASSED,
    make_tool_wrapper_patience_and_submit_silent,
)


def _wrapper():
    return make_tool_wrapper_patience_and_submit_silent(
        {"bash": 1.0, "submit": 3.0}, submit_tool_name="submit"
    )


def _run(wrapper, tool_name, patience, response):
    request = SimpleNamespace(
        tool_call={"name": tool_name, "id": "call-1"},
        runtime=SimpleNamespace(state={"updated_user_patience": patience}),
    )
    return wrapper.wrap_tool_call(request, lambda _req: response)


class TestSubmitAlwaysTerminal:
    def test_submit_with_budget_left_marks_passed_sentinel(self):
        response = ToolMessage(content="Submitted.", tool_call_id="call-1", name="submit")
        command = _run(_wrapper(), "submit", 5.0, response)
        assert command.update["updated_user_patience"] == PATIENCE_SUBMIT_PASSED

    def test_submit_after_block_marks_exhausted_sentinel(self):
        response = ToolMessage(content="Submitted.", tool_call_id="call-1", name="submit")
        command = _run(_wrapper(), "submit", -1.0, response)
        assert command.update["updated_user_patience"] == PATIENCE_SUBMIT_EXHAUSTED

    def test_submit_is_never_blocked_even_when_over_cost(self):
        # submit's own cost (3.0) exceeds remaining budget (1.0), but submit must
        # still run — the wrapper only blocks non-submit tools.
        response = ToolMessage(content="Submitted.", tool_call_id="call-1", name="submit")
        command = _run(_wrapper(), "submit", 1.0, response)
        assert command.update["updated_user_patience"] == PATIENCE_SUBMIT_PASSED


class TestNonSubmitToolBlocking:
    def test_blocked_tool_message_names_the_configured_submit_tool(self):
        response = ToolMessage(content="ignored", tool_call_id="call-1", name="bash")
        command = _run(_wrapper(), "bash", 0.5, response)
        assert command.update["updated_user_patience"] == PATIENCE_BLOCKED
        blocked_message = command.update["messages"][0]
        assert "call submit now" in blocked_message.content.lower()

    def test_live_budget_tool_records_cost(self):
        response = ToolMessage(content="output", tool_call_id="call-1", name="bash")
        command = _run(_wrapper(), "bash", 5.0, response)
        assert command.update["tool_called_patience"] == [1.0]


class TestCommandReturningTool:
    def test_command_update_is_preserved_and_cost_recorded(self):
        tool_msg = ToolMessage(content="wrote file", tool_call_id="c1", name="write_query")
        response = Command(update={"messages": [tool_msg], "files": {"answer.sql": "x"}})
        request = SimpleNamespace(
            tool_call={"name": "write_query", "id": "c1"},
            runtime=SimpleNamespace(state={"updated_user_patience": 5.0}),
        )
        out = _wrapper().wrap_tool_call(request, lambda _req: response)
        assert isinstance(out, Command)
        assert out.update["files"] == {"answer.sql": "x"}
        # "write_query" is absent from the {"bash": 1.0, "submit": 3.0} cost
        # table passed to _wrapper(), so it must default to 0.0.
        assert out.update["tool_called_patience"] == [0.0]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/eval_framework/agents/test_maintenance_silent_submit.py -v`
Expected: FAIL with `ImportError: cannot import name 'make_tool_wrapper_patience_and_submit_silent'`

- [ ] **Step 3: Implement the factory**

Append to the end of `src/conversation2sql/eval_framework/agents/bird_baseline/agent_callback.py` (after the existing `tool_wrapper_patience_and_submit = make_tool_wrapper_patience_and_submit(TOOL_COSTS)` line):

```python


def make_tool_wrapper_patience_and_submit_silent(
    tool_costs: dict[str, float], submit_tool_name: str = "submit"
):
    """Build a patience/submit tool-wrapper middleware for a *silent* submit tool.

    Unlike ``make_tool_wrapper_patience_and_submit`` (bird_baseline's
    ``submit_sql``, which reports ``passed`` and lets a failing submit retry),
    ``submit_tool_name`` here carries no pass/fail signal at all: calling it
    always ends the episode. This is maintenance_agent's silent-submit design
    (see docs/superpowers/specs/2026-07-03-maintenance-agent-baseline-design.md)
    — the agent never learns whether its SQL was graded correct, so there is
    no retry-on-failed-submit branch to preserve.
    """

    @wrap_tool_call
    def tool_wrapper_patience_and_submit_silent(
            request: ToolCallRequest,
            handler: Callable[[ToolCallRequest], ToolMessage | Command],
    ) -> ToolMessage | Command:
        tool_name = request.tool_call["name"]
        cost = tool_costs.get(tool_name, 0.0)
        runtime: ToolRuntime[TaskData, CustomAgentState] = request.runtime
        user_patience = runtime.state["updated_user_patience"]

        # Block any non-submit tool whose cost would exceed the remaining
        # budget. The submit tool is always allowed so the agent can finalize
        # even when out of budget.
        if user_patience < cost and tool_name != submit_tool_name:
            return Command(
                update={
                    "messages": [
                        ToolMessage(
                            content=f"Budget exhausted ({user_patience:.1f} remaining). "
                                    f"You MUST call {submit_tool_name} now.",
                            tool_call_id=request.tool_call["id"],
                            name=tool_name,
                        )
                    ],
                    'updated_user_patience': PATIENCE_BLOCKED
                },
            )

        response = handler(request)

        if tool_name == submit_tool_name:
            # No passed/failed signal exists for a silent submit — calling it
            # is always terminal. Only the *reason* differs: a voluntary
            # submit with budget left is a clean finish; a submit reached
            # only after being blocked is a forced finalize. Reusing the two
            # bird_baseline sentinels keeps check_budget_limit's message
            # wording and downstream turn-classification code working
            # unmodified — no correctness signal is attached to either
            # sentinel here, unlike in bird_baseline where PASSED specifically
            # means "SQL was graded correct."
            terminal_patience = (
                PATIENCE_SUBMIT_EXHAUSTED if user_patience < 0 else PATIENCE_SUBMIT_PASSED
            )
            return Command(
                update={
                    "messages": [response.model_copy(deep=True)],
                    "updated_user_patience": terminal_patience,
                },
            )

        if isinstance(response, Command) and isinstance(response.update, dict):
            prior = response.update.get("tool_called_patience", [])
            merged = {**response.update, "tool_called_patience": [*prior, cost]}
            return replace(response, update=merged)

        return Command(
            update={
                "messages": [response.model_copy(deep=True)],
                "tool_called_patience": [cost],
            },
        )

    return tool_wrapper_patience_and_submit_silent
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/eval_framework/agents/test_maintenance_silent_submit.py -v`
Expected: PASS (7 tests)

- [ ] **Step 5: Commit**

```bash
git add src/conversation2sql/eval_framework/agents/bird_baseline/agent_callback.py tests/eval_framework/agents/test_maintenance_silent_submit.py
git commit -m "feat(maintenance_agent): add silent-submit patience middleware factory"
```

---

### Task 2: maintenance_agent package skeleton + workspace materialization

**Files:**
- Create: `src/conversation2sql/eval_framework/agents/maintenance_agent/__init__.py` (empty)
- Create: `src/conversation2sql/eval_framework/agents/maintenance_agent/catalog_seed.py`
- Test: `tests/eval_framework/agents/maintenance_agent/__init__.py` (empty)
- Test: `tests/eval_framework/agents/maintenance_agent/conftest.py`
- Test: `tests/eval_framework/agents/maintenance_agent/test_catalog_seed.py`

**Interfaces:**
- Consumes: `deep_agent.catalog_seed.materialize_catalog_dir(task: TaskData) -> Path` (existing), `bird_baseline.tools.USER_TOOL_COSTS` (existing).
- Produces: `materialize_maintenance_workspace(task: TaskData) -> Path` and `maintenance_tool_costs() -> dict[str, float]` (keys: `bash`, `write_query`, `comment_on_issue`, `run_tests`, `submit`). Both consumed by Task 5's `agent_code.py` and Task 3's tool factories.

- [ ] **Step 1: Create the empty package files**

```bash
mkdir -p /workspaces/conversation2SQL/src/conversation2sql/eval_framework/agents/maintenance_agent
touch /workspaces/conversation2SQL/src/conversation2sql/eval_framework/agents/maintenance_agent/__init__.py
mkdir -p /workspaces/conversation2SQL/tests/eval_framework/agents/maintenance_agent
touch /workspaces/conversation2SQL/tests/eval_framework/agents/maintenance_agent/__init__.py
```

- [ ] **Step 2: Write the test conftest**

Create `tests/eval_framework/agents/maintenance_agent/conftest.py`:

```python
"""Shared fixtures for the maintenance_agent tests.

Re-expose the existing tools-test fixtures (`task_data`, `make_chat_model`,
`column_meanings`, `masked_agent_kb`) so the maintenance_agent tests can reuse
the same minimal `TaskData` construction without duplicating it — mirrors
tests/eval_framework/agents/deep_agent/conftest.py.
"""
from __future__ import annotations

from tests.eval_framework.tools.conftest import (  # noqa: F401
    column_meanings,
    make_chat_model,
    masked_agent_kb,
    task_data,
)
```

- [ ] **Step 3: Write the failing tests**

Create `tests/eval_framework/agents/maintenance_agent/test_catalog_seed.py`:

```python
from pathlib import Path

import pytest

from conversation2sql.eval_framework.agents.maintenance_agent import catalog_seed


def _make_catalog(root: Path, db: str) -> None:
    tables = root / db / "tables"
    tables.mkdir(parents=True)
    (tables / "users.md").write_text("# users\n")
    (tables / "_foreign_key_constraints.md").write_text("no fks\n")
    (root / db / "database_overview.md").write_text("overview\n")


def test_materialize_nests_catalog_under_docs(task_data, tmp_path):
    _make_catalog(tmp_path, "mydb")
    task_data.deep_catalog_root = str(tmp_path)
    task_data.selected_database = "mydb"

    out = catalog_seed.materialize_maintenance_workspace(task_data)

    assert (out / "docs" / "database_overview.md").read_text().startswith("overview\n")
    assert (out / "docs" / "tables" / "users.md").read_text() == "# users\n"
    assert (out / "docs" / "knowledge_base" / "active_user.md").exists()
    # nothing left at the old flat deep_agent locations
    assert not (out / "database_overview.md").exists()
    assert not (out / "tables").exists()
    assert not (out / "knowledge_base").exists()


def test_materialize_skips_docs_overview_when_absent(task_data, tmp_path):
    tables = tmp_path / "mydb" / "tables"
    tables.mkdir(parents=True)
    (tables / "users.md").write_text("# users\n")
    task_data.deep_catalog_root = str(tmp_path)
    task_data.selected_database = "mydb"

    out = catalog_seed.materialize_maintenance_workspace(task_data)

    assert not (out / "docs" / "database_overview.md").exists()
    assert (out / "docs" / "tables" / "users.md").exists()


def test_materialize_writes_issue_with_task_question_and_empty_comments(task_data, tmp_path):
    _make_catalog(tmp_path, "mydb")
    task_data.deep_catalog_root = str(tmp_path)
    task_data.selected_database = "mydb"
    task_data.task_question = "How many active users do we have?"

    out = catalog_seed.materialize_maintenance_workspace(task_data)

    issue = (out / "ISSUE.md").read_text()
    assert issue == "# Issue\n\nHow many active users do we have?\n\n## Comments\n"


def test_materialize_writes_empty_stub_query(task_data, tmp_path):
    _make_catalog(tmp_path, "mydb")
    task_data.deep_catalog_root = str(tmp_path)
    task_data.selected_database = "mydb"

    out = catalog_seed.materialize_maintenance_workspace(task_data)

    stub = (out / "queries" / "answer.sql").read_text()
    assert stub == "-- TODO: replace this stub with your SQL query.\n"


def test_materialize_writes_test_contract_reference(task_data, tmp_path):
    _make_catalog(tmp_path, "mydb")
    task_data.deep_catalog_root = str(tmp_path)
    task_data.selected_database = "mydb"

    out = catalog_seed.materialize_maintenance_workspace(task_data)

    contract = (out / "tests" / "test_contract.py").read_text()
    assert "def test_query_is_written" in contract
    assert "def test_query_parses" in contract


def test_materialize_missing_tables_dir_raises(task_data, tmp_path):
    (tmp_path / "mydb").mkdir()  # db dir exists but no tables/
    task_data.deep_catalog_root = str(tmp_path)
    task_data.selected_database = "mydb"
    with pytest.raises(FileNotFoundError):
        catalog_seed.materialize_maintenance_workspace(task_data)


def test_maintenance_tool_costs_five_tools():
    costs = catalog_seed.maintenance_tool_costs()
    assert set(costs) == {"bash", "write_query", "comment_on_issue", "run_tests", "submit"}
    assert costs["bash"] == 1.0
    assert costs["write_query"] == 1.0
    assert costs["run_tests"] == 1.0
```

- [ ] **Step 4: Run tests to verify they fail**

Run: `uv run pytest tests/eval_framework/agents/maintenance_agent/test_catalog_seed.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'conversation2sql.eval_framework.agents.maintenance_agent.catalog_seed'`

- [ ] **Step 5: Implement catalog_seed.py**

Create `src/conversation2sql/eval_framework/agents/maintenance_agent/catalog_seed.py`:

```python
"""Materialize one task's maintenance workspace into a temp dir.

Layout (the agent's cwd):
  <tmp>/ISSUE.md                     (ticket body + growing "## Comments" thread)
  <tmp>/docs/database_overview.md    (deep_agent's catalog, nested under docs/)
  <tmp>/docs/tables/<table>.md
  <tmp>/docs/tables/_foreign_key_constraints.md
  <tmp>/docs/knowledge_base/<node>.md
  <tmp>/queries/answer.sql           (empty stub; the agent's deliverable)
  <tmp>/tests/test_contract.py       (read-only reference; see maintenance_tools.run_tests)

Reuses deep_agent.catalog_seed.materialize_catalog_dir for the docs/ content
(table/KB rendering, per-task masked-KB faithfulness) and nests its flat
output one level under docs/, per the framing spec's workspace anatomy.
"""
from __future__ import annotations

import shutil
from pathlib import Path

from conversation2sql.eval_framework.agents.bird_baseline.tools import USER_TOOL_COSTS
from conversation2sql.eval_framework.agents.deep_agent.catalog_seed import (
    materialize_catalog_dir,
)
from conversation2sql.eval_framework.state import TaskData

STUB_QUERY_CONTENT = "-- TODO: replace this stub with your SQL query.\n"

ISSUE_TEMPLATE = "# Issue\n\n{task_question}\n\n## Comments\n"

TEST_CONTRACT_REFERENCE = r'''"""Visible contract test for queries/answer.sql.

Executability only: checks structure, never correctness, so it can never leak
the ground-truth answer. This file is a read-only reference; the sandboxed
bash tool in this environment cannot execute scripts directly, so the
equivalent check is exposed as the `run_tests` tool instead.
"""
import pathlib
import re

TARGET = pathlib.Path(__file__).parent.parent / "queries" / "answer.sql"


def _strip_sql_comments(text: str) -> str:
    text = re.sub(r"--.*", "", text)
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
    return text.strip()


def test_query_is_written():
    """queries/answer.sql must no longer be the empty stub."""
    assert _strip_sql_comments(TARGET.read_text())


def test_query_parses():
    """queries/answer.sql must EXPLAIN successfully against the database."""
    # Run via the `run_tests` tool in this environment.
'''


def materialize_maintenance_workspace(task: TaskData) -> Path:
    """Write this task's maintenance workspace to a fresh temp dir and return it."""
    catalog_dir = materialize_catalog_dir(task)

    docs_dir = catalog_dir / "docs"
    docs_dir.mkdir()
    for name in ("database_overview.md", "tables", "knowledge_base"):
        src = catalog_dir / name
        if src.exists():
            shutil.move(str(src), str(docs_dir / name))

    (catalog_dir / "ISSUE.md").write_text(
        ISSUE_TEMPLATE.format(task_question=task.task_question), encoding="utf-8"
    )

    queries_dir = catalog_dir / "queries"
    queries_dir.mkdir()
    (queries_dir / "answer.sql").write_text(STUB_QUERY_CONTENT, encoding="utf-8")

    tests_dir = catalog_dir / "tests"
    tests_dir.mkdir()
    (tests_dir / "test_contract.py").write_text(TEST_CONTRACT_REFERENCE, encoding="utf-8")

    return catalog_dir


def maintenance_tool_costs() -> dict[str, float]:
    """Bird-coin cost map the patience middleware consults for maintenance_agent."""
    return {
        "bash": 1.0,
        "write_query": 1.0,
        "comment_on_issue": USER_TOOL_COSTS["ask_user"],
        "run_tests": 1.0,
        "submit": USER_TOOL_COSTS["submit_sql"],
    }
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `uv run pytest tests/eval_framework/agents/maintenance_agent/test_catalog_seed.py -v`
Expected: PASS (7 tests)

- [ ] **Step 7: Commit**

```bash
git add src/conversation2sql/eval_framework/agents/maintenance_agent/__init__.py \
        src/conversation2sql/eval_framework/agents/maintenance_agent/catalog_seed.py \
        tests/eval_framework/agents/maintenance_agent/__init__.py \
        tests/eval_framework/agents/maintenance_agent/conftest.py \
        tests/eval_framework/agents/maintenance_agent/test_catalog_seed.py
git commit -m "feat(maintenance_agent): workspace materialization (ISSUE.md, docs/, queries/, tests/)"
```

---

### Task 3: The four new maintenance tools

**Files:**
- Create: `src/conversation2sql/eval_framework/agents/maintenance_agent/tools/__init__.py`
- Create: `src/conversation2sql/eval_framework/agents/maintenance_agent/tools/maintenance_tools.py`
- Test: `tests/eval_framework/agents/maintenance_agent/tools/__init__.py` (empty)
- Test: `tests/eval_framework/agents/maintenance_agent/tools/test_maintenance_tools.py`

**Interfaces:**
- Consumes: `deep_agent.tools.bash_tool.BASH_TOOL_SPECS` (existing), `bird_baseline.tools.bird_interact_user_tools.ask_user_impl` (existing), `bird_baseline.tools.utils_db_execute._execute_query` (existing), `bird_baseline.agent_code_state.CustomAgentState` (existing, reused as the runtime state type — `MaintenanceAgentCustomState` from Task 5 is a subtype).
- Produces: `return_tool_write_query(catalog_dir: Path)`, `return_tool_run_tests(catalog_dir: Path, db_dsn: str)`, `return_tool_comment_on_issue(catalog_dir: Path, model_user_parsing: BaseChatModel, model_user_generator: BaseChatModel)`, `run_tests_impl(query_path: Path, db_dsn: str) -> dict`, module-level `@tool submit`, and `MA_TOOL_SPECS: dict[str, ToolSpec]` / `MA_TOOL_COSTS: dict[str, float]` (module `tools/__init__.py`). All consumed by Task 4's prompts.py and Task 5's agent_code.py.

- [ ] **Step 1: Create the empty test package file**

```bash
mkdir -p /workspaces/conversation2SQL/src/conversation2sql/eval_framework/agents/maintenance_agent/tools
mkdir -p /workspaces/conversation2SQL/tests/eval_framework/agents/maintenance_agent/tools
touch /workspaces/conversation2SQL/tests/eval_framework/agents/maintenance_agent/tools/__init__.py
```

- [ ] **Step 2: Write the failing tests**

Create `tests/eval_framework/agents/maintenance_agent/tools/test_maintenance_tools.py`:

```python
import json

import psycopg2

from conversation2sql.eval_framework.agents.maintenance_agent.tools import (
    maintenance_tools as mt,
)
from conversation2sql.eval_framework.agents.maintenance_agent.tools.maintenance_tools import (
    return_tool_comment_on_issue,
    return_tool_run_tests,
    return_tool_write_query,
    run_tests_impl,
    submit,
)


class _Runtime:
    """Stub LangGraph runtime exposing only `.context` (all comment_on_issue reads)."""

    def __init__(self, context):
        self.context = context


class TestWriteQuery:
    def test_writes_full_content_to_fixed_path(self, tmp_path):
        (tmp_path / "queries").mkdir()
        (tmp_path / "queries" / "answer.sql").write_text("-- stub\n")
        tool = return_tool_write_query(tmp_path)

        tool.func(content="SELECT 1;")

        assert (tmp_path / "queries" / "answer.sql").read_text() == "SELECT 1;"

    def test_overwrite_replaces_not_appends(self, tmp_path):
        (tmp_path / "queries").mkdir()
        (tmp_path / "queries" / "answer.sql").write_text("SELECT 1;")
        tool = return_tool_write_query(tmp_path)

        tool.func(content="SELECT 2;")

        assert (tmp_path / "queries" / "answer.sql").read_text() == "SELECT 2;"

    def test_ignores_path_like_content_in_argument(self, tmp_path):
        # The target path is fixed in the tool closure, not derived from the
        # argument — a filename-shaped `content` string is written verbatim as
        # text, never treated as a path.
        (tmp_path / "queries").mkdir()
        (tmp_path / "queries" / "answer.sql").write_text("-- stub\n")
        tool = return_tool_write_query(tmp_path)

        tool.func(content="../../etc/passwd")

        assert (tmp_path / "queries" / "answer.sql").read_text() == "../../etc/passwd"
        assert not (tmp_path / "etc").exists()


class TestRunTestsImpl:
    def test_stub_fails(self, tmp_path):
        path = tmp_path / "answer.sql"
        path.write_text("-- TODO: replace this stub with your SQL query.\n")

        result = run_tests_impl(path, db_dsn="postgresql://x")

        assert result["passed"] is False
        assert "stub" in result["message"]

    def test_valid_query_passes(self, tmp_path, monkeypatch):
        path = tmp_path / "answer.sql"
        path.write_text("SELECT 1;")
        monkeypatch.setattr(mt, "_execute_query", lambda query, db_dsn: ([], ()))

        result = run_tests_impl(path, db_dsn="postgresql://x")

        assert result["passed"] is True

    def test_invalid_query_fails_structurally(self, tmp_path, monkeypatch):
        path = tmp_path / "answer.sql"
        path.write_text("SELEKT 1;")

        def _raise(query, db_dsn):
            raise psycopg2.DatabaseError("syntax error")

        monkeypatch.setattr(mt, "_execute_query", _raise)

        result = run_tests_impl(path, db_dsn="postgresql://x")

        assert result["passed"] is False
        assert "does not parse" in result["message"]


class TestRunTestsTool:
    def test_tool_serializes_impl_result_to_json(self, tmp_path, monkeypatch):
        (tmp_path / "queries").mkdir()
        (tmp_path / "queries" / "answer.sql").write_text("SELECT 1;")
        monkeypatch.setattr(mt, "_execute_query", lambda query, db_dsn: ([], ()))
        tool = return_tool_run_tests(tmp_path, db_dsn="postgresql://x")

        raw = tool.func()

        assert json.loads(raw)["passed"] is True


class TestCommentOnIssue:
    def test_appends_question_and_answer_to_issue_md(
        self, tmp_path, make_chat_model, task_data
    ):
        (tmp_path / "ISSUE.md").write_text("# Issue\n\nq\n\n## Comments\n")
        model_user_parsing = make_chat_model("<s>AMB</s>")
        model_user_generator = make_chat_model("<s>Use the users table.</s>")
        tool = return_tool_comment_on_issue(
            tmp_path, model_user_parsing, model_user_generator
        )

        answer = tool.func(question="Which table?", runtime=_Runtime(task_data))

        assert answer == "Use the users table."
        issue = (tmp_path / "ISSUE.md").read_text()
        assert "**Agent:** Which table?" in issue
        assert "**Author:** Use the users table." in issue


class TestSubmit:
    def test_returns_confirmation_with_no_pass_fail_signal(self):
        result = submit.func()
        assert "passed" not in result.lower()
        assert "submitted" in result.lower()
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `uv run pytest tests/eval_framework/agents/maintenance_agent/tools/test_maintenance_tools.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'conversation2sql.eval_framework.agents.maintenance_agent.tools.maintenance_tools'`

- [ ] **Step 4: Implement maintenance_tools.py**

Create `src/conversation2sql/eval_framework/agents/maintenance_agent/tools/maintenance_tools.py`:

```python
"""The four new maintenance_agent tools: write_query, run_tests, comment_on_issue,
submit. `bash` is reused unchanged from deep_agent.tools.bash_tool.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import psycopg2
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import tool
from langgraph.prebuilt import ToolRuntime

from conversation2sql.eval_framework.agents.bird_baseline.agent_code_state import (
    CustomAgentState,
)
from conversation2sql.eval_framework.agents.bird_baseline.tools import USER_TOOL_COSTS
from conversation2sql.eval_framework.agents.bird_baseline.tools.bird_interact_user_tools import (
    ask_user_impl,
)
from conversation2sql.eval_framework.agents.bird_baseline.tools.utils_db_execute import (
    _execute_query,
)
from conversation2sql.eval_framework.agents.tool_specs import (
    ToolSpec,
    stamp_cost_in_descriptions,
)
from conversation2sql.eval_framework.state import TaskData

MAINTENANCE_TOOL_SPECS: dict[str, ToolSpec] = {
    "write_query": ToolSpec(
        "write_query", 1.0,
        "overwrite queries/answer.sql with the full SQL query text (not a diff)",
    ),
    "comment_on_issue": ToolSpec(
        "comment_on_issue", USER_TOOL_COSTS["ask_user"],
        "post a clarification question to the issue thread and get the author's reply",
    ),
    "run_tests": ToolSpec(
        "run_tests", 1.0,
        "check that queries/answer.sql is written and EXPLAINs successfully "
        "(structure only, not correctness)",
    ),
    "submit": ToolSpec(
        "submit", USER_TOOL_COSTS["submit_sql"],
        "submit your work for review; ends the episode with no pass/fail feedback",
    ),
}


def _strip_sql_comments(text: str) -> str:
    text = re.sub(r"--.*", "", text)
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
    return text.strip()


# ---------------------------------------------------------------------------
# write_query — cost: 1.0
# ---------------------------------------------------------------------------
def return_tool_write_query(catalog_dir: Path):
    """Build the per-task `write_query` tool bound to its fixed target path."""
    target = catalog_dir / "queries" / "answer.sql"

    @tool
    def write_query(content: str) -> str:
        """Overwrite queries/answer.sql with the full SQL query text.

        This always replaces the WHOLE file — there is no partial edit.
        Include the complete query, not a diff.

        Args:
            content: The full SQL query text to write to queries/answer.sql.

        Returns:
            A confirmation message.
        """
        target.write_text(content, encoding="utf-8")
        return f"Wrote {len(content)} characters to queries/answer.sql."

    stamp_cost_in_descriptions([write_query], MAINTENANCE_TOOL_SPECS)
    return write_query


# ---------------------------------------------------------------------------
# run_tests — cost: 1.0. Structural check only, never a correctness oracle.
# ---------------------------------------------------------------------------
def run_tests_impl(query_path: Path, db_dsn: str) -> dict:
    content = query_path.read_text(encoding="utf-8")
    stripped = _strip_sql_comments(content)
    if not stripped:
        return {
            "passed": False,
            "message": "queries/answer.sql is still the empty stub.",
        }
    try:
        _execute_query(query=f"EXPLAIN {stripped}", db_dsn=db_dsn)
    except psycopg2.DatabaseError as e:
        return {
            "passed": False,
            "message": f"queries/answer.sql does not parse: {e}",
        }
    return {
        "passed": True,
        "message": "queries/answer.sql is non-empty and parses successfully.",
    }


def return_tool_run_tests(catalog_dir: Path, db_dsn: str):
    """Build the per-task `run_tests` tool bound to its fixed target path + DSN."""
    query_path = catalog_dir / "queries" / "answer.sql"

    @tool
    def run_tests() -> str:
        """Run the visible structural check on queries/answer.sql: (1) the file
        is no longer the empty stub, (2) it EXPLAINs successfully against the
        database. This checks structure only — it never reveals whether the
        query is semantically correct.

        Returns:
            A JSON string with `passed` (bool) and `message` (str).
        """
        return json.dumps(run_tests_impl(query_path, db_dsn), indent=2)

    stamp_cost_in_descriptions([run_tests], MAINTENANCE_TOOL_SPECS)
    return run_tests


# ---------------------------------------------------------------------------
# comment_on_issue — cost: USER_TOOL_COSTS["ask_user"] (2.0)
# ---------------------------------------------------------------------------
def return_tool_comment_on_issue(
    catalog_dir: Path,
    model_user_parsing: BaseChatModel,
    model_user_generator: BaseChatModel,
):
    """Build the per-task `comment_on_issue` tool bound to its ISSUE.md path."""
    issue_path = catalog_dir / "ISSUE.md"

    @tool
    def comment_on_issue(
        question: str,
        runtime: ToolRuntime[TaskData, CustomAgentState],
    ) -> str:
        """Post a comment on the issue thread asking the ticket author a
        clarification question. Use this when the request is ambiguous.
        Ask one question at a time.

        Args:
            question: The clarification question to post.

        Returns:
            The author's reply.
        """
        answer = ask_user_impl(
            clarification_question=question,
            task=runtime.context,
            model_user_parsing=model_user_parsing,
            model_user_generator=model_user_generator,
        )["user_answer"]
        with issue_path.open("a", encoding="utf-8") as f:
            f.write(f"\n**Agent:** {question}\n\n**Author:** {answer}\n")
        return answer

    stamp_cost_in_descriptions([comment_on_issue], MAINTENANCE_TOOL_SPECS)
    return comment_on_issue


# ---------------------------------------------------------------------------
# submit — cost: USER_TOOL_COSTS["submit_sql"] (3.0). Always terminal, silent —
# see make_tool_wrapper_patience_and_submit_silent in bird_baseline/agent_callback.py.
# ---------------------------------------------------------------------------
@tool
def submit() -> str:
    """Submit your work for review. This ends the episode immediately: you
    will receive no pass/fail feedback and cannot retry. Only call this once
    queries/answer.sql is complete — verify with run_tests first.

    Returns:
        A confirmation string. No correctness feedback is given.
    """
    return "Submitted. No further actions will be taken."


stamp_cost_in_descriptions([submit], MAINTENANCE_TOOL_SPECS)
```

Create `src/conversation2sql/eval_framework/agents/maintenance_agent/tools/__init__.py`:

```python
"""Single source of truth: per-tool cost + prompt summary for maintenance_agent."""
from conversation2sql.eval_framework.agents.deep_agent.tools.bash_tool import BASH_TOOL_SPECS
from conversation2sql.eval_framework.agents.maintenance_agent.tools.maintenance_tools import (
    MAINTENANCE_TOOL_SPECS,
    return_tool_comment_on_issue,
    return_tool_run_tests,
    return_tool_write_query,
    run_tests_impl,
    submit,
)
from conversation2sql.eval_framework.agents.tool_specs import ToolSpec

MA_TOOL_SPECS: dict[str, ToolSpec] = {**BASH_TOOL_SPECS, **MAINTENANCE_TOOL_SPECS}
MA_TOOL_COSTS: dict[str, float] = {name: spec.cost for name, spec in MA_TOOL_SPECS.items()}

__all__ = [
    "MA_TOOL_SPECS",
    "MA_TOOL_COSTS",
    "return_tool_write_query",
    "return_tool_run_tests",
    "return_tool_comment_on_issue",
    "run_tests_impl",
    "submit",
]
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/eval_framework/agents/maintenance_agent/tools/test_maintenance_tools.py -v`
Expected: PASS (9 tests)

- [ ] **Step 6: Commit**

```bash
git add src/conversation2sql/eval_framework/agents/maintenance_agent/tools/ \
        tests/eval_framework/agents/maintenance_agent/tools/
git commit -m "feat(maintenance_agent): write_query, run_tests, comment_on_issue, submit tools"
```

---

### Task 4: Prompt template

**Files:**
- Create: `src/conversation2sql/eval_framework/agents/maintenance_agent/prompts.py`
- Test: `tests/eval_framework/agents/maintenance_agent/test_prompts.py`

**Interfaces:**
- Consumes: `maintenance_agent.tools.MA_TOOL_SPECS` (Task 3), `tool_specs.format_cost` (existing), `agents.utils.utils_build_messages` (existing).
- Produces: `build_maintenance_agent_messages(params: dict) -> list[dict]`. `params` keys used: `total_budget`, `amb_user_query`, `enable_ask_user` (default `False` inside the template — the runner in Task 5 passes `True` by default). Consumed by Task 5's `agent_code.py`.

- [ ] **Step 1: Write the failing tests**

Create `tests/eval_framework/agents/maintenance_agent/test_prompts.py`:

```python
from conversation2sql.eval_framework.agents.maintenance_agent.prompts import (
    build_maintenance_agent_messages,
)


def _render(**overrides):
    params = {
        "total_budget": 20,
        "amb_user_query": "How many active users?",
    }
    params.update(overrides)
    return build_maintenance_agent_messages(params)


def test_prompt_describes_workspace_layout():
    system = _render()[0]["content"]
    assert "ISSUE.md" in system
    assert "docs/database_overview.md" in system
    assert "docs/tables/" in system
    assert "docs/knowledge_base/" in system
    assert "queries/answer.sql" in system
    assert "tests/test_contract.py" in system


def test_prompt_describes_tools_and_costs():
    system = _render()[0]["content"]
    assert "write_query" in system
    assert "run_tests" in system
    assert "submit" in system
    assert "bash" in system
    assert "psql" in system


def test_prompt_includes_ticket_and_budget():
    msgs = _render()
    joined = " ".join(m["content"] for m in msgs)
    assert "How many active users?" in joined
    assert "20" in joined


def test_comment_on_issue_absent_by_default():
    system = _render()[0]["content"]
    assert "comment_on_issue" not in system


def test_comment_on_issue_surfaced_when_enabled():
    system = _render(enable_ask_user=True)[0]["content"]
    assert "comment_on_issue" in system
    assert "ambiguous" in system


def test_submit_is_described_as_silent_and_terminal():
    system = _render()[0]["content"]
    assert "no pass/fail feedback" in system
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/eval_framework/agents/maintenance_agent/test_prompts.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'conversation2sql.eval_framework.agents.maintenance_agent.prompts'`

- [ ] **Step 3: Implement prompts.py**

Create `src/conversation2sql/eval_framework/agents/maintenance_agent/prompts.py`:

```python
"""maintenance_agent prompt templates — inline Jinja2 with typed params."""
from __future__ import annotations

from conversation2sql.eval_framework.agents.maintenance_agent.tools import MA_TOOL_SPECS
from conversation2sql.eval_framework.agents.tool_specs import format_cost
from conversation2sql.eval_framework.agents.utils import utils_build_messages

_MAINTENANCE_AGENT_SYSTEM = r"""
You are a data engineer resolving a ticket filed against a SQL reporting application.

Task description:
A colleague filed a ticket describing a report they want. The ticket may be ambiguous
or missing details. Your job is to deliver a correct SQL query at queries/answer.sql
that satisfies the request.

The workspace (your current working directory) is laid out as:
- ISSUE.md: the ticket, with a running "## Comments" thread of your questions and the
  author's replies.
- docs/database_overview.md: a high-level description of the database, plus an index of
  every external-knowledge entry (name and its exact filename under docs/knowledge_base/)
  — read it first.
- docs/tables/: one Markdown file per table (DDL, columns with descriptions and examples).
- docs/tables/_foreign_key_constraints.md lists every PK/FK.
- docs/knowledge_base/<file>.md: full definition of one external-knowledge entry (domain
  term or formula the ticket may rely on). Look up the exact filename in
  docs/database_overview.md's Knowledge Base index rather than guessing one from a term's
  name.
- queries/answer.sql: your deliverable. Starts as an empty stub; write your query here.
- tests/test_contract.py: a reference copy of the structural check run_tests performs.

Available tools and costs:
- bash: {{ tool_specs['bash'].summary }}. Cost: {{ tool_specs['bash'].cost }}
- write_query: {{ tool_specs['write_query'].summary }}. Cost: {{ tool_specs['write_query'].cost }}
- run_tests: {{ tool_specs['run_tests'].summary }}. Cost: {{ tool_specs['run_tests'].cost }}
{% if enable_ask_user %}- comment_on_issue: {{ tool_specs['comment_on_issue'].summary }}. Cost: {{ tool_specs['comment_on_issue'].cost }}
{% endif %}- submit: {{ tool_specs['submit'].summary }}. Cost: {{ tool_specs['submit'].cost }}

How to use the bash tool:
- It runs ONE read-only command in the workspace (your cwd). Allowed commands: cat, ls, find, grep, head, tail, wc, psql. You may chain them with a pipe (|), but ;, &&, redirection (> <), backticks and $(...) are rejected.
- Explore the docs first, e.g.: `cat docs/database_overview.md`, then `ls docs/tables/`, then `cat docs/tables/<table>.md`, then `cat docs/tables/_foreign_key_constraints.md`. When the ticket needs a defined term or formula, `cat docs/knowledge_base/<filename>.md` using the exact filename from docs/database_overview.md's Knowledge Base index.
- To locate a table/column/knowledge-base name or description without opening every file by hand, use `grep -l "<term>" docs/tables/*.md` or `find . -iname "*<term>*"` instead of `ls`-ing and `cat`-ing each one.
- Query the database by running psql: `psql -c "SELECT ... FROM ... WHERE ...;"`. Do NOT add any connection flags — credentials are pre-injected as environment variables. The connection is read-only, so only SELECT-style queries work (writes and DDL are rejected). psql meta-commands like \dt, \d <table>, and \l are allowed.
- Every command returns `exit=<code>` followed by `--- stdout ---` and `--- stderr ---`. Read the exit code and stderr to diagnose failures before retrying.

How to use write_query and run_tests:
- write_query always overwrites the WHOLE contents of queries/answer.sql — include your complete query, not a diff.
- run_tests checks that queries/answer.sql is no longer the empty stub and that it EXPLAINs successfully. It never tells you whether your query is semantically correct — only whether it is structurally usable. Use it as a red→green sanity check before submitting, not as a correctness oracle.

Important strategy tips:
- First explore the workspace — read docs/ and probe with SELECT queries via psql before writing your final query.
{% if enable_ask_user %}- If the ticket is ambiguous, use comment_on_issue to ask the author a clarification question before committing to a query.
- Ask one clarification question at a time.
{% endif %}- Be efficient with your actions to conserve budget.
- Run run_tests after write_query and before submit to catch structural mistakes early.
- submit ends the episode immediately with no pass/fail feedback and cannot be undone — only call it once queries/answer.sql is complete.
- If a submission would fail, you will not be told; there is no retry after submit. Verify with run_tests first.
- Keep track of the remaining budget and prioritize actions accordingly.
- When several knowledge base entries have similar-sounding names (e.g.
  multiple "Classification…", "Confidence" or "Coherence" entries), cat
  every candidate and compare definitions before picking one, do not assume the first plausible match is correct.
- Do not add extra columns/joins/filters just because a table happens to expose them, and do not aggregate or deduplicate
  rows beyond what the ticket asks for. In particular, never JOIN a table
  you don't reference in SELECT/WHERE/GROUP BY, an unused INNER JOIN can
  silently drop rows that lack a match in that table, corrupting COUNT/AVG.
- Match the aggregation scope to the ticket exactly: if it asks for one
  aggregate (e.g. an overall average) plus a separate count of rows meeting a
  condition, compute the average over all rows and use
  `COUNT(*) FILTER (WHERE condition)` for the conditional count in the same
  query — don't add a WHERE/CTE filter that silently restricts the other
  aggregates to the same subset unless the ticket asks for that.
- PostgreSQL has no `MEDIAN()` aggregate function — for a median use
  `PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY <col>)`.
- A column alias defined in the SELECT list cannot be referenced in the WHERE
  clause of that same SELECT (Postgres evaluates WHERE before aliases exist).
  You'll get a "column does not exist" error. If you need to filter on a
  computed/CASE expression, repeat the full expression in WHERE, or wrap the
  query in a subquery/CTE and filter in the outer query using the alias.
"""
_MAINTENANCE_AGENT_USER = """
Ticket:
{{ amb_user_query }}

[SYSTEM NOTE: You have a total action budget of {{ total_budget }} units. Each action consumes budget. If the budget runs out, you must submit.]
"""


def build_maintenance_agent_messages(params: dict) -> list[dict]:
    # Render the tool list straight from MA_TOOL_SPECS (the single source of
    # truth for each tool's cost + summary) instead of hardcoding the wording
    # or the numbers in the template, where they could drift out of sync.
    params = {
        # Default the ask_user gate to False so an omitted key renders the
        # non-ambiguous prompt deterministically (not via Jinja's undefined).
        "enable_ask_user": False,
        **params,
        "tool_specs": {
            name: {"summary": spec.summary, "cost": format_cost(spec.cost)}
            for name, spec in MA_TOOL_SPECS.items()
        },
    }
    return utils_build_messages(_MAINTENANCE_AGENT_SYSTEM, _MAINTENANCE_AGENT_USER, params)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/eval_framework/agents/maintenance_agent/test_prompts.py -v`
Expected: PASS (6 tests)

- [ ] **Step 5: Commit**

```bash
git add src/conversation2sql/eval_framework/agents/maintenance_agent/prompts.py \
        tests/eval_framework/agents/maintenance_agent/test_prompts.py
git commit -m "feat(maintenance_agent): prompt template describing the workspace and tools"
```

---

### Task 5: agent_code_state.py + agent_code.py (run_agent_maintenance)

**Files:**
- Create: `src/conversation2sql/eval_framework/agents/maintenance_agent/agent_code_state.py`
- Create: `src/conversation2sql/eval_framework/agents/maintenance_agent/agent_code.py`
- Test: `tests/eval_framework/agents/maintenance_agent/test_agent_code.py`

**Interfaces:**
- Consumes: `make_tool_wrapper_patience_and_submit_silent` (Task 1), `materialize_maintenance_workspace`/`maintenance_tool_costs` (Task 2), the four tool factories + `submit` (Task 3), `build_maintenance_agent_messages` (Task 4), `deep_agent.tools.bash_tool.build_pg_env`/`return_tool_bash` (existing), `bird_baseline.agent_code.utils_process_agent_response` (existing), `bird_baseline.tools.submit_sql_impl` (existing).
- Produces: `run_agent_maintenance(single_task, model_agent, model_user_parsing, model_user_generator, *, enable_ask_user=True) -> dict` — same call signature shape as `run_agent_deep_agent`, required by `main_pipe_workflow._process_one`'s generic `runner(task, model_agent, model_user_parsing, model_user_generator, enable_ask_user=...)` dispatch. Also `_build_maintenance_tools` and `_build_maintenance_middleware` (unit-testable helpers). Consumed by Task 6's pipeline wiring.

- [ ] **Step 1: Write the failing tests**

Create `tests/eval_framework/agents/maintenance_agent/test_agent_code.py`:

```python
from types import SimpleNamespace

import pytest
from langchain_core.messages import ToolMessage

from conversation2sql.eval_framework.agents.maintenance_agent import agent_code
from conversation2sql.eval_framework.agents.maintenance_agent.catalog_seed import (
    maintenance_tool_costs,
)


def test_middleware_includes_silent_patience_budget(task_data):
    mws = agent_code._build_maintenance_middleware()
    names = [type(m).__name__ for m in mws]
    assert "tool_wrapper_patience_and_submit_silent" in names


def test_bash_tool_call_is_charged_maintenance_cost(task_data):
    mws = agent_code._build_maintenance_middleware()
    tool_wrapper = next(
        m for m in mws if type(m).__name__ == "tool_wrapper_patience_and_submit_silent"
    )
    request = SimpleNamespace(
        tool_call={"name": "bash", "id": "call-1"},
        runtime=SimpleNamespace(state={"updated_user_patience": 10.0}),
    )
    response = ToolMessage(content="output", tool_call_id="call-1", name="bash")
    out = tool_wrapper.wrap_tool_call(request, lambda _req: response)
    assert out.update["tool_called_patience"] == [maintenance_tool_costs()["bash"]]


def test_tools_are_bash_write_run_submit_when_ask_user_disabled(task_data, tmp_path):
    (tmp_path / "queries").mkdir()
    (tmp_path / "queries" / "answer.sql").write_text("-- stub\n")
    tools = agent_code._build_maintenance_tools(
        task_data,
        model_user_parsing=None,
        model_user_generator=None,
        catalog_dir=tmp_path,
        pg_env={},
        enable_ask_user=False,
    )
    names = {t.name for t in tools}
    assert names == {"bash", "write_query", "run_tests", "submit"}
    assert "comment_on_issue" not in names


def test_tools_include_comment_on_issue_when_enabled(task_data, make_chat_model, tmp_path):
    (tmp_path / "queries").mkdir()
    (tmp_path / "queries" / "answer.sql").write_text("-- stub\n")
    tools = agent_code._build_maintenance_tools(
        task_data,
        model_user_parsing=make_chat_model("<s>x</s>"),
        model_user_generator=make_chat_model("<s>y</s>"),
        catalog_dir=tmp_path,
        pg_env={},
        enable_ask_user=True,
    )
    names = {t.name for t in tools}
    assert names == {"bash", "write_query", "run_tests", "submit", "comment_on_issue"}


def test_catalog_dir_cleaned_up_on_exception(
    task_data, make_chat_model, tmp_path, monkeypatch
):
    import conversation2sql.eval_framework.agents.maintenance_agent.agent_code as ac

    created = {}

    def fake_materialize(task):
        d = tmp_path / "workspace"
        d.mkdir()
        (d / "queries").mkdir()
        (d / "queries" / "answer.sql").write_text("-- stub\n")
        created["dir"] = d
        return d

    class _Boom:
        def invoke(self, *a, **k):
            raise RuntimeError("boom")

    monkeypatch.setattr(ac, "materialize_maintenance_workspace", fake_materialize)
    monkeypatch.setattr(ac, "create_agent", lambda *a, **k: _Boom())

    with pytest.raises(RuntimeError):
        ac.run_agent_maintenance(
            task_data,
            model_agent=make_chat_model("x"),
            model_user_parsing=make_chat_model("<s>x</s>"),
            model_user_generator=make_chat_model("<s>y</s>"),
        )
    assert not created["dir"].exists()


def test_predicted_sql_read_from_disk_after_run(
    task_data, make_chat_model, tmp_path, monkeypatch
):
    import conversation2sql.eval_framework.agents.maintenance_agent.agent_code as ac

    def fake_materialize(task):
        d = tmp_path / "workspace"
        d.mkdir()
        (d / "queries").mkdir()
        (d / "queries" / "answer.sql").write_text("SELECT 1;")
        return d

    monkeypatch.setattr(ac, "materialize_maintenance_workspace", fake_materialize)
    monkeypatch.setattr(ac, "build_pg_env", lambda dsn: {})

    class _Recorder:
        def invoke(self, state, *a, **k):
            return {"messages": []}

    monkeypatch.setattr(ac, "create_agent", lambda *a, **k: _Recorder())
    monkeypatch.setattr(
        ac, "submit_sql_impl", lambda **kwargs: {"passed": True, "message": "ok"}
    )
    monkeypatch.setattr(
        ac, "utils_process_agent_response", lambda *a, **k: {"execution_accuracy": False}
    )

    output = ac.run_agent_maintenance(
        task_data,
        model_agent=make_chat_model("x"),
        model_user_parsing=make_chat_model("<s>x</s>"),
        model_user_generator=make_chat_model("<s>y</s>"),
    )

    assert output["predicted_sql"] == "SELECT 1;"


def test_execution_accuracy_overridden_from_hidden_grader_not_from_run_tests(
    task_data, make_chat_model, tmp_path, monkeypatch
):
    # Regression: utils_process_agent_response's generic `passed`-key scan
    # would otherwise pick up run_tests' *structural* pass/fail (also a dict
    # tool message with a "passed" key) instead of the real grade. Pin that
    # the hidden submit_sql_impl result always wins.
    import conversation2sql.eval_framework.agents.maintenance_agent.agent_code as ac

    def fake_materialize(task):
        d = tmp_path / "workspace"
        d.mkdir()
        (d / "queries").mkdir()
        (d / "queries" / "answer.sql").write_text("SELECT 1;")
        return d

    monkeypatch.setattr(ac, "materialize_maintenance_workspace", fake_materialize)
    monkeypatch.setattr(ac, "build_pg_env", lambda dsn: {})

    class _Recorder:
        def invoke(self, state, *a, **k):
            return {"messages": []}

    monkeypatch.setattr(ac, "create_agent", lambda *a, **k: _Recorder())
    # Hidden grader says the SQL is WRONG...
    monkeypatch.setattr(
        ac, "submit_sql_impl", lambda **kwargs: {"passed": False, "message": "wrong"}
    )
    # ...even though utils_process_agent_response's own (unrelated) scan would
    # have reported True (simulating a run_tests structural pass leaking in).
    monkeypatch.setattr(
        ac, "utils_process_agent_response", lambda *a, **k: {"execution_accuracy": True}
    )

    output = ac.run_agent_maintenance(
        task_data,
        model_agent=make_chat_model("x"),
        model_user_parsing=make_chat_model("<s>x</s>"),
        model_user_generator=make_chat_model("<s>y</s>"),
    )

    assert output["execution_accuracy"] is False


def test_system_and_user_travel_together_in_initial_state(
    task_data, make_chat_model, tmp_path, monkeypatch
):
    import conversation2sql.eval_framework.agents.maintenance_agent.agent_code as ac

    def fake_materialize(task):
        d = tmp_path / "workspace"
        d.mkdir()
        (d / "queries").mkdir()
        (d / "queries" / "answer.sql").write_text("-- stub\n")
        return d

    monkeypatch.setattr(ac, "materialize_maintenance_workspace", fake_materialize)
    monkeypatch.setattr(ac, "build_pg_env", lambda dsn: {})

    captured = {}

    class _Recorder:
        def invoke(self, state, *a, **k):
            captured["state"] = state
            return {"messages": []}

    monkeypatch.setattr(
        ac,
        "create_agent",
        lambda *a, **k: (captured.setdefault("kwargs", k), _Recorder())[1],
    )
    monkeypatch.setattr(
        ac, "submit_sql_impl", lambda **kwargs: {"passed": False, "message": "x"}
    )
    monkeypatch.setattr(ac, "utils_process_agent_response", lambda *a, **k: {})

    ac.run_agent_maintenance(
        task_data,
        model_agent=make_chat_model("x"),
        model_user_parsing=make_chat_model("<s>x</s>"),
        model_user_generator=make_chat_model("<s>y</s>"),
    )

    assert "system_prompt" not in captured["kwargs"]
    msgs = captured["state"]["messages"]
    assert msgs[0]["role"] == "system"
    assert msgs[1]["role"] == "user"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/eval_framework/agents/maintenance_agent/test_agent_code.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'conversation2sql.eval_framework.agents.maintenance_agent.agent_code'`

- [ ] **Step 3: Implement agent_code_state.py**

Create `src/conversation2sql/eval_framework/agents/maintenance_agent/agent_code_state.py`:

```python
"""maintenance_agent state: bird patience fields, mirrors DeepAgentCustomState."""
from __future__ import annotations

from conversation2sql.eval_framework.agents.bird_baseline.agent_code_state import (
    CustomAgentState,
)


class MaintenanceAgentCustomState(CustomAgentState):
    """The three patience fields (from CustomAgentState), nothing more.

    Mirrors DeepAgentCustomState: the workspace lives on disk (ISSUE.md,
    docs/, queries/, tests/), not in agent state, and the system prompt
    travels as the leading message in the initial state.
    """
```

- [ ] **Step 4: Implement agent_code.py**

Create `src/conversation2sql/eval_framework/agents/maintenance_agent/agent_code.py`:

```python
"""maintenance_agent baseline: bash + write_query + run_tests + comment_on_issue +
silent submit, under the bird patience budget."""
from __future__ import annotations

import shutil
from pathlib import Path

from langchain.agents import create_agent
from langchain.agents.middleware import (
    ModelRetryMiddleware,
    ToolRetryMiddleware,
)
from langchain_core.language_models import BaseChatModel

from conversation2sql.eval_framework.agents.bird_baseline.agent_callback import (
    check_budget_limit,
    make_tool_wrapper_patience_and_submit_silent,
    sanitize_thinking_history,
    wrap_model_append_tool_message,
)
from conversation2sql.eval_framework.agents.bird_baseline.agent_code import (
    utils_process_agent_response,
)
from conversation2sql.eval_framework.agents.bird_baseline.tools import submit_sql_impl
from conversation2sql.eval_framework.agents.deep_agent.tools.bash_tool import (
    build_pg_env,
    return_tool_bash,
)
from conversation2sql.eval_framework.agents.maintenance_agent.agent_code_state import (
    MaintenanceAgentCustomState,
)
from conversation2sql.eval_framework.agents.maintenance_agent.catalog_seed import (
    maintenance_tool_costs,
    materialize_maintenance_workspace,
)
from conversation2sql.eval_framework.agents.maintenance_agent.prompts import (
    build_maintenance_agent_messages,
)
from conversation2sql.eval_framework.agents.maintenance_agent.tools.maintenance_tools import (
    return_tool_comment_on_issue,
    return_tool_run_tests,
    return_tool_write_query,
    submit,
)
from conversation2sql.eval_framework.state import TaskData
from conversation2sql.logger import get_logger

logger = get_logger(__name__)


def _build_maintenance_tools(
    single_task: TaskData,
    model_user_parsing: BaseChatModel | None,
    model_user_generator: BaseChatModel | None,
    catalog_dir: Path,
    pg_env: dict[str, str],
    *,
    enable_ask_user: bool,
) -> list:
    tools: list = [
        return_tool_bash(catalog_dir, pg_env),
        return_tool_write_query(catalog_dir),
        return_tool_run_tests(catalog_dir, single_task.db_dsn),
        submit,
    ]
    # Only surface comment_on_issue when it's enabled — binding it without
    # also describing it in the prompt hides the tool from the model.
    if enable_ask_user:
        assert model_user_parsing is not None and model_user_generator is not None, (
            "enable_ask_user=True requires the user-sim models"
        )
        tools.append(
            return_tool_comment_on_issue(
                catalog_dir, model_user_parsing, model_user_generator
            )
        )
    return tools


def _build_maintenance_middleware() -> list:
    mws: list = [
        ModelRetryMiddleware(max_delay=60.0, on_failure="error"),
        ToolRetryMiddleware(max_delay=60.0, on_failure="error"),
    ]
    # Patience budget — appended last, same relative order as bird_baseline /
    # deep_agent. Costed from maintenance_tool_costs(), and using the *silent*
    # submit factory: this baseline's `submit` carries no pass/fail signal, so
    # there is no retry-on-failed-submit branch (see Task 1).
    mws += [
        check_budget_limit,
        sanitize_thinking_history,
        wrap_model_append_tool_message,
        make_tool_wrapper_patience_and_submit_silent(
            maintenance_tool_costs(), submit_tool_name="submit"
        ),
    ]
    return mws


def run_agent_maintenance(
    single_task: TaskData,
    model_agent: BaseChatModel,
    model_user_parsing: BaseChatModel | None,
    model_user_generator: BaseChatModel | None,
    *,
    enable_ask_user: bool = True,
) -> dict:
    messages = build_maintenance_agent_messages(
        params={
            "total_budget": single_task.task_budget,
            "amb_user_query": single_task.task_question,
            "enable_ask_user": enable_ask_user,
        }
    )
    catalog_dir = materialize_maintenance_workspace(single_task)
    try:
        pg_env = build_pg_env(single_task.db_dsn)
        agent = create_agent(
            model_agent,
            _build_maintenance_tools(
                single_task,
                model_user_parsing,
                model_user_generator,
                catalog_dir,
                pg_env,
                enable_ask_user=enable_ask_user,
            ),
            state_schema=MaintenanceAgentCustomState,
            context_schema=TaskData,
            middleware=_build_maintenance_middleware(),  # pyrefly: ignore
        )
        agent_state = {
            "messages": messages,
            "initial_user_patience": single_task.task_budget,
            "updated_user_patience": single_task.task_budget,
            "tool_called_patience": [],
        }
        response = agent.invoke(agent_state, context=single_task)  # pyrefly: ignore
        # The workspace *is* the state: no tool-call parsing needed. This is
        # unconditional — whether the episode ended via an explicit `submit`
        # call or via forced budget exhaustion, the file's final on-disk
        # content is the predicted SQL.
        predicted_sql = (catalog_dir / "queries" / "answer.sql").read_text(
            encoding="utf-8"
        )
        # Hidden grading: never exposed to the agent (silent submit). Computed
        # purely so the pipeline can report execution_accuracy.
        grading = submit_sql_impl(
            sql=predicted_sql,
            sol_sqls=single_task.sol_sql,
            db_dsn=single_task.db_dsn,
            conditions=single_task.sql_query_conditions,
        )
        output = utils_process_agent_response(
            response,  # pyrefly: ignore
            tool_costs=maintenance_tool_costs(),
        )
        output["predicted_sql"] = predicted_sql
        # Never trust utils_process_agent_response's own `execution_accuracy`
        # derivation here: run_tests also returns a dict tool message with a
        # `passed` key (a structural check), which would otherwise be picked
        # up as the correctness signal. The real grade always comes from the
        # hidden submit_sql_impl call above, run on the final on-disk file.
        output["execution_accuracy"] = grading["passed"]
        return output
    finally:
        shutil.rmtree(catalog_dir, ignore_errors=True)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/eval_framework/agents/maintenance_agent/test_agent_code.py -v`
Expected: PASS (7 tests)

- [ ] **Step 6: Commit**

```bash
git add src/conversation2sql/eval_framework/agents/maintenance_agent/agent_code_state.py \
        src/conversation2sql/eval_framework/agents/maintenance_agent/agent_code.py \
        tests/eval_framework/agents/maintenance_agent/test_agent_code.py
git commit -m "feat(maintenance_agent): run_agent_maintenance entry point"
```

---

### Task 6: Pipeline wiring (config, registry, dispatch, vLLM tool-calling flags)

**Files:**
- Modify: `src/conversation2sql/config_input.py:12`
- Modify: `src/conversation2sql/eval_framework/agents/__init__.py`
- Modify: `src/conversation2sql/eval_framework/main_pipe_workflow.py` (imports, `_resolve_baseline_settings`, `_process_one`)
- Modify: `src/conversation2sql/presets.py` (`_TOOL_BASELINES`)
- Modify: `tests/eval_framework/test_main_pipe_workflow.py`
- Modify: `tests/test_presets.py`
- Test: `tests/eval_framework/agents/maintenance_agent/test_pipeline_wiring.py`

**Interfaces:**
- Consumes: `run_agent_maintenance` (Task 5).
- Produces: the `maintenance_agent` baseline is fully launchable via `conv2sql run --baseline maintenance_agent` / `just eval --baseline maintenance_agent`.

- [ ] **Step 1: Write the failing tests**

Edit `tests/eval_framework/test_main_pipe_workflow.py` — extend the existing parametrize list (around line 21-27):

```python
    @pytest.mark.parametrize("baseline,expected_amb,expected_user_sim", [
        ("no_tool", False, False),
        ("tools_only", False, False),
        ("tools_user", False, True),
        ("bird_full", True, True),
        ("deep_agent", False, False),
        ("maintenance_agent", True, True),
    ])
```

Then append a new test class at the end of the file (after `TestConcurrencyConfig`):

```python


@patch("conversation2sql.eval_framework.main_pipe_workflow.run_agent_maintenance")
@patch("conversation2sql.eval_framework.main_pipe_workflow.run_agent_bird_baseline")
@patch("conversation2sql.eval_framework.main_pipe_workflow.run_baseline_no_tool")
@patch("conversation2sql.eval_framework.main_pipe_workflow.load_bird_interact_as_tasks")
@patch("conversation2sql.eval_framework.main_pipe_workflow.utils_create_model")
class TestMaintenanceAgentDispatch:
    def test_maintenance_agent_dispatches_with_ask_user_true(
        self, mock_create, mock_load, mock_no_tool, mock_agent, mock_maintenance, configs,
    ):
        cp, cr, cpred, cu = configs
        cp.baseline = "maintenance_agent"
        mock_load.return_value = [_fake_task()]
        mock_maintenance.return_value = _stub_response()
        mock_create.return_value = MagicMock()

        workflow_evaluation_pipeline(cp, cr, cpred, cu)

        mock_maintenance.assert_called_once()
        assert mock_maintenance.call_args.kwargs["enable_ask_user"] is True
        mock_agent.assert_not_called()
        mock_no_tool.assert_not_called()
        assert mock_load.call_args.kwargs["make_data_ambiguous"] is True
```

Create `tests/eval_framework/agents/maintenance_agent/test_pipeline_wiring.py`:

```python
from conversation2sql.eval_framework.main_pipe_workflow import (
    _resolve_baseline_settings,
)
from conversation2sql.eval_framework.agents import run_agent_maintenance


def test_maintenance_agent_baseline_resolves_to_runner():
    # maintenance_agent's entire point is ambiguity resolution through the
    # issue thread, so — unlike deep_agent — it defaults ambiguity on.
    forced_amb, runner, needs_user_sim = _resolve_baseline_settings("maintenance_agent")
    assert forced_amb is True
    assert needs_user_sim is True
    assert runner is run_agent_maintenance
```

Edit `tests/test_presets.py` — add one line inside `test_baseline_uses_tools` (after the existing `deep_agent` assertion):

```python
def test_baseline_uses_tools():
    assert baseline_uses_tools("no_tool") is False
    assert baseline_uses_tools("tools_only") is True
    assert baseline_uses_tools("tools_user") is True
    assert baseline_uses_tools("bird_full") is True
    assert baseline_uses_tools("deep_agent") is True
    assert baseline_uses_tools("maintenance_agent") is True
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/eval_framework/test_main_pipe_workflow.py tests/eval_framework/agents/maintenance_agent/test_pipeline_wiring.py tests/test_presets.py -v`
Expected: FAIL — `test_resolves[maintenance_agent-True-True]` raises `ValueError: Unknown baseline: 'maintenance_agent'`; `TestMaintenanceAgentDispatch` fails the same way; `test_pipeline_wiring.py` fails with `ImportError: cannot import name 'run_agent_maintenance'`; `test_baseline_uses_tools` fails its last assertion.

- [ ] **Step 3: Wire the baseline through every registration point**

Edit `src/conversation2sql/config_input.py:12`:

```python
    baseline: Literal['no_tool', 'tools_only', 'tools_user', 'bird_full', 'deep_agent', 'maintenance_agent'] = 'bird_full'
```

Edit `src/conversation2sql/eval_framework/agents/__init__.py`:

```python
from conversation2sql.eval_framework.agents.bird_baseline.agent_code import run_agent_bird_baseline
from conversation2sql.eval_framework.agents.deep_agent.agent_code import run_agent_deep_agent
from conversation2sql.eval_framework.agents.maintenance_agent.agent_code import run_agent_maintenance
from conversation2sql.eval_framework.agents.no_tool_baseline.baseline_model import run_baseline_no_tool

__all__ = [
    "run_agent_bird_baseline",
    "run_agent_deep_agent",
    "run_agent_maintenance",
    "run_baseline_no_tool",
]
```

Edit `src/conversation2sql/eval_framework/main_pipe_workflow.py` — the import block (lines 19-23):

```python
from conversation2sql.eval_framework.agents import (
    run_agent_bird_baseline,
    run_agent_deep_agent,
    run_agent_maintenance,
    run_baseline_no_tool,
)
```

`_resolve_baseline_settings` (lines 33-58) — update the docstring and table:

```python
def _resolve_baseline_settings(baseline: str) -> tuple[bool, Callable, bool]:
    """Map ConfigPipeline.baseline → (make_data_ambiguous, runner, needs_user_sim).

    no_tool             -> clean query, no agent loop, no user-sim
    tools_only          -> clean query, agent without ask_user, no user-sim
    tools_user          -> clean query, agent with ask_user, user-sim required
    bird_full           -> ambiguous query, agent with ask_user, user-sim required
    deep_agent          -> clean query, bash catalog agent, no ask_user, no user-sim
    maintenance_agent   -> ambiguous query, bash+write_query+run_tests+comment_on_issue
                            agent, user-sim required (ambiguity resolution via the
                            on-disk issue thread is the point of this baseline)
    """
    table = {
        "no_tool": (False, run_baseline_no_tool, False),
        "tools_only": (False, run_agent_bird_baseline, False),
        "tools_user": (False, run_agent_bird_baseline, True),
        "bird_full": (True, run_agent_bird_baseline, True),
        # deep_agent swaps the schema tools for the on-disk catalog + bash tool.
        # Default is the clean (non-ambiguous) query with no ask_user / user-sim,
        # to validate bash + KB reading in isolation; ambiguity + ask_user can be
        # re-enabled here later (flip to (True, ..., True) and add to the
        # enable_ask_user set below).
        "deep_agent": (False, run_agent_deep_agent, False),
        # maintenance_agent evolves deep_agent into the software-maintenance
        # framing (see docs/superpowers/specs/2026-07-03-maintenance-agent-baseline-design.md).
        # Ambiguity resolution through the on-disk issue thread is the core
        # measured skill, so — unlike deep_agent — it defaults ambiguous + on.
        "maintenance_agent": (True, run_agent_maintenance, True),
    }
    if baseline not in table:
        raise ValueError(
            f"Unknown baseline: {baseline!r}; expected one of {list(table)}"
        )
    return table[baseline]
```

`_process_one` (around line 252) — add `"maintenance_agent"` to the `enable_ask_user` set:

```python
                    response = await asyncio.to_thread(
                        runner,
                        task,
                        model_agent,
                        model_user_parsing,
                        model_user_generator,
                        enable_ask_user=(
                            baseline in ("tools_user", "bird_full", "maintenance_agent")
                        ),
                    )
```

Edit `src/conversation2sql/presets.py` — the `_TOOL_BASELINES` frozenset:

```python
_TOOL_BASELINES = frozenset(
    {"tools_only", "tools_user", "bird_full", "deep_agent", "maintenance_agent"}
)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/eval_framework/test_main_pipe_workflow.py tests/eval_framework/agents/maintenance_agent/test_pipeline_wiring.py tests/test_presets.py -v`
Expected: PASS (all tests, including the 6-row `test_resolves` parametrize and the new `TestMaintenanceAgentDispatch`)

- [ ] **Step 5: Run the full test suite and type checker**

Run: `uv run pytest tests/`
Expected: PASS (no regressions in any existing suite)

Run: `uv run pyrefly check`
Expected: no new errors attributable to `maintenance_agent/` or the modified files

- [ ] **Step 6: Commit**

```bash
git add src/conversation2sql/config_input.py \
        src/conversation2sql/presets.py \
        src/conversation2sql/eval_framework/agents/__init__.py \
        src/conversation2sql/eval_framework/main_pipe_workflow.py \
        tests/eval_framework/test_main_pipe_workflow.py \
        tests/test_presets.py \
        tests/eval_framework/agents/maintenance_agent/test_pipeline_wiring.py
git commit -m "feat(maintenance_agent): wire the baseline into config, registry, and dispatch"
```

---

## Self-Review Notes

- **Spec coverage:** every design-doc section maps to a task — middleware (Task 1), workspace (Task 2), tools (Task 3), prompt (Task 4), runner + grading override (Task 5), pipeline registration incl. the `presets.py`/`_TOOL_BASELINES` point the design doc didn't mention but is required for `just eval --baseline maintenance_agent` to select correct vLLM tool-calling flags (Task 6).
- **Placeholder scan:** no TBD/TODO markers; every step has complete, runnable code.
- **Type consistency:** `run_agent_maintenance`'s signature matches the `runner(task, model_agent, model_user_parsing, model_user_generator, enable_ask_user=...)` shape `_process_one` calls positionally-then-kwarg; `MaintenanceAgentCustomState`/`CustomAgentState` used consistently across Tasks 3, 5; `maintenance_tool_costs()` keys (`bash`, `write_query`, `comment_on_issue`, `run_tests`, `submit`) match `MAINTENANCE_TOOL_SPECS`/`MA_TOOL_SPECS` keys and the tools actually bound in `_build_maintenance_tools`.
