# deep_agent bash tool — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace `deep_agent`'s deepagents virtual filesystem (and `execute_sql`) with a single read-only `bash` tool that explores a per-task on-disk catalog and runs SQL via `psql`, trimming the agent to 3 tools (`bash` + `submit_sql` + `ask_user`).

**Architecture:** Each run materializes the task's catalog (tables + masked KB + overview) into a fresh temp dir, sets it as the `bash` tool's `cwd`, and injects DB credentials via env so `psql -c "…"` works without a connection string. A command whitelist + a shell-metacharacter denylist + the existing psql host-command guardrail keep it read-only. The patience budget, `submit_sql`, `ask_user`, and the `subagents` ablation are unchanged.

**Tech Stack:** Python 3.12, `uv`, LangChain `create_agent` + middleware, `deepagents` (for `SubAgentMiddleware` only), `psycopg2`, pytest (`asyncio_mode=auto`).

## Global Constraints

- Run everything through `uv run` (e.g. `uv run pytest`, `uv run pyrefly check`).
- `db_dsn` format is a URI: `postgresql://user:pass@host:port/db`.
- The eval DB user may be a superuser; the `SELECT pg_read_file(...)` leak is **deferred by decision** — mark it with a `# TODO(unmasked-kb-leak):` comment, do not solve it.
- There is **no** `deep_enable_summarization` in the code (only stale docs). The flags to remove are `deep_enable_todos` and `deep_enable_fs_write`. Keep `deep_enable_subagents`.
- Reuse, don't reimplement: import `_violates_psql_guardrail` / `PSQL_GUARDRAIL_REFUSAL` from `bird_baseline.tools.bird_interact_env_tools`; import `submit_sql` / `return_tool_ask_user` / `TOOL_COSTS` from `bird_baseline.tools`.
- Spec: `docs/superpowers/specs/2026-06-30-deep-agent-bash-tool-design.md`.

---

### Task 1: `bash_tool.py` — the read-only shell tool (isolated, no existing code touched)

**Files:**
- Create: `src/conversation2sql/eval_framework/agents/deep_agent/bash_tool.py`
- Test: `tests/eval_framework/agents/deep_agent/test_bash_tool.py`

**Interfaces:**
- Consumes: `_violates_psql_guardrail`, `PSQL_GUARDRAIL_REFUSAL` from `bird_baseline.tools.bird_interact_env_tools`.
- Produces:
  - `build_pg_env(db_dsn: str) -> dict[str, str]`
  - `return_tool_bash(catalog_dir: Path, pg_env: dict[str, str])` → a LangChain `@tool` named `bash` taking `command: str`.
  - `ALLOWED_COMMANDS: frozenset[str]`, `WHITELIST_REFUSAL: str`, `METACHAR_REFUSAL: str` (for tests).

- [ ] **Step 1: Write the failing tests**

Create `tests/eval_framework/agents/deep_agent/test_bash_tool.py`:

```python
import os
from pathlib import Path

import pytest

from conversation2sql.eval_framework.agents.deep_agent import bash_tool
from conversation2sql.eval_framework.agents.bird_baseline.tools.bird_interact_env_tools import (
    PSQL_GUARDRAIL_REFUSAL,
)


def test_build_pg_env_maps_dsn_and_sets_readonly():
    env = bash_tool.build_pg_env("postgresql://alice:secret@db.local:5544/shop")
    assert env["PGHOST"] == "db.local"
    assert env["PGPORT"] == "5544"
    assert env["PGUSER"] == "alice"
    assert env["PGPASSWORD"] == "secret"
    assert env["PGDATABASE"] == "shop"
    assert "default_transaction_read_only=on" in env["PGOPTIONS"]
    # inherits the parent environment
    assert "PATH" in env


@pytest.fixture
def catalog(tmp_path) -> Path:
    (tmp_path / "tables").mkdir()
    (tmp_path / "tables" / "users.md").write_text("# users\nid, name\n")
    (tmp_path / "database_overview.md").write_text("shop db\n")
    return tmp_path


def _run(catalog: Path, command: str) -> str:
    bash = bash_tool.return_tool_bash(catalog, {**os.environ})
    return bash.invoke({"command": command})


def test_allowed_read_command_runs(catalog):
    out = _run(catalog, "cat tables/users.md")
    assert "exit=0" in out
    assert "id, name" in out


def test_pipe_of_allowed_commands_runs(catalog):
    out = _run(catalog, "cat tables/users.md | grep name")
    assert "exit=0" in out
    assert "name" in out


def test_non_whitelisted_command_is_refused_without_spawning(catalog):
    out = _run(catalog, "echo pwned")
    assert "Refused" in out
    assert "pwned" not in out  # never executed


def test_write_command_is_refused(catalog):
    out = _run(catalog, "rm -rf tables")
    assert "Refused" in out
    assert (catalog / "tables").exists()  # nothing deleted


def test_shell_metacharacters_are_refused(catalog):
    for bad in ["cat tables/users.md; rm -rf .", "cat $(whoami)", "cat a > b", "cat a && rm b"]:
        out = _run(catalog, bad)
        assert "Refused" in out, bad


def test_psql_host_meta_command_is_refused(catalog):
    out = _run(catalog, 'psql -c "\\! id"')
    assert out == PSQL_GUARDRAIL_REFUSAL


def test_unbalanced_quotes_are_refused(catalog):
    out = _run(catalog, 'cat "unclosed')
    assert "Refused" in out
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/eval_framework/agents/deep_agent/test_bash_tool.py -v`
Expected: FAIL with `ModuleNotFoundError: ... deep_agent.bash_tool` (module not created yet).

- [ ] **Step 3: Write the implementation**

Create `src/conversation2sql/eval_framework/agents/deep_agent/bash_tool.py`:

```python
"""A single read-only `bash` tool for the deep_agent.

Explores the per-task catalog (cwd) with a small whitelist of read commands and
runs read-only SQL via `psql` (credentials injected via env). Replaces the
deepagents virtual filesystem and `execute_sql`.
"""
from __future__ import annotations

import os
import shlex
import subprocess
from pathlib import Path

from langchain_core.tools import tool
from psycopg2.extensions import parse_dsn

from conversation2sql.eval_framework.agents.bird_baseline.tools.bird_interact_env_tools import (
    PSQL_GUARDRAIL_REFUSAL,
    _violates_psql_guardrail,
)

# Read-only commands the agent may run (psql is read-only via PGOPTIONS below).
ALLOWED_COMMANDS: frozenset[str] = frozenset(
    {"cat", "ls", "find", "grep", "head", "tail", "wc", "psql"}
)
# Shell control constructs we never allow (a first-token whitelist alone does not
# stop `;`/`&&`/command-substitution). Conservative by design: a quoted
# occurrence is also refused — an acceptable false-positive for read-only use.
_FORBIDDEN_METACHARS: tuple[str, ...] = (";", "&", ">", "<", "`", "$(", "${", "\n")

_TIMEOUT_S = 30
_STDOUT_CAP = 8000
_STDERR_CAP = 2000

WHITELIST_REFUSAL = (
    "Refused: only read-only commands are allowed "
    f"({', '.join(sorted(ALLOWED_COMMANDS))}), optionally joined with a pipe (|)."
)
METACHAR_REFUSAL = (
    "Refused: shell control characters (; & > < ` $( ${ newline) are not allowed; "
    "use a single read-only command or a pipeline of them."
)


def build_pg_env(db_dsn: str) -> dict[str, str]:
    """Inherit os.environ and add PG* vars from the DSN + a read-only PGOPTIONS,
    so `psql -c "SELECT …"` connects with no credentials in the command."""
    info = parse_dsn(db_dsn)
    env = {**os.environ}
    if info.get("host"):
        env["PGHOST"] = str(info["host"])
    if info.get("port"):
        env["PGPORT"] = str(info["port"])
    if info.get("user"):
        env["PGUSER"] = str(info["user"])
    if info.get("password"):
        env["PGPASSWORD"] = str(info["password"])
    if info.get("dbname"):
        env["PGDATABASE"] = str(info["dbname"])
    env["PGOPTIONS"] = "-c default_transaction_read_only=on -c statement_timeout=60s"
    return env


def _segments(tokens: list[str]) -> list[list[str]]:
    """Split a shlex token list on bare `|` tokens into pipeline segments."""
    segments: list[list[str]] = [[]]
    for t in tokens:
        if t == "|":
            segments.append([])
        else:
            segments[-1].append(t)
    return segments


def _refusal(command: str) -> str | None:
    """Return a refusal message if `command` is not an allowed read-only shell
    line, else None. Order: metachars → tokenizing → per-segment whitelist →
    psql host-command guardrail."""
    if any(m in command for m in _FORBIDDEN_METACHARS):
        return METACHAR_REFUSAL
    try:
        tokens = shlex.split(command)
    except ValueError:
        return WHITELIST_REFUSAL  # unbalanced quotes, etc.
    if not tokens:
        return WHITELIST_REFUSAL
    has_psql = False
    for seg in _segments(tokens):
        if not seg or seg[0] not in ALLOWED_COMMANDS:
            return WHITELIST_REFUSAL
        if seg[0] == "psql":
            has_psql = True
    # TODO(unmasked-kb-leak): a superuser DB role can still read the on-disk
    # (unmasked) catalog via `psql -c "SELECT pg_read_file('…')"`. The guardrail
    # below only blocks backslash host meta-commands. Deferred — see the spec's
    # Risks section (path-containment, revoke pg_read_file, or block it here).
    if has_psql and _violates_psql_guardrail(command):
        return PSQL_GUARDRAIL_REFUSAL
    return None


def return_tool_bash(catalog_dir: Path, pg_env: dict[str, str]):
    """Build the per-task `bash` tool bound to its catalog cwd and DB env."""

    @tool
    def bash(command: str) -> str:
        """Run a read-only shell command to explore the database catalog in your
        working directory (cat, ls, find, grep, head, tail, wc; pipes allowed), or
        run a read-only SQL query with `psql -c "SELECT …"` (writes are rejected)."""
        refusal = _refusal(command)
        if refusal is not None:
            return refusal
        try:
            proc = subprocess.run(
                ["bash", "-lc", command],
                cwd=str(catalog_dir),
                env=pg_env,
                capture_output=True,
                text=True,
                timeout=_TIMEOUT_S,
            )
        except subprocess.TimeoutExpired:
            return f"Command timed out after {_TIMEOUT_S}s."
        out = (proc.stdout or "")[:_STDOUT_CAP]
        err = (proc.stderr or "")[:_STDERR_CAP]
        return f"exit={proc.returncode}\n--- stdout ---\n{out}\n--- stderr ---\n{err}"

    return bash
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/eval_framework/agents/deep_agent/test_bash_tool.py -v`
Expected: PASS (all 8 tests).

- [ ] **Step 5: Commit**

```bash
git add src/conversation2sql/eval_framework/agents/deep_agent/bash_tool.py \
        tests/eval_framework/agents/deep_agent/test_bash_tool.py
git commit -m "feat(deep_agent): add read-only bash tool (catalog + psql)"
```

---

### Task 2: `catalog_seed.py` — materialize the per-task catalog dir

**Files:**
- Create: `src/conversation2sql/eval_framework/agents/deep_agent/catalog_seed.py`
- Test: `tests/eval_framework/agents/deep_agent/test_catalog_seed.py`

(`filesystem_seed.py` is left in place this task; Task 3 deletes it — so the tree stays green.)

**Interfaces:**
- Consumes: `TaskData`; `linearize_prerequisites` from `agents.utils_kb_linearize`; `TOOL_COSTS` from `bird_baseline.tools`.
- Produces:
  - `materialize_catalog_dir(task: TaskData) -> Path`
  - `deep_tool_costs() -> dict[str, float]`  (keys: `bash`, `submit_sql`, `ask_user`)

- [ ] **Step 1: Write the failing tests**

Create `tests/eval_framework/agents/deep_agent/test_catalog_seed.py`:

```python
from pathlib import Path

import pytest

from conversation2sql.eval_framework.agents.deep_agent import catalog_seed


def _make_catalog(root: Path, db: str) -> None:
    tables = root / db / "tables"
    tables.mkdir(parents=True)
    (tables / "users.md").write_text("# users\n")
    (tables / "_foreign_key_constraints.md").write_text("no fks\n")
    (root / db / "database_overview.md").write_text("overview\n")


def test_materialize_copies_tables_overview_and_renders_masked_kb(task_data, tmp_path):
    _make_catalog(tmp_path, "mydb")
    task_data.deep_catalog_root = str(tmp_path)
    task_data.selected_database = "mydb"

    out = catalog_seed.materialize_catalog_dir(task_data)

    assert (out / "tables" / "users.md").read_text() == "# users\n"
    assert (out / "tables" / "_foreign_key_constraints.md").exists()
    assert (out / "database_overview.md").read_text() == "overview\n"
    # one KB file per surviving (masked) node, none for absent names
    assert (out / "knowledge_base" / "active_user.md").exists()
    assert (out / "knowledge_base" / "revenue.md").exists()
    assert not (out / "knowledge_base" / "secret_kb.md").exists()


def test_materialize_missing_tables_dir_raises(task_data, tmp_path):
    (tmp_path / "mydb").mkdir()  # db dir exists but no tables/
    task_data.deep_catalog_root = str(tmp_path)
    task_data.selected_database = "mydb"
    with pytest.raises(FileNotFoundError):
        catalog_seed.materialize_catalog_dir(task_data)


def test_materialize_skips_overview_when_absent(task_data, tmp_path):
    tables = tmp_path / "mydb" / "tables"
    tables.mkdir(parents=True)
    (tables / "users.md").write_text("# users\n")
    task_data.deep_catalog_root = str(tmp_path)
    task_data.selected_database = "mydb"
    out = catalog_seed.materialize_catalog_dir(task_data)
    assert not (out / "database_overview.md").exists()
    assert (out / "tables" / "users.md").exists()


def test_deep_tool_costs_three_tools():
    costs = catalog_seed.deep_tool_costs()
    assert set(costs) == {"bash", "submit_sql", "ask_user"}
    assert costs["bash"] == 1.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/eval_framework/agents/deep_agent/test_catalog_seed.py -v`
Expected: FAIL with `ModuleNotFoundError: ... deep_agent.catalog_seed`.

- [ ] **Step 3: Write the implementation**

Create `src/conversation2sql/eval_framework/agents/deep_agent/catalog_seed.py`:

```python
"""Materialize one task's catalog into a temp dir for the deep_agent bash tool.

Layout (the bash tool's cwd):
  <tmp>/database_overview.md            (copied from disk if present)
  <tmp>/tables/<table>.md               (copied verbatim)
  <tmp>/tables/_foreign_key_constraints.md
  <tmp>/knowledge_base/<node>.md        (re-rendered from masked_agent_kb)

The KB is re-rendered from `masked_agent_kb` (not read from disk), so masked
prerequisites never appear — faithful per-sample masking, no leak.
"""
from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from conversation2sql.eval_framework.agents.bird_baseline.tools import TOOL_COSTS
from conversation2sql.eval_framework.agents.utils_kb_linearize import (
    linearize_prerequisites,
)
from conversation2sql.eval_framework.state import TaskData


def _catalog_db_dir(task: TaskData) -> Path:
    return Path(task.deep_catalog_root) / task.selected_database


def materialize_catalog_dir(task: TaskData) -> Path:
    """Write this task's catalog to a fresh temp dir and return that dir."""
    db_dir = _catalog_db_dir(task)
    if not (db_dir / "tables").is_dir():
        raise FileNotFoundError(
            f"deep_agent tables catalog not found for database "
            f"'{task.selected_database}' at {db_dir / 'tables'}. Generate it with "
            f"scripts/generate_catalog.py (--database {task.selected_database} "
            f"--output-dir {task.deep_catalog_root})."
        )
    out = Path(tempfile.mkdtemp(prefix="deep_catalog_"))

    tables_out = out / "tables"
    tables_out.mkdir()
    for path in sorted((db_dir / "tables").glob("*.md")):
        shutil.copyfile(path, tables_out / path.name)

    overview = db_dir / "database_overview.md"
    if overview.is_file():
        shutil.copyfile(overview, out / "database_overview.md")

    kb_out = out / "knowledge_base"
    kb_out.mkdir()
    for name in task.masked_agent_kb:
        content = linearize_prerequisites(name, task.masked_agent_kb)
        (kb_out / f"{name}.md").write_text(content, encoding="utf-8")

    return out


def deep_tool_costs() -> dict[str, float]:
    """Bird-coin cost map the patience middleware consults for the deep_agent.

    `bash` is a flat read cost; submit/ask_user reuse the shared table."""
    return {
        "bash": 1.0,
        "submit_sql": TOOL_COSTS["submit_sql"],
        "ask_user": TOOL_COSTS["ask_user"],
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/eval_framework/agents/deep_agent/test_catalog_seed.py -v`
Expected: PASS (all 4 tests).

- [ ] **Step 5: Commit**

```bash
git add src/conversation2sql/eval_framework/agents/deep_agent/catalog_seed.py \
        tests/eval_framework/agents/deep_agent/test_catalog_seed.py
git commit -m "feat(deep_agent): add catalog_seed (materialize per-task dir + costs)"
```

---

### Task 3: Rewire `agent_code.py`, state, prompts; delete `filesystem_seed.py`

This is the atomic flip: the agent stops using the deepagents filesystem and `execute_sql`, and starts using `bash` + the materialized dir. It ends green with the full suite.

**Files:**
- Modify: `src/conversation2sql/eval_framework/agents/deep_agent/agent_code.py` (full rewrite of the FS-related parts)
- Modify: `src/conversation2sql/eval_framework/agents/deep_agent/agent_code_state.py`
- Modify: `src/conversation2sql/eval_framework/agents/deep_agent/prompts.py`
- Delete: `src/conversation2sql/eval_framework/agents/deep_agent/filesystem_seed.py`
- Modify: `tests/eval_framework/agents/deep_agent/test_agent_code.py`

**Interfaces:**
- Consumes: `return_tool_bash`, `build_pg_env` (Task 1); `materialize_catalog_dir`, `deep_tool_costs` (Task 2); `submit_sql`, `return_tool_ask_user` from `bird_baseline.tools`; `execute_sql`, `submit_sql` (subagent only).
- Produces: `run_agent_deep_agent(...)` unchanged signature; `_build_deep_tools(task, model_user_parsing, model_user_generator, catalog_dir, pg_env)`; `_build_deep_middleware(task, model_agent)`; `_split_system_prompt`, `_capture_system_message` retained.

- [ ] **Step 1: Rewrite the deep_agent tests first (failing)**

Replace the whole contents of `tests/eval_framework/agents/deep_agent/test_agent_code.py` with:

```python
from pathlib import Path

from conversation2sql.eval_framework.agents.deep_agent import agent_code
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from deepagents import SubAgentMiddleware


def _task(task_data, **flags):
    for k, v in flags.items():
        setattr(task_data, k, v)
    return task_data


def _fake_model() -> FakeListChatModel:
    # deepagents introspects the model when compiling a subagent, so the
    # middleware tests need a real BaseChatModel rather than a MagicMock.
    return FakeListChatModel(responses=["ok"])


def test_middleware_minimal_by_default(task_data):
    mws = agent_code._build_deep_middleware(_task(task_data), _fake_model())
    assert not any(isinstance(m, SubAgentMiddleware) for m in mws)
    from conversation2sql.eval_framework.agents.bird_baseline.agent_callback import (
        tool_wrapper_patience_and_submit,
    )
    assert tool_wrapper_patience_and_submit in mws


def test_middleware_subagents_flag_adds_component(task_data):
    mws = agent_code._build_deep_middleware(
        _task(task_data, deep_enable_subagents=True), _fake_model()
    )
    assert any(isinstance(m, SubAgentMiddleware) for m in mws)


def test_split_system_prompt_separates_system_from_conversation():
    messages = [
        {"role": "system", "content": "SYS"},
        {"role": "user", "content": "hello"},
    ]
    system_prompt, convo = agent_code._split_system_prompt(messages)
    assert system_prompt == "SYS"
    assert convo == [{"role": "user", "content": "hello"}]
    assert all(m["role"] != "system" for m in convo)


def test_tools_are_bash_submit_ask_user(task_data, make_chat_model, tmp_path):
    tools = agent_code._build_deep_tools(
        _task(task_data),
        model_user_parsing=make_chat_model("<s>x</s>"),
        model_user_generator=make_chat_model("<s>y</s>"),
        catalog_dir=tmp_path,
        pg_env={},
    )
    names = {t.name for t in tools}
    assert names == {"bash", "submit_sql", "ask_user"}
    assert "execute_sql" not in names
    assert "read_file" not in names
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/eval_framework/agents/deep_agent/test_agent_code.py -v`
Expected: FAIL (`_build_deep_tools` signature lacks `catalog_dir`/`pg_env`; `execute_sql` still present; etc.).

- [ ] **Step 3: Rewrite `agent_code.py`**

Replace the whole file `src/conversation2sql/eval_framework/agents/deep_agent/agent_code.py` with:

```python
"""deep_agent baseline: one read-only bash tool + bird patience budget."""
from __future__ import annotations

import shutil
from pathlib import Path
from typing import Callable

from langchain.agents import create_agent
from langchain.agents.middleware import (
    ModelRetryMiddleware,
    ModelResponse,
    ToolRetryMiddleware,
    ExtendedModelResponse,
    ModelRequest,
    wrap_model_call,
)
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage
from langgraph.types import Command
from deepagents import SubAgentMiddleware
from deepagents.backends.state import StateBackend
from deepagents.middleware.subagents import SubAgent

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
from conversation2sql.eval_framework.agents.deep_agent.bash_tool import (
    build_pg_env,
    return_tool_bash,
)
from conversation2sql.eval_framework.agents.deep_agent.catalog_seed import (
    deep_tool_costs,
    materialize_catalog_dir,
)
from conversation2sql.eval_framework.agents.deep_agent.prompts import (
    build_deep_agent_messages,
)
from conversation2sql.eval_framework.state import TaskData
from conversation2sql.logger import get_logger

logger = get_logger(__name__)


# Minimal general-purpose subagent. deepagents' SubAgentMiddleware requires at
# least one subagent (each needing a model); tuning subagent prompts/tools is out
# of scope (subagents are off by default — this only exists so the
# deep_enable_subagents flag can be toggled).
def _general_subagent(model_agent: BaseChatModel) -> SubAgent:
    return {
        "name": "general",
        "description": "A general-purpose subagent for an isolated, self-contained sub-task.",
        "system_prompt": (
            "You are a focused sub-agent. Complete the assigned sub-task using the "
            "tools available to you and return a single concise result."
        ),
        "model": model_agent,
        "tools": [execute_sql, submit_sql],
    }


@wrap_model_call(state_schema=DeepAgentCustomState)
def _capture_system_message(
    request: ModelRequest,
    handler: Callable[[ModelRequest], ModelResponse],
) -> ModelResponse:
    """Capture the complete system message (template + any middleware additions)
    on the first model call, for faithful output logging."""
    response = handler(request)
    if not request.state.get("captured_system_prompt") and request.system_message:
        raw = request.system_message.content
        if isinstance(raw, list):
            raw = "\n\n".join(
                b.get("text", str(b)) if isinstance(b, dict) and b.get("type") == "text" else str(b)
                for b in raw
            )
        return ExtendedModelResponse(
            model_response=response,
            command=Command(update={"captured_system_prompt": raw}),
        )
    return response


def _split_system_prompt(messages: list[dict]) -> tuple[str | None, list[dict]]:
    """Split a built message list into (system_prompt, conversation).

    The system prompt travels via create_agent's `system_prompt=` so it becomes
    the single leading system turn; only non-system turns belong in the initial
    state (two leading system messages make vLLM reject the request)."""
    system = next((m["content"] for m in messages if m["role"] == "system"), None)
    convo = [m for m in messages if m["role"] != "system"]
    return system, convo


def _build_deep_tools(
    single_task: TaskData,
    model_user_parsing: BaseChatModel,
    model_user_generator: BaseChatModel,
    catalog_dir: Path,
    pg_env: dict[str, str],
) -> list:
    return [
        return_tool_bash(catalog_dir, pg_env),
        submit_sql,
        return_tool_ask_user(model_user_parsing, model_user_generator),
    ]


def _build_deep_middleware(single_task: TaskData, model_agent: BaseChatModel) -> list:
    mws: list = [
        ModelRetryMiddleware(max_delay=60.0, on_failure="error"),
        ToolRetryMiddleware(max_delay=60.0, on_failure="error"),
    ]
    if single_task.deep_enable_subagents:
        mws.append(
            SubAgentMiddleware(
                backend=StateBackend(),
                subagents=[_general_subagent(model_agent)],
            )
        )
    # Patience budget — appended last, same relative order as bird_baseline.
    mws += [
        check_budget_limit,
        sanitize_thinking_history,
        _capture_system_message,
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
            "enable_subagents": single_task.deep_enable_subagents,
        }
    )
    system_prompt, convo = _split_system_prompt(messages)
    catalog_dir = materialize_catalog_dir(single_task)
    try:
        pg_env = build_pg_env(single_task.db_dsn)
        agent = create_agent(
            model_agent,
            _build_deep_tools(
                single_task,
                model_user_parsing,
                model_user_generator,
                catalog_dir,
                pg_env,
            ),
            system_prompt=system_prompt,
            state_schema=DeepAgentCustomState,
            context_schema=TaskData,
            middleware=_build_deep_middleware(single_task, model_agent),  # pyrefly: ignore
        )
        agent_state = {
            "messages": convo,
            "initial_user_patience": single_task.task_budget,
            "updated_user_patience": single_task.task_budget,
            "tool_called_patience": [],
            "captured_system_prompt": "",
        }
        response = agent.invoke(agent_state, context=single_task)  # pyrefly: ignore
        full_system = response.get("captured_system_prompt") or system_prompt
        if full_system:
            response["messages"].insert(0, SystemMessage(content=full_system))
        predicted_sql = _extract_predicted_sql(response["messages"])
        output = utils_process_agent_response(
            response,  # pyrefly: ignore
            tool_costs=deep_tool_costs(),
        )
        output["predicted_sql"] = predicted_sql
        return output
    finally:
        shutil.rmtree(catalog_dir, ignore_errors=True)
```

- [ ] **Step 4: Rewrite `agent_code_state.py`**

Replace the whole file `src/conversation2sql/eval_framework/agents/deep_agent/agent_code_state.py` with:

```python
"""deep_agent state: bird patience fields + captured system prompt (no FS)."""
from __future__ import annotations

from conversation2sql.eval_framework.agents.bird_baseline.agent_code_state import (
    CustomAgentState,
)


class DeepAgentCustomState(CustomAgentState):
    """The three patience fields (from CustomAgentState) plus the full system
    message captured on the first model call, for faithful output logging."""

    captured_system_prompt: str
```

- [ ] **Step 5: Rewrite the system prompt in `prompts.py`**

Replace the `_DEEP_AGENT_SYSTEM` string in `src/conversation2sql/eval_framework/agents/deep_agent/prompts.py` with:

```python
_DEEP_AGENT_SYSTEM = """
You are a helpful PostgreSQL agent that interacts with a user and a database to solve the user's ambiguous question.

You have ONE shell tool, `bash`, for two purposes:

1. Explore the database catalog — it is laid out as files in your current working directory:
   - database_overview.md   — a high-level description of the database (read it first).
   - tables/                — one Markdown file per table (DDL, columns with descriptions);
                              tables/_foreign_key_constraints.md lists every PK/FK.
   - knowledge_base/        — one Markdown file per external-knowledge entry for this task.
   Use read-only commands: ls, cat, find, grep, head, tail, wc (you may pipe them, e.g.
   `grep -ril revenue knowledge_base | head`). Commands run in the catalog directory.

2. Run read-only SQL against the live database with psql, e.g.
   `psql -c "SELECT count(*) FROM users"`. No connection details are needed and writes
   are rejected. Use this to test a query before submitting.

You also have:
- ask_user: ask the user ONE clarifying question when their intent is ambiguous.
- submit_sql: submit your final SQL for grading (this ends the task).
{% if enable_subagents %}- task: delegate an isolated sub-task to an ephemeral subagent.
{% endif %}
Each action costs bird-coins from a fixed budget; be efficient. The interaction
ends when you submit the correct SQL or the budget runs out.

Strategy:
- Read database_overview.md and `ls tables`, then read the relevant table files.
- grep the catalog for relevant table/column/knowledge names instead of reading everything.
- If the user's intent is ambiguous, ask one clarifying question before committing to SQL.
- Test SQL with `psql -c "…"` before submit_sql when useful.
- Track your remaining budget and submit before it runs out.
"""
```

- [ ] **Step 6: Delete the old filesystem seed module**

```bash
git rm src/conversation2sql/eval_framework/agents/deep_agent/filesystem_seed.py
```

- [ ] **Step 7: Verify nothing else references the removed symbols**

Run: `grep -rn "filesystem_seed\|build_db_filesystem\|CustomFilesystemMiddleware\|_build_fs_middleware\|FS_TOOL_COSTS" src/ tests/`
Expected: no matches. If any appear, fix them before continuing.

- [ ] **Step 8: Run the deep_agent tests**

Run: `uv run pytest tests/eval_framework/agents/deep_agent/ -v`
Expected: PASS (bash, catalog_seed, and the 4 agent_code tests). `test_config_flags.py` still references the old flags — it is fixed in Task 4; if it fails here on `deep_enable_todos`/`deep_enable_fs_write`, that is expected and addressed next. To confirm only that is red, run the agent_code file alone:
Run: `uv run pytest tests/eval_framework/agents/deep_agent/test_agent_code.py -v`
Expected: PASS.

- [ ] **Step 9: Commit**

```bash
git add src/conversation2sql/eval_framework/agents/deep_agent/ \
        tests/eval_framework/agents/deep_agent/test_agent_code.py
git commit -m "feat(deep_agent): swap deepagents FS for bash tool; drop fs middleware"
```

---

### Task 4: Remove the `deep_enable_todos` and `deep_enable_fs_write` flags

Nothing references these in code after Task 3, so they can be deleted cleanly.

**Files:**
- Modify: `src/conversation2sql/config_input.py:47,49`
- Modify: `src/conversation2sql/eval_framework/state.py:78,80`
- Modify: `src/conversation2sql/eval_framework/dataset_readers/bird_interact_reader.py:257,259,434,436`
- Modify: `bash_scripts/utils/utils_evaluate.sh:98,100`
- Modify: `tests/eval_framework/agents/deep_agent/test_config_flags.py`

**Interfaces:**
- Produces: `ConfigReader` / `TaskData` / the reader keep only `deep_enable_subagents` among the deep flags.

- [ ] **Step 1: Update the config-flags test first (failing)**

Replace the whole contents of `tests/eval_framework/agents/deep_agent/test_config_flags.py` with:

```python
from conversation2sql.config_input import ConfigPipeline, ConfigReader
from conversation2sql.eval_framework.state import TaskData


def test_config_reader_deep_subagents_default_false():
    cfg = ConfigReader()
    assert cfg.deep_enable_subagents is False


def test_config_reader_drops_removed_deep_flags():
    cfg = ConfigReader()
    assert not hasattr(cfg, "deep_enable_todos")
    assert not hasattr(cfg, "deep_enable_fs_write")


def test_pipeline_accepts_deep_agent_baseline():
    cfg = ConfigPipeline(baseline="deep_agent")
    assert cfg.baseline == "deep_agent"


def test_taskdata_carries_deep_subagents(make_minimal_task_kwargs):
    task = TaskData(**make_minimal_task_kwargs())
    assert task.deep_enable_subagents is False


def test_config_reader_deep_catalog_root_default_is_lite_catalog():
    cfg = ConfigReader()
    assert cfg.deep_catalog_root == "data/bird_interact/catalog_bird_interact_lite"


def test_taskdata_carries_deep_catalog_root(make_minimal_task_kwargs):
    task = TaskData(**make_minimal_task_kwargs(deep_catalog_root="/some/root"))
    assert task.deep_catalog_root == "/some/root"
```

Note on `test_config_reader_drops_removed_deep_flags`: `ConfigReader` is a Pydantic model without `extra="allow"`, so removed fields are truly absent (`hasattr` is False). `TaskData` uses `extra="allow"`, so we do **not** assert absence on it — we only assert the surviving flag.

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/eval_framework/agents/deep_agent/test_config_flags.py -v`
Expected: FAIL on `test_config_reader_drops_removed_deep_flags` (fields still present).

- [ ] **Step 3: Remove the fields from `config_input.py`**

In `src/conversation2sql/config_input.py`, delete these two lines (keep the `deep_enable_subagents` line between them):

```python
    deep_enable_todos: bool = False  # add deepagents planning/write_todos middleware
    deep_enable_fs_write: bool = False  # expose write_file/edit_file (default: read-only FS)
```

Result — that block reads:

```python
    # --- deep_agent baseline ablations (only meaningful when baseline='deep_agent') ---
    deep_enable_subagents: bool = False  # add deepagents subagents (task tool) middleware
```

- [ ] **Step 4: Remove the fields from `state.py`**

In `src/conversation2sql/eval_framework/state.py`, delete the `deep_enable_todos` and `deep_enable_fs_write` lines so the block reads:

```python
    # deep_agent baseline ablations (only meaningful when baseline='deep_agent'),
    # threaded from ConfigReader exactly like the enable_* flags above.
    deep_enable_subagents: bool = False
```

- [ ] **Step 5: Remove the params from the reader**

In `src/conversation2sql/eval_framework/dataset_readers/bird_interact_reader.py`, delete the two signature lines (`deep_enable_todos: bool = False,` and `deep_enable_fs_write: bool = False,`) leaving `deep_enable_subagents: bool = False,`, and delete the two passing lines (`deep_enable_todos=deep_enable_todos,` and `deep_enable_fs_write=deep_enable_fs_write,`) leaving `deep_enable_subagents=deep_enable_subagents,`.

(The reader is called via `load_bird_interact_as_tasks(**config_reader.model_dump())`; with the fields gone from `ConfigReader`, they won't be in the dump, and the reader's `**kwargs` would absorb any stragglers anyway.)

- [ ] **Step 6: Remove the run-dir slug lines in the bash launcher**

In `bash_scripts/utils/utils_evaluate.sh`, delete these two lines (keep the `__subagents` line between them):

```bash
  [[ "${EXTRA:-}" =~ --deep_enable_todos[[:space:]]+true ]] && slug="${slug}__todos"
  [[ "${EXTRA:-}" =~ --deep_enable_fs_write[[:space:]]+true ]] && slug="${slug}__fswrite"
```

- [ ] **Step 7: Run config-flags test to verify it passes**

Run: `uv run pytest tests/eval_framework/agents/deep_agent/test_config_flags.py -v`
Expected: PASS.

- [ ] **Step 8: Verify no lingering references**

Run: `grep -rn "deep_enable_todos\|deep_enable_fs_write" src/ tests/ bash_scripts/`
Expected: no matches (README/CLAUDE.md docs are handled in Task 5).

- [ ] **Step 9: Commit**

```bash
git add src/conversation2sql/config_input.py \
        src/conversation2sql/eval_framework/state.py \
        src/conversation2sql/eval_framework/dataset_readers/bird_interact_reader.py \
        bash_scripts/utils/utils_evaluate.sh \
        tests/eval_framework/agents/deep_agent/test_config_flags.py
git commit -m "refactor(deep_agent): drop deep_enable_todos and deep_enable_fs_write flags"
```

---

### Task 5: Docs refresh + full verification

**Files:**
- Modify: `src/conversation2sql/eval_framework/agents/deep_agent/CLAUDE.md`
- Modify: `src/conversation2sql/eval_framework/agents/deep_agent/README.md`
- Modify: `src/conversation2sql/eval_framework/agents/CLAUDE.md`

- [ ] **Step 1: Update `deep_agent/CLAUDE.md`**

Rewrite it to describe the new design. Required content changes:
- Intro: the agent explores via a single read-only `bash` tool over an on-disk per-task catalog (cwd), and runs SQL via `psql` through the same tool — not a deepagents virtual filesystem, not `execute_sql`.
- Key files: replace the `filesystem_seed.py` bullet with `catalog_seed.py` (`materialize_catalog_dir(task)` → temp dir with `database_overview.md`, `tables/*.md`, `knowledge_base/<node>.md`; `deep_tool_costs()`); add a `bash_tool.py` bullet (`return_tool_bash`, `build_pg_env`, whitelist + metachar denylist + reused `_violates_psql_guardrail`; `# TODO(unmasked-kb-leak)`).
- Tools (3): `bash` + `submit_sql` + `ask_user`. Remove the FS-tools section.
- Ablation flags table: keep only the `deep_enable_subagents` / `SubAgentMiddleware` / `__subagents` row. Remove the `todos`, `summarization`, and `fs_write` rows.

- [ ] **Step 2: Update `deep_agent/README.md`**

Same substance: bash tool + `psql`; 3 tools; ablation table keeps only `deep_enable_subagents`/`__subagents`; remove the `__todos`, `__summar`, `__fswrite` rows and any `--deep_enable_todos`/`--deep_enable_fs_write` examples (use `--deep_enable_subagents true` as the example).

- [ ] **Step 3: Update `agents/CLAUDE.md`**

In the variants table, change the `deep_agent` row's tools cell from `FS (read)` to `bash (catalog + psql)`. Update the prose paragraph that says deep_agent "swaps the schema tools for a deepagents virtual filesystem (`ls`/`read_file`/`grep`/`glob`…)" to say it uses a single read-only `bash` tool over an on-disk per-task catalog (and `psql` for SQL).

- [ ] **Step 4: Run the full suite and type check**

Run: `uv run pytest tests/`
Expected: PASS (no failures, no errors).

Run: `uv run pyrefly check`
Expected: no new errors in `deep_agent/` files (bash_tool.py, catalog_seed.py, agent_code.py, agent_code_state.py, prompts.py).

- [ ] **Step 5: Commit**

```bash
git add src/conversation2sql/eval_framework/agents/deep_agent/CLAUDE.md \
        src/conversation2sql/eval_framework/agents/deep_agent/README.md \
        src/conversation2sql/eval_framework/agents/CLAUDE.md
git commit -m "docs(deep_agent): document bash tool + psql; trim ablations to subagents"
```

---

## Self-Review

**Spec coverage:**
- Per-task masked catalog dir incl. `database_overview.md` → Task 2 ✅
- Direct subprocess, cwd = per-task dir, no docker → Task 1 ✅
- Replace FS tools in-place, keep subagents → Task 3 ✅
- Read-only whitelist + psql guardrail → Task 1 ✅
- bash runs psql (3 tools), generic char cap → Task 1 ✅
- submit_sql stays separate → Task 3 (tool list) ✅
- Remove fs_write + todos flags (no summarization in code) → Task 4 ✅
- Rename filesystem_seed.py → catalog_seed.py → Tasks 2 (create) + 3 (delete) ✅
- Flat bash=1.0, keep captured_system_prompt → Tasks 2 + 3 ✅
- `# TODO(unmasked-kb-leak)` in code → Task 1 ✅
- Tests + docs → Tasks 1–5 ✅

**Placeholder scan:** No "TBD"/"handle edge cases"/"similar to" — all steps carry full code or exact edits.

**Type consistency:** `return_tool_bash(catalog_dir: Path, pg_env: dict[str,str])` used identically in `bash_tool.py`, `_build_deep_tools`, and tests. `materialize_catalog_dir(task) -> Path` and `deep_tool_costs() -> dict[str,float]` consistent across `catalog_seed.py`, `agent_code.py`, and tests. `_build_deep_tools` 5-arg signature matches its call site and the test.
