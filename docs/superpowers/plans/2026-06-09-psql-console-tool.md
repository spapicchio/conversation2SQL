# Single `psql_console` Tool (Ablation) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an ablation (default off) that replaces the four typed DB tools (`execute_sql`, `get_schema`, `get_table_names`, `get_table_schema`) with a single read-only `psql_console` tool that runs SQL or psql backslash meta-commands in a real subprocess.

**Architecture:** A new `psql_console_impl(command, db_dsn)` (guardrail + `psql -X -c` subprocess under a read-only `PGOPTIONS`) wrapped by a `@tool`, gated by a new `enable_psql_console` flag threaded `ConfigReader → TaskData → run_agent_bird_baseline` exactly like `enable_table_schema_tools`. The two flags are mutually exclusive (config validator + agent guard). The prompt and run-slug reflect the ablation when on.

**Tech Stack:** Python 3.12, Pydantic v2, LangChain/LangGraph `@tool`, `subprocess`, pytest (`uv run pytest`), bash.

**Spec:** `docs/superpowers/specs/2026-06-09-psql-console-tool-design.md`

---

### Task 1: `psql_console` guardrail + impl + tool

**Files:**
- Modify: `src/conversation2sql/eval_framework/agents/bird_baseline/tools/bird_interact_env_tools.py`
- Modify: `src/conversation2sql/eval_framework/agents/bird_baseline/tools/__init__.py`
- Test: `tests/eval_framework/tools/test_bird_interact_env_tools.py` (append)

- [ ] **Step 1: Write the failing guardrail + subprocess tests**

Append to `tests/eval_framework/tools/test_bird_interact_env_tools.py`. First add the import near the other env-tool imports at the top (after line 29):

```python
from conversation2sql.eval_framework.agents.bird_baseline.tools import (
    psql_console_impl,
)
from conversation2sql.eval_framework.agents.bird_baseline.tools.bird_interact_env_tools import (
    PSQL_GUARDRAIL_REFUSAL,
)
```

Then append these tests at the end of the file:

```python
# ---------------------------------------------------------------------------
# psql_console_impl
# ---------------------------------------------------------------------------
import subprocess
from types import SimpleNamespace


class TestPsqlGuardrail:
    """Host-reaching backslash meta-commands must be refused before psql spawns."""

    def test_shell_escape_is_refused_without_spawning(self):
        with patch.object(env_tools.subprocess, "run") as run_mock:
            out = psql_console_impl("\\! rm -rf /", db_dsn="dsn")
        assert out == PSQL_GUARDRAIL_REFUSAL
        run_mock.assert_not_called()

    def test_copy_to_file_is_refused(self):
        with patch.object(env_tools.subprocess, "run") as run_mock:
            out = psql_console_impl("\\copy t TO '/tmp/x.csv'", db_dsn="dsn")
        assert out == PSQL_GUARDRAIL_REFUSAL
        run_mock.assert_not_called()

    def test_output_redirect_is_refused(self):
        with patch.object(env_tools.subprocess, "run") as run_mock:
            assert psql_console_impl("\\o /tmp/x", db_dsn="dsn") == PSQL_GUARDRAIL_REFUSAL
        run_mock.assert_not_called()

    def test_include_file_is_refused(self):
        with patch.object(env_tools.subprocess, "run") as run_mock:
            assert psql_console_impl("\\i /etc/passwd", db_dsn="dsn") == PSQL_GUARDRAIL_REFUSAL
        run_mock.assert_not_called()

    def test_g_with_pipe_argument_is_refused(self):
        with patch.object(env_tools.subprocess, "run") as run_mock:
            assert psql_console_impl("SELECT 1 \\g | sh", db_dsn="dsn") == PSQL_GUARDRAIL_REFUSAL
        run_mock.assert_not_called()

    def test_bare_g_is_allowed(self):
        # Bare \g just re-runs the buffer — harmless, must reach psql.
        with patch.object(
            env_tools.subprocess, "run",
            return_value=SimpleNamespace(returncode=0, stdout="ok", stderr=""),
        ):
            assert psql_console_impl("SELECT 1 \\g", db_dsn="dsn") == "ok"

    def test_dt_inspection_command_is_allowed(self):
        with patch.object(
            env_tools.subprocess, "run",
            return_value=SimpleNamespace(returncode=0, stdout="list", stderr=""),
        ):
            assert psql_console_impl("\\dt", db_dsn="dsn") == "list"


class TestPsqlConsoleImpl:
    def test_argv_and_readonly_env(self):
        with patch.object(
            env_tools.subprocess, "run",
            return_value=SimpleNamespace(returncode=0, stdout="rows", stderr=""),
        ) as run_mock:
            out = psql_console_impl("SELECT 1;", db_dsn="postgresql://x/y")
        assert out == "rows"
        args, kwargs = run_mock.call_args
        assert args[0] == ["psql", "postgresql://x/y", "-X", "-c", "SELECT 1;"]
        assert "default_transaction_read_only=on" in kwargs["env"]["PGOPTIONS"]
        assert "statement_timeout=60s" in kwargs["env"]["PGOPTIONS"]
        assert kwargs["timeout"] == env_tools.PSQL_TIMEOUT_S

    def test_nonzero_exit_returns_stderr(self):
        with patch.object(
            env_tools.subprocess, "run",
            return_value=SimpleNamespace(returncode=1, stdout="", stderr="ERROR: boom"),
        ):
            assert psql_console_impl("SELECT bad;", db_dsn="dsn") == "ERROR: boom"

    def test_timeout_is_translated(self):
        with patch.object(
            env_tools.subprocess, "run",
            side_effect=subprocess.TimeoutExpired(cmd="psql", timeout=60),
        ):
            out = psql_console_impl("SELECT pg_sleep(99);", db_dsn="dsn")
        assert "timed out" in out.lower()

    def test_long_output_is_truncated(self):
        big = "x" * (env_tools.MAX_RESULT_LENGTH + 50)
        with patch.object(
            env_tools.subprocess, "run",
            return_value=SimpleNamespace(returncode=0, stdout=big, stderr=""),
        ):
            out = psql_console_impl("SELECT 1;", db_dsn="dsn")
        assert out.endswith(env_tools.TRUNCATION_NOTICE)
        assert len(out) == env_tools.MAX_RESULT_LENGTH + len(env_tools.TRUNCATION_NOTICE)


def test_psql_console_select_real_db():
    db_dsn = "postgresql://root:123123@localhost:5433/solar_panel"
    out = psql_console_impl("SELECT sitekey FROM plants LIMIT 1;", db_dsn)
    assert "sitekey" in out


def test_psql_console_dt_real_db():
    db_dsn = "postgresql://root:123123@localhost:5433/solar_panel"
    out = psql_console_impl("\\dt", db_dsn)
    assert "plants" in out


def test_psql_console_write_rejected_by_readonly_real_db():
    db_dsn = "postgresql://root:123123@localhost:5433/solar_panel"
    out = psql_console_impl("CREATE TABLE _should_not_exist (id int);", db_dsn)
    assert "read-only" in out.lower()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/eval_framework/tools/test_bird_interact_env_tools.py -k psql -v`
Expected: FAIL — `ImportError: cannot import name 'psql_console_impl'`.

- [ ] **Step 3: Implement the guardrail + impl + tool**

In `bird_interact_env_tools.py`, add `import os`, `import subprocess` to the imports at the top (alongside `import json`, `import re`). After the `DB_TOOL_COSTS` dict, add `"psql_console": 1.0` as a new entry:

```python
DB_TOOL_COSTS: dict[str, float] = {
    "execute_sql": 1.0,
    "get_schema": 1.0,
    "get_table_names": 0.5,
    "get_table_schema": 0.5,
    "get_all_column_meanings": 1.0,
    "get_column_meaning": 0.5,
    "get_all_external_knowledge_names": 0.5,
    "get_knowledge_definition": 0.5,
    "get_all_knowledge_definitions": 1.0,
    # Single read-only psql terminal tool (ablation). Replaces the four DB
    # tools above when enable_psql_console is set; flat cost like execute_sql.
    "psql_console": 1.0,
}
```

Then add this block after the `ExecuteSQLResponse` class (before the pure `*_impl` section):

```python
# ---------------------------------------------------------------------------
# psql_console (single read-only terminal tool — ablation)
# ---------------------------------------------------------------------------
PSQL_TIMEOUT_S = 60

# Read-only protects the DATABASE, not the dev container. psql's `\!` runs host
# shell commands, and `\o`/`\copy`/`\i`/`\e`/`\w`/`\s` (and `\g`/`\gx` WITH an
# argument) read or write the host filesystem / pipe to a shell. These are
# refused before psql is ever spawned. Bare `\g`/`\gx` (no argument) just re-run
# the query buffer and are allowed; read-only inspection commands (\dt \d \l
# \df …) are allowed.
_PSQL_HOST_REACHING = frozenset(
    {"!", "o", "out", "copy", "i", "ir", "e", "ef", "ev", "w", "s"}
)
_PSQL_META_RE = re.compile(r"\\(!|[A-Za-z]+)")

PSQL_GUARDRAIL_REFUSAL = (
    "Refused: psql meta-commands that reach the host shell or filesystem "
    "(\\!, \\o, \\copy, \\i, \\e, \\w, \\s, and \\g/\\gx with a file or pipe) "
    "are not allowed. Use plain SQL or read-only inspection commands such as "
    "\\dt, \\d <table>, \\l, \\df."
)


def _violates_psql_guardrail(command: str) -> bool:
    """True if ``command`` uses a host-reaching psql meta-command.

    Conservative by design: a SQL string literal that happens to contain e.g.
    ``\\copy`` is also refused. That is an acceptable false-positive for a
    read-only inspection tool.
    """
    for match in _PSQL_META_RE.finditer(command):
        name = match.group(1)
        if name in _PSQL_HOST_REACHING:
            return True
        if name in ("g", "gx"):
            # Dangerous only with a trailing argument (\g <file> or \g |cmd) on
            # the same line; bare \g just re-runs the buffer.
            tail = command[match.end():].split("\n", 1)[0].strip()
            if tail:
                return True
    return False


def psql_console_impl(command: str, db_dsn: str) -> str:
    if _violates_psql_guardrail(command):
        return PSQL_GUARDRAIL_REFUSAL

    env = {
        **os.environ,
        "PGOPTIONS": "-c default_transaction_read_only=on -c statement_timeout=60s",
    }
    try:
        proc = subprocess.run(
            ["psql", db_dsn, "-X", "-c", command],
            env=env,
            capture_output=True,
            text=True,
            timeout=PSQL_TIMEOUT_S,
        )
    except subprocess.TimeoutExpired:
        return f"Error: psql timed out after {PSQL_TIMEOUT_S}s."

    output = proc.stdout if proc.returncode == 0 else (proc.stderr or proc.stdout)
    if len(output) > MAX_RESULT_LENGTH:
        output = output[:MAX_RESULT_LENGTH] + TRUNCATION_NOTICE
    return output
```

Finally add the `@tool` wrapper at the end of the "Database execution tools" section (right after `get_table_schema`):

```python
@tool
def psql_console(command: str, runtime: ToolRuntime[TaskData, CustomAgentState]) -> str:
    """Run ONE PostgreSQL statement OR one psql backslash meta-command in a
    read-only psql session. Use SQL (SELECT/EXPLAIN/WITH) to query data, or
    inspection meta-commands like \\dt (list tables), \\d <table> (describe a
    table), \\l (list databases), \\df (list functions). The session is
    read-only — writes and DDL are rejected by the server. Cost: 1 bird-coin.

    Args:
        command: One SQL statement or one psql backslash meta-command.

    Returns:
        The psql output on success, or the error text on failure.
    """
    return psql_console_impl(command=command, db_dsn=runtime.context.db_dsn)
```

- [ ] **Step 4: Export from the tools package**

In `tools/__init__.py`, add `psql_console`, `psql_console_impl` to the import from `bird_interact_env_tools` (alongside `execute_sql`, etc.) and to `__all__` (in the "env tools" groups):

```python
from conversation2sql.eval_framework.agents.bird_baseline.tools.bird_interact_env_tools import (
    execute_sql,
    psql_console,
    ...
    execute_sql_impl,
    psql_console_impl,
    ...
)
```

And in `__all__` add `"psql_console",` to the `@tool` group and `"psql_console_impl",` to the `*_impl` group.

- [ ] **Step 5: Run the psql tests (mocked + guardrail pass; real-DB need the container)**

Run: `uv run pytest tests/eval_framework/tools/test_bird_interact_env_tools.py -k "psql and not real_db" -v`
Expected: PASS (all guardrail + mocked-subprocess tests).

Run: `uv run pytest tests/eval_framework/tools/test_bird_interact_env_tools.py -k "psql and real_db" -v`
Expected: PASS if the `localhost:5433` Postgres container is up (same precondition as the existing `*_real_db` tests). If the DB is down these error like the existing real-DB tests — note it and move on.

- [ ] **Step 6: Commit**

```bash
git add src/conversation2sql/eval_framework/agents/bird_baseline/tools/bird_interact_env_tools.py \
        src/conversation2sql/eval_framework/agents/bird_baseline/tools/__init__.py \
        tests/eval_framework/tools/test_bird_interact_env_tools.py
git commit -m "feat(tools): add read-only psql_console tool with host-escape guardrail

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 2: `enable_psql_console` config flag + mutual-exclusion validator

**Files:**
- Modify: `src/conversation2sql/config_input.py:22-33`
- Test: `tests/test_config_input.py` (create)

- [ ] **Step 1: Write the failing validator test**

Create `tests/test_config_input.py`:

```python
import pytest

from conversation2sql.config_input import ConfigReader


def test_psql_console_defaults_off():
    assert ConfigReader().enable_psql_console is False


def test_psql_console_alone_is_allowed():
    cfg = ConfigReader(enable_psql_console=True)
    assert cfg.enable_psql_console is True
    assert cfg.enable_table_schema_tools is False


def test_both_db_tool_ablations_are_mutually_exclusive():
    with pytest.raises(ValueError, match="mutually exclusive"):
        ConfigReader(enable_psql_console=True, enable_table_schema_tools=True)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_config_input.py -v`
Expected: FAIL — `AttributeError`/`TypeError` (no `enable_psql_console`), and the mutual-exclusion test does not raise.

- [ ] **Step 3: Add the field + validator**

In `config_input.py`, change the import line 3 to include `model_validator`:

```python
from pydantic import BaseModel, Field, model_validator
```

Add the field right after `enable_table_schema_tools` (line 33) inside `ConfigReader`, then the validator:

```python
    enable_psql_console: bool = False  # Ablation: replace the DB tools (execute_sql/get_schema/get_table_*) with a single read-only psql terminal tool (psql_console). Mutually exclusive with enable_table_schema_tools.

    @model_validator(mode="after")
    def _check_db_tool_ablation_exclusivity(self) -> "ConfigReader":
        if self.enable_psql_console and self.enable_table_schema_tools:
            raise ValueError(
                "enable_psql_console and enable_table_schema_tools are mutually "
                "exclusive; enable at most one DB-tool ablation."
            )
        return self
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_config_input.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add src/conversation2sql/config_input.py tests/test_config_input.py
git commit -m "feat(config): add enable_psql_console flag, mutually exclusive with table-schema tools

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 3: Thread the flag onto `TaskData` + one-time warning in the reader

**Files:**
- Modify: `src/conversation2sql/eval_framework/state.py:64`
- Modify: `src/conversation2sql/eval_framework/dataset_readers/bird_interact_reader.py:228-242`, `:386-388`
- Test: covered indirectly; no new isolated test (reader needs the dataset on disk). Verified via Task 4's agent-layer test and the full suite.

- [ ] **Step 1: Add the `TaskData` field**

In `state.py`, after `enable_table_schema_tools: bool = False` (line 64) add:

```python
    # When True the agent gets the single read-only psql_console tool INSTEAD of
    # execute_sql/get_schema/get_table_* (ablation). Mutually exclusive with
    # enable_table_schema_tools (enforced in ConfigReader + run_agent_bird_baseline).
    enable_psql_console: bool = False
```

- [ ] **Step 2: Add the reader parameter, the one-time warning, and thread it onto `TaskData`**

In `bird_interact_reader.py`, add the parameter to `load_bird_interact_as_tasks` after `enable_table_schema_tools: bool = False,` (line 239):

```python
    enable_psql_console: bool = False,
```

Immediately after `dataset_path = Path(dataset_path)` (line 249), add the one-time warning (fires once per load, not per task):

```python
    if enable_psql_console:
        logger.warning(
            "PSQL-CONSOLE ABLATION ENABLED: replacing execute_sql/get_schema/"
            "get_table_* with the single read-only psql_console tool."
        )
```

In the `TaskData(...)` construction, after `enable_table_schema_tools=enable_table_schema_tools,` (line 387) add:

```python
                enable_psql_console=enable_psql_console,
```

- [ ] **Step 3: Verify the package still imports and config flows through**

Run: `uv run python -c "from conversation2sql.config_input import ConfigReader; from conversation2sql.eval_framework.dataset_readers import load_bird_interact_as_tasks; import inspect; assert 'enable_psql_console' in inspect.signature(load_bird_interact_as_tasks).parameters; assert 'enable_psql_console' in ConfigReader().model_dump(); print('ok')"`
Expected: prints `ok` (confirms `ConfigReader().model_dump()` — which is how `main_pipe_workflow` calls the reader — carries the new key, and the reader accepts it).

- [ ] **Step 4: Commit**

```bash
git add src/conversation2sql/eval_framework/state.py \
        src/conversation2sql/eval_framework/dataset_readers/bird_interact_reader.py
git commit -m "feat(reader): thread enable_psql_console onto TaskData with one-time warning

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 4: Wire the tool selection into the agent (with mutual-exclusion guard)

**Files:**
- Modify: `src/conversation2sql/eval_framework/agents/bird_baseline/agent_code.py:33-46` (imports), `:78-91` (tool list)
- Test: `tests/eval_framework/agents/test_psql_console_wiring.py` (create)

- [ ] **Step 1: Write the failing tool-selection tests**

Create `tests/eval_framework/agents/test_psql_console_wiring.py`:

```python
import pytest

from conversation2sql.eval_framework.agents.bird_baseline.agent_code import (
    _select_db_tools,
)
from conversation2sql.eval_framework.state import TaskData


def _task(**flags) -> TaskData:
    # model_construct skips validation/required fields: _select_db_tools only
    # reads the two ablation booleans off the task.
    return TaskData.model_construct(
        enable_psql_console=flags.get("psql", False),
        enable_table_schema_tools=flags.get("table_tools", False),
    )


def test_default_db_tools():
    names = {t.name for t in _select_db_tools(_task())}
    assert names == {"execute_sql", "get_schema"}


def test_table_schema_tools_added():
    names = {t.name for t in _select_db_tools(_task(table_tools=True))}
    assert names == {"execute_sql", "get_schema", "get_table_names", "get_table_schema"}


def test_psql_console_replaces_db_tools():
    names = {t.name for t in _select_db_tools(_task(psql=True))}
    assert names == {"psql_console"}


def test_both_flags_raise():
    with pytest.raises(ValueError, match="mutually exclusive"):
        _select_db_tools(_task(psql=True, table_tools=True))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/eval_framework/agents/test_psql_console_wiring.py -v`
Expected: FAIL — `ImportError: cannot import name '_select_db_tools'`.

- [ ] **Step 3: Add `_select_db_tools` and use it in the tool list**

In `agent_code.py`, add `psql_console` to the tools import block (after `execute_sql,` on line 34):

```python
from conversation2sql.eval_framework.agents.bird_baseline.tools import (
    execute_sql,
    psql_console,
    get_all_column_meanings,
    ...
)
```

Add this helper just above `def run_agent_bird_baseline(`:

```python
def _select_db_tools(single_task: TaskData) -> list:
    """Pick the database-facing tools for this task.

    Default: execute_sql + get_schema (+ get_table_names/get_table_schema when
    enable_table_schema_tools). When enable_psql_console is set, a single
    read-only psql_console tool replaces all of them. The two ablation flags are
    mutually exclusive (also enforced in ConfigReader).
    """
    if single_task.enable_psql_console and single_task.enable_table_schema_tools:
        raise ValueError(
            "enable_psql_console and enable_table_schema_tools are mutually "
            "exclusive; enable at most one DB-tool ablation."
        )
    if single_task.enable_psql_console:
        return [psql_console]
    db_tools = [execute_sql, get_schema]
    if single_task.enable_table_schema_tools:
        db_tools.extend([get_table_names, get_table_schema])
    return db_tools
```

Replace the current tool-list construction (lines 78-89, from `tools = [` through the `if single_task.enable_table_schema_tools:` block) with:

```python
    tools = [
        *_select_db_tools(single_task),
        get_all_column_meanings,
        get_column_meaning,
        get_all_external_knowledge_names,
        get_knowledge_definition,
        get_all_knowledge_definitions,
        submit_sql,
    ]
    if enable_ask_user:
        tools.append(return_tool_ask_user(model_user_parsing, model_user_generator))
```

Also add `enable_psql_console` to the prompt params dict (after `"enable_table_schema_tools": single_task.enable_table_schema_tools,` on line 74):

```python
            "enable_psql_console": single_task.enable_psql_console,
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/eval_framework/agents/test_psql_console_wiring.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add src/conversation2sql/eval_framework/agents/bird_baseline/agent_code.py \
        tests/eval_framework/agents/test_psql_console_wiring.py
git commit -m "feat(agent): select psql_console vs typed DB tools via _select_db_tools

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 5: Render the tool in the prompt

**Files:**
- Modify: `src/conversation2sql/eval_framework/agents/bird_baseline/prompts.py:31-42`
- Test: `tests/eval_framework/agents/test_psql_console_wiring.py` (append)

- [ ] **Step 1: Write the failing prompt tests**

Append to `tests/eval_framework/agents/test_psql_console_wiring.py`:

```python
from conversation2sql.eval_framework.agents.bird_baseline.prompts import (
    build_bird_interact_agent_messages,
)


def _system_prompt(**params) -> str:
    base = {
        "total_budget": 20,
        "amb_user_query": "q",
        "enable_ask_user": False,
        "enable_table_schema_tools": False,
        "enable_psql_console": False,
    }
    base.update(params)
    return build_bird_interact_agent_messages(base)[0]["content"]


def test_prompt_default_lists_execute_sql_not_psql():
    text = _system_prompt()
    assert "execute_sql" in text
    assert "psql_console" not in text


def test_prompt_psql_mode_lists_psql_not_execute_sql():
    text = _system_prompt(enable_psql_console=True)
    assert "psql_console" in text
    assert "execute_sql" not in text
    assert "get_schema" not in text
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/eval_framework/agents/test_psql_console_wiring.py -k prompt -v`
Expected: FAIL — `test_prompt_psql_mode_lists_psql_not_execute_sql` fails (no `psql_console` in the rendered prompt; `execute_sql` still present).

- [ ] **Step 3: Update the prompt template**

In `prompts.py`, replace the tool-list lines (currently lines 31-35, from `Available tools and costs:` through the `enable_table_schema_tools` block) with:

```jinja
Available tools and costs:
{% if enable_psql_console %}- psql_console: run ONE PostgreSQL statement or one read-only psql meta-command (\dt list tables, \d <table> describe, \l list databases, \df list functions). Read-only session. Cost: 1
{% else %}- execute_sql: execute a PostgreSQL query. Cost: 1
- get_schema: get the full database schema. Cost: 1
{% if enable_table_schema_tools %}- get_table_names: list all table names in the database. Cost: 0.5
- get_table_schema: get one table's schema (CREATE TABLE, sample rows, and foreign keys to joinable tables). Cost: 0.5
{% endif %}{% endif %}- get_all_column_meanings: get all column meanings. Cost: 1
```

(The `- get_all_column_meanings` line already exists immediately after; keep only one copy — the diff replaces up to and including the reintroduced `get_all_column_meanings` line.)

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/eval_framework/agents/test_psql_console_wiring.py -v`
Expected: PASS (all 6 tests in the file).

- [ ] **Step 5: Commit**

```bash
git add src/conversation2sql/eval_framework/agents/bird_baseline/prompts.py \
        tests/eval_framework/agents/test_psql_console_wiring.py
git commit -m "feat(prompt): render psql_console line in psql-console ablation mode

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 6: Reflect the ablation in the run-directory slug

**Files:**
- Modify: `bash_scripts/utils/utils_evaluate.sh:91-93`
- Test: manual bash assertion (the slug builder is bash; no pytest harness).

- [ ] **Step 1: Add the slug suffix**

In `bash_scripts/utils/utils_evaluate.sh`, immediately after the `slug="${slug}__iter${num_iterations}"` line (line 91) and before the DEBUG line, add:

```bash
  # Ablation: the single read-only psql_console tool is passed via --extra
  # (not a VARIANT), so detect it directly from EXTRA and tag the slug.
  [[ "${EXTRA:-}" =~ --enable_psql_console[[:space:]]+true ]] && slug="${slug}__psql"
```

- [ ] **Step 2: Verify the slug logic**

Run:

```bash
bash -c '
EXTRA="--enable_psql_console true"; slug="base__iter1"
[[ "${EXTRA:-}" =~ --enable_psql_console[[:space:]]+true ]] && slug="${slug}__psql"
echo "with: $slug"
EXTRA=""; slug="base__iter1"
[[ "${EXTRA:-}" =~ --enable_psql_console[[:space:]]+true ]] && slug="${slug}__psql"
echo "without: $slug"
'
```

Expected:
```
with: base__iter1__psql
without: base__iter1
```

- [ ] **Step 3: Commit**

```bash
git add bash_scripts/utils/utils_evaluate.sh
git commit -m "feat(eval): tag run slug with __psql when psql_console ablation is set

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 7: Update CLAUDE.md docs + full test/type pass

**Files:**
- Modify: `src/conversation2sql/eval_framework/agents/bird_baseline/tools/CLAUDE.md`
- Modify: `src/conversation2sql/eval_framework/agents/bird_baseline/CLAUDE.md`

- [ ] **Step 1: Document the tool in tools/CLAUDE.md**

In `tools/CLAUDE.md`, add a row to the "Tool costs" table:

```markdown
| `psql_console` | 1.0 | `bird_interact_env_tools.py` |
```

And add a bullet under "Gotchas":

```markdown
- `psql_console` (ablation, `enable_psql_console`, default off) runs a real `psql -X -c <command>` subprocess under `PGOPTIONS='-c default_transaction_read_only=on -c statement_timeout=60s'` — SELECT + read-only meta-commands (`\dt`, `\d`, `\l`, `\df`) work; writes/DDL are rejected by the server. A guardrail refuses host-reaching meta-commands (`\!`, `\o`, `\copy`, `\i`, `\e`, `\w`, `\s`, and `\g`/`\gx` with a file/pipe arg) **before** psql is spawned. When on it **replaces** `execute_sql`/`get_schema`/`get_table_names`/`get_table_schema`; it is **mutually exclusive** with `enable_table_schema_tools` (enforced in `ConfigReader` and `run_agent_bird_baseline._select_db_tools`).
```

- [ ] **Step 2: Document the flag in bird_baseline/CLAUDE.md**

In `bird_baseline/CLAUDE.md`, under the `agent_code.py` bullets (next to the `enable_table_schema_tools` entry), add:

```markdown
  - `enable_psql_console` (read from `single_task`, default `False`) — when `True` a single read-only `psql_console` tool **replaces** `execute_sql`/`get_schema`/`get_table_*`. Mutually exclusive with `enable_table_schema_tools` (raises in `ConfigReader` and in `_select_db_tools`). Ablation knob via `--extra "--enable_psql_console true"`; the run slug gets a `__psql` suffix.
```

- [ ] **Step 3: Run the full test suite + type check**

Run: `uv run pytest tests/`
Expected: PASS (real-DB tests require the `localhost:5433` container, same as before; if it is down, only the pre-existing `*_real_db` tests are affected — note it).

Run: `uv run pyrefly check`
Expected: no new errors introduced by these files.

- [ ] **Step 4: Commit**

```bash
git add src/conversation2sql/eval_framework/agents/bird_baseline/tools/CLAUDE.md \
        src/conversation2sql/eval_framework/agents/bird_baseline/CLAUDE.md
git commit -m "docs: document psql_console tool and enable_psql_console ablation

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Self-review notes

- **Spec coverage:** scope/gating (Tasks 2-4), tool + read-only + guardrail (Task 1), mutual-exclusion error (Tasks 2 + 4), one-time warning (Task 3), prompt (Task 5), slug (Task 6), tests mirroring the existing file (Task 1), docs (Task 7). All spec sections map to a task.
- **Type/name consistency:** `psql_console`, `psql_console_impl`, `PSQL_GUARDRAIL_REFUSAL`, `PSQL_TIMEOUT_S`, `_select_db_tools`, `enable_psql_console` are used identically across tasks.
- **Known conservative behavior:** the guardrail can false-positive on a SQL string literal containing a denylisted token (e.g. `SELECT '\copy'`) — documented in code and acceptable for a read-only inspection tool.
