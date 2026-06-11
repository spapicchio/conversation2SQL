# Python UDF Ablation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an `enable_python_udf` ablation flag that gives the agent a `create_python_udf` tool — allowing it to define `plpython3u` UDFs in PostgreSQL and call them from SELECT queries.

**Architecture:** A new `@tool` wraps a pure `create_python_udf_impl` function (auto-namespaced with a sanitised `instance_id` prefix to avoid parallel-run collisions). After `agent.invoke()` returns, `cleanup_python_udfs_impl` drops all prefixed functions via `pg_proc` query. Everything mirrors the existing `enable_psql_console` ablation pattern.

**Tech Stack:** Python 3.12, psycopg2, LangChain `@tool` / LangGraph `ToolRuntime`, PostgreSQL `plpython3u`, Pydantic v2, pytest, uv.

---

### Task 1: Config & State Wiring

**Files:**
- Modify: `src/conversation2sql/config_input.py`
- Modify: `src/conversation2sql/eval_framework/state.py`

- [ ] **Step 1: Add flag to `ConfigReader`**

In `config_input.py`, add after `enable_psql_console`:

```python
enable_python_udf: bool = False  # Ablation: add create_python_udf tool (plpython3u). Additive — compatible with all other DB-tool variants.
```

- [ ] **Step 2: Thread flag onto `TaskData`**

In `state.py`, add after `enable_psql_console`:

```python
# When True the agent gets the create_python_udf tool (plpython3u ablation).
# Additive — compatible with enable_psql_console and enable_table_schema_tools.
enable_python_udf: bool = False
```

- [ ] **Step 3: Smoke-test the new field**

Run:
```bash
uv run python -c "
from conversation2sql.config_input import ConfigReader
from conversation2sql.eval_framework.state import TaskData, ColumnMeaningEntry, ExternalKnowledgeEntry
r = ConfigReader()
assert r.enable_python_udf is False
print('ConfigReader OK')
"
```
Expected: `ConfigReader OK`

- [ ] **Step 4: Commit**

```bash
git add src/conversation2sql/config_input.py src/conversation2sql/eval_framework/state.py
git commit -m "feat: add enable_python_udf flag to ConfigReader and TaskData"
```

---

### Task 2: `_safe_instance_prefix` — test then implement

**Files:**
- Modify: `src/conversation2sql/eval_framework/agents/bird_baseline/tools/bird_interact_env_tools.py`
- Modify: `tests/eval_framework/tools/test_bird_interact_env_tools.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/eval_framework/tools/test_bird_interact_env_tools.py`:

```python
# ---------------------------------------------------------------------------
# _safe_instance_prefix
# ---------------------------------------------------------------------------
from conversation2sql.eval_framework.agents.bird_baseline.tools.bird_interact_env_tools import (
    _safe_instance_prefix,
)

def test_safe_instance_prefix_simple():
    assert _safe_instance_prefix("solar_panel_m_5") == "solar_panel_m_5"

def test_safe_instance_prefix_hyphens_and_dots():
    assert _safe_instance_prefix("alien-db.task.1") == "alien_db_task_1"

def test_safe_instance_prefix_uppercase():
    assert _safe_instance_prefix("SolarPanel_M_5") == "solarpanel_m_5"

def test_safe_instance_prefix_truncation():
    long_id = "a" * 50
    result = _safe_instance_prefix(long_id)
    assert len(result) == 30
    assert result == "a" * 30

def test_safe_instance_prefix_default_max_len():
    # default max_len is 30
    result = _safe_instance_prefix("x" * 31)
    assert len(result) == 30
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
uv run pytest tests/eval_framework/tools/test_bird_interact_env_tools.py::test_safe_instance_prefix_simple -v
```
Expected: `ImportError` or `AttributeError` — `_safe_instance_prefix` does not exist yet.

- [ ] **Step 3: Implement `_safe_instance_prefix`**

In `bird_interact_env_tools.py`, add after the `KNOWLEDGE_VISIBLE_FIELDS` constant (before the logger line):

```python
def _safe_instance_prefix(instance_id: str, max_len: int = 30) -> str:
    """Return a SQL-safe prefix derived from instance_id for UDF namespacing.

    Lowercases and replaces any character outside [a-z0-9] with '_', then
    truncates to max_len. max_len=30 leaves ≥32 chars for '_<function_name>'
    within PostgreSQL's 63-char identifier limit.
    """
    safe = re.sub(r"[^a-z0-9]", "_", instance_id.lower())
    return safe[:max_len]
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
uv run pytest tests/eval_framework/tools/test_bird_interact_env_tools.py -k "safe_instance_prefix" -v
```
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add src/conversation2sql/eval_framework/agents/bird_baseline/tools/bird_interact_env_tools.py \
        tests/eval_framework/tools/test_bird_interact_env_tools.py
git commit -m "feat: add _safe_instance_prefix for UDF namespacing"
```

---

### Task 3: `UDFParameter`, `create_python_udf_impl`, `cleanup_python_udfs_impl` — test then implement

**Files:**
- Modify: `src/conversation2sql/eval_framework/agents/bird_baseline/tools/bird_interact_env_tools.py`
- Create: `tests/eval_framework/tools/test_python_udf_tools.py`

These tests require a live PostgreSQL connection to `localhost:5433/solar_panel` (same pattern as `test_long_run_timeout_real_db`). They also require `plpython3u` to be installed in the server — run `SELECT * FROM pg_available_extensions WHERE name = 'plpython3u'` to verify.

- [ ] **Step 1: Write failing tests**

Create `tests/eval_framework/tools/test_python_udf_tools.py`:

```python
"""Tests for Python UDF creation and cleanup impl functions.

Requires a live PostgreSQL server at localhost:5433/solar_panel with
plpython3u available. Run:
    SELECT * FROM pg_available_extensions WHERE name = 'plpython3u';
to confirm before running these tests.
"""
from __future__ import annotations

import psycopg2
import pytest

from conversation2sql.eval_framework.agents.bird_baseline.tools.bird_interact_env_tools import (
    UDFParameter,
    _safe_instance_prefix,
    cleanup_python_udfs_impl,
    create_python_udf_impl,
)

DB_DSN = "postgresql://root:123123@localhost:5433/solar_panel"
TEST_PREFIX = "pytest_udf_test"


@pytest.fixture(autouse=True)
def cleanup_test_udfs():
    """Drop any leftover test UDFs before and after each test."""
    cleanup_python_udfs_impl(DB_DSN, TEST_PREFIX)
    yield
    cleanup_python_udfs_impl(DB_DSN, TEST_PREFIX)


# ---------------------------------------------------------------------------
# create_python_udf_impl
# ---------------------------------------------------------------------------

def test_create_python_udf_impl_returns_qualified_name():
    result = create_python_udf_impl(
        function_name="add_nums",
        parameters=[
            UDFParameter(name="x", pg_type="FLOAT8"),
            UDFParameter(name="y", pg_type="FLOAT8"),
        ],
        return_type="FLOAT8",
        python_body="return x + y",
        db_dsn=DB_DSN,
        instance_id="pytest_udf_test_run",
    )
    assert result == "pytest_udf_test_run_add_nums"


def test_create_python_udf_impl_callable_from_select():
    prefix_safe = "pytest_udf_test_run"
    create_python_udf_impl(
        function_name="add_nums",
        parameters=[
            UDFParameter(name="x", pg_type="FLOAT8"),
            UDFParameter(name="y", pg_type="FLOAT8"),
        ],
        return_type="FLOAT8",
        python_body="return x + y",
        db_dsn=DB_DSN,
        instance_id="pytest_udf_test_run",
    )
    conn = psycopg2.connect(DB_DSN)
    try:
        with conn.cursor() as cur:
            cur.execute(f"SELECT {prefix_safe}_add_nums(3.0, 4.0)")
            assert cur.fetchone()[0] == pytest.approx(7.0)
    finally:
        conn.close()


def test_create_python_udf_impl_no_params():
    result = create_python_udf_impl(
        function_name="get_pi",
        parameters=[],
        return_type="FLOAT8",
        python_body="import math\nreturn math.pi",
        db_dsn=DB_DSN,
        instance_id="pytest_udf_test_run",
    )
    assert result == "pytest_udf_test_run_get_pi"
    conn = psycopg2.connect(DB_DSN)
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT pytest_udf_test_run_get_pi()")
            assert cur.fetchone()[0] == pytest.approx(3.14159, rel=1e-5)
    finally:
        conn.close()


def test_create_python_udf_impl_sanitizes_function_name():
    result = create_python_udf_impl(
        function_name="My-Func!",
        parameters=[],
        return_type="INTEGER",
        python_body="return 42",
        db_dsn=DB_DSN,
        instance_id="pytest_udf_test_run",
    )
    assert result == "pytest_udf_test_run_my_func_"


def test_create_python_udf_impl_error_returns_message():
    result = create_python_udf_impl(
        function_name="bad_func",
        parameters=[UDFParameter(name="x", pg_type="NOT_A_REAL_TYPE")],
        return_type="INTEGER",
        python_body="return 1",
        db_dsn=DB_DSN,
        instance_id="pytest_udf_test_run",
    )
    assert isinstance(result, str)
    assert "error" in result.lower() or "type" in result.lower()


# ---------------------------------------------------------------------------
# cleanup_python_udfs_impl
# ---------------------------------------------------------------------------

def test_cleanup_removes_prefixed_functions():
    conn = psycopg2.connect(DB_DSN)
    try:
        with conn.cursor() as cur:
            cur.execute(
                "CREATE EXTENSION IF NOT EXISTS plpython3u"
            )
            cur.execute(
                "CREATE OR REPLACE FUNCTION pytest_udf_test_fn1() "
                "RETURNS INTEGER AS $$ return 1 $$ LANGUAGE plpython3u"
            )
            cur.execute(
                "CREATE OR REPLACE FUNCTION pytest_udf_test_fn2() "
                "RETURNS INTEGER AS $$ return 2 $$ LANGUAGE plpython3u"
            )
        conn.commit()
    finally:
        conn.close()

    cleanup_python_udfs_impl(DB_DSN, TEST_PREFIX)

    conn = psycopg2.connect(DB_DSN)
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT proname FROM pg_proc WHERE proname LIKE %s",
                (f"{TEST_PREFIX}_%",),
            )
            assert cur.fetchall() == []
    finally:
        conn.close()


def test_cleanup_does_not_touch_other_functions():
    conn = psycopg2.connect(DB_DSN)
    try:
        with conn.cursor() as cur:
            cur.execute(
                "CREATE EXTENSION IF NOT EXISTS plpython3u"
            )
            cur.execute(
                "CREATE OR REPLACE FUNCTION other_prefix_fn() "
                "RETURNS INTEGER AS $$ return 99 $$ LANGUAGE plpython3u"
            )
        conn.commit()
    finally:
        conn.close()

    cleanup_python_udfs_impl(DB_DSN, TEST_PREFIX)

    conn = psycopg2.connect(DB_DSN)
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT proname FROM pg_proc WHERE proname = 'other_prefix_fn'")
            assert cur.fetchone() is not None
        # manual teardown
        with conn.cursor() as cur:
            cur.execute("DROP FUNCTION IF EXISTS other_prefix_fn()")
        conn.commit()
    finally:
        conn.close()


def test_cleanup_noop_when_no_functions_exist():
    # Should not raise even when there's nothing to drop
    cleanup_python_udfs_impl(DB_DSN, "nonexistent_prefix_xyz")
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
uv run pytest tests/eval_framework/tools/test_python_udf_tools.py -v 2>&1 | head -30
```
Expected: `ImportError` — `UDFParameter`, `create_python_udf_impl`, `cleanup_python_udfs_impl` do not exist yet.

- [ ] **Step 3: Add `UDFParameter` to `bird_interact_env_tools.py`**

After the `ExecuteSQLResponse` class definition (around line 122), add:

```python
class UDFParameter(BaseModel):
    """One parameter for a Python UDF: a name and its PostgreSQL type."""
    name: str
    pg_type: str
```

- [ ] **Step 4: Add `create_python_udf_impl` to `bird_interact_env_tools.py`**

Add after the `psql_console_impl` function, before the `apply_column_comments_impl` function:

```python
def create_python_udf_impl(
    function_name: str,
    parameters: list[UDFParameter],
    return_type: str,
    python_body: str,
    db_dsn: str,
    instance_id: str,
) -> str:
    """Create a plpython3u UDF in PostgreSQL and return its qualified name.

    The function is registered under ``<safe_prefix>_<sanitised_name>`` where
    ``safe_prefix`` is derived from ``instance_id`` via ``_safe_instance_prefix``,
    ensuring parallel conversations on the same database do not collide.

    Returns the qualified name on success, or the error message on failure.
    """
    safe_name = re.sub(r"[^a-z0-9]", "_", function_name.lower())
    prefix = _safe_instance_prefix(instance_id)
    qualified = f"{prefix}_{safe_name}"[:63]

    param_str = ", ".join(f"{p.name} {p.pg_type}" for p in parameters)
    ddl = (
        f"CREATE OR REPLACE FUNCTION {qualified}({param_str}) "
        f"RETURNS {return_type} AS $$ {python_body} $$ LANGUAGE plpython3u"
    )

    conn = psycopg2.connect(db_dsn)
    try:
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS plpython3u")
            cur.execute(ddl)
        conn.commit()
        return qualified
    except psycopg2.DatabaseError as exc:
        conn.rollback()
        return f"Error: {exc}"
    finally:
        conn.close()
```

- [ ] **Step 5: Add `cleanup_python_udfs_impl` to `bird_interact_env_tools.py`**

Add immediately after `create_python_udf_impl`:

```python
def cleanup_python_udfs_impl(db_dsn: str, prefix: str) -> None:
    """Drop all plpython3u UDFs whose name starts with ``prefix_``.

    Called unconditionally after each agent run when enable_python_udf=True.
    Uses ``pg_get_function_identity_arguments`` to build the exact DROP
    signature so overloaded functions are handled correctly.
    No-op when no matching functions exist.
    """
    conn = psycopg2.connect(db_dsn)
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT proname, pg_get_function_identity_arguments(oid) "
                "FROM pg_proc WHERE proname LIKE %s",
                (f"{prefix}_%",),
            )
            for proname, args in cur.fetchall():
                cur.execute(f"DROP FUNCTION IF EXISTS {proname}({args})")
        conn.commit()
    finally:
        conn.close()
```

- [ ] **Step 6: Run tests to confirm they pass**

```bash
uv run pytest tests/eval_framework/tools/test_python_udf_tools.py -v
```
Expected: all tests pass. (If `plpython3u` is not installed, step through the next step first.)

> **If `plpython3u` is missing:** Connect to the container and run:
> ```bash
> apt-get install -y postgresql-plpython3-<version>
> # then inside psql: CREATE EXTENSION plpython3u;
> ```
> Or verify with: `SELECT name FROM pg_available_extensions WHERE name = 'plpython3u';`

- [ ] **Step 7: Commit**

```bash
git add src/conversation2sql/eval_framework/agents/bird_baseline/tools/bird_interact_env_tools.py \
        tests/eval_framework/tools/test_python_udf_tools.py
git commit -m "feat: add UDFParameter, create_python_udf_impl, cleanup_python_udfs_impl"
```

---

### Task 4: `DB_TOOL_SPECS` entry + `@tool` wrapper

**Files:**
- Modify: `src/conversation2sql/eval_framework/agents/bird_baseline/tools/bird_interact_env_tools.py`

- [ ] **Step 1: Add spec entry to `DB_TOOL_SPECS`**

In `bird_interact_env_tools.py`, add to the `DB_TOOL_SPECS` dict (after the `psql_console` entry):

```python
"create_python_udf": ToolSpec(
    "create_python_udf",
    1.0,
    "create a PostgreSQL Python UDF (plpython3u) for complex formulas; returns the callable function name",
),
```

- [ ] **Step 2: Add the `@tool` wrapper**

Add after the `@tool psql_console` function (the `@tool\ndef psql_console(...)` block), before the `# Column meaning tools` section:

```python
@tool
def create_python_udf(
    function_name: str,
    parameters: list[UDFParameter],
    return_type: str,
    python_body: str,
    runtime: ToolRuntime[TaskData, CustomAgentState],
) -> str:
    """Create a PostgreSQL Python (plpython3u) user-defined function and return
    its callable name. Use it to implement formulas that are hard to express in
    pure SQL, then call the returned name inside execute_sql or psql_console.

    Args:
        function_name: Desired function name (auto-prefixed for isolation).
        parameters: List of {name, pg_type} parameter definitions.
        return_type: PostgreSQL return type of the function.
        python_body: Python code body ending with `return <value>`.
            The body runs inside PostgreSQL via plpython3u.
            Import standard library modules with normal `import` statements.

    PostgreSQL type → Python type mapping for parameters and return type:

        PostgreSQL type              Python type inside body
        ----------------------------------------------------
        INTEGER / SMALLINT           int
        BIGINT                       int
        REAL / FLOAT4                float
        DOUBLE PRECISION / FLOAT8    float
        NUMERIC / DECIMAL            decimal.Decimal  (import decimal)
        TEXT / VARCHAR / CHAR        str
        BOOLEAN                      bool
        DATE                         datetime.date    (import datetime)
        TIMESTAMP                    datetime.datetime
        TIMESTAMPTZ                  datetime.datetime (timezone-aware)
        BYTEA                        bytes
        any[] (array type)           list
        composite type               dict  (keys = column names)
        NULL (any nullable type)     None

    Returns:
        The qualified function name to use in SELECT queries, e.g.
        ``instance_42_my_formula``. On error returns the error message.
    """
    return create_python_udf_impl(
        function_name=function_name,
        parameters=parameters,
        return_type=return_type,
        python_body=python_body,
        db_dsn=runtime.context.db_dsn,
        instance_id=runtime.context.instance_id,
    )
```

- [ ] **Step 3: Verify no import errors**

```bash
uv run python -c "from conversation2sql.eval_framework.agents.bird_baseline.tools.bird_interact_env_tools import create_python_udf; print('OK')"
```
Expected: `OK`

- [ ] **Step 4: Commit**

```bash
git add src/conversation2sql/eval_framework/agents/bird_baseline/tools/bird_interact_env_tools.py
git commit -m "feat: add create_python_udf @tool and DB_TOOL_SPECS entry"
```

---

### Task 5: `__init__.py` exports and cost stamping

**Files:**
- Modify: `src/conversation2sql/eval_framework/agents/bird_baseline/tools/__init__.py`

- [ ] **Step 1: Add imports**

In `__init__.py`, add to the `from bird_interact_env_tools import (...)` block:

```python
    create_python_udf,
    UDFParameter,
    _safe_instance_prefix,
    create_python_udf_impl,
    cleanup_python_udfs_impl,
```

- [ ] **Step 2: Add `create_python_udf` to `stamp_cost_in_descriptions`**

In the `stamp_cost_in_descriptions([...], TOOL_SPECS)` call, add `create_python_udf` to the list:

```python
stamp_cost_in_descriptions(
    [
        execute_sql,
        psql_console,
        create_python_udf,   # ← add this line
        get_schema,
        get_table_names,
        get_table_schema,
        get_all_column_meanings,
        get_column_meaning,
        get_all_external_knowledge_names,
        get_knowledge_definition,
        get_all_knowledge_definitions,
        submit_sql,
    ],
    TOOL_SPECS,
)
```

- [ ] **Step 3: Add to `__all__`**

In `__all__`, add under the `# env tools (LangGraph @tool wrappers)` section:

```python
    "create_python_udf",
```

And under `# env tools (pure *_impl functions)`:

```python
    "create_python_udf_impl",
    "cleanup_python_udfs_impl",
    "_safe_instance_prefix",
```

And under `# env tools (types / constants)`:

```python
    "UDFParameter",
```

- [ ] **Step 4: Verify imports resolve**

```bash
uv run python -c "
from conversation2sql.eval_framework.agents.bird_baseline.tools import (
    create_python_udf, cleanup_python_udfs_impl, _safe_instance_prefix, UDFParameter
)
print('all exports OK')
"
```
Expected: `all exports OK`

- [ ] **Step 5: Commit**

```bash
git add src/conversation2sql/eval_framework/agents/bird_baseline/tools/__init__.py
git commit -m "feat: export create_python_udf, cleanup_python_udfs_impl, UDFParameter"
```

---

### Task 6: Agent wiring — tool list + cleanup call

**Files:**
- Modify: `src/conversation2sql/eval_framework/agents/bird_baseline/agent_code.py`

- [ ] **Step 1: Add import**

In `agent_code.py`, add to the tools import block:

```python
from conversation2sql.eval_framework.agents.bird_baseline.tools import (
    ...
    create_python_udf,
    cleanup_python_udfs_impl,
    _safe_instance_prefix,
)
```

(Add `create_python_udf`, `cleanup_python_udfs_impl`, `_safe_instance_prefix` to the existing import from `tools`.)

- [ ] **Step 2: Add tool to the tool list**

In `run_agent_bird_baseline`, after the `if enable_ask_user:` block (around line 111), add:

```python
    if single_task.enable_python_udf:
        tools.append(create_python_udf)
```

- [ ] **Step 3: Add prompt param**

In the `build_bird_interact_agent_messages` params dict, add:

```python
        "enable_python_udf": single_task.enable_python_udf,
```

- [ ] **Step 4: Add cleanup call**

After `response: CustomAgentState = agent.invoke(agent_state, context=single_task)`, add:

```python
    if single_task.enable_python_udf:
        cleanup_python_udfs_impl(
            db_dsn=single_task.db_dsn,
            prefix=_safe_instance_prefix(single_task.instance_id),
        )
```

- [ ] **Step 5: Verify no import errors**

```bash
uv run python -c "from conversation2sql.eval_framework.agents.bird_baseline.agent_code import run_agent_bird_baseline; print('OK')"
```
Expected: `OK`

- [ ] **Step 6: Commit**

```bash
git add src/conversation2sql/eval_framework/agents/bird_baseline/agent_code.py
git commit -m "feat: wire create_python_udf into agent tool list + cleanup after invoke"
```

---

### Task 7: Prompt template

**Files:**
- Modify: `src/conversation2sql/eval_framework/agents/bird_baseline/prompts.py`

- [ ] **Step 1: Add tool line to the tools list block**

In `_BIRD_AGENT_SYSTEM`, the tools list ends with `{% if enable_table_schema_tools %}...{% endif %}{% endif %}` then continues with `- get_all_column_meanings`. Add a `create_python_udf` line between the DB-tools block and the column-meaning tools:

Find this line:
```
{% endif %}{% endif %}- get_all_column_meanings: {{ tool_specs['get_all_column_meanings'].summary }}. Cost: {{ tool_specs['get_all_column_meanings'].cost }}
```

Replace with:
```
{% endif %}{% endif %}{% if enable_python_udf %}- create_python_udf: {{ tool_specs['create_python_udf'].summary }}. Cost: {{ tool_specs['create_python_udf'].cost }}
{% endif %}- get_all_column_meanings: {{ tool_specs['get_all_column_meanings'].summary }}. Cost: {{ tool_specs['get_all_column_meanings'].cost }}
```

- [ ] **Step 2: Add strategy tip**

In `_BIRD_AGENT_SYSTEM`, after the `- If a submission fails and budget remains, debug and try again.` line, add:

```
{% if enable_python_udf %}- Use create_python_udf to implement complex formulas as Python functions, then call the returned name in execute_sql or psql_console.
{% endif %}
```

- [ ] **Step 3: Pass `enable_python_udf` in `build_bird_interact_agent_messages`**

The `enable_python_udf` param is already added to `build_bird_interact_agent_messages` in Task 6 Step 3. Verify it's present in the `params` dict passed to `utils_build_messages`. No extra change needed.

- [ ] **Step 4: Test prompt rendering**

```bash
uv run python -c "
from conversation2sql.eval_framework.agents.bird_baseline.prompts import build_bird_interact_agent_messages
msgs = build_bird_interact_agent_messages({
    'total_budget': 10,
    'amb_user_query': 'test',
    'enable_ask_user': False,
    'enable_table_schema_tools': False,
    'enable_psql_console': False,
    'enable_python_udf': True,
})
system = msgs[0].content if hasattr(msgs[0], 'content') else msgs[0]['content']
assert 'create_python_udf' in system, 'tool not in prompt'
assert 'plpython3u' in system, 'type mapping not in prompt'
print('prompt OK — create_python_udf present')

# Verify it is absent when flag is False
msgs2 = build_bird_interact_agent_messages({
    'total_budget': 10,
    'amb_user_query': 'test',
    'enable_ask_user': False,
    'enable_table_schema_tools': False,
    'enable_psql_console': False,
    'enable_python_udf': False,
})
system2 = msgs2[0].content if hasattr(msgs2[0], 'content') else msgs2[0]['content']
assert 'create_python_udf' not in system2, 'tool should not appear when flag is False'
print('prompt OK — create_python_udf absent when disabled')
"
```
Expected: two `OK` lines.

- [ ] **Step 5: Commit**

```bash
git add src/conversation2sql/eval_framework/agents/bird_baseline/prompts.py
git commit -m "feat: add create_python_udf to agent prompt when enable_python_udf=True"
```

---

### Task 8: Bash slug suffix

**Files:**
- Modify: `bash_scripts/utils/utils_evaluate.sh`

- [ ] **Step 1: Add slug suffix line**

In `build_run_slug()`, after the `__psql` line (line 94):

```bash
  [[ "${EXTRA:-}" =~ --enable_psql_console[[:space:]]+true ]] && slug="${slug}__psql"
```

Add immediately after:

```bash
  [[ "${EXTRA:-}" =~ --enable_python_udf[[:space:]]+true ]] && slug="${slug}__pyudf"
```

- [ ] **Step 2: Verify the slug is correct**

```bash
EXTRA="--enable_python_udf true" bash -c '
source bash_scripts/utils/utils_evaluate.sh 2>/dev/null || true
# Just test the regex directly
EXTRA="--enable_python_udf true"
slug="test_base"
[[ "${EXTRA:-}" =~ --enable_python_udf[[:space:]]+true ]] && slug="${slug}__pyudf"
echo "slug: $slug"
'
```
Expected: `slug: test_base__pyudf`

- [ ] **Step 3: Commit**

```bash
git add bash_scripts/utils/utils_evaluate.sh
git commit -m "feat: add __pyudf slug suffix for enable_python_udf ablation"
```

---

### Task 9: Final test suite

- [ ] **Step 1: Run full test suite**

```bash
uv run pytest tests/ -v 2>&1 | tail -30
```
Expected: all existing tests pass; new UDF tests pass.

- [ ] **Step 2: Run type check**

```bash
uv run pyrefly check 2>&1 | tail -20
```
Expected: no new errors introduced.

- [ ] **Step 3: Smoke-test the flag end-to-end (optional, requires running models)**

```bash
uv run conv2sql run --config configs/eval_pipeline_config.yaml \
  --debug true --enable_python_udf true 2>&1 | head -40
```
Expected: runs one task without error; run dir slug contains `__pyudf`.
