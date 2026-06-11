# Python UDF Ablation Design

**Date:** 2026-06-11
**Branch:** feat/python-udf-ablation (to be created)
**Status:** Approved

## Overview

Add a new ablation flag `enable_python_udf` that equips the agent with a
`create_python_udf` tool. The tool lets the agent define a PostgreSQL
`plpython3u` function to implement complex formulas, then call it from
subsequent `SELECT` queries via `execute_sql`. The existing DB tools are kept
unchanged — this tool is additive.

**Motivation:** Some BIRD-Interact Query tasks involve formulas that are
cumbersome in pure SQL (custom scoring, multi-step arithmetic, string
normalization). Python UDFs let the agent offload that logic to Python code
running inside Postgres, then compose the result into a SELECT.

**Ablation pattern:** mirrors `enable_psql_console`. Flag off = baseline
behaviour; flag on = baseline + `create_python_udf`. Run-dir slug gets a
`__pyudf` suffix when the flag is set.

---

## Flag & Config

`ConfigReader` gains:

```python
enable_python_udf: bool = False
```

Threaded onto `TaskData` as `enable_python_udf: bool` (same pattern as
`enable_psql_console`). No mutual-exclusion with other ablation flags — the
tool is additive and compatible with all existing DB-tool variants.

CLI override: `--extra "--enable_python_udf true"`.

---

## Tool Interface

### Input schema

```python
class UDFParameter(BaseModel):
    name: str      # parameter name, e.g. "x"
    pg_type: str   # PostgreSQL type, e.g. "FLOAT8"

@tool
def create_python_udf(
    function_name: str,          # desired name; will be prefixed automatically
    parameters: list[UDFParameter],
    return_type: str,            # PostgreSQL return type, e.g. "FLOAT8"
    python_body: str,            # Python statements ending with `return <value>`
    runtime: ToolRuntime[TaskData, CustomAgentState],
) -> str: ...
```

### Behaviour (`create_python_udf_impl`)

1. Sanitize `function_name`: lowercase, replace `[^a-z0-9]` → `_`.
2. Build qualified name: `<safe_prefix>_<function_name>`, where
   `safe_prefix = _safe_instance_prefix(instance_id)` (see Namespacing below).
   Truncate qualified name to 63 chars (PostgreSQL identifier limit).
3. `CREATE EXTENSION IF NOT EXISTS plpython3u` (idempotent; requires superuser —
   `root` in both dev containers qualifies).
4. `CREATE OR REPLACE FUNCTION <qualified_name>(<params>) RETURNS <return_type>
   AS $$ <python_body> $$ LANGUAGE plpython3u`.
5. Return the qualified name as a string so the agent can call it in a
   subsequent `SELECT`.

On any `psycopg2.DatabaseError` the tool returns the error message without
raising, consistent with `execute_sql`.

### Tool description (type mapping embedded)

The `@tool` docstring includes the full PostgreSQL ↔ Python type table so the
agent knows what to write in `pg_type` / `return_type`:

```
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
TIMESTAMPTZ                  datetime.datetime  (timezone-aware)
BYTEA                        bytes
any[] (array type)           list
composite type               dict  (keys = column names)
NULL (any nullable type)     None
```

### Cost

**1 patience** — same as `execute_sql`. Registered in `DB_TOOL_SPECS` /
`DB_TOOL_COSTS` in `bird_interact_env_tools.py`.

---

## Namespacing

```python
def _safe_instance_prefix(instance_id: str, max_len: int = 30) -> str:
    safe = re.sub(r"[^a-z0-9]", "_", instance_id.lower())
    return safe[:max_len]
```

`max_len = 30` leaves at least 32 chars for `_` + `function_name` within
PostgreSQL's 63-char limit. Each parallel conversation has a unique
`instance_id`, so prefixes never collide across concurrent runs against the
same database.

---

## Cleanup

`cleanup_python_udfs_impl(db_dsn, prefix)` is called unconditionally in
`run_agent_bird_baseline` after `agent.invoke()` returns (even if the agent
never called `create_python_udf` — the query returns nothing and is a no-op).

```python
def cleanup_python_udfs_impl(db_dsn: str, prefix: str) -> None:
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

Uses `pg_get_function_identity_arguments` to build the exact signature, so
overloaded functions are dropped correctly without knowing argument types in
advance. No state tracking in `CustomAgentState` is needed.

---

## Wiring Summary

| File | Change |
|------|--------|
| `config_input.py` | Add `enable_python_udf: bool = False` to `ConfigReader` |
| `eval_framework/state.py` | Add `enable_python_udf: bool = False` to `TaskData` |
| `tools/bird_interact_env_tools.py` | Add `UDFParameter`, `_safe_instance_prefix`, `create_python_udf_impl`, `cleanup_python_udfs_impl`, `create_python_udf` `@tool`, entry in `DB_TOOL_SPECS` |
| `tools/__init__.py` | Export `create_python_udf`, `cleanup_python_udfs_impl` |
| `tools/tool_specs.py` | Add `create_python_udf` spec (cost 1.0) |
| `agent_code.py` | Add `create_python_udf` to tool list when `enable_python_udf=True`; call `cleanup_python_udfs_impl` after `agent.invoke()` |
| `prompts.py` | Add Jinja2 block rendering the tool entry when `enable_python_udf=True` |
| `bash_scripts/utils/utils_evaluate.sh` | Add `__pyudf` slug suffix in `build_run_slug()` when `--enable_python_udf true` is in `$EXTRA` (mirrors the `__psql` line) |

---

## Testing

- `test_create_python_udf_impl`: creates a simple UDF, calls it via psycopg2,
  verifies result; tests error handling (bad body, bad types).
- `test_cleanup_python_udfs_impl`: creates two prefixed functions, calls
  cleanup, verifies both are gone; verifies non-prefixed functions are
  untouched.
- `test_safe_instance_prefix`: edge cases (hyphens, dots, long IDs, uppercase).
- Integration smoke: `enable_python_udf=True` with a real DB, agent creates a
  UDF and calls it in a SELECT.
