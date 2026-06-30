# Single `psql_console` tool (ablation) — design

**Date:** 2026-06-09
**Status:** approved (pending spec review)

## Motivation

Today the agent explores Postgres through four typed tools — `execute_sql`
(SELECT-only), `get_schema`, `get_table_names`, `get_table_schema`. The
`.devcontainer/README.md` shows that the dev container has a real `psql` client
on `PATH` (installed via `postgresql-client`). This ablation asks: does giving
the agent a *single* terminal-style tool that runs `psql` — SQL **and** backslash
meta-commands (`\dt`, `\d table`, `\l`, `\df`) — help or hurt versus the typed
tools? It is a controlled ablation, default **off**; the baseline is unchanged.

## Scope

- **Replaces** (only when enabled): `execute_sql`, `get_schema`,
  `get_table_names`, `get_table_schema`.
- **Untouched** (both modes): column-meaning tools (`get_column_meaning`,
  `get_all_column_meanings`), KB tools (`get_*_knowledge_*`), `ask_user`,
  `submit_sql`. These read metadata files, not Postgres, so `psql` cannot reach
  them.
- **Not** wired into a `presets.py` VARIANT. Driven ad-hoc via
  `--extra "--enable_psql_console true"`, exactly like `enable_table_schema_tools`.

## Gating flag: `enable_psql_console`

A new boolean, default `False`, threaded identically to the existing
`enable_table_schema_tools`:

- `ConfigReader.enable_psql_console: bool = False` (`config_input.py`).
- Passed through `load_bird_interact_as_tasks` (`bird_interact_reader.py`) onto
  each `TaskData`.
- `TaskData.enable_psql_console: bool = False` (`state.py`).
- Read in `run_agent_bird_baseline` via `single_task.enable_psql_console`.

### Mutual exclusion (error, not precedence)

`enable_psql_console` and `enable_table_schema_tools` must not both be `True`.

- **Authoritative check:** a `model_validator(mode="after")` on `ConfigReader`
  raises `ValueError` when both are set — fails fast at config-parse time, before
  any task runs or any model is launched.
- **Defense-in-depth:** `run_agent_bird_baseline` also asserts the two are not
  both set on `single_task`, so a hand-built `TaskData` in a test or future
  caller can't silently produce an inconsistent tool list.

### Visibility: one-time warning

When tasks are loaded with `enable_psql_console=True`, emit a single
`logger.warning(...)` in `load_bird_interact_as_tasks` (once, after the flag is
known — **not** per task, which would emit thousands of lines). Message makes the
ablation obvious in the log, e.g.:

```
PSQL-CONSOLE ABLATION ENABLED: replacing execute_sql/get_schema/get_table_* with the single read-only psql_console tool.
```

## The tool

`psql_console(command: str)` — one SQL statement **or** one backslash
meta-command. Cost **1.0** bird-coin.

- Add `"psql_console": 1.0` to `DB_TOOL_COSTS` in `bird_interact_env_tools.py`.
  Because `TOOL_COSTS = {**DB_TOOL_COSTS, **USER_TOOL_COSTS}` already feeds the
  patience middleware, the flat cost is charged with **no** middleware change.
- The `@tool` wrapper (`psql_console`) is a thin shell over a pure
  `psql_console_impl(command, db_dsn)`, matching the existing
  `*_impl`/`@tool` split so the logic is unit-testable without the LangGraph
  runtime.

### `psql_console_impl(command: str, db_dsn: str) -> str`

1. **Guardrail (host safety) — applied first, before spawning anything.**
   Read-only protects the *database* but not the *dev container*: psql's `\!`
   runs host shell commands, and `\copy` / `\o` / `\i` / `\g <file>` / `\e`
   / `\w` / `\s` touch the host filesystem. If `command` contains any
   denylisted meta-command, return a short refusal string and do **not** spawn
   psql. Denylist tokens (matched as backslash commands): `\!`, `\o`, `\copy`,
   `\i`, `\g` *with a file argument*, `\e`, `\w`, `\s`. Read-only inspection
   meta-commands (`\dt \d \l \df \dn \dv \z \conninfo \d+` …) pass through, as
   does plain SQL.
2. **Execution.** Real subprocess, no shell:
   `subprocess.run(["psql", db_dsn, "-X", "-c", command], env=…, capture_output=True, text=True, timeout=60)`.
   `command` is a single argv element, so there is no shell-injection surface.
   - `-X` ignores `~/.psqlrc` for deterministic output.
   - `env` adds `PGOPTIONS="-c default_transaction_read_only=on -c statement_timeout=60s"`:
     the server rejects writes/DDL and caps query time; SELECT + meta-commands
     still work. `db_dsn` already carries credentials
     (`postgresql://root:123123@localhost:5432/<db>`).
   - `timeout=60` is a wall-clock backstop in case the statement-timeout GUC is
     bypassed (e.g. a meta-command that does not run a cancellable query).
3. **Return value.** On exit code 0, return `stdout`. On nonzero exit, return
   `stderr` (so the agent sees the Postgres error and can self-correct, exactly
   like `execute_sql` surfaces its error). On `subprocess.TimeoutExpired`, return
   a timeout message. Truncate the returned text to the existing
   `MAX_RESULT_LENGTH` (500) and append the existing `TRUNCATION_NOTICE` when it
   overflows.

## Tool-list assembly (`run_agent_bird_baseline`)

```
base tools: get_all_column_meanings, get_column_meaning,
            get_all_external_knowledge_names, get_knowledge_definition,
            get_all_knowledge_definitions, submit_sql
if enable_psql_console:
    add psql_console            # replaces the 4 DB tools
else:
    add execute_sql, get_schema
    if enable_table_schema_tools:
        add get_table_names, get_table_schema
if enable_ask_user:
    add ask_user
```

The mutual-exclusion guard means `enable_psql_console` and
`enable_table_schema_tools` are never both true here.

## Prompt (`prompts.py`)

Pass `enable_psql_console` into the Jinja params. In the "Available tools and
costs" block:

- When `enable_psql_console`: list a single `psql_console` line (runs SQL or
  `\d`/`\dt`/`\l` inspection; read-only; Cost 1) **instead of** the
  `execute_sql` / `get_schema` / `get_table_names` / `get_table_schema` lines.
- When off: unchanged (current behavior, including the
  `enable_table_schema_tools` block).

The column-meaning, KB, `ask_user`, and `submit_sql` lines are unchanged in both
modes.

## Run-directory slug (`bash_scripts/utils/utils_evaluate.sh`)

`build_run_slug` already runs with the `EXTRA` env var in scope (exported by the
justfile, consumed by `eval_payload.sh`). Add: if `EXTRA` contains
`--enable_psql_console true`, append `__psql` to the slug so the ablation is
visible in the run directory name. Example:

```
tools_user__Qwen3.5-9B__ddl__lin__iter1__psql
```

Detection is a simple bash match on the `EXTRA` string
(`[[ "${EXTRA:-}" =~ --enable_psql_console[[:space:]]+true ]]`).

## Testing (mirrors `tests/eval_framework/tools/test_bird_interact_env_tools.py`)

- **Pure guardrail unit tests** (no subprocess): denylisted commands (`\!`,
  `\copy …`, `\o file`, …) return the refusal and never spawn psql; allowed
  inputs (`\dt`, `SELECT 1;`, `\d users`) pass the guard.
- **`subprocess.run`-mocked tests:** argv is
  `["psql", db_dsn, "-X", "-c", command]`; env includes the read-only +
  statement-timeout `PGOPTIONS`; exit 0 → stdout returned; nonzero → stderr
  returned; long output truncated with the notice; `TimeoutExpired` → timeout
  message.
- **Config validation test:** `ConfigReader(enable_psql_console=True,
  enable_table_schema_tools=True)` raises `ValueError`.
- **One real-DB integration test** (against `localhost:5433/solar_panel`, like
  the existing real-DB tests): a `SELECT` and a `\dt` return sane output, and a
  write (`CREATE TABLE …`) is rejected by the read-only session.

## Docs to update

- `eval_framework/agents/bird_baseline/tools/CLAUDE.md` — add `psql_console`
  (cost 1.0), its read-only + guardrail behavior, and the
  `enable_psql_console` gating / mutual-exclusion-with-`enable_table_schema_tools`
  note.
- `eval_framework/agents/bird_baseline/CLAUDE.md` — document the
  `enable_psql_console` flag alongside `enable_table_schema_tools`.

## Out of scope

- No write/DDL support (benchmark answers are reads; per-call subprocesses make
  session-scoped scratch state useless anyway).
- No `presets.py` VARIANT entry; ad-hoc `--extra` only.
- No change to the patience-budget middleware (flat cost rides the existing
  `TOOL_COSTS` path).
```
