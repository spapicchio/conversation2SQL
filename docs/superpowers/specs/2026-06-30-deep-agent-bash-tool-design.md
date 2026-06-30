# deep_agent: bash-based catalog exploration + SQL execution

**Date:** 2026-06-30
**Status:** Approved, ready for implementation
**Area:** `src/conversation2sql/eval_framework/agents/deep_agent/`

## Problem

The `deep_agent` baseline currently explores the database through `deepagents`'
`FilesystemMiddleware`: at startup `build_db_filesystem` seeds an **in-memory
virtual filesystem** (a `StateBackend` holding `FileData` dicts) and exposes four
tools (`ls`/`read_file`/`glob`/`grep`). This pulls in a non-trivial stack —
`CustomFilesystemMiddleware`, `StateBackend`, the `files` state field, system-prompt
injection plumbing (`_capture_system_message`, `_split_system_prompt`), and a
`deep_enable_fs_write` ablation — to do what is essentially "read some Markdown
files." The author wants something simpler.

## Goal

Replace the virtual filesystem with **one `bash` tool** that runs read-only shell
commands via `subprocess.run`, with `cwd` bound to a freshly-materialized,
per-task catalog directory on disk. The same tool also runs SQL via `psql`,
absorbing `execute_sql`. The result is a **3-tool agent**: `bash` + `submit_sql` +
`ask_user` (down from the current 4). The patience budget, `submit_sql`, and
`ask_user` are unchanged. Of the four deepagents ablations, only `subagents`
survives; `todos`, `summarization`, and `fs_write` are removed.

### Non-goals

- No change to `submit_sql`, `ask_user`, the patience-budget middleware, or the
  surviving `subagents` ablation.
- No docker / containerization (explicitly rejected — see Decisions).
- No path-containment guardrail (whitelist + non-advertised cwd is deemed
  sufficient — see Decisions).

## Decisions (from brainstorming)

1. **Masked KB:** materialize a per-task catalog dir; the KB is re-rendered from
   `masked_agent_kb` exactly as today, so masked prerequisites never appear on
   disk. The on-disk catalog path is never advertised to the model.
2. **Execution:** direct `subprocess.run` with `cwd` = the per-task dir. **No
   docker** (the eval runs on SLURM where docker-in-docker is often unavailable;
   the files are static Markdown).
3. **Scope:** replace the FS tools *in place* inside `deep_agent`. Keep only the
   `subagents` ablation. **Remove `deep_enable_fs_write`** (bash is read-only),
   **`deep_enable_todos`**, and **`deep_enable_summarization`** — trimming the
   agent to the essentials.
4. **Guardrails:** read-only command **whitelist** (`cat`, `ls`, `find`, `grep`,
   `head`, `tail`, `wc`, `psql`). No separate path-containment check.
5. **SQL:** `bash` runs `psql` (3 tools total), accepting the loss of
   `execute_sql`'s row/cell result truncation in favor of a generic char cap.
   `submit_sql` **cannot** be folded in — it is the scored terminal action.
6. **Include `database_overview.md`** in the materialized dir (new vs. current
   behavior, which omitted it).
7. **Rename** `filesystem_seed.py` → `catalog_seed.py`.
8. Flat `bash` cost = `1.0` (tunable). Keep `captured_system_prompt`.

## Why `submit_sql` stays a separate tool

`submit_sql_impl` executes both the predicted and the ground-truth SQL, compares
result sets (honoring `sql_query_conditions["order"]`), sets `passed`, feeds
`_extract_predicted_sql`, and trips the patience sentinel that ends the run. This
is Python evaluation logic, not a shell command, so it cannot live inside `bash`.

## Design

### 1. Per-task catalog materialization (`catalog_seed.py`)

Rename `filesystem_seed.py` → `catalog_seed.py`. Replace `build_db_filesystem`
(returns `dict[str, FileData]`) with:

```python
def materialize_catalog_dir(task: TaskData) -> Path:
    """Write this task's catalog to a fresh temp dir and return it.

    Layout (cwd the bash tool runs in):
      <tmp>/database_overview.md            # copied from disk (NEW)
      <tmp>/tables/<table>.md               # copied verbatim
      <tmp>/tables/_foreign_key_constraints.md
      <tmp>/knowledge_base/<node>.md        # re-rendered from masked_agent_kb
    """
```

- `tempfile.mkdtemp(prefix="deep_catalog_")`.
- Tables: copy every `<catalog>/<db>/tables/*.md` verbatim.
- `database_overview.md`: copy from `<catalog>/<db>/database_overview.md` if
  present (skip silently if absent — older catalogs may lack it).
- Knowledge base: one `<tmp>/knowledge_base/<node>.md` per node in
  `masked_agent_kb`, rendered via `linearize_prerequisites` (unchanged logic).
- Keep the same guard: raise `FileNotFoundError` if `<catalog>/<db>/tables/` is
  missing, with the existing actionable message.

### 2. The `bash` tool (`return_tool_bash`)

A factory closing over the per-task cwd and a DB env dict (mirrors
`return_tool_ask_user`):

```python
def return_tool_bash(catalog_dir: Path, pg_env: dict[str, str]):
    @tool
    def bash(command: str) -> str:
        """Run a read-only shell command to explore the catalog in your working
        directory, or psql to run a read-only SQL query (SELECT/WITH/EXPLAIN)."""
        ...
    return bash
```

Behavior:
- **Whitelist:** split `command` on `|`; the first token of each segment must be
  in `{cat, ls, find, grep, head, tail, wc, psql}`. Otherwise return a clear
  refusal naming the allowed commands (no subprocess spawned).
- **psql guardrail:** for any segment whose command is `psql`, run
  `bird_baseline`'s `_violates_psql_guardrail` against the command string and
  refuse host-reaching backslash meta-commands (`\!`, `\copy`, `\i`, `\o`, `\e`,
  `\w`, `\s`, `\g`/`\gx` with a file/pipe) **before** spawning. This closes the
  *backslash* host-file-read escape. It does **not** close the
  `SELECT pg_read_file('…')` server-side escape (the eval role may be superuser);
  leave a `# TODO(unmasked-kb-leak):` comment in `return_tool_bash` pointing at
  this — to be solved later (see Risks).
- **Execution:**
  `subprocess.run(["bash","-lc",command], cwd=catalog_dir, env=pg_env,
  capture_output=True, text=True, timeout=30)`.
- **Output:** `f"exit={rc}\n--- stdout ---\n{out[:8000]}\n--- stderr ---\n{err[:2000]}"`.
  On `TimeoutExpired`, return a clear timeout message.

### 3. DB environment (`pg_env`)

Built once per task from the task DSN (parse `task`'s DSN / `db_dsn_template`
result for the selected database) into:

```
PGHOST, PGPORT, PGUSER, PGPASSWORD, PGDATABASE
PGOPTIONS = "-c default_transaction_read_only=on -c statement_timeout=60s"
```

so `psql -c "SELECT …"` connects with no credentials visible to the model and
writes are rejected server-side. Reuse whatever DSN-parsing helper
`bird_baseline` already uses for `psql_console` if one exists; otherwise parse
with `psycopg2.extensions.parse_dsn` / `urllib`.

### 4. Wiring (`agent_code.py`)

- `_build_deep_tools(task, model_user_parsing, model_user_generator, catalog_dir,
  pg_env)` → `[return_tool_bash(catalog_dir, pg_env), submit_sql,
  return_tool_ask_user(...)]`.
- `_build_deep_middleware`: delete `_build_fs_middleware` and the
  `CustomFilesystemMiddleware`/`StateBackend` imports. Drop the `todos` and
  `summarization` gates (and the `TodoListMiddleware`/`SummarizationMiddleware`
  imports). Keep `ModelRetryMiddleware`, `ToolRetryMiddleware`, the `subagents`
  gate, and the patience stack (`check_budget_limit` → `sanitize_thinking_history`
  → `_capture_system_message` → `wrap_model_append_tool_message` →
  `tool_wrapper_patience_and_submit`).
- `run_agent_deep_agent`:
  1. `catalog_dir = materialize_catalog_dir(task)`
  2. build `pg_env`
  3. build tools + middleware, `create_agent(...)`
  4. initial state: patience fields + `captured_system_prompt: ""` (no `files`)
  5. `try: agent.invoke(...) finally: shutil.rmtree(catalog_dir, ignore_errors=True)`
  6. rest (system-prompt injection, `_extract_predicted_sql`,
     `utils_process_agent_response`) unchanged.
- Remove the FS-tool description constants (`FILESYSTEM_SYSTEM_PROMPT`,
  `FS_TOOL_DESCRIPTIONS`, `_FS_READ_TOOLS`/`_FS_WRITE_TOOLS`, `_build_fs_middleware`).

### 5. State (`agent_code_state.py`)

`DeepAgentCustomState` no longer mixes in deepagents' `FilesystemState`/`files`.
It carries the three patience fields (from `CustomAgentState`) +
`captured_system_prompt`. The `subagents` middleware's state keys merge
automatically via `create_agent`.

### 6. Costs (`catalog_seed.py`)

Replace `FS_TOOL_COSTS` / `deep_tool_costs(enable_fs_write=...)` with:

```python
def deep_tool_costs() -> dict[str, float]:
    return {"bash": 1.0, "submit_sql": TOOL_COSTS["submit_sql"],
            "ask_user": TOOL_COSTS["ask_user"]}
```

(`enable_fs_write` parameter removed.)

### 7. Prompts (`prompts.py`)

Rewrite `_DEEP_AGENT_SYSTEM`:
- Describe the catalog as files in the **current working directory**
  (`database_overview.md`, `tables/`, `knowledge_base/`), explored with `bash`
  (`ls`, `cat`, `find`, `grep`, `head`, `tail`, `wc`).
- Describe running read-only SQL via `psql -c "SELECT …"` through the same `bash`
  tool (no connection details needed; writes rejected).
- Keep `ask_user` and `submit_sql` guidance.
- Remove the `enable_fs_write` and `enable_todos` blocks. Keep the
  `enable_subagents` conditional line.

### 8. Ablation flag removal

Remove three flags — `deep_enable_fs_write`, `deep_enable_todos`, and
`deep_enable_summarization` — from `ConfigReader`/`TaskData`, their threading, the
prompt blocks, the middleware gates, and their run-dir slug detection
(`__fswrite`, `__todos`, `__summar`) in `bash_scripts/utils/utils_evaluate.sh`.
Leave `deep_enable_subagents` (and its `__subagents` slug) intact.

## Testing

Update `tests/eval_framework/agents/deep_agent/test_agent_code.py` and
`test_config_flags.py`:
- Tool list is `{bash, submit_sql, ask_user}`; FS tools (`ls`/`read_file`/…) gone.
- `bash` whitelist: allowed commands run; a non-whitelisted command (e.g. `rm`,
  `echo`) is refused without spawning.
- psql guardrail: `psql -c "\\! id"` (and `\copy`/`\o`) refused.
- `materialize_catalog_dir`: produces `database_overview.md`, `tables/*.md`,
  `tables/_foreign_key_constraints.md`, and `knowledge_base/<node>.md`; a node
  masked out of `masked_agent_kb` does **not** appear (no leak); missing
  `tables/` raises `FileNotFoundError`.
- Cleanup: the temp dir is removed after `run_agent_deep_agent` (including on
  exception).
- `deep_enable_fs_write`, `deep_enable_todos`, `deep_enable_summarization` removed
  (config no longer accepts them); `deep_enable_subagents` still works.

Run `uv run pytest tests/` and `uv run pyrefly check`.

## Docs to refresh

- `src/conversation2sql/eval_framework/agents/deep_agent/CLAUDE.md` (the whole FS
  section, tool count, ablation table — keep only `deep_enable_subagents`).
- `src/conversation2sql/eval_framework/agents/CLAUDE.md` (the variants table row
  for `deep_agent`: FS tools → bash).
- `src/conversation2sql/eval_framework/agents/deep_agent/README.md`.

## Risks / trade-offs

- **Loss of smart result truncation:** raw `psql` output gets only an 8000-char
  cap, not `execute_sql`'s row-based + giant-cell truncation. Accepted for
  simplicity; the char cap still protects the context window.
- **Whitelist, not a jail:** `default_transaction_read_only` blocks DB *writes*
  but not server-side file reads. `psql -c "SELECT pg_read_file('…')"` is a
  superuser-gated function, and the eval DB user is `root` (per CLAUDE.md), which
  may be a superuser — so this is a plausible way to read the unmasked on-disk
  catalog. **Deferred by decision:** flagged in code with a
  `# TODO(unmasked-kb-leak):` comment in `return_tool_bash`, to be solved later
  (e.g. path-containment check, revoking `pg_read_file` from the eval role, or
  blocking `pg_read_file`/`pg_ls_dir`/`COPY … FROM PROGRAM` in the psql guardrail).
- **Flat `bash` cost** changes the per-action economics vs. the old per-tool
  costs; the patience budget formula is unchanged. `bash=1.0` is a tunable knob.
