# deep_agent

A BIRD-Interact evaluation baseline built on LangChain's [`deepagents`](https://docs.langchain.com/oss/python/deepagents/overview)
package. It parallels `bird_full` (ambiguous query + `ask_user`, under the
bird-coin patience budget) but **swaps the `get_schema`-style DB tools for a
virtual filesystem**: the agent explores the database by reading `/db/*` files
via deepagents' `ls`/`read_file`/`grep`/`glob`.

## Why `create_agent` + middleware, not `create_deep_agent`

We build the agent with `create_agent` (the same primitive as `bird_baseline`)
and compose deepagents' `FilesystemMiddleware` ourselves. `create_deep_agent`
only removes its built-in tools via a **globally-registered** `HarnessProfile`
keyed by model provider — that global state is unworkable for per-run ablations
(two concurrent runs with different flags would clobber each other). Composing
middleware explicitly keeps every toggle local to one agent build.

## Key files

- `agent_code.py` — `run_agent_deep_agent` (entry point) plus the unit-testable
  helpers `_build_fs_middleware`, `_build_deep_tools`, `_build_deep_middleware`.
- `filesystem_seed.py` — `build_db_filesystem(task)` renders the three `/db/*`
  seed files; `FS_TOOL_COSTS` / `deep_tool_costs(enable_fs_write=...)` is the
  cost table fed to `utils_process_agent_response`.
- `agent_code_state.py` — `DeepAgentCustomState` merges deepagents'
  `FilesystemState` (`files`) with `bird_baseline`'s `CustomAgentState` (the
  three patience fields).
- `prompts.py` — `build_deep_agent_messages`; the system prompt advertises the
  `/db` filesystem (and never the schema tools) with conditional blocks per flag.

## Filesystem seed (`/db/*`, one file per concern)

| Path | Source | Renderer |
|------|--------|----------|
| `/db/schema.sql` | `TaskData.ddl_database_schema` | raw DDL |
| `/db/column_meanings.md` | `TaskData.column_meanings` | `get_all_column_meanings_impl` |
| `/db/knowledge_base.md` | `TaskData.masked_agent_kb` | `linearize_kb` if `is_kb_linearized`, else `get_all_knowledge_definitions_impl` |

An empty KB / empty column-meanings map yields a `"(none)"` sentinel file (not an
omitted path), so `ls /db` always shows the same three paths.

## Tools (4)

Read-only FS subset (`ls`/`read_file`/`grep`/`glob`) + the **reused** `execute_sql`,
`ask_user` (via `return_tool_ask_user`), and `submit_sql` from `bird_baseline`.
The FS `execute` tool is always dropped (it errors on the non-sandbox
`StateBackend`); `write_file`/`edit_file` appear only under `deep_enable_fs_write`.

## Patience budget (reused unchanged)

The same middleware stack as `bird_baseline`, appended last in the same relative
order: `check_budget_limit` → `sanitize_thinking_history` →
`wrap_model_append_tool_message` → `tool_wrapper_patience_and_submit`. Do not
reimplement it — it is imported from `bird_baseline/agent_callback.py`.

## Ablation flags (all default `False` = minimal agent)

Threaded `ConfigReader` → `TaskData` exactly like `enable_psql_console`. Each is
additive deepagents middleware, gated in `_build_deep_middleware`, with a matching
prompt block and a run-dir slug suffix (in `bash_scripts/utils/utils_evaluate.sh`,
detected from `--extra`).

| Flag | Adds | Slug |
|------|------|------|
| `deep_enable_todos` | `TodoListMiddleware` (`write_todos`) | `__todos` |
| `deep_enable_subagents` | `SubAgentMiddleware` (`task`) | `__subagents` |
| `deep_enable_summarization` | `SummarizationMiddleware` (needs a model — reuses `model_agent`) | `__summar` |
| `deep_enable_fs_write` | `write_file`/`edit_file` | `__fswrite` |

`SubAgentMiddleware` requires ≥1 subagent (each needing a `model` and `tools`);
`_general_subagent` is a minimal general-purpose one — tuning subagents is out of
scope (off by default). Drive any flag via
`just eval --baseline deep_agent --extra "--deep_enable_todos true"`.
