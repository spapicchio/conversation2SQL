# Deep Agent baseline — design spec

**Date:** 2026-06-24
**Status:** Approved (pending spec review)

## Goal

Add a new evaluation agent to `eval_framework/agents/` built on LangChain's
[`deepagents`](https://docs.langchain.com/oss/python/deepagents/overview) package.
The agent has a deliberately small surface: it explores the Postgres database
**through a virtual filesystem** (deepagents' built-in `ls`/`read_file`/`grep`/`glob`)
instead of the `get_schema`-style tools, and it talks to the simulated user via the
existing human-in-the-loop `ask_user` tool. It runs under the **same bird-coin
patience budget** as `bird_baseline`, so its results are directly comparable.

deepagents ships many features on by default (planning/todos, subagents,
summarization, filesystem writes). We strip the agent down to the minimum and
expose each default as an individual ablation flag, so they can be switched on
and measured one at a time later.

## Decisions (from brainstorming)

| Decision | Choice |
|----------|--------|
| Tool set | Read-only filesystem + `ask_user` + `submit_sql` + `execute_sql` |
| Budget | Reuse the existing bird-coin patience-budget middleware/state |
| Filesystem layout | One file per concern (whole-file dumps) |
| deepagents defaults | Minimal by default; each default an `enable_*` ablation flag |

## Architecture

### Package & wiring

- New package `src/conversation2sql/eval_framework/agents/deep_agent/`:
  - `__init__.py`
  - `agent_code.py` — entry function `run_agent_deep_agent`
  - `prompts.py` — system/user prompt (Jinja2, same pattern as `bird_baseline/prompts.py`)
  - `filesystem_seed.py` — builds the seeded `dict[str, str]` of DB-info files from a `TaskData`
- New `ConfigPipeline.baseline` literal value `"deep_agent"`.
- `main_pipe_workflow._resolve_baseline_settings`: add
  `"deep_agent": (True, run_agent_deep_agent, True)` — it **parallels `bird_full`**
  (ambiguous user query + `ask_user` enabled).
- `main_pipe_workflow` runner dispatch: add a branch that calls
  `run_agent_deep_agent(...)` with the same model arguments `run_agent_bird_baseline`
  receives, with `enable_ask_user=True`.
- Register `run_agent_deep_agent` in `agents/__init__.py`.
- Add `deepagents` to `pyproject.toml` dependencies (`uv add deepagents`); re-lock.

### Tools (4 total)

1. **Filesystem (deepagents built-in):** read-only subset — `ls`, `read_file`,
   `grep`, `glob`. Seeded with the DB-info files (below). These replace
   `get_schema` / `get_all_column_meanings` / `get_column_meaning` /
   `get_*_knowledge_*`.
2. **`execute_sql`** — reused verbatim from `bird_baseline/tools` (read-only
   `SELECT`/`WITH`/`EXPLAIN`; reads `runtime.context`).
3. **`ask_user`** — reused via `return_tool_ask_user(model_user_parsing, model_user_generator)`.
4. **`submit_sql`** — reused verbatim; the terminal grader.

Each filesystem read tool plus the three reused tools must have a bird-coin cost
so the patience middleware can account for them. Add a new `FS_TOOL_COSTS`
mapping (e.g. `read_file`/`grep`/`glob`/`ls` ≈ 0.5 each — final values set during
implementation) and merge it into the agent's `TOOL_COSTS` view alongside the
existing `DB_TOOL_COSTS`/`USER_TOOL_COSTS`. When `deep_enable_fs_write` is on,
`write_file`/`edit_file` also need cost entries.

### Filesystem seed (one file per concern)

`filesystem_seed.py` builds a `dict[str, str]` (virtual path → contents) from a
`TaskData`, seeded into the deep agent's initial state `files`:

| Path | Source | Renderer |
|------|--------|----------|
| `/db/schema.sql` | `TaskData.ddl_database_schema` | raw DDL string |
| `/db/column_meanings.md` | `TaskData.column_meanings` | reuse `get_all_column_meanings_impl` formatting |
| `/db/knowledge_base.md` | `TaskData.masked_agent_kb` | reuse `get_all_knowledge_definitions_impl` / `linearize_kb` |

Edge cases: an empty KB or empty column-meanings map yields a file containing a
`"(none)"` sentinel rather than being omitted, so `ls /db` always shows the same
three paths (predictable for the agent and for tests).

### State & budget integration

- Merged state schema `DeepAgentCustomState`: deepagents' state (`files`, plus
  `todos` when planning is enabled) **+** the three patience fields from
  `CustomAgentState` (`initial_user_patience`, `updated_user_patience`,
  `tool_called_patience`).
- Append the existing patience middleware from
  `bird_baseline/agent_callback.py` to the deep agent's middleware list, in the
  same relative order used by `bird_baseline`:
  `check_budget_limit` → `sanitize_thinking_history` →
  `wrap_model_append_tool_message` → `tool_wrapper_patience_and_submit`.
  The budget invariant is unchanged; `submit_sql` stays terminal.
- Initial agent state seeds `files` (from `filesystem_seed`) + the patience
  fields (`initial_user_patience = updated_user_patience = task_budget`,
  `tool_called_patience = []`).
- Output is normalised through the existing
  `utils_process_agent_response(response, tool_costs=TOOL_COSTS)` and
  `_extract_predicted_sql`, so the JSONL record shape matches the other agents.

### Ablation flags (minimal by default)

Four new flags on `ConfigReader` (and threaded onto `TaskData`, exactly like
`enable_psql_console`), all default `False`:

| Flag | Toggles |
|------|---------|
| `deep_enable_todos` | deepagents planning / `write_todos` middleware + tool |
| `deep_enable_subagents` | deepagents subagents (`task` tool) middleware |
| `deep_enable_summarization` | deepagents summarization middleware |
| `deep_enable_fs_write` | adds `write_file` / `edit_file` to the filesystem toolset |

With all four `False`, the agent is exactly: read-only FS + `execute_sql` +
`ask_user` + `submit_sql`, under the patience budget. `agent_code.py` builds the
deepagents middleware/tool list conditionally from these flags.

Each flag also:
- Renders a corresponding guidance block in `deep_agent/prompts.py`.
- Gets a run-dir slug suffix in `presets.py` (matching the `__psql` /
  `__strictpsql` pattern) so runs are distinguishable on disk.

### Prompts

`deep_agent/prompts.py` (new, same Jinja2 + Pydantic-params pattern as
`bird_baseline/prompts.py`):
- System prompt instructs the agent to explore `/db/*` via `ls`/`read_file`/`grep`
  (explicitly: there are no `get_schema`-style tools).
- Reuses the ask_user + budget-awareness guidance from `bird_baseline`.
- Conditional blocks per ablation flag (todos/subagents/summarization/fs_write)
  so the prompt only mentions a capability when it is enabled.

## Testing (TDD)

New tests under `tests/eval_framework/agents/deep_agent/`:

- `test_filesystem_seed.py` — given a `TaskData` fixture, the three `/db/*` files
  are built with the expected whole-file content; empty-KB / empty-column-meanings
  yield the `"(none)"` sentinel.
- `test_deep_agent_tools.py` — `FS_TOOL_COSTS` is merged into the agent's
  `TOOL_COSTS` view; the read-only FS tool set is wired; cost entries exist for
  every tool the agent can call.
- `test_agent_code.py` — ablation flags add/remove the right middleware and tools
  (todos/subagents/summarization off by default; `deep_enable_fs_write` adds
  write tools); the patience middleware is present in the stack. Model is mocked
  as in the existing wrapper tests.

Verification: `uv run pytest tests/` and `uv run pyrefly check` must pass.

## Open implementation risk

`deepagents` is **not yet installed**. The exact API for (a) appending custom
middleware to a `create_deep_agent` graph, (b) supplying a merged `state_schema`
(or otherwise combining deepagents' state with the patience fields), and (c)
selecting/excluding which built-in middleware and filesystem tools are included,
must be verified against the actually-installed version. The implementation
plan's **first step installs `deepagents` and pins the real signatures** before
any agent code is written; the conditional-middleware design above may need
small adjustments to match the real API surface (e.g. `builtin_tools` /
`excluded_middleware` parameter names).

## Out of scope

- Tuning/optimising subagent prompts (subagents are off by default; enabling and
  configuring them is future work).
- Non-Postgres backends.
- Changes to the existing `bird_baseline` / `no_tool` agents.
