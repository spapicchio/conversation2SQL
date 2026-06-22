# bird_baseline

Interactive LangGraph agent for the BIRD-Interact benchmark.

## Key files

- `agent_code.py` — `run_agent_bird_baseline`: builds and invokes the agent graph. Middleware stack (outermost first): `ModelRetryMiddleware` → `ToolRetryMiddleware` → `ModelCallLimitMiddleware` → `ToolCallLimitMiddleware` → `check_budget_limit` → `wrap_model_append_tool_message` → `tool_wrapper_patience_and_submit`.
  - `enable_ask_user: bool` (kwarg) — when `False` the `ask_user` tool is omitted and the system prompt renders without ambiguity-related guidance. The middleware stack is unchanged.
  - `enable_table_schema_tools` (read from `single_task`, i.e. the `TaskData`/`ConfigReader` flag, default `False`) — when `True` the granular `get_table_names` + `get_table_schema` tools are added to the tool list and rendered in the prompt; off keeps the baseline's single `get_schema`. Ablation knob via `--extra "--enable_table_schema_tools true"`.
  - `enable_psql_console` (read from `single_task`, default `False`) — when `True` a single read-only `psql_console` tool **replaces** `execute_sql`/`get_schema`/`get_table_*` (tool selection lives in `_select_db_tools`). Mutually exclusive with `enable_table_schema_tools` (raises in `ConfigReader` and in `_select_db_tools`). Ablation knob via `--extra "--enable_psql_console true"`; the run slug gets a `__psql` suffix and the reader logs a one-time warning.
  - `enable_psql_strict_inspection` (read from `single_task`, default `False`) — sub-ablation of `enable_psql_console`: when set, `psql_console` runs in **strict inspection** mode (allowlist) — only SQL (`SELECT`/`WITH`/`EXPLAIN`), `\h`, and the informational `\d`-family (`\d*`, `\l`, `\sf`, `\sv`, `\z`) are allowed; every other meta-command is refused with `PSQL_STRICT_REFUSAL`, and `\?` returns only psql's *Informational* section (via `_extract_informational_help`). Off = legacy denylist (full rollback). No-op without `enable_psql_console`. Drive via `--extra "--enable_psql_console true --enable_psql_strict_inspection true"`; slug gets a `__strictpsql` suffix.
- `agent_code_state.py` — `CustomAgentState` extends `AgentState` with three fields: `initial_user_patience` (fixed budget), `updated_user_patience` (decremented each turn), `tool_called_patience` (list of costs for the current turn, reset to `[]` after each model call via `Overwrite`).
- `agent_callback.py` — the three middleware functions that implement the patience-budget invariant (see below).
- `prompts.py` — `build_bird_interact_agent_messages`: Jinja2-rendered system + user prompt.

## Patience-budget invariant

Three middleware hooks work together — **read `agent_callback.py` before touching any of them**:

1. `check_budget_limit` (`@before_model`): jumps to `end` once a terminal sentinel is set (`updated_user_patience <= PATIENCE_SUBMIT_EXHAUSTED`). It words the closing turn by reason: a passing submit (`PATIENCE_SUBMIT_PASSED`) ends **silently** (the `submit_sql` ToolMessage is the record); a forced/out-of-budget submit (`PATIENCE_SUBMIT_EXHAUSTED`) appends the `"User patience exhausted"` note.
2. `tool_wrapper_patience_and_submit` (`@wrap_tool_call`): blocks tools whose cost > remaining budget (always allows `submit_sql`, setting `PATIENCE_BLOCKED`). On a terminal `submit_sql` it sets a reason-specific sentinel: `PATIENCE_SUBMIT_PASSED` (`-3`) when `passed=True`, `PATIENCE_SUBMIT_EXHAUSTED` (`-2`) when not passed but out of budget. A non-passing submit with budget left falls through to retry.
3. `wrap_model_append_tool_message` (`@wrap_model_call`): deducts `sum(tool_called_patience)` from `updated_user_patience` (clamped at `PATIENCE_BLOCKED`), clears `tool_called_patience` via `Overwrite([])`, and appends `[SYSTEM NOTE: Remaining budget: x/y]` to the last ToolMessage.

The three terminal/floor sentinels live as named constants at the top of `agent_callback.py`. The state reducer keeps the smallest write, so a passing submit (`-3`) wins over a forced one (`-2`) if both occur in one super-step.

## Gotchas

- `submit_sql` is always allowed even when `updated_user_patience < cost` — this is intentional so the agent can always finalize.
- On a failed `submit_sql` (wrong SQL, budget > 0) the agent retries; cost is still recorded.
- `ModelCallLimitMiddleware` and `ToolCallLimitMiddleware` are both set to `task_budget + 5` / `task_budget * 2` as a hard safety net above the patience system.
