# bird_baseline

Interactive LangGraph agent for the BIRD-Interact benchmark.

## Key files

- `agent_code.py` — `run_agent_bird_baseline`: builds and invokes the agent graph. Middleware stack (outermost first): `ModelRetryMiddleware` → `ToolRetryMiddleware` → `ModelCallLimitMiddleware` → `ToolCallLimitMiddleware` → `check_budget_limit` → `wrap_model_append_tool_message` → `tool_wrapper_patience_and_submit`.
  - `enable_ask_user: bool` (kwarg) — when `False` the `ask_user` tool is omitted and the system prompt renders without ambiguity-related guidance. The middleware stack is unchanged.
- `agent_code_state.py` — `CustomAgentState` extends `AgentState` with three fields: `initial_user_patience` (fixed budget), `updated_user_patience` (decremented each turn), `tool_called_patience` (list of costs for the current turn, reset to `[]` after each model call via `Overwrite`).
- `agent_callback.py` — the three middleware functions that implement the patience-budget invariant (see below).
- `prompts.py` — `build_bird_interact_agent_messages`: Jinja2-rendered system + user prompt.

## Patience-budget invariant

Three middleware hooks work together — **read `agent_callback.py` before touching any of them**:

1. `check_budget_limit` (`@before_model`): jumps to `end` if `updated_user_patience < -1`.
2. `tool_wrapper_patience_and_submit` (`@wrap_tool_call`): blocks tools whose cost > remaining budget (always allows `submit_sql`); on `submit_sql` success or budget exhaustion sets `updated_user_patience = -2` (terminal).
3. `wrap_model_append_tool_message` (`@wrap_model_call`): deducts `sum(tool_called_patience)` from `updated_user_patience`, clears `tool_called_patience` via `Overwrite([])`, and appends `[SYSTEM NOTE: Remaining budget: x/y]` to the last ToolMessage.

## Gotchas

- `submit_sql` is always allowed even when `updated_user_patience < cost` — this is intentional so the agent can always finalize.
- On a failed `submit_sql` (wrong SQL, budget > 0) the agent retries; cost is still recorded.
- `ModelCallLimitMiddleware` and `ToolCallLimitMiddleware` are both set to `task_budget + 5` / `task_budget * 2` as a hard safety net above the patience system.
