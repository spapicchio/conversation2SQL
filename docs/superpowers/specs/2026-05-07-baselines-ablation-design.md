# BIRD-Interact Baselines Ablation — Design

**Date:** 2026-05-07
**Status:** Approved (architecture, components, data flow, error handling, testing) — pending user review of this written spec

## Problem

The repo currently has one full BIRD-Interact agent (`bird_baseline`) and a stub `no_tool_baseline` that is wired up but broken (calls `utils_process_agent_response` on a raw `AIMessage`, and renders the empty system prompt as the user prompt). For the research, we need a clean ablation across two axes — **DB tools** and **user interaction (ask_user + ambiguity)** — to attribute the contribution of each component of the full agent.

## Goals

1. Define four ablation baselines selected by a single config knob, with consistent result-record shape so one analysis script reads all four.
2. Reuse the existing middleware/budget machinery (do **not** fork the agent graph).
3. Fix the `no_tool_baseline` evaluation path so its accuracy numbers are directly comparable to the agent baselines.
4. Keep config priority (defaults → env → YAML → CLI) intact; nothing else in the pipeline changes.

## Non-goals

- Refactoring the patience middleware.
- Changing `submit_sql_impl` or the user simulator.
- Introducing new tools.
- Migrating the system prompt from the messages list to `create_agent`'s `system_prompt=` parameter (cleaner per current LangChain docs but out of scope here).

## The four baselines

Ablation matrix — each row is one value of `ConfigPipeline.baseline`:

| `baseline` | Query | DB tools | `ask_user` | `submit_sql` | `make_data_ambiguous` | Forced budget formula |
|---|---|---|---|---|---|---|
| `no_tool` | clean | — | — | text-only (parsed offline) | `False` | n/a (no agent loop) |
| `tools_only` | clean | ✅ | — | ✅ | `False` | `6 + 2*patience` |
| `tools_user` | clean | ✅ | ✅ | ✅ | `False` | `6 + 2*patience` |
| `bird_full` | ambiguous | ✅ | ✅ | ✅ | `True` | `6 + 2*m_amb + 2*patience` |

**Research signal per baseline:**

- `no_tool` — lower bound: classical text-to-SQL with full schema in context.
- `tools_only` — value of agentic DB exploration alone (no ambiguity, no user).
- `tools_user` — value of having `ask_user` available even when the query is unambiguous (does the agent over-ask?).
- `bird_full` — full BIRD-Interact agent (paper baseline).

## Architecture

```
ConfigPipeline.baseline ─┐
                          ├──► _resolve_baseline_settings() in main_pipe_workflow.py
                          │       returns (make_data_ambiguous, runner, needs_user_sim)
                          ├──► overrides ConfigReader.make_data_ambiguous (warn if user set inconsistently)
                          ├──► _init_models conditionally constructs user-sim models
                          └──► dispatch:
                                 baseline == 'no_tool'  → run_baseline_no_tool
                                 otherwise              → run_agent_bird_baseline(enable_ask_user=...)
```

**Key choices** (decided during brainstorming):

- **One parameterizable agent** — `bird_baseline/agent_code.py` takes `enable_ask_user: bool` and conditionally appends `ask_user` to the tool list. Middleware stack unchanged.
- **One Jinja system prompt** with `{% if enable_ask_user %}` blocks — single source of truth for tool-cost listings and strategy hints.
- **Baseline-aware budget** — `_calculate_initial_budget` accepts `count_ambiguity: bool`; threaded from `make_data_ambiguous` in the reader.
- **Subfolder-per-baseline output** — `<output_dir>/<baseline>/<YYYY_MM_DD>/<HH_MM_SS>/results.jsonl`.
- **Conditional user-sim init** — skip `model_user_parsing` and `model_user_generator` for `no_tool` and `tools_only`.

## Components & file changes

### Config

**`src/conversation2sql/config_input.py`**

```python
class ConfigPipeline(BaseModel):
    debug: bool = True
    mode: str = 'a-interact'
    output_folder: str = "results"
    concurrency: int = Field(default=5, ...)
    baseline: Literal['no_tool', 'tools_only', 'tools_user', 'bird_full'] = 'bird_full'
```

We do **not** couple `baseline` and `make_data_ambiguous` inside Pydantic — `ConfigReader` must remain runnable from its own `__main__`. The coupling lives in `_resolve_baseline_settings()`.

### Agent dispatch (parameterizable)

**`src/conversation2sql/eval_framework/agents/bird_baseline/agent_code.py`** — `run_agent_bird_baseline` gains a kwarg:

```python
def run_agent_bird_baseline(
    single_task: TaskData,
    model_agent: BaseChatModel,
    model_user_parsing: BaseChatModel | None,
    model_user_generator: BaseChatModel | None,
    *,
    enable_ask_user: bool,
) -> CustomAgentState:
    ...
    tools = [
        execute_sql, get_schema, get_all_column_meanings, get_column_meaning,
        get_all_external_knowledge_names, get_knowledge_definition,
        get_all_knowledge_definitions, submit_sql,
    ]
    if enable_ask_user:
        assert model_user_parsing is not None and model_user_generator is not None, \
            "ask_user requires user-simulator models"
        tools.append(return_tool_ask_user(model_user_parsing, model_user_generator))
    ...
```

The `messages` initial state and middleware stack are unchanged.

**`src/conversation2sql/eval_framework/agents/bird_baseline/prompts.py`** — `_BIRD_AGENT_SYSTEM` becomes Jinja-conditional. The `ask_user` line in the cost table and the "ask clarifying questions" strategy bullet are wrapped:

```jinja
{% if enable_ask_user %}
- ask_user: ask the user a clarification question. Cost: 2
{% endif %}
```

`build_bird_interact_agent_messages` passes `enable_ask_user` through `params`.

### No-tool baseline (fix + wire-up)

**`src/conversation2sql/eval_framework/agents/no_tool_baseline/baseline_model.py`** — rename `run_baseline_model` → `run_baseline_no_tool`. New flow:

1. `model_agent.invoke(messages)` → captures `AIMessage`.
2. `extract_sql_from_response(text) -> str | None` — regex on fenced `` ``` ``/`` ```sql `` blocks; returns the last block; tolerates trailing whitespace; returns `None` on empty/missing.
3. `submit_sql_impl(sql, single_task)` if `sql is not None` else synthesize `{"passed": False, "error": "no_sql_block_found", "raw_text": text[:2000]}`.
4. Assemble result dict mirroring the agent baselines' shape (see "Result record shape" below) so `_save_record` and the `_smaller.jsonl` projection work without a special case.

**`src/conversation2sql/eval_framework/agents/no_tool_baseline/prompts.py`** — fix existing bug: change `utils_build_messages(_BASE_MODEL_SYSTEM, _BASE_MODEL_SYSTEM, params)` to `utils_build_messages(_BASE_MODEL_SYSTEM, _BASE_MODEL_USER, params)`.

### Pipeline orchestration

**`src/conversation2sql/eval_framework/main_pipe_workflow.py`**

New helper:

```python
def _resolve_baseline_settings(baseline: str) -> tuple[bool, Callable, bool]:
    """Returns (make_data_ambiguous, runner, needs_user_sim)."""
    table = {
        'no_tool':    (False, run_baseline_no_tool, False),
        'tools_only': (False, run_agent_bird_baseline, False),
        'tools_user': (False, run_agent_bird_baseline, True),
        'bird_full':  (True,  run_agent_bird_baseline, True),
    }
    return table[baseline]
```

`_init_models(config_predictor, config_user, needs_user_sim)` skips user-sim model construction when `needs_user_sim=False` and returns `(model_agent, (None, None))` in that case.

In the main loop:

```python
if config_pipeline.baseline == 'no_tool':
    response = run_baseline_no_tool(task, model_agent)
else:
    response = run_agent_bird_baseline(
        task, model_agent, model_user_parsing, model_user_generator,
        enable_ask_user=(config_pipeline.baseline in ('tools_user', 'bird_full')),
    )
```

`make_data_ambiguous` from the dispatch table overrides `config_reader.make_data_ambiguous` before `load_bird_interact_as_tasks` is called; one `logger.warning(...)` if the user set it inconsistently. The saved `config.yaml` snapshot reflects the override (so the snapshot matches what actually ran).

Output path: `Path(output_folder) / baseline / date / time / 'results.jsonl'`.

**`src/conversation2sql/eval_framework/agents/__init__.py`** — re-export `run_baseline_no_tool` alongside `run_agent_bird_baseline`.

### Budget

**`src/conversation2sql/eval_framework/dataset_readers/bird_interact_reader.py`**

```python
def _calculate_initial_budget(line: dict, user_patience: int, count_ambiguity: bool) -> float:
    if not count_ambiguity:
        return 6.0 + 2.0 * user_patience
    critical = len(line.get("user_query_ambiguity", {}).get("critical_ambiguity", []))
    knowledge = len(line.get("knowledge_ambiguity", []))
    m_amb = critical + knowledge
    return 6.0 + 2.0 * m_amb + 2.0 * user_patience
```

`load_bird_interact_as_tasks(...)` passes `count_ambiguity=make_data_ambiguous` into the call site.

## Data flow

### Common prefix (all baselines)

```
main.py → PydanticParser → workflow_evaluation_pipeline
  ├─ _resolve_baseline_settings(baseline)
  ├─ override config_reader.make_data_ambiguous (with warning if inconsistent)
  ├─ _init_models(..., needs_user_sim)
  ├─ load_bird_interact_as_tasks(...)
  └─ for each TaskData: runner(task, models...) → _save_record(...)
```

### Per-baseline divergence

**`no_tool`** — single-shot text-to-SQL.
```
TaskData (clean query, full DDL)
  → build_omnisql_prompt(schema, question)
  → model_agent.invoke(messages) → AIMessage(text)
    ├─ extract_sql_from_response(text) → sql | None
    ├─ submit_sql_impl(sql, task) if sql else synthetic-failure dict
    └─ assemble result dict mirroring agent shape
```

**`tools_only`** — agent with DB+submit tools, no `ask_user`.
```
TaskData (clean query, full KB, schema, budget = 6 + 2*patience)
  → run_agent_bird_baseline(task, model_agent, None, None, enable_ask_user=False)
  → create_agent(tools=[...DB+submit, no ask_user...], middleware=[...standard...])
  → terminates on submit_sql success or budget exhaustion
```

**`tools_user`** — same as `tools_only` plus `ask_user`. Tool list includes `return_tool_ask_user(...)`. The user simulator runs against the **clean** query — the experimental signal is whether the agent unnecessarily asks.

**`bird_full`** — unchanged. Ambiguous query, masked KB, full tools, full ambiguity-aware budget.

### Result record shape (consistent across baselines)

Every baseline emits a result dict with the same top-level keys, so a single downstream analysis script reads all four:

| Field | `no_tool` | `tools_*` / `bird_*` |
|---|---|---|
| `execution_accuracy` | from `submit_sql_impl(extracted_sql)` (or `False` on no-SQL-block) | from final `submit_sql` ToolMessage |
| `tool_calls_in_order` | `[]` | as-is |
| `initial_user_patience` / `updated_user_patience` | `None` | from agent state |
| `messages` | `[user, ai, synthetic_tool_msg]` (synthetic msg has `tool_name="submit_sql_offline"`) | as-is |
| `total_cost`, `total_tokens`, `mean_*` | from the single `AIMessage` | as-is |
| `config_pipeline.baseline` | embedded by `_save_record` | embedded by `_save_record` |

The synthetic `submit_sql_offline` ToolMessage in `no_tool` keeps the `_smaller.jsonl` projection valid and lets analysis scripts treat all rows uniformly.

## Error handling

1. **`no_tool` — no fenced SQL block.** Record `execution_accuracy=False` with synthetic ToolMessage `{"passed": False, "error": "no_sql_block_found", "raw_text": text[:2000]}`. Don't raise — a malformed model output is a result, not a pipeline error.
2. **`no_tool` — SQL parse/runtime error.** `submit_sql_impl` already catches and returns `passed=False` with the DB error. No new handling needed.
3. **`tools_only` — agent never calls `submit_sql`.** Existing budget middleware terminates the loop; `_process_agent_response` returns `passed=False`. Correct as-is — that's "agent gave up."
4. **`tools_user` — agent calls `ask_user` despite a clean query.** Not an error; the experimental signal we want to measure. `tool_calls_in_order` surfaces the count.
5. **Inconsistent config** (`baseline=tools_only` + `make_data_ambiguous=True`). Override + one `logger.warning(...)`. Saved `config.yaml` reflects the override.
6. **`enable_ask_user=True` but user-sim models are `None`.** Assertion at agent-construction time fails fast.

**Preserved behavior:**
- `_save_record` still writes `results_error.jsonl` and re-raises mid-run errors.
- `submit_sql_impl` still runs `clean_up_sqls` after each evaluation (the `no_tool` baseline goes through the same path).
- Global `litellm.suppress_debug_info` and warning filter in `main.py` apply to all baselines.

**Explicitly not guarded:** missing Postgres connection (already raises early), output-folder collisions (subfolder-per-baseline isolates them).

## Testing

### Unit (pure, fast)

- `tests/eval_framework/agents/test_no_tool_baseline.py` — `extract_sql_from_response`: ` ```sql `-fenced, ` ``` `-fenced, multiple blocks (returns last), no block, trailing prose, whitespace-only block. ~6 cases.
- `tests/eval_framework/dataset_readers/test_budget.py` — `_calculate_initial_budget(..., count_ambiguity=True)` matches current behavior; `count_ambiguity=False` returns `6 + 2*patience` regardless of `m_amb`.

### Mocked dispatch (fast, no DB / no model)

- `tests/eval_framework/test_main_pipe_workflow.py` — parametric over four `baseline` values with mocks. Asserts: correct runner invoked, `enable_ask_user` correctly set, user-sim models constructed iff needed, `make_data_ambiguous` reaching the reader matches the dispatch table even on inconsistent input, output path includes `<baseline>/`.

### Tool-list assembly (mocked model)

- `tests/eval_framework/agents/test_bird_baseline_agent.py` — extend with `enable_ask_user=False` (tools = 8 minus `ask_user` = 7) and `enable_ask_user=True` (regression for `bird_full` = 8 tools). Build the agent without invoking it.

### Integration (gated)

- `tests/eval_framework/integration/test_baselines_smoke.py`, marked `@pytest.mark.integration`, run manually:
  ```bash
  uv run pytest tests/eval_framework/integration/ -m integration
  ```
  One task per baseline against a tiny model + the live Postgres container. Asserts result-dict shape and `_smaller.jsonl` projection succeeds. Does not assert SQL correctness.

### Type checking & gate

- `uv run pyrefly check` must pass (the `Literal` enum and `BaseChatModel | None` are the new type-surface).
- `uv run pytest tests/` must pass before any PR (per repo `CLAUDE.md`).

## Out of scope

- Migrating the system prompt to `create_agent`'s `system_prompt=` parameter.
- Adding a `concurrency`-aware runner (each baseline still runs sequentially).
- Aggregated cross-baseline reporting / dashboards.
