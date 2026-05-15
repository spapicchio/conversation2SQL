# no_tool_baseline

Single-shot text-to-SQL baseline — no tools, no user interaction.

## Entry function

`run_baseline_no_tool(single_task, model_agent)` in `baseline_model.py`:
1. Renders the OmniSQL-style prompt via `build_omnisql_prompt` (Jinja2, schema + question).
2. Calls `model_agent.invoke(messages)` once.
3. Extracts SQL from the model's fenced code block via `extract_sql_from_response`.
4. If SQL was extracted, calls `submit_sql_impl(sql, sol_sqls, db_dsn, conditions)` against Postgres for `passed`.
5. Returns a result dict whose top-level keys mirror `run_agent_bird_baseline` (a synthetic
   `submit_sql_offline` ToolMessage stands in for the missing real submit).

## When to use

Lower-bound baseline for the ablation. No patience-budget mechanics; no agent loop.

## Gotchas

- `extract_sql_from_response` returns the **last** fenced block (handles draft-then-refine).
- When no fenced block is present, `execution_accuracy=False` is recorded with
  `error="no_sql_block_found"` — the task is not retried.
