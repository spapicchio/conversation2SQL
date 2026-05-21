# eval_framework

Orchestrates the full evaluation pipeline and defines shared state types.

## Key files

- `main_pipe_workflow.py` — `workflow_evaluation_pipeline` is the top-level entry point called from `main.py`. It initialises models, loads tasks, runs the agent per task, and writes JSONL output. Results land in `<output_dir>/results.jsonl` (full) and `results_smaller.jsonl` (projected subset of fields).
- `state.py` — all shared Pydantic types: `TaskData` (one evaluation task), `ColumnMeaningEntry`, `ExternalKnowledgeEntry`, `FollowUpPayload`, `EvaluationMetrics`, `EvaluationOutput`.

## TaskData

`TaskData` is the **immutable context** passed into each agent run. Key fields:
- `task_question` — either `amb_user_query` or `not_ambiguos_query` depending on `make_data_ambiguous`
- `task_budget` — bird-coin budget: `6 + 2*m_amb + 2*user_patience`
- `masked_agent_kb` — KB with ambiguous entries deleted; `dict[str, ExternalKnowledgeEntry]` encoding the DAG via `children_knowledge` IDs
- `is_kb_linearized` — when `True` the KB tools call `agents/utils_kb_linearize.py` to produce a flat string at tool-call time; the reader no longer pre-computes any string representation
- `sql_query_conditions` — dict with `"order": bool` flag that affects `submit_sql_impl` comparison

## Output records

`_save_record` appends one JSON line per task. The `_smaller.jsonl` projection keeps only the fields listed in `keep_vars` (see `main_pipe_workflow.py`). `execution_accuracy` comes from `submit_sql`'s `passed` field.

## Gotchas

- `config_pipeline.debug = True` breaks after the first task — useful for single-task smoke tests.
- Errors mid-run append to `results_error.jsonl` and re-raise; already-written results are preserved.
- The pipeline saves a `config.yaml` snapshot to the output folder at the start of each run.
