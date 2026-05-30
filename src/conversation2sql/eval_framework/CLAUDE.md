# eval_framework

Orchestrates the full evaluation pipeline and defines shared state types.

## Key files

- `main_pipe_workflow.py` — `workflow_evaluation_pipeline` is the top-level entry point called from `cli.py`/`main.py`. It initialises models, loads tasks, runs the agent for every (iteration, task) pair, and writes JSONL output. It runs `num_iterations` passes of the dataset (collapsed to 1 when predictor `temperature <= 0`) and writes one file per iteration: `<output_dir>/results_iter{i}.jsonl`. Returns `None` — records are streamed to disk, not accumulated in memory.
- `state.py` — all shared Pydantic types: `TaskData` (one evaluation task), `ColumnMeaningEntry`, `ExternalKnowledgeEntry`, `FollowUpPayload`, `EvaluationMetrics`, `EvaluationOutput`.

## TaskData

`TaskData` is the **immutable context** passed into each agent run. Key fields:
- `task_question` — either `amb_user_query` or `not_ambiguos_query` depending on `make_data_ambiguous`
- `task_budget` — bird-coin budget: `6 + 2*m_amb + 2*user_patience`
- `masked_agent_kb` — KB with ambiguous entries deleted; `dict[str, ExternalKnowledgeEntry]` encoding the DAG via `children_knowledge` IDs
- `is_kb_linearized` — when `True` the KB tools call `agents/utils_kb_linearize.py` to produce a flat string at tool-call time; the reader no longer pre-computes any string representation
- `sql_query_conditions` — dict with `"order": bool` flag that affects `submit_sql_impl` comparison

## Output records

`_save_record` appends one JSON line per (task, iteration). Each record carries an `iteration` field (0..num_iterations-1) and lands in `results_iter{iteration}.jsonl`. Aggregation across iterations (mean/std, pass@k, etc.) is done downstream in the explorer app, not in the pipeline. `execution_accuracy` comes from the last `submit_sql` ToolMessage's `passed` field.

## Gotchas

- `config_pipeline.debug = True` breaks after the first task — useful for single-task smoke tests.
- Errors mid-run append to `results_error.jsonl` and re-raise; already-written results are preserved.
- The pipeline saves a `config.yaml` snapshot to the output folder at the start of each run.
