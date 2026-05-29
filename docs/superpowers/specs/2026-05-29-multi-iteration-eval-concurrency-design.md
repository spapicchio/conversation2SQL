# Multi-iteration evaluation with maximized concurrency

**Date:** 2026-05-29
**Component:** `src/conversation2sql/eval_framework/main_pipe_workflow.py`, `src/conversation2sql/config_input.py`

## Problem

The evaluation pipeline runs the configured variant (`ConfigPipeline.baseline`) over the
dataset exactly once, producing a single `results.jsonl`. Because the agent (and, for
interactive baselines, the user simulator) is non-deterministic at `temperature > 0`, a
single pass gives one point estimate of execution accuracy with no measure of variance.

We want to run the dataset **N times** for one variant so that execution accuracy can be
analysed for statistical relevance — while keeping the run as fast as possible by
**maximizing concurrency** across all iterations.

## Scope

- **In scope:** repeat the dataset N times within one invocation; tag each result record
  with its iteration index; write one results file per iteration; share the concurrency
  pool across all iterations.
- **Out of scope:** computing aggregate statistics (mean/std/CI, pass@k, pass^k,
  per-task pass rate). Aggregation is done downstream in the **explorer app**, which reads
  the per-iteration result files. No metric computation is added to the pipeline.
- **Out of scope:** sweeping multiple baselines/models in one invocation. One variant per
  invocation, as today. Multiple variants remain the job of the `justfile`/shell matrix.
- **Out of scope:** seed plumbing. Variation across iterations comes purely from
  user-set `temperature > 0` and inherent API non-determinism.

## Design

### 1. Config

Add to `ConfigPipeline` in `config_input.py`:

```python
num_iterations: int = Field(default=1, ge=1)  # repeat the dataset N times for statistical relevance
```

`Field(ge=1)` requires importing `Field` from `pydantic`. Default `1` preserves current
behavior.

### 2. Effective-iteration resolution (temperature=0 collapse)

When the predictor temperature is `0`, repeated iterations are deterministic and add no
information, so the run collapses to a single iteration regardless of `num_iterations`.

```python
def _resolve_iterations(num_iterations: int, predictor_temperature: float) -> int:
    if predictor_temperature <= 0 and num_iterations > 1:
        logger.warning(
            "predictor temperature=%s; iterations are deterministic — "
            "collapsing num_iterations=%s to 1",
            predictor_temperature, num_iterations,
        )
        return 1
    return num_iterations
```

**Decision (confirmed):** gating is on the **predictor** temperature only — the agent under
evaluation. A non-zero user-simulator temperature does **not** justify multiple iterations.

`workflow_evaluation_pipeline` computes `effective_iterations = _resolve_iterations(...)`
after resolving baseline settings and passes it to `_run_tasks_concurrently`.

### 3. Concurrency — flatten all (iteration, task) pairs (Approach A)

`_run_tasks_concurrently` gains a `num_iterations` parameter and builds the full
cross-product of coroutines:

```python
coros = [
    _process_one(task, iteration)
    for iteration in range(num_iterations)
    for task in dataset
]
await tqdm.asyncio.tqdm.gather(
    *coros, desc=f"Inference with {baseline} x{num_iterations}"
)
```

The existing `asyncio.Semaphore(concurrency)` now bounds **all N×T units together**, so the
pool stays saturated across iteration boundaries — no idle gap while one iteration drains
before the next starts. Models are built once (`_init_models`) and the dataset is read once.

`_process_one` gains an `iteration: int` argument. It:
- runs the agent exactly as today (via `asyncio.to_thread(runner, ...)`),
- adds `"iteration": iteration` to `task_output`,
- writes the record to that iteration's file (section 4),
- **returns `None`** (see "Memory" below) — the full record is already on disk.

#### Memory: lightweight return

`gather` retains whatever each coroutine returns, so returning the full `task_output` would
hold all N×T record dicts (each carrying the complete message history) in memory at once.
Nothing consumes them: records are streamed to disk by `_save_record`, and
`workflow_evaluation_pipeline`'s return value is ignored by `main_launch_eval`.

Therefore `_process_one` returns `None` and `_run_tasks_concurrently` no longer builds a
result list — it just awaits the `gather` for its side effects (disk writes). Peak memory is
then bounded by `concurrency` in-flight agent runs, independent of N and T.
`workflow_evaluation_pipeline` accordingly returns `None` instead of `list[dict]`.

**Verify during implementation:** confirm no test or caller relies on the return value of
`workflow_evaluation_pipeline` / `_run_tasks_concurrently`. If one does, return a tiny
summary (e.g. count of records written) rather than the full dicts.

#### Approaches considered

- **A (chosen):** flatten all `(iteration, task)` pairs into one shared semaphore-bounded
  `gather`. Highest utilization; small diff; one combined progress bar.
- **B (rejected):** nested loop calling the existing per-task concurrency once per
  iteration. Concurrency drains between iterations — directly conflicts with the goal.
- **C (rejected):** external shell/justfile loop running the whole pipeline N times.
  Re-inits models and re-reads the dataset each time; no shared pool; the user wants this
  inside the workflow.

### 4. Output layout — one file per iteration

```python
def _iter_file(output_folder: Path, iteration: int) -> Path:
    return output_folder / f"results_iter{iteration}.jsonl"
```

**Decision (confirmed):** uniform naming for **all** runs, including `num_iterations=1`
(→ `results_iter0.jsonl`). This replaces today's single `results.jsonl`. Uniform naming
keeps the explorer app's file glob simple. Downstream readers of `results.jsonl` must be
updated to read `results_iter*.jsonl` (explorer app change is separate scope).

Writes stay guarded by a single `threading.Lock`; appends are tiny JSON lines and already
execute on the event-loop thread. Each record also carries its `iteration` field, so the
data is self-describing even independent of filename.

### 5. Error handling & config snapshot (unchanged)

- On any exception during the gather, the pipeline still writes `results_error.jsonl` and
  re-raises. Lines already written across all per-iteration files are preserved.
- `_save_configs_as_yaml` still writes a single `config.yaml` (the config is identical for
  every iteration), now reflecting `num_iterations`.

## Testing

- Extend the workflow tests:
  - `num_iterations=3` with predictor `temperature>0` → 3 files
    (`results_iter0.jsonl`..`results_iter2.jsonl`), each with `len(dataset)` records, each
    record carrying the correct `iteration` value.
  - predictor `temperature=0` with `num_iterations=3` → collapses to 1 file
    (`results_iter0.jsonl`).
  - `num_iterations=1` → single `results_iter0.jsonl`.
- Check `tests/` for existing workflow coverage and adapt any test that asserts on the
  `results.jsonl` filename.
- Run `uv run pytest tests/` (per CLAUDE.md) after the change.

## Risks / notes

- **Filename rename** is a breaking change for anything reading `results.jsonl`; the
  explorer app and any analysis scripts must be pointed at `results_iter*.jsonl`.
- **Memory:** addressed by the lightweight-return design (section 3, "Memory") — coroutines
  return `None`, so peak memory is bounded by `concurrency` in-flight runs, not N×T record
  dicts. The N×T *coroutine objects* themselves are still created up front; they are
  lightweight, but for very large sweeps a worker-pool / `asyncio.Queue` pattern (drained by
  `concurrency` workers) would bound that too. Deferred — adopt only if large sweeps prove it
  necessary.
