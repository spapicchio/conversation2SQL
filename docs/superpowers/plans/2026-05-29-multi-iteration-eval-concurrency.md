# Multi-iteration evaluation with maximized concurrency — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run the configured eval variant over the dataset N times in one invocation, maximizing concurrency, writing one results file per iteration, so execution accuracy can be analysed for statistical relevance downstream.

**Architecture:** Add `num_iterations` to `ConfigPipeline`. In `main_pipe_workflow.py`, flatten all `(iteration, task)` pairs into a single `asyncio.Semaphore`-bounded `gather` (Approach A) so the concurrency pool stays saturated across iterations. Each unit writes to `results_iter{i}.jsonl` and tags the record with its `iteration`. Coroutines return `None` and the pipeline returns `None` (results stream to disk; nothing consumes the return), bounding peak memory by `concurrency`. When predictor `temperature <= 0`, collapse to a single iteration.

**Tech Stack:** Python 3.12, Pydantic v2, asyncio, pytest (`asyncio_mode=auto`), run everything via `uv run`.

**Spec:** `docs/superpowers/specs/2026-05-29-multi-iteration-eval-concurrency-design.md`

**Out of scope (do NOT touch):** metric aggregation (lives in the explorer app); sweeping multiple baselines/models; seed plumbing. NOTE for handoff: the explorer loader (`src/conversation2sql/explorer/`) currently reads `results.jsonl`; after this change the pipeline emits `results_iter{i}.jsonl`, so the explorer must be pointed at `results_iter*.jsonl` in a **separate** change.

---

### Task 1: Add `num_iterations` to `ConfigPipeline`

**Files:**
- Modify: `src/conversation2sql/config_input.py:1-13`
- Test: `tests/eval_framework/test_main_pipe_workflow.py` (extend `TestConcurrencyConfig`)

- [ ] **Step 1: Write the failing tests**

In `tests/eval_framework/test_main_pipe_workflow.py`, add `pytest`-based validation imports at the top if missing (`pytest` is already imported) and a new import:

```python
from pydantic import ValidationError
```

Then extend the existing `TestConcurrencyConfig` class (around line 139) with:

```python
    def test_num_iterations_defaults_to_1(self):
        assert ConfigPipeline().num_iterations == 1

    def test_num_iterations_rejects_zero(self):
        with pytest.raises(ValidationError):
            ConfigPipeline(num_iterations=0)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/eval_framework/test_main_pipe_workflow.py::TestConcurrencyConfig -v`
Expected: FAIL — `test_num_iterations_defaults_to_1` raises `AttributeError` (no `num_iterations`); `test_num_iterations_rejects_zero` fails because `num_iterations=0` is accepted (unknown kwarg ignored or no constraint).

- [ ] **Step 3: Add the field**

In `src/conversation2sql/config_input.py`, change the import on line 3 and add the field to `ConfigPipeline`:

```python
from pydantic import BaseModel, Field
```

```python
class ConfigPipeline(BaseModel):
    debug: bool = True
    output_folder: str = "results"
    baseline: Literal['no_tool', 'tools_only', 'tools_user', 'bird_full'] = 'bird_full'
    concurrency: int = 1
    num_iterations: int = Field(default=1, ge=1)  # repeat the dataset N times for statistical relevance
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/eval_framework/test_main_pipe_workflow.py::TestConcurrencyConfig -v`
Expected: PASS (all four tests in the class).

- [ ] **Step 5: Commit**

```bash
git add src/conversation2sql/config_input.py tests/eval_framework/test_main_pipe_workflow.py
git commit -m "feat: add num_iterations to ConfigPipeline"
```

---

### Task 2: Add `_resolve_iterations` helper (temperature=0 collapse)

**Files:**
- Modify: `src/conversation2sql/eval_framework/main_pipe_workflow.py` (add helper near `_resolve_baseline_settings`, ~line 49)
- Test: `tests/eval_framework/test_main_pipe_workflow.py` (new class)

- [ ] **Step 1: Write the failing tests**

In `tests/eval_framework/test_main_pipe_workflow.py`, add to the import block at the top:

```python
from conversation2sql.eval_framework.main_pipe_workflow import _resolve_iterations
```

Add a new test class (place it after `TestResolveBaselineSettings`):

```python
class TestResolveIterations:
    def test_passthrough_when_temperature_positive(self):
        assert _resolve_iterations(5, 0.7) == 5

    def test_collapses_to_one_when_temperature_zero(self):
        assert _resolve_iterations(5, 0.0) == 1

    def test_single_iteration_stays_one_when_temperature_zero(self):
        assert _resolve_iterations(1, 0.0) == 1

    def test_collapses_when_temperature_negative(self):
        assert _resolve_iterations(3, -0.1) == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/eval_framework/test_main_pipe_workflow.py::TestResolveIterations -v`
Expected: FAIL with `ImportError: cannot import name '_resolve_iterations'`.

- [ ] **Step 3: Implement the helper**

In `src/conversation2sql/eval_framework/main_pipe_workflow.py`, add after `_resolve_baseline_settings` (after line 48):

```python
def _resolve_iterations(num_iterations: int, predictor_temperature: float) -> int:
    """Collapse to a single iteration when the predictor is deterministic.

    With temperature <= 0 the agent's output is deterministic, so repeating the
    dataset adds no information — we run it once regardless of num_iterations.
    """
    if predictor_temperature <= 0 and num_iterations > 1:
        logger.warning(
            "predictor temperature=%s; iterations are deterministic — "
            "collapsing num_iterations=%s to 1",
            predictor_temperature,
            num_iterations,
        )
        return 1
    return num_iterations
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/eval_framework/test_main_pipe_workflow.py::TestResolveIterations -v`
Expected: PASS (all four tests).

- [ ] **Step 5: Commit**

```bash
git add src/conversation2sql/eval_framework/main_pipe_workflow.py tests/eval_framework/test_main_pipe_workflow.py
git commit -m "feat: add _resolve_iterations temperature=0 collapse helper"
```

---

### Task 3: Flatten concurrency, per-iteration files, lightweight return

This is the core change. `_run_tasks_concurrently` runs all `(iteration, task)` pairs in one semaphore-bounded `gather`, writes each record to `results_iter{i}.jsonl` with an `iteration` field, and returns `None`. `workflow_evaluation_pipeline` resolves the effective iteration count, drops the `file_result = .../results.jsonl` variable, fixes the error path, and returns `None`.

**Files:**
- Modify: `src/conversation2sql/eval_framework/main_pipe_workflow.py:51-176`
- Test: `tests/eval_framework/test_main_pipe_workflow.py` (update existing, add new iteration tests)

- [ ] **Step 1: Update the existing concurrency/results tests to the new contract**

In `tests/eval_framework/test_main_pipe_workflow.py`, add to the top import block:

```python
import json
```

Replace `test_results_written_to_output_folder` (lines ~123-136) body assertion and comment:

```python
    def test_results_written_to_output_folder(
        self, mock_create, mock_load, mock_no_tool, mock_agent, configs, tmp_path,
    ):
        # Python writes results_iter{i}.jsonl directly into the given output_folder.
        cp, cr, cpred, cu = configs
        cp.baseline = "tools_only"
        mock_load.return_value = [_fake_task()]
        mock_agent.return_value = _stub_response()
        mock_create.return_value = MagicMock()

        workflow_evaluation_pipeline(cp, cr, cpred, cu)

        assert (tmp_path / "results_iter0.jsonl").exists()
```

Replace `test_all_tasks_processed_with_concurrency` (lines ~149-161) so it no longer relies on the return value:

```python
    def test_all_tasks_processed_with_concurrency(
        self, mock_create, mock_load, mock_no_tool, mock_agent, configs
    ):
        cp, cr, cpred, cu = configs
        cp = cp.model_copy(update={"baseline": "no_tool", "concurrency": 4})
        mock_load.return_value = [_fake_task() for _ in range(4)]
        mock_no_tool.return_value = _stub_response()
        mock_create.return_value = MagicMock()

        workflow_evaluation_pipeline(cp, cr, cpred, cu)

        assert mock_no_tool.call_count == 4
        lines = (Path(cp.output_folder) / "results_iter0.jsonl").read_text().splitlines()
        assert len(lines) == 4
```

- [ ] **Step 2: Add new multi-iteration tests**

Add a new class to `tests/eval_framework/test_main_pipe_workflow.py` (after `TestConcurrency`), reusing the same patch decorators:

```python
@patch("conversation2sql.eval_framework.main_pipe_workflow.run_agent_bird_baseline")
@patch("conversation2sql.eval_framework.main_pipe_workflow.run_baseline_no_tool")
@patch("conversation2sql.eval_framework.main_pipe_workflow.load_bird_interact_as_tasks")
@patch("conversation2sql.eval_framework.main_pipe_workflow.utils_create_model")
class TestIterations:
    def test_n_iterations_write_one_file_each_when_temperature_positive(
        self, mock_create, mock_load, mock_no_tool, mock_agent, configs
    ):
        cp, cr, cpred, cu = configs
        cp = cp.model_copy(update={"baseline": "no_tool", "num_iterations": 3})
        cpred = cpred.model_copy(update={"temperature": 0.7})
        mock_load.return_value = [_fake_task() for _ in range(2)]
        mock_no_tool.return_value = _stub_response()
        mock_create.return_value = MagicMock()

        workflow_evaluation_pipeline(cp, cr, cpred, cu)

        assert mock_no_tool.call_count == 6
        out = Path(cp.output_folder)
        for i in range(3):
            lines = (out / f"results_iter{i}.jsonl").read_text().splitlines()
            assert len(lines) == 2
            record = json.loads(lines[0])
            assert record["iteration"] == i

    def test_temperature_zero_collapses_to_single_iteration(
        self, mock_create, mock_load, mock_no_tool, mock_agent, configs
    ):
        cp, cr, cpred, cu = configs
        cp = cp.model_copy(update={"baseline": "no_tool", "num_iterations": 3})
        # cpred.temperature defaults to 0.0
        mock_load.return_value = [_fake_task() for _ in range(2)]
        mock_no_tool.return_value = _stub_response()
        mock_create.return_value = MagicMock()

        workflow_evaluation_pipeline(cp, cr, cpred, cu)

        assert mock_no_tool.call_count == 2
        out = Path(cp.output_folder)
        assert (out / "results_iter0.jsonl").exists()
        assert not (out / "results_iter1.jsonl").exists()
```

- [ ] **Step 3: Run the updated/new tests to verify they fail**

Run: `uv run pytest tests/eval_framework/test_main_pipe_workflow.py::TestDispatch::test_results_written_to_output_folder tests/eval_framework/test_main_pipe_workflow.py::TestConcurrency tests/eval_framework/test_main_pipe_workflow.py::TestIterations -v`
Expected: FAIL — `results_iter0.jsonl` does not exist yet (pipeline still writes `results.jsonl`); `record["iteration"]` KeyError; `TestIterations` writes only one file.

- [ ] **Step 4: Rewrite `_run_tasks_concurrently`**

In `src/conversation2sql/eval_framework/main_pipe_workflow.py`, replace the whole `_run_tasks_concurrently` function (lines 130-176) with:

```python
async def _run_tasks_concurrently(
    dataset: list[TaskData],
    runner: Callable,
    model_agent,
    model_user_parsing,
    model_user_generator,
    baseline: str,
    config_predictor: ConfigPredictor,
    config_user: ConfigUserSimulator,
    config_pipeline: ConfigPipeline,
    config_reader: ConfigReader,
    output_folder: Path,
    concurrency: int,
    num_iterations: int,
) -> None:
    sem = asyncio.Semaphore(concurrency)
    file_lock = threading.Lock()

    async def _process_one(task: TaskData, iteration: int) -> None:
        async with sem:
            if baseline == "no_tool":
                response = await asyncio.to_thread(runner, task, model_agent)
            else:
                response = await asyncio.to_thread(
                    runner,
                    task,
                    model_agent,
                    model_user_parsing,
                    model_user_generator,
                    enable_ask_user=(baseline in ("tools_user", "bird_full")),
                )
        task_output = {
            "config_predictor": config_predictor.model_dump(),
            "config_user": config_user.model_dump(),
            "config_pipeline": config_pipeline.model_dump(),
            "config_reader": config_reader.model_dump(),
            **task.model_dump(),
            **response,
            "iteration": iteration,
        }
        file_result = output_folder / f"results_iter{iteration}.jsonl"
        with file_lock:
            _save_record(response=task_output, output_path_jsonl=file_result)
        logger.info(
            f"Saved response for task_id={task.instance_id} iteration={iteration} to {file_result}"
        )

    # Flatten all (iteration, task) pairs into one shared semaphore-bounded gather
    # so the concurrency pool stays saturated across iteration boundaries.
    coros = [
        _process_one(task, iteration)
        for iteration in range(num_iterations)
        for task in dataset
    ]
    await tqdm.asyncio.tqdm.gather(
        *coros, desc=f"Inference with {baseline} x{num_iterations}"
    )
```

Note: `"iteration"` is placed **after** the `**` spreads so it can never be clobbered by a key from `task` or `response`.

- [ ] **Step 5: Update `workflow_evaluation_pipeline` to wire it up**

In `src/conversation2sql/eval_framework/main_pipe_workflow.py`:

(a) Change the signature return annotation (line 56) from `) -> list[dict]:` to `) -> None:`.

(b) Delete the `file_result = output_folder / "results.jsonl"` line (line 86).

(c) After `_init_models(...)` (i.e. after line 93) add:

```python
    effective_iterations = _resolve_iterations(
        config_pipeline.num_iterations, config_predictor.temperature
    )
```

(d) Replace the `try/except` block (lines 102-127) with:

```python
    try:
        asyncio.run(
            _run_tasks_concurrently(
                dataset=dataset,
                runner=runner,
                model_agent=model_agent,
                model_user_parsing=model_user_parsing,
                model_user_generator=model_user_generator,
                baseline=config_pipeline.baseline,
                config_predictor=config_predictor,
                config_user=config_user,
                config_pipeline=config_pipeline,
                config_reader=config_reader,
                output_folder=output_folder,
                concurrency=config_pipeline.concurrency,
                num_iterations=effective_iterations,
            )
        )
    except Exception as e:
        logger.error(f"Error occurred: {e}")
        response_error = {"error": str(e)}
        output = output_folder / "results_error.jsonl"
        _save_record(response=response_error, output_path_jsonl=output)
        logger.info(f"Saved ERROR to {output}")
        raise e
```

(Note: the function now falls off the end returning `None`; there is no `return result`.)

- [ ] **Step 6: Run the targeted tests to verify they pass**

Run: `uv run pytest tests/eval_framework/test_main_pipe_workflow.py -v`
Expected: PASS — all classes (`TestResolveBaselineSettings`, `TestResolveIterations`, `TestDispatch`, `TestConcurrencyConfig`, `TestConcurrency`, `TestIterations`).

- [ ] **Step 7: Commit**

```bash
git add src/conversation2sql/eval_framework/main_pipe_workflow.py tests/eval_framework/test_main_pipe_workflow.py
git commit -m "feat: run N iterations per variant with flattened concurrency and per-iteration files"
```

---

### Task 4: Fix the integration smoke test and update docs

The integration smoke test (`test_baselines_smoke.py`) reads the return value of the pipeline, which is now `None`. Point it at the per-iteration file instead. Then update the stale CLAUDE.md.

**Files:**
- Modify: `tests/eval_framework/integration/test_baselines_smoke.py:33-39`
- Modify: `src/conversation2sql/eval_framework/CLAUDE.md`

- [ ] **Step 1: Rewrite the smoke test to read records from disk**

Replace lines 33-39 of `tests/eval_framework/integration/test_baselines_smoke.py` with:

```python
    workflow_evaluation_pipeline(cp, cr, cpred, cu)

    import json
    lines = (tmp_path / "results_iter0.jsonl").read_text().splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    missing = REQUIRED_KEYS - record.keys()
    assert not missing, f"Missing keys for {baseline}: {missing}"
    assert isinstance(record["execution_accuracy"], bool)
```

- [ ] **Step 2: Verify the smoke test at least imports/collects**

Run: `uv run pytest tests/eval_framework/integration/test_baselines_smoke.py --collect-only -q`
Expected: collection succeeds (4 parametrized cases listed). The tests themselves are `@pytest.mark.integration` and require live Postgres + API keys, so they are skipped/deselected in the normal suite — collection is the check here.

- [ ] **Step 3: Update CLAUDE.md**

In `src/conversation2sql/eval_framework/CLAUDE.md`, update the two stale references:

Replace the `main_pipe_workflow.py` bullet (under "Key files"):

```markdown
- `main_pipe_workflow.py` — `workflow_evaluation_pipeline` is the top-level entry point called from `cli.py`/`main.py`. It initialises models, loads tasks, runs the agent for every (iteration, task) pair, and writes JSONL output. It runs `num_iterations` passes of the dataset (collapsed to 1 when predictor `temperature <= 0`) and writes one file per iteration: `<output_dir>/results_iter{i}.jsonl`. Returns `None` — records are streamed to disk, not accumulated in memory.
```

Replace the "Output records" section:

```markdown
## Output records

`_save_record` appends one JSON line per (task, iteration). Each record carries an `iteration` field (0..num_iterations-1) and lands in `results_iter{iteration}.jsonl`. Aggregation across iterations (mean/std, pass@k, etc.) is done downstream in the explorer app, not in the pipeline. `execution_accuracy` comes from the last `submit_sql` ToolMessage's `passed` field.
```

- [ ] **Step 4: Run the full suite**

Run: `uv run pytest tests/`
Expected: PASS (integration tests deselected/skipped without the `-m integration` marker and live services).

- [ ] **Step 5: Type-check**

Run: `uv run pyrefly check`
Expected: no new errors introduced by the changed files.

- [ ] **Step 6: Commit**

```bash
git add tests/eval_framework/integration/test_baselines_smoke.py src/conversation2sql/eval_framework/CLAUDE.md
git commit -m "test: read smoke-test records from per-iteration file; refresh CLAUDE.md"
```

---

## Final verification

- [ ] Run `uv run pytest tests/` — full suite green.
- [ ] Manual smoke (optional, needs Postgres + API keys): `uv run python main.py --config configs/eval_pipeline_config.yaml --num_iterations 3 --predictor_temperature 0.7 --debug` and confirm `results_iter0.jsonl`..`results_iter2.jsonl` appear in the output folder, each with the same task count and an `iteration` field per record.
- [ ] Confirm with the user that the explorer app update (glob `results_iter*.jsonl` instead of `results.jsonl`) is tracked as a separate follow-up.
