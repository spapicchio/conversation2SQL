# Resume / recover failed eval instances — design

**Date:** 2026-06-01
**Status:** Approved (Approach A), ready for implementation planning

## Problem

When a `just eval` run partially fails, some `(instance_id, iteration)` pairs never
produce a successful result. Today the failures are isolated and logged to
`<run_dir>/results_error.jsonl` (`main_pipe_workflow.py` `_process_one`), but there is
no way to re-run *only* the missing work — the reader
(`load_bird_interact_as_tasks`) always loads the full dataset, and every launch
creates a fresh timestamped output directory.

We want a one-command recovery: `just recover <run_dir>` that re-runs only what is
missing and appends the results into the original run directory, leaving that one
folder complete.

## Decisions (from brainstorming)

- **Recovery scope:** *errored + missing*. Re-run any `(instance_id, iteration)` pair
  that lacks a successful result in `results_iter*.jsonl`. This naturally covers both
  logged errors and tasks the run never reached (e.g. a crash/kill partway through).
- **Output destination:** append into the **same** `<run_dir>`, so that folder becomes
  complete. (Mutates the original run; accepted.)
- **Trigger:** a `just recover <run_dir>` recipe, built on a pipeline-level `--resume`
  flag.
- **Skip granularity:** per `(instance_id, iteration)` pair, so a run that crashed
  mid-iteration is correctly resumed.

## Approach A — replay the saved snapshot (chosen)

The pipeline already writes a `config.yaml` snapshot into the run directory at start
(`_save_configs_as_yaml`), using sections `pipeline / reader / predictor / user`.
Those are exactly the section names `PydanticParser` derives from the four config
classes (`cli_parser.py` `_derive_section_name`, `_read_yaml`), so the snapshot can be
fed back verbatim via `--config`, with CLI flags overriding only what must change
(`output_folder`, `resume`, fresh vLLM api-base).

This faithfully reproduces the *exact* original run, including any ad-hoc `--extra`
overrides — which is why it was preferred over reconstructing the named
`--variant`/`--model-profile` presets (variant-inversion cannot represent ad-hoc runs).

## Architecture

### Layer 1 — pipeline `--resume` (core, independently usable)

- Add `resume: bool = False` to `ConfigPipeline` (`config_input.py`).
- In `_run_tasks_concurrently` (`main_pipe_workflow.py`), before building the coroutine
  list:
  - If `resume` is set, scan `output_folder/results_iter{i}.jsonl` for
    `i in range(num_iterations)` and collect the set of already-successful
    `(instance_id, iteration)` pairs.
  - Parsing is robust: read line-by-line, `json.loads` each line inside try/except,
    skip blank/malformed lines (a crash can leave a truncated trailing line). Only
    records carrying both `instance_id` and `iteration` count as completed.
  - Skip any `(task.instance_id, iteration)` pair already in the completed set when
    building `coros`. Log `resume: skipping N completed, running M`.
- "Errored + missing" falls out for free: error records live only in
  `results_error.jsonl`, never in `results_iter*`, so anything errored or never-reached
  is absent from the completed set and is re-run. No duplicates arise because completed
  pairs are skipped.
- Snapshot preservation: when `resume` is true, write the start-of-run snapshot to
  `config_resume_<HH_MM_SS>.yaml` instead of overwriting the `config.yaml` we just read
  from. (`_save_configs_as_yaml` gains a target-filename parameter or the caller picks
  the name.)

Helper to add:

```python
def _load_completed_pairs(output_folder: Path, num_iterations: int) -> set[tuple[str, int]]:
    """Successful (instance_id, iteration) pairs already on disk in results_iter*.jsonl."""
```

The completed-pairs computation runs once (not per task); the existing per-task error
isolation in `_process_one` is unchanged.

### Layer 2 — `just recover <run_dir>`

- **presets CLI subcommand `recover-config --run-dir <dir>`** (`presets.py`): reads
  `<dir>/config.yaml`, pulls `predictor.model_provider` and `predictor.model_name`,
  reverse-maps the model name to its profile via a new helper
  `profile_for_model_name(name) -> str` (model names are unique across
  `MODEL_PROFILES`; raise on unknown), then emits NUL-delimited fields mirroring
  `server-config`:

  ```
  provider \0 model_name \0 max_model_len \0 thinking \0 <vllm serve args...>
  ```

  Thinking and serve args are resolved from the snapshot's `predictor.enable_thinking`
  and the profile, reusing `resolve_server_args`.

- **`bash_scripts/recover_payload.sh`** (small sibling of `eval_payload.sh`):
  - Requires `RESUME_DIR=<run_dir>`.
  - Calls `presets.py recover-config --run-dir "$RESUME_DIR"` to get provider + server
    config.
  - For `hosted_vllm`: source `vllm_server.sh`, `start_vllm_server` with the resolved
    args; the helper exports `PREDICTOR_VLLM_API_BASE`.
  - For external providers: no server; unset the api-base vars.
  - Invoke:
    ```
    uv run conv2sql run \
      --config "$RESUME_DIR/config.yaml" \
      --output_folder "$RESUME_DIR" \
      --resume true \
      [--predictor_vllm_api_base "$PREDICTOR_VLLM_API_BASE"]
    ```
  - Honour `DRY_RUN=1` to print the resolved command and exit (parity with
    `eval_payload.sh`).

- **`just recover run_dir=...` recipe** (`justfile`): dispatches `recover_payload.sh`
  through `submit_and_log.sh` (same tmux/slurm path as `eval`), exporting
  `RESUME_DIR=<run_dir>`. Because `output_folder` is the existing dir, no new
  timestamped directory is created.

## Testing

- Unit: `_load_completed_pairs` — multi-iteration files, a malformed/truncated trailing
  line, and absence of a file (returns empty).
- Unit: `profile_for_model_name` round-trips every entry in `MODEL_PROFILES` and raises
  on an unknown name.
- Integration: run the pipeline once with a pre-seeded `results_iter0.jsonl` containing
  a subset of instances and `resume=True`; assert only the missing instances are
  executed and appended (no duplicates for already-present pairs).
- `recover-config` CLI emits the expected NUL-delimited fields for a sample snapshot.
- Run `uv run pytest tests/` after implementation (per project CLAUDE.md).

## Known wart (documented, not fixed)

Stale lines for now-fixed instances remain in `results_error.jsonl` after a successful
recovery. Downstream aggregation (explorer) already treats `results_iter*` as the
source of truth, so this is cosmetic.

## Out of scope

- New-folder-then-merge output mode (rejected in favour of in-place append).
- A pure manual `--reader_only_instance_ids` filter (not needed; resume derives the set
  from disk).
- Deduplicating/rotating `results_error.jsonl`.
