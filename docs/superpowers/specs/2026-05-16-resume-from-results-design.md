# Resume from prior `results.jsonl`

## Problem

`workflow_evaluation_pipeline` writes one JSON line per task to `<output_folder>/results.jsonl`. When an exception is raised mid-run, the error is appended to `results_error.jsonl` and re-raised. Re-launching the script today starts from scratch, repeating every already-completed task. For long runs (hundreds of tasks against a Qwen3.5-9B vLLM server) this is expensive in both compute and wall-clock time.

We want a way to resume: skip already-processed tasks, continue from where the failure occurred, and preserve the previous results.

## User-facing contract

Add a single optional CLI flag:

```bash
uv run conv2sql run --config ... --resume_from /path/to/prior/results.jsonl
```

Behavior when set:

1. Read the prior `results.jsonl` and the sibling `config.yaml`.
2. Validate that the prior run's *identity fields* match the current configuration. On any mismatch, raise `ResumeConfigMismatchError` with a per-field diff and exit **before** loading any model.
3. Copy the prior `results.jsonl` into the current run's `output_folder`. The caller is responsible for ensuring `output_folder` is distinct from the prior run's folder (the usual convention — timestamped subfolders set by the bash launcher — already guarantees this); `prepare_resume` enforces this defensively by refusing to overwrite a non-empty destination.
4. Load the dataset, filter out any `TaskData` whose `instance_id` appears in the seeded file, and run the standard pipeline on the remainder.
5. New records continue to append to the seeded `output_folder/results.jsonl`.

When `--resume_from` is omitted, behavior is identical to today.

## Identity fields

Two configurations are considered "the same experiment" iff all of the following fields are equal:

| Section     | Field                                                                                          |
|-------------|------------------------------------------------------------------------------------------------|
| `pipeline`  | `baseline`                                                                                     |
| `reader`    | `dataset_name_jsonl`, `make_data_ambiguous`, `database_schema_type`, `is_kb_linearized`, `read_only_gt_kb`, `read_only_gt_tables` |
| `predictor` | `model_name`, `model_provider`                                                                 |

Everything else (sampling parameters, `max_new_tokens`, `user_patience_budget`, user-simulator model, output folder, debug flag, vLLM API base, etc.) is **not** part of the identity. A mismatch on any non-identity field does not block resume and does not produce a warning — the assumption is that you may legitimately tune those between runs (e.g., bump `max_new_tokens` after an OOM).

## Code structure

### New field on `ConfigPipeline`

`src/conversation2sql/config_input.py`:

```python
class ConfigPipeline(BaseModel):
    debug: bool = True
    output_folder: str = "results"
    baseline: Literal['no_tool', 'tools_only', 'tools_user', 'bird_full'] = 'bird_full'
    resume_from: str | None = None  # path to a prior results.jsonl to resume from
```

The `PydanticParser` auto-generates `--pipeline_resume_from`; since the field name is unique across all four configs, the bare `--resume_from` also works.

### New module `eval_framework/resume.py`

```python
from pathlib import Path
import json
import shutil
import yaml

IDENTITY_FIELDS: dict[str, list[str]] = {
    "pipeline":  ["baseline"],
    "reader":    ["dataset_name_jsonl", "make_data_ambiguous",
                  "database_schema_type", "is_kb_linearized",
                  "read_only_gt_kb", "read_only_gt_tables"],
    "predictor": ["model_name", "model_provider"],
}


class ResumeConfigMismatchError(RuntimeError):
    """Raised when --resume_from points at a run whose config diverges on identity fields."""


def validate_resume_config(prior_config_path: Path, current: dict[str, dict]) -> None:
    """Compare identity fields between prior config.yaml and current config dicts.

    `current` shape: {"pipeline": {...}, "reader": {...}, "predictor": {...}, "user": {...}}
    Raises ResumeConfigMismatchError listing ALL mismatches at once.
    """


def prepare_resume(resume_from: Path, output_folder: Path) -> set[str]:
    """Copy prior results.jsonl into output_folder and return processed instance_ids.

    Raises:
        FileNotFoundError    — resume_from does not exist
        RuntimeError         — output_folder/results.jsonl already exists and is non-empty
                               (prevents silent double-counting on accidental re-resume)
    """
```

### Wiring in `main_pipe_workflow.py`

Inside `workflow_evaluation_pipeline`, immediately after `_save_configs_as_yaml` and before `_init_models`:

```python
done_ids: set[str] = set()
if config_pipeline.resume_from:
    resume_path = Path(config_pipeline.resume_from)
    validate_resume_config(
        prior_config_path=resume_path.parent / "config.yaml",
        current={
            "pipeline":  config_pipeline.model_dump(),
            "reader":    config_reader.model_dump(),
            "predictor": config_predictor.model_dump(),
            "user":      config_user.model_dump(),
        },
    )
    done_ids = prepare_resume(resume_path, output_folder)
    logger.info("Resuming: skipping %d already-processed tasks from %s",
                len(done_ids), resume_path)
```

After `load_bird_interact_as_tasks`, before the `debug` slice:

```python
if done_ids:
    dataset = [t for t in dataset if t.instance_id not in done_ids]

if config_pipeline.debug:
    dataset = dataset[:10]
```

This ordering means `debug` always selects the first 10 *remaining* tasks, which is the intuitive behavior for a debug resume.

## Failure modes

All raise with actionable messages and exit before any model is loaded:

| Condition                                                | Exception                       |
|----------------------------------------------------------|---------------------------------|
| `resume_from` path does not exist                        | `FileNotFoundError`             |
| Sibling `config.yaml` missing                            | `FileNotFoundError`             |
| Any identity field differs                               | `ResumeConfigMismatchError`     |
| `output_folder/results.jsonl` exists and non-empty       | `RuntimeError`                  |

Corrupt JSON lines in the prior file are logged at `WARNING` and skipped — recovery should be tolerant of a partially-written final line if the previous run died mid-write.

`results_error.jsonl` in the prior folder is ignored. Its records have no `instance_id` to skip and no business in the seeded file.

## Tests

`tests/eval_framework/test_resume.py`:

1. `validate_resume_config` passes when identity fields match.
2. Each identity field mismatch raises with that field name in the message.
3. Multiple mismatches → all listed in a single error.
4. Non-identity fields differ (e.g., `temperature`, `max_new_tokens`) → no raise.
5. `prepare_resume` copies the file and returns the correct set of `instance_id`s.
6. `prepare_resume` raises when destination `results.jsonl` already has content.
7. `prepare_resume` tolerates a malformed final line (logs + skips).
8. Integration: stub dataset with 5 tasks + pre-seeded `results.jsonl` containing 2 of those `instance_id`s → only 3 tasks are dispatched to the runner.
9. Integration with `debug=True`: 20-task stub dataset, 5 pre-seeded → debug slice operates on the 15 remaining and yields 10.

## Out of scope

- Automatic resumption (no flag). Explicit `--resume_from` only, per the user's requirement.
- Resumption from anything other than `results.jsonl` (no checkpointing of partial agent state mid-task — a task either completed and was recorded, or it didn't).
- Multi-file merging (no support for `--resume_from path1.jsonl --resume_from path2.jsonl`).
- Warnings on non-identity field drift.
