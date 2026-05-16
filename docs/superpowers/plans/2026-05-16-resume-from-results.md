# Resume From Prior `results.jsonl` Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `--resume_from <path>` CLI option to `workflow_evaluation_pipeline` so a failed evaluation run can resume from a prior `results.jsonl`, skipping already-processed `instance_id`s and continuing in a new output folder.

**Architecture:** A new helper module `src/conversation2sql/eval_framework/resume.py` exposes two pure functions: `validate_resume_config` (hard-fails on identity-field mismatch between prior `config.yaml` and current run) and `prepare_resume` (copies the prior `results.jsonl` into the new output folder and returns the set of completed `instance_id`s). `ConfigPipeline` gains an optional `resume_from` field; `workflow_evaluation_pipeline` calls these helpers and filters the loaded dataset before the existing debug slice. Non-identity fields (sampling params, max tokens, user-sim model, debug, output folder) are not compared — users may legitimately change them between runs.

**Tech Stack:** Python 3.12, Pydantic v2, PyYAML, pytest (`asyncio_mode = "auto"`), `uv run` for all commands.

---

## File Structure

- **Create** `src/conversation2sql/eval_framework/resume.py` — identity-field constant, exception type, `validate_resume_config`, `prepare_resume`. ~80 lines.
- **Create** `tests/eval_framework/test_resume.py` — unit tests for both functions.
- **Modify** `src/conversation2sql/config_input.py` — add `resume_from: str | None = None` field to `ConfigPipeline`.
- **Modify** `src/conversation2sql/eval_framework/main_pipe_workflow.py` — wire the two helpers in, filter dataset by `done_ids`.
- **Modify** `tests/eval_framework/test_main_pipe_workflow.py` — add one integration test for the resume code path.

---

## Task 1: Add `resume_from` field to `ConfigPipeline`

**Files:**
- Modify: `src/conversation2sql/config_input.py:9-12`

- [ ] **Step 1: Add the field**

Replace the existing `ConfigPipeline` body so it reads:

```python
class ConfigPipeline(BaseModel):
    debug: bool = True
    output_folder: str = "results"
    baseline: Literal['no_tool', 'tools_only', 'tools_user', 'bird_full'] = 'bird_full'
    resume_from: str | None = None  # path to a prior results.jsonl to resume from
```

- [ ] **Step 2: Verify Pydantic parses the new field**

Run: `uv run python -c "from conversation2sql.config_input import ConfigPipeline; print(ConfigPipeline(resume_from='/tmp/x.jsonl').resume_from)"`

Expected: `/tmp/x.jsonl`

- [ ] **Step 3: Verify CLI parser exposes the flag**

Run: `uv run conv2sql run --help 2>&1 | grep -E "resume_from|pipeline_resume"`

Expected: at least one line mentioning `resume_from` (either bare or `--pipeline_resume_from`).

- [ ] **Step 4: Commit**

```bash
git add src/conversation2sql/config_input.py
git commit -m "feat: add ConfigPipeline.resume_from field for evaluation resume"
```

---

## Task 2: Create `resume.py` skeleton with constants and exception

**Files:**
- Create: `src/conversation2sql/eval_framework/resume.py`

- [ ] **Step 1: Create the file**

```python
"""Helpers to resume `workflow_evaluation_pipeline` from a prior results.jsonl.

The pipeline writes one JSON line per completed task to `<output_folder>/results.jsonl`.
On failure mid-run, re-launching with `--resume_from <path-to-prior-results.jsonl>`
will copy the prior file into the new output folder, skip already-processed
`instance_id`s, and continue from where the previous run stopped.

Two helpers are exported:
    - validate_resume_config(prior_config_path, current) -> None
    - prepare_resume(resume_from, output_folder) -> set[str]

A run is considered "the same experiment" iff all `IDENTITY_FIELDS` match between
the prior config.yaml and the current configuration. Non-identity fields may
legitimately change between runs (sampling params, max_new_tokens, user-sim
model, output_folder, debug).
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import yaml

from conversation2sql.logger import get_logger

logger = get_logger(__name__)

IDENTITY_FIELDS: dict[str, list[str]] = {
    "pipeline":  ["baseline"],
    "reader":    [
        "dataset_name_jsonl",
        "make_data_ambiguous",
        "database_schema_type",
        "is_kb_linearized",
        "read_only_gt_kb",
        "read_only_gt_tables",
    ],
    "predictor": ["model_name", "model_provider"],
}


class ResumeConfigMismatchError(RuntimeError):
    """Raised when --resume_from points at a run whose config diverges on identity fields."""
```

- [ ] **Step 2: Verify it imports**

Run: `uv run python -c "from conversation2sql.eval_framework.resume import IDENTITY_FIELDS, ResumeConfigMismatchError; print(sorted(IDENTITY_FIELDS))"`

Expected: `['pipeline', 'predictor', 'reader']`

- [ ] **Step 3: Commit**

```bash
git add src/conversation2sql/eval_framework/resume.py
git commit -m "feat: add resume.py skeleton with identity fields constant"
```

---

## Task 3: Test `validate_resume_config` — matching configs pass

**Files:**
- Create: `tests/eval_framework/test_resume.py`

- [ ] **Step 1: Write the failing test**

```python
from pathlib import Path

import pytest
import yaml

from conversation2sql.eval_framework.resume import (
    IDENTITY_FIELDS,
    ResumeConfigMismatchError,
    validate_resume_config,
)


def _identity_payload() -> dict[str, dict]:
    """Return a complete config dict with every identity field populated."""
    return {
        "pipeline":  {"baseline": "no_tool", "debug": True, "output_folder": "results"},
        "reader":    {
            "dataset_name_jsonl":   "data/foo.jsonl",
            "make_data_ambiguous":  False,
            "database_schema_type": "ddl",
            "is_kb_linearized":     False,
            "read_only_gt_kb":      True,
            "read_only_gt_tables":  True,
            "user_patience_budget": 10,
        },
        "predictor": {"model_name": "Qwen/Qwen3.5-9B", "model_provider": "hosted_vllm",
                      "temperature": 1.0, "max_new_tokens": 2000},
        "user":      {"model_name": "gpt-3.5-turbo", "model_provider": "openai"},
    }


def _write_yaml(path: Path, data: dict) -> Path:
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


def test_identical_configs_pass(tmp_path):
    payload = _identity_payload()
    prior = _write_yaml(tmp_path / "config.yaml", payload)
    validate_resume_config(prior, payload)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/eval_framework/test_resume.py::test_identical_configs_pass -v`

Expected: FAIL with `ImportError` on `validate_resume_config` (function not yet defined).

- [ ] **Step 3: Add minimal implementation to `resume.py`**

Append to `src/conversation2sql/eval_framework/resume.py`:

```python
def validate_resume_config(prior_config_path: Path, current: dict[str, dict]) -> None:
    """Raise ResumeConfigMismatchError if any identity field differs.

    Args:
        prior_config_path: path to config.yaml saved by the prior run.
        current: dict shaped {"pipeline": {...}, "reader": {...}, "predictor": {...}, "user": {...}}
                 — typically built by calling model_dump() on each of the four config models.

    Raises:
        FileNotFoundError: prior_config_path does not exist.
        ResumeConfigMismatchError: one or more identity fields differ. The message
                                   lists ALL mismatches at once (not just the first).
    """
    if not prior_config_path.exists():
        raise FileNotFoundError(
            f"Cannot resume: prior config.yaml not found at {prior_config_path}"
        )

    with prior_config_path.open("r", encoding="utf-8") as f:
        prior = yaml.safe_load(f) or {}

    diffs: list[str] = []
    for section, fields in IDENTITY_FIELDS.items():
        prior_section = prior.get(section, {}) or {}
        curr_section = current.get(section, {}) or {}
        for field in fields:
            prior_val = prior_section.get(field)
            curr_val = curr_section.get(field)
            if prior_val != curr_val:
                diffs.append(f"{section}.{field}: prior={prior_val!r}, current={curr_val!r}")

    if diffs:
        raise ResumeConfigMismatchError(
            "Cannot resume: identity-field mismatch between prior run and current config:\n  - "
            + "\n  - ".join(diffs)
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/eval_framework/test_resume.py::test_identical_configs_pass -v`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/conversation2sql/eval_framework/resume.py tests/eval_framework/test_resume.py
git commit -m "feat: validate_resume_config passes on identity-field match"
```

---

## Task 4: Test `validate_resume_config` — each identity field mismatch raises

**Files:**
- Modify: `tests/eval_framework/test_resume.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/eval_framework/test_resume.py`:

```python
@pytest.mark.parametrize("section,field,new_value", [
    ("pipeline",  "baseline",             "bird_full"),
    ("reader",    "dataset_name_jsonl",   "data/other.jsonl"),
    ("reader",    "make_data_ambiguous",  True),
    ("reader",    "database_schema_type", "toon"),
    ("reader",    "is_kb_linearized",     True),
    ("reader",    "read_only_gt_kb",      False),
    ("reader",    "read_only_gt_tables",  False),
    ("predictor", "model_name",           "Qwen/Qwen3.5-14B"),
    ("predictor", "model_provider",       "openai"),
])
def test_single_identity_mismatch_raises(tmp_path, section, field, new_value):
    payload = _identity_payload()
    prior = _write_yaml(tmp_path / "config.yaml", payload)

    mutated = _identity_payload()
    mutated[section][field] = new_value

    with pytest.raises(ResumeConfigMismatchError) as exc:
        validate_resume_config(prior, mutated)
    assert f"{section}.{field}" in str(exc.value)


def test_multiple_mismatches_all_reported(tmp_path):
    payload = _identity_payload()
    prior = _write_yaml(tmp_path / "config.yaml", payload)

    mutated = _identity_payload()
    mutated["pipeline"]["baseline"] = "bird_full"
    mutated["predictor"]["model_name"] = "different-model"

    with pytest.raises(ResumeConfigMismatchError) as exc:
        validate_resume_config(prior, mutated)
    msg = str(exc.value)
    assert "pipeline.baseline" in msg
    assert "predictor.model_name" in msg


def test_non_identity_field_difference_does_not_raise(tmp_path):
    payload = _identity_payload()
    prior = _write_yaml(tmp_path / "config.yaml", payload)

    mutated = _identity_payload()
    mutated["predictor"]["temperature"] = 0.2          # not identity
    mutated["predictor"]["max_new_tokens"] = 4000      # not identity
    mutated["reader"]["user_patience_budget"] = 20     # not identity
    mutated["pipeline"]["debug"] = False               # not identity
    mutated["pipeline"]["output_folder"] = "/other"    # not identity
    mutated["user"]["model_name"] = "gpt-4"            # not identity

    validate_resume_config(prior, mutated)  # must not raise


def test_missing_prior_config_raises_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError):
        validate_resume_config(tmp_path / "missing.yaml", _identity_payload())
```

- [ ] **Step 2: Run tests to verify they pass**

Run: `uv run pytest tests/eval_framework/test_resume.py -v`

Expected: all 12+ tests PASS (1 from Task 3 + 9 parametrized + 3 new).

- [ ] **Step 3: Commit**

```bash
git add tests/eval_framework/test_resume.py
git commit -m "test: cover identity-field mismatch and non-identity drift cases"
```

---

## Task 5: Test `prepare_resume` — copy file and extract instance_ids

**Files:**
- Modify: `tests/eval_framework/test_resume.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/eval_framework/test_resume.py`:

```python
import json
from conversation2sql.eval_framework.resume import prepare_resume


def _write_jsonl(path: Path, records: list[dict]) -> Path:
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    return path


def test_prepare_resume_copies_file_and_returns_ids(tmp_path):
    src_dir = tmp_path / "prior"
    src_dir.mkdir()
    src = _write_jsonl(src_dir / "results.jsonl", [
        {"instance_id": "task_a", "execution_accuracy": True},
        {"instance_id": "task_b", "execution_accuracy": False},
        {"instance_id": "task_c", "execution_accuracy": True},
    ])

    dest_dir = tmp_path / "new"
    dest_dir.mkdir()

    done = prepare_resume(src, dest_dir)

    assert done == {"task_a", "task_b", "task_c"}
    dest_file = dest_dir / "results.jsonl"
    assert dest_file.exists()
    assert dest_file.read_text() == src.read_text()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/eval_framework/test_resume.py::test_prepare_resume_copies_file_and_returns_ids -v`

Expected: FAIL with `ImportError` on `prepare_resume`.

- [ ] **Step 3: Implement `prepare_resume`**

Append to `src/conversation2sql/eval_framework/resume.py`:

```python
def prepare_resume(resume_from: Path, output_folder: Path) -> set[str]:
    """Copy prior results.jsonl into output_folder and return processed instance_ids.

    Args:
        resume_from: path to a prior `results.jsonl` written by the pipeline.
        output_folder: destination directory for the resumed run. Must already
                       exist (the workflow creates it via _save_configs_as_yaml).

    Returns:
        Set of `instance_id` values found in the prior file. Tasks with these
        ids should be skipped when iterating the dataset.

    Raises:
        FileNotFoundError: resume_from does not exist.
        RuntimeError:      output_folder/results.jsonl already exists and is non-empty
                           (prevents silent double-counting on accidental re-resume).
    """
    if not resume_from.exists():
        raise FileNotFoundError(
            f"Cannot resume: prior results.jsonl not found at {resume_from}"
        )

    dest = output_folder / "results.jsonl"
    if dest.exists() and dest.stat().st_size > 0:
        raise RuntimeError(
            f"Cannot resume: destination {dest} already exists and is non-empty. "
            "Refusing to overwrite to avoid double-counting."
        )

    done_ids: set[str] = set()
    with resume_from.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                logger.warning(
                    "prepare_resume: skipping malformed JSON at %s:%d (%s)",
                    resume_from, lineno, e,
                )
                continue
            instance_id = record.get("instance_id")
            if instance_id is None:
                logger.warning(
                    "prepare_resume: record at %s:%d has no instance_id; skipping",
                    resume_from, lineno,
                )
                continue
            done_ids.add(str(instance_id))

    shutil.copyfile(resume_from, dest)
    logger.info(
        "prepare_resume: seeded %s with %d records from %s",
        dest, len(done_ids), resume_from,
    )
    return done_ids
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/eval_framework/test_resume.py::test_prepare_resume_copies_file_and_returns_ids -v`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/conversation2sql/eval_framework/resume.py tests/eval_framework/test_resume.py
git commit -m "feat: prepare_resume copies prior results.jsonl and returns instance_ids"
```

---

## Task 6: Test `prepare_resume` — edge cases (missing, occupied, corrupt)

**Files:**
- Modify: `tests/eval_framework/test_resume.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/eval_framework/test_resume.py`:

```python
def test_prepare_resume_missing_source_raises(tmp_path):
    dest_dir = tmp_path / "new"
    dest_dir.mkdir()
    with pytest.raises(FileNotFoundError):
        prepare_resume(tmp_path / "nonexistent.jsonl", dest_dir)


def test_prepare_resume_refuses_non_empty_destination(tmp_path):
    src = _write_jsonl(tmp_path / "src.jsonl", [{"instance_id": "a"}])
    dest_dir = tmp_path / "dest"
    dest_dir.mkdir()
    (dest_dir / "results.jsonl").write_text('{"instance_id": "x"}\n', encoding="utf-8")

    with pytest.raises(RuntimeError, match="already exists"):
        prepare_resume(src, dest_dir)


def test_prepare_resume_empty_file_returns_empty_set(tmp_path):
    src = tmp_path / "src.jsonl"
    src.write_text("", encoding="utf-8")
    dest_dir = tmp_path / "dest"
    dest_dir.mkdir()

    assert prepare_resume(src, dest_dir) == set()
    assert (dest_dir / "results.jsonl").exists()


def test_prepare_resume_skips_malformed_lines(tmp_path, caplog):
    src = tmp_path / "src.jsonl"
    src.write_text(
        '{"instance_id": "task_a"}\n'
        'not valid json\n'
        '{"instance_id": "task_b"}\n'
        '{"no_id_here": true}\n',
        encoding="utf-8",
    )
    dest_dir = tmp_path / "dest"
    dest_dir.mkdir()

    import logging
    with caplog.at_level(logging.WARNING):
        done = prepare_resume(src, dest_dir)

    assert done == {"task_a", "task_b"}
    assert any("malformed JSON" in r.message for r in caplog.records)
    assert any("no instance_id" in r.message for r in caplog.records)
```

- [ ] **Step 2: Run tests to verify they pass**

Run: `uv run pytest tests/eval_framework/test_resume.py -v`

Expected: all tests PASS.

- [ ] **Step 3: Commit**

```bash
git add tests/eval_framework/test_resume.py
git commit -m "test: cover prepare_resume edge cases (missing/occupied/corrupt)"
```

---

## Task 7: Wire resume helpers into `workflow_evaluation_pipeline`

**Files:**
- Modify: `src/conversation2sql/eval_framework/main_pipe_workflow.py:74-88`

- [ ] **Step 1: Add import**

Open `src/conversation2sql/eval_framework/main_pipe_workflow.py`. After the existing import:

```python
from conversation2sql.eval_framework.dataset_readers import load_bird_interact_as_tasks
```

add:

```python
from conversation2sql.eval_framework.resume import prepare_resume, validate_resume_config
```

- [ ] **Step 2: Insert resume block after `_save_configs_as_yaml`**

In `workflow_evaluation_pipeline`, locate the block (currently lines 67-76):

```python
    _save_configs_as_yaml(
        output_folder=output_folder,
        config_pipeline=config_pipeline,
        config_reader=config_reader,
        config_predictor=config_predictor,
        config_user=config_user,
    )

    file_result = output_folder / 'results.jsonl'
```

Replace with:

```python
    _save_configs_as_yaml(
        output_folder=output_folder,
        config_pipeline=config_pipeline,
        config_reader=config_reader,
        config_predictor=config_predictor,
        config_user=config_user,
    )

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
        logger.info(
            "Resuming: skipping %d already-processed tasks from %s",
            len(done_ids), resume_path,
        )

    file_result = output_folder / 'results.jsonl'
```

- [ ] **Step 3: Filter dataset by `done_ids` before debug slice**

Locate the block (currently lines 83-88):

```python
    # read dataset
    dataset: list[TaskData] = load_bird_interact_as_tasks(**config_reader.model_dump())

    if config_pipeline.debug:
        dataset = dataset[:10]
        logger.info("Debug mode is ON - using only the first 10 tasks from the dataset")
```

Replace with:

```python
    # read dataset
    dataset: list[TaskData] = load_bird_interact_as_tasks(**config_reader.model_dump())

    if done_ids:
        before = len(dataset)
        dataset = [t for t in dataset if t.instance_id not in done_ids]
        logger.info(
            "Resume filter: %d -> %d tasks after skipping completed instance_ids",
            before, len(dataset),
        )

    if config_pipeline.debug:
        dataset = dataset[:10]
        logger.info("Debug mode is ON - using only the first 10 tasks from the dataset")
```

- [ ] **Step 4: Verify existing tests still pass**

Run: `uv run pytest tests/eval_framework/test_main_pipe_workflow.py -v`

Expected: all 4 existing tests PASS (resume_from defaults to None, so the new code is a no-op).

- [ ] **Step 5: Commit**

```bash
git add src/conversation2sql/eval_framework/main_pipe_workflow.py
git commit -m "feat: wire resume_from into workflow_evaluation_pipeline"
```

---

## Task 8: Integration test — pipeline skips pre-seeded instance_ids

**Files:**
- Modify: `tests/eval_framework/test_main_pipe_workflow.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/eval_framework/test_main_pipe_workflow.py`:

```python
import json
import yaml as _yaml


def _make_task(instance_id: str):
    t = MagicMock()
    t.instance_id = instance_id
    t.model_dump.return_value = {
        "instance_id": instance_id,
        "selected_database": "db1",
        "amb_user_query": "q?",
        "sol_sql": "SELECT 1",
        "sql_query_conditions": {},
        "not_ambiguos_query": "q",
        "gt_knowledge_base": [],
        "category": "easy",
        "ddl_database_schema": "CREATE TABLE t (id INT);",
        "masked_agent_kb_linearized": {},
        "user_query_ambiguity": {},
    }
    return t


@patch("conversation2sql.eval_framework.main_pipe_workflow.run_agent_bird_baseline")
@patch("conversation2sql.eval_framework.main_pipe_workflow.run_baseline_no_tool")
@patch("conversation2sql.eval_framework.main_pipe_workflow.load_bird_interact_as_tasks")
@patch("conversation2sql.eval_framework.main_pipe_workflow.utils_create_model")
def test_resume_skips_already_processed_instance_ids(
    mock_create, mock_load, mock_no_tool, mock_agent, tmp_path,
):
    # Build a "prior" run folder with config.yaml + results.jsonl for 2 of 5 tasks.
    prior_dir = tmp_path / "prior"
    prior_dir.mkdir()
    prior_payload = {
        "pipeline":  {"baseline": "no_tool", "debug": False, "output_folder": str(prior_dir),
                      "resume_from": None},
        "reader":    ConfigReader(make_data_ambiguous=False, is_kb_linearized=False,
                                  read_only_gt_kb=True, read_only_gt_tables=True).model_dump(),
        "predictor": ConfigPredictor().model_dump(),
        "user":      ConfigUserSimulator().model_dump(),
    }
    (prior_dir / "config.yaml").write_text(_yaml.safe_dump(prior_payload), encoding="utf-8")
    prior_results = prior_dir / "results.jsonl"
    with prior_results.open("w", encoding="utf-8") as f:
        for iid in ("task_1", "task_2"):
            f.write(json.dumps({"instance_id": iid, "execution_accuracy": True}) + "\n")

    # Current run points at the prior results.jsonl.
    new_out = tmp_path / "new"
    new_out.mkdir()
    cp = ConfigPipeline(
        debug=False, output_folder=str(new_out), baseline="no_tool",
        resume_from=str(prior_results),
    )
    cr = ConfigReader(make_data_ambiguous=False, is_kb_linearized=False,
                      read_only_gt_kb=True, read_only_gt_tables=True)
    cpred = ConfigPredictor()
    cu = ConfigUserSimulator()

    mock_load.return_value = [_make_task(f"task_{i}") for i in range(1, 6)]
    mock_no_tool.return_value = _stub_response()
    mock_create.return_value = MagicMock()

    workflow_evaluation_pipeline(cp, cr, cpred, cu)

    # Only the 3 not-yet-processed tasks should have been dispatched to the runner.
    dispatched_ids = [
        call.args[0].instance_id for call in mock_no_tool.call_args_list
    ]
    assert dispatched_ids == ["task_3", "task_4", "task_5"]

    # The new output folder's results.jsonl should be seeded with the 2 prior records
    # plus the 3 new ones (5 total lines).
    new_results = new_out / "results.jsonl"
    assert new_results.exists()
    lines = [json.loads(l) for l in new_results.read_text().splitlines() if l.strip()]
    assert len(lines) == 5
    assert {l["instance_id"] for l in lines[:2]} == {"task_1", "task_2"}
```

- [ ] **Step 2: Run the test**

Run: `uv run pytest tests/eval_framework/test_main_pipe_workflow.py::test_resume_skips_already_processed_instance_ids -v`

Expected: PASS

- [ ] **Step 3: Run the full pipeline test module to confirm no regressions**

Run: `uv run pytest tests/eval_framework/test_main_pipe_workflow.py tests/eval_framework/test_resume.py -v`

Expected: all tests PASS.

- [ ] **Step 4: Commit**

```bash
git add tests/eval_framework/test_main_pipe_workflow.py
git commit -m "test: integration test for resume_from in workflow_evaluation_pipeline"
```

---

## Task 9: Integration test — debug slice operates on post-filter dataset

**Files:**
- Modify: `tests/eval_framework/test_main_pipe_workflow.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/eval_framework/test_main_pipe_workflow.py`:

```python
@patch("conversation2sql.eval_framework.main_pipe_workflow.run_agent_bird_baseline")
@patch("conversation2sql.eval_framework.main_pipe_workflow.run_baseline_no_tool")
@patch("conversation2sql.eval_framework.main_pipe_workflow.load_bird_interact_as_tasks")
@patch("conversation2sql.eval_framework.main_pipe_workflow.utils_create_model")
def test_resume_with_debug_takes_first_ten_remaining(
    mock_create, mock_load, mock_no_tool, mock_agent, tmp_path,
):
    # Prior run completed 5 tasks; dataset has 20; debug=True should yield 10 of the
    # 15 remaining (not 10 of 20).
    prior_dir = tmp_path / "prior"
    prior_dir.mkdir()
    prior_payload = {
        "pipeline":  {"baseline": "no_tool", "debug": True, "output_folder": str(prior_dir),
                      "resume_from": None},
        "reader":    ConfigReader(make_data_ambiguous=False, is_kb_linearized=False,
                                  read_only_gt_kb=True, read_only_gt_tables=True).model_dump(),
        "predictor": ConfigPredictor().model_dump(),
        "user":      ConfigUserSimulator().model_dump(),
    }
    (prior_dir / "config.yaml").write_text(_yaml.safe_dump(prior_payload), encoding="utf-8")
    prior_results = prior_dir / "results.jsonl"
    with prior_results.open("w", encoding="utf-8") as f:
        for i in range(1, 6):
            f.write(json.dumps({"instance_id": f"task_{i}"}) + "\n")

    new_out = tmp_path / "new"
    new_out.mkdir()
    cp = ConfigPipeline(
        debug=True, output_folder=str(new_out), baseline="no_tool",
        resume_from=str(prior_results),
    )
    cr = ConfigReader(make_data_ambiguous=False, is_kb_linearized=False,
                      read_only_gt_kb=True, read_only_gt_tables=True)
    cpred = ConfigPredictor()
    cu = ConfigUserSimulator()

    mock_load.return_value = [_make_task(f"task_{i}") for i in range(1, 21)]
    mock_no_tool.return_value = _stub_response()
    mock_create.return_value = MagicMock()

    workflow_evaluation_pipeline(cp, cr, cpred, cu)

    dispatched_ids = [c.args[0].instance_id for c in mock_no_tool.call_args_list]
    assert dispatched_ids == [f"task_{i}" for i in range(6, 16)]
    assert len(dispatched_ids) == 10
```

- [ ] **Step 2: Run the test**

Run: `uv run pytest tests/eval_framework/test_main_pipe_workflow.py::test_resume_with_debug_takes_first_ten_remaining -v`

Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add tests/eval_framework/test_main_pipe_workflow.py
git commit -m "test: debug slice operates on dataset after resume filter"
```

---

## Task 10: Integration test — identity-field mismatch aborts before model load

**Files:**
- Modify: `tests/eval_framework/test_main_pipe_workflow.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/eval_framework/test_main_pipe_workflow.py`:

```python
from conversation2sql.eval_framework.resume import ResumeConfigMismatchError


@patch("conversation2sql.eval_framework.main_pipe_workflow.run_agent_bird_baseline")
@patch("conversation2sql.eval_framework.main_pipe_workflow.run_baseline_no_tool")
@patch("conversation2sql.eval_framework.main_pipe_workflow.load_bird_interact_as_tasks")
@patch("conversation2sql.eval_framework.main_pipe_workflow.utils_create_model")
def test_resume_with_mismatched_baseline_aborts_before_model_load(
    mock_create, mock_load, mock_no_tool, mock_agent, tmp_path,
):
    prior_dir = tmp_path / "prior"
    prior_dir.mkdir()
    prior_payload = {
        "pipeline":  {"baseline": "no_tool", "debug": False, "output_folder": str(prior_dir),
                      "resume_from": None},
        "reader":    ConfigReader(make_data_ambiguous=False, is_kb_linearized=False,
                                  read_only_gt_kb=True, read_only_gt_tables=True).model_dump(),
        "predictor": ConfigPredictor().model_dump(),
        "user":      ConfigUserSimulator().model_dump(),
    }
    (prior_dir / "config.yaml").write_text(_yaml.safe_dump(prior_payload), encoding="utf-8")
    prior_results = prior_dir / "results.jsonl"
    prior_results.write_text(json.dumps({"instance_id": "task_1"}) + "\n", encoding="utf-8")

    new_out = tmp_path / "new"
    new_out.mkdir()
    # current baseline differs from prior
    cp = ConfigPipeline(
        debug=False, output_folder=str(new_out), baseline="bird_full",
        resume_from=str(prior_results),
    )
    cr = ConfigReader(make_data_ambiguous=True, is_kb_linearized=False,
                      read_only_gt_kb=True, read_only_gt_tables=True)
    cpred = ConfigPredictor()
    cu = ConfigUserSimulator()

    with pytest.raises(ResumeConfigMismatchError) as exc:
        workflow_evaluation_pipeline(cp, cr, cpred, cu)
    assert "pipeline.baseline" in str(exc.value)

    # No model loaded, no dataset loaded, no tasks dispatched.
    mock_create.assert_not_called()
    mock_load.assert_not_called()
    mock_no_tool.assert_not_called()
    mock_agent.assert_not_called()
```

- [ ] **Step 2: Run the test**

Run: `uv run pytest tests/eval_framework/test_main_pipe_workflow.py::test_resume_with_mismatched_baseline_aborts_before_model_load -v`

Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add tests/eval_framework/test_main_pipe_workflow.py
git commit -m "test: identity mismatch aborts before any model is loaded"
```

---

## Task 11: Final verification — full suite and type-check

**Files:** none modified.

- [ ] **Step 1: Run the entire test suite**

Run: `uv run pytest tests/`

Expected: all tests PASS (no regressions in any other module).

- [ ] **Step 2: Run the type checker**

Run: `uv run pyrefly check`

Expected: clean exit. If new errors appear, they should reference only the files modified in this plan — fix them and re-run.

- [ ] **Step 3: Smoke-check the CLI help**

Run: `uv run conv2sql run --help 2>&1 | grep -i resume`

Expected: at least one line mentioning the `resume_from` / `--pipeline_resume_from` flag.

- [ ] **Step 4: Final commit (only if Steps 1-3 needed any fixes; otherwise skip)**

```bash
git status
# If clean, nothing to commit.
# If type-fixes were needed:
git add -p
git commit -m "fix: resolve type-checker findings in resume implementation"
```

---

## Self-review notes

- **Spec coverage:** Every section of `docs/superpowers/specs/2026-05-16-resume-from-results-design.md` maps to a task — user-facing contract (Tasks 1, 7), identity fields (Tasks 2, 3, 4), code structure (Tasks 1, 2, 3, 5, 7), failure modes (Tasks 4, 6, 10), tests (Tasks 3-6, 8-10).
- **Type consistency:** `validate_resume_config(prior_config_path: Path, current: dict[str, dict]) -> None` and `prepare_resume(resume_from: Path, output_folder: Path) -> set[str]` are used with the same signatures in every task. The `current` dict is always shaped `{"pipeline": .., "reader": .., "predictor": .., "user": ..}`. `done_ids: set[str]` is the type carried through `workflow_evaluation_pipeline`.
- **No placeholders:** Every code/test step contains the actual code or command.
