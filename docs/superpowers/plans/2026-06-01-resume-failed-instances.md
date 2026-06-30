# Resume / Recover Failed Eval Instances — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `just recover <run_dir>` command that re-runs only the missing/errored `(instance_id, iteration)` pairs of a previous eval run and appends them into the same run directory.

**Architecture:** A pipeline-level `--resume` flag scans the run dir's `results_iter*.jsonl`, computes the already-completed `(instance_id, iteration)` pairs, and skips them. A `just recover` recipe replays the run's own `config.yaml` snapshot via `--config` (overriding `output_folder`, `resume`, and a fresh vLLM api-base), reverse-mapping the model name to its server profile in `presets.py`. The snapshot is made round-trippable by fixing its user-simulator section key.

**Tech Stack:** Python 3.12, Pydantic, Typer/argparse CLI (`PydanticParser`), pytest (asyncio auto-mode), `just`, bash, vLLM.

---

## Background the engineer needs

- **Run the pipeline:** `uv run conv2sql run --config <yaml> [flags]`. Always use `uv run`.
- **Run tests:** `uv run pytest tests/...`. `pytest.ini_options` sets `asyncio_mode = "auto"`.
- **Type-check:** `uv run pyrefly check`.
- **Output layout:** a run writes `<run_dir>/results_iter{i}.jsonl` (one successful record per `(task, iteration)`), `<run_dir>/results_error.jsonl` (per-task failures: `{"instance_id","iteration","error"}`), and a `<run_dir>/config.yaml` snapshot. `<run_dir>` is the leaf dir that holds `config.yaml`, e.g. `results/2026-06-01/13-55-45/no_tool__Qwen3.5-9B__ddl/`.
- **Config sections:** `PydanticParser` derives YAML section names from class names: `ConfigPipeline→pipeline`, `ConfigReader→reader`, `ConfigPredictor→predictor`, `ConfigUserSimulator→user_simulator`. CLI flags beat YAML beats env beats defaults.
- **presets.py constraint:** the module must have **no third-party imports at module level** (bash runs `server-config` with the system `python3`, no venv). Any `yaml` use must be a function-local import.
- **vLLM launch:** `bash_scripts/utils/vllm_server.sh` `start_vllm_server MODEL_NAME MAX_MODEL_LEN [serve args...]` exports `PREDICTOR_VLLM_API_BASE` / `USER_SIMULATOR_VLLM_API_BASE`. `eval_payload.sh` is the reference for how server args are resolved and passed.
- **Dispatch:** `bash bash_scripts/submit_and_log.sh <payload.sh> [job_name]` snapshots the exported environment (`export -p`) — so any env var exported by the `just` recipe (e.g. `RESUME_DIR`) reaches the payload — then runs it in tmux (local) or sbatch (slurm).

---

## File Structure

- **Modify** `src/conversation2sql/config_input.py` — add `resume` field to `ConfigPipeline`.
- **Modify** `src/conversation2sql/eval_framework/main_pipe_workflow.py` — fix snapshot key, add `_load_completed_pairs`, thread `resume` through `_run_tasks_concurrently`, write a separate snapshot file on resume.
- **Modify** `src/conversation2sql/presets.py` — add `profile_for_model_name()` and the `recover-config` CLI subcommand.
- **Create** `bash_scripts/recover_payload.sh` — resume payload (start vLLM if needed, replay snapshot with `--resume`).
- **Modify** `justfile` — add `recover` recipe + `dispatch_recover` variable.
- **Tests:** `tests/eval_framework/test_main_pipe_workflow.py`, `tests/test_presets.py`, `tests/test_recover_payload_dryrun.py` (new).

---

## Task 1: Fix the snapshot user-simulator section key

Makes the `config.yaml` snapshot round-trip through `PydanticParser` (the foundation for replaying it via `--config`).

**Files:**
- Modify: `src/conversation2sql/eval_framework/main_pipe_workflow.py:284`
- Test: `tests/eval_framework/test_main_pipe_workflow.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/eval_framework/test_main_pipe_workflow.py`:

```python
def test_saved_snapshot_roundtrips_through_parser(tmp_path):
    from conversation2sql.eval_framework.main_pipe_workflow import _save_configs_as_yaml
    from conversation2sql.cli_parser import PydanticParser

    cp = ConfigPipeline(output_folder=str(tmp_path), baseline="tools_user")
    cr = ConfigReader()
    cpred = ConfigPredictor(model_name="some/model")
    cu = ConfigUserSimulator(model_name="user/model", model_provider="openai")
    _save_configs_as_yaml(tmp_path, cp, cr, cpred, cu)

    parser = PydanticParser(
        [ConfigPipeline, ConfigReader, ConfigPredictor, ConfigUserSimulator]
    )
    rp, rr, rpred, ruser = parser.parse_args_and_config(
        ["--config", str(tmp_path / "config.yaml")]
    )
    # The user-simulator section must survive the round-trip.
    assert ruser.model_name == "user/model"
    assert ruser.model_provider == "openai"
    assert rpred.model_name == "some/model"
    assert rp.baseline == "tools_user"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/eval_framework/test_main_pipe_workflow.py::test_saved_snapshot_roundtrips_through_parser -v`
Expected: FAIL — `ruser.model_name` is the default `"gpt-3.5-turbo"`, not `"user/model"` (the `user` key is ignored by the parser).

- [ ] **Step 3: Rename the snapshot key**

In `src/conversation2sql/eval_framework/main_pipe_workflow.py`, in `_save_configs_as_yaml`, change the `configs` dict key:

```python
    configs = {
        "pipeline": config_pipeline.model_dump(mode="json"),
        "reader": config_reader.model_dump(mode="json"),
        "predictor": config_predictor.model_dump(mode="json"),
        "user_simulator": config_user.model_dump(mode="json"),
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/eval_framework/test_main_pipe_workflow.py::test_saved_snapshot_roundtrips_through_parser -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/conversation2sql/eval_framework/main_pipe_workflow.py tests/eval_framework/test_main_pipe_workflow.py
git commit -m "fix(eval): write snapshot user-simulator section under its parser key

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 2: Add the `resume` config field

**Files:**
- Modify: `src/conversation2sql/config_input.py:9-14`
- Test: `tests/eval_framework/test_main_pipe_workflow.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/eval_framework/test_main_pipe_workflow.py`, inside `class TestConcurrencyConfig`:

```python
    def test_resume_defaults_to_false(self):
        assert ConfigPipeline().resume is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/eval_framework/test_main_pipe_workflow.py::TestConcurrencyConfig::test_resume_defaults_to_false -v`
Expected: FAIL — `AttributeError: 'ConfigPipeline' object has no attribute 'resume'`

- [ ] **Step 3: Add the field**

In `src/conversation2sql/config_input.py`, add to `ConfigPipeline` (after `num_iterations`):

```python
    resume: bool = False  # skip (instance_id, iteration) pairs already in output_folder/results_iter*.jsonl and run only the missing ones
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/eval_framework/test_main_pipe_workflow.py::TestConcurrencyConfig::test_resume_defaults_to_false -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/conversation2sql/config_input.py tests/eval_framework/test_main_pipe_workflow.py
git commit -m "feat(eval): add ConfigPipeline.resume flag

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 3: Add `_load_completed_pairs` helper

Scans the run dir for already-successful `(instance_id, iteration)` pairs, robust to missing files and malformed/truncated lines.

**Files:**
- Modify: `src/conversation2sql/eval_framework/main_pipe_workflow.py`
- Test: `tests/eval_framework/test_main_pipe_workflow.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/eval_framework/test_main_pipe_workflow.py` (top-level, near the other helper tests). Note the import line added to the existing import block:

```python
def test_load_completed_pairs_reads_all_iterations(tmp_path):
    from conversation2sql.eval_framework.main_pipe_workflow import _load_completed_pairs

    (tmp_path / "results_iter0.jsonl").write_text(
        json.dumps({"instance_id": "a", "iteration": 0}) + "\n"
        + json.dumps({"instance_id": "b", "iteration": 0}) + "\n"
        + "{ this is a truncated line\n"  # crash can leave a partial trailing line
    )
    (tmp_path / "results_iter1.jsonl").write_text(
        json.dumps({"instance_id": "a", "iteration": 1}) + "\n"
    )

    pairs = _load_completed_pairs(tmp_path, num_iterations=2)
    assert pairs == {("a", 0), ("b", 0), ("a", 1)}


def test_load_completed_pairs_missing_files_return_empty(tmp_path):
    from conversation2sql.eval_framework.main_pipe_workflow import _load_completed_pairs

    assert _load_completed_pairs(tmp_path, num_iterations=3) == set()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/eval_framework/test_main_pipe_workflow.py -k load_completed_pairs -v`
Expected: FAIL — `ImportError: cannot import name '_load_completed_pairs'`

- [ ] **Step 3: Implement the helper**

In `src/conversation2sql/eval_framework/main_pipe_workflow.py`, add this function (place it just above `_run_tasks_concurrently`):

```python
def _load_completed_pairs(
    output_folder: Path, num_iterations: int
) -> set[tuple[str, int]]:
    """Successful (instance_id, iteration) pairs already on disk.

    Reads output_folder/results_iter{i}.jsonl for i in range(num_iterations).
    Parsing is line-by-line and defensive: blank lines, malformed JSON (a crash
    can leave a truncated trailing line), and records missing instance_id are
    skipped. Error records live only in results_error.jsonl and are deliberately
    NOT counted as completed, so errored/never-reached tasks are re-run.
    """
    completed: set[tuple[str, int]] = set()
    for iteration in range(num_iterations):
        path = output_folder / f"results_iter{iteration}.jsonl"
        if not path.exists():
            continue
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                instance_id = record.get("instance_id")
                if instance_id is not None:
                    completed.add((instance_id, iteration))
    return completed
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/eval_framework/test_main_pipe_workflow.py -k load_completed_pairs -v`
Expected: PASS (both tests)

- [ ] **Step 5: Commit**

```bash
git add src/conversation2sql/eval_framework/main_pipe_workflow.py tests/eval_framework/test_main_pipe_workflow.py
git commit -m "feat(eval): add _load_completed_pairs scanner for resume

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 4: Wire `resume` into the task scheduler + separate snapshot on resume

**Files:**
- Modify: `src/conversation2sql/eval_framework/main_pipe_workflow.py` (`workflow_evaluation_pipeline`, `_run_tasks_concurrently`, `_save_configs_as_yaml`)
- Test: `tests/eval_framework/test_main_pipe_workflow.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/eval_framework/test_main_pipe_workflow.py`. First extend the existing `_fake_task` helper to take an id (change its definition):

```python
def _fake_task(instance_id="task_1"):
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
        "user_query_ambiguity": {},
    }
    return t
```

Then add the resume test class:

```python
@patch("conversation2sql.eval_framework.main_pipe_workflow.run_agent_bird_baseline")
@patch("conversation2sql.eval_framework.main_pipe_workflow.run_baseline_no_tool")
@patch("conversation2sql.eval_framework.main_pipe_workflow.load_bird_interact_as_tasks")
@patch("conversation2sql.eval_framework.main_pipe_workflow.utils_create_model")
class TestResume:
    def test_resume_runs_only_missing_pairs(
        self, mock_create, mock_load, mock_no_tool, mock_agent, configs, tmp_path
    ):
        cp, cr, cpred, cu = configs
        cp = cp.model_copy(update={"baseline": "no_tool", "resume": True, "debug": False})
        # cpred.temperature defaults to 0.0 -> a single iteration (0).
        # Pre-seed task_1 as already complete for iteration 0.
        (tmp_path / "results_iter0.jsonl").write_text(
            json.dumps({"instance_id": "task_1", "iteration": 0}) + "\n"
        )
        mock_load.return_value = [_fake_task("task_1"), _fake_task("task_2")]
        mock_no_tool.return_value = _stub_response()
        mock_create.return_value = MagicMock()

        workflow_evaluation_pipeline(cp, cr, cpred, cu)

        # Only the missing task_2 runs.
        assert mock_no_tool.call_count == 1

    def test_resume_writes_snapshot_to_separate_file(
        self, mock_create, mock_load, mock_no_tool, mock_agent, configs, tmp_path
    ):
        cp, cr, cpred, cu = configs
        cp = cp.model_copy(update={"baseline": "no_tool", "resume": True, "debug": False})
        original = "predictor:\n  model_name: ORIGINAL\n"
        (tmp_path / "config.yaml").write_text(original)
        mock_load.return_value = [_fake_task("task_1")]
        mock_no_tool.return_value = _stub_response()
        mock_create.return_value = MagicMock()

        workflow_evaluation_pipeline(cp, cr, cpred, cu)

        # Original snapshot is preserved (not clobbered); a resume snapshot is added.
        assert (tmp_path / "config.yaml").read_text() == original
        resume_snaps = list(tmp_path.glob("config_resume_*.yaml"))
        assert len(resume_snaps) == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/eval_framework/test_main_pipe_workflow.py::TestResume -v`
Expected: FAIL — `test_resume_runs_only_missing_pairs` fails with `call_count == 2` (no filtering yet); `test_resume_writes_snapshot_to_separate_file` fails because `config.yaml` is overwritten and no `config_resume_*.yaml` exists.

- [ ] **Step 3: Implement resume in the pipeline**

3a. In `src/conversation2sql/eval_framework/main_pipe_workflow.py`, add the import at the top (after `from pathlib import Path`):

```python
from datetime import datetime
```

3b. Give `_save_configs_as_yaml` a `filename` parameter. Change its signature and the path line:

```python
def _save_configs_as_yaml(
    output_folder: Path,
    config_pipeline: ConfigPipeline,
    config_reader: ConfigReader,
    config_predictor: ConfigPredictor,
    config_user: ConfigUserSimulator,
    filename: str = "config.yaml",
) -> None:
    output_folder.mkdir(parents=True, exist_ok=True)
    configs = {
        "pipeline": config_pipeline.model_dump(mode="json"),
        "reader": config_reader.model_dump(mode="json"),
        "predictor": config_predictor.model_dump(mode="json"),
        "user_simulator": config_user.model_dump(mode="json"),
    }
    config_path = output_folder / filename
    with config_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(configs, f, sort_keys=False, allow_unicode=True)
    logger.info(f"Saved configs to {config_path}")
```

3c. In `workflow_evaluation_pipeline`, replace the existing `_save_configs_as_yaml(...)` call (the block that starts with `# save config in the output folder`) with a resume-aware filename:

```python
    # save config in the output folder; on resume keep the original snapshot intact
    snapshot_name = (
        f"config_resume_{datetime.now().strftime('%H_%M_%S')}.yaml"
        if config_pipeline.resume
        else "config.yaml"
    )
    _save_configs_as_yaml(
        output_folder=output_folder,
        config_pipeline=config_pipeline,
        config_reader=config_reader,
        config_predictor=config_predictor,
        config_user=config_user,
        filename=snapshot_name,
    )
```

3d. Pass `resume` into `_run_tasks_concurrently`. In the `asyncio.run(_run_tasks_concurrently(...))` call, add the argument:

```python
                resume=config_pipeline.resume,
```

3e. Update `_run_tasks_concurrently`'s signature — add a parameter after `num_iterations: int`:

```python
    num_iterations: int,
    resume: bool = False,
```

3f. In `_run_tasks_concurrently`, replace the `coros = [...]` comprehension with a resume-aware version:

```python
    completed: set[tuple[str, int]] = (
        _load_completed_pairs(output_folder, num_iterations) if resume else set()
    )
    pairs = [
        (task, iteration)
        for iteration in range(num_iterations)
        for task in dataset
        if (task.instance_id, iteration) not in completed
    ]
    if resume:
        total = num_iterations * len(dataset)
        logger.info(
            "resume: skipping %d completed, running %d/%d (instance_id, iteration) pairs",
            len(completed),
            len(pairs),
            total,
        )
    coros = [_process_one(task, iteration) for task, iteration in pairs]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/eval_framework/test_main_pipe_workflow.py::TestResume -v`
Expected: PASS (both tests)

- [ ] **Step 5: Run the full workflow test module (no regressions)**

Run: `uv run pytest tests/eval_framework/test_main_pipe_workflow.py -v`
Expected: PASS (all tests, including the previously-existing ones)

- [ ] **Step 6: Commit**

```bash
git add src/conversation2sql/eval_framework/main_pipe_workflow.py tests/eval_framework/test_main_pipe_workflow.py
git commit -m "feat(eval): resume skips completed pairs and preserves original snapshot

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 5: Add `profile_for_model_name` to presets

**Files:**
- Modify: `src/conversation2sql/presets.py`
- Test: `tests/test_presets.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_presets.py`:

```python
def test_profile_for_model_name_roundtrips_all_profiles():
    from conversation2sql.presets import MODEL_PROFILES, profile_for_model_name

    for name, prof in MODEL_PROFILES.items():
        assert profile_for_model_name(prof["predictor_model_name"]) == name


def test_profile_for_model_name_unknown_raises():
    import pytest
    from conversation2sql.presets import profile_for_model_name

    with pytest.raises(ValueError):
        profile_for_model_name("no/such-model")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_presets.py -k profile_for_model_name -v`
Expected: FAIL — `ImportError: cannot import name 'profile_for_model_name'`

- [ ] **Step 3: Implement the helper**

In `src/conversation2sql/presets.py`, add (just below `_require_profile`):

```python
def profile_for_model_name(model_name: str) -> str:
    """Reverse-map a predictor model name to its profile key.

    Model names are unique across MODEL_PROFILES. Used by `recover-config` to
    relaunch the right vLLM server for a snapshot that only records the model
    name, not the profile.
    """
    for name, prof in MODEL_PROFILES.items():
        if prof["predictor_model_name"] == model_name:
            return name
    raise ValueError(
        f"No model profile for model_name {model_name!r}; "
        f"known: {', '.join(sorted(p['predictor_model_name'] for p in MODEL_PROFILES.values()))}"
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_presets.py -k profile_for_model_name -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/conversation2sql/presets.py tests/test_presets.py
git commit -m "feat(presets): add profile_for_model_name reverse lookup

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 6: Add the `recover-config` CLI subcommand to presets

Reads a run's `config.yaml`, reverse-maps the model name to a profile, and emits NUL-delimited fields for bash, mirroring `server-config` but with `provider` first.

**Files:**
- Modify: `src/conversation2sql/presets.py` (`main`, new `_cmd_recover_config`)
- Test: `tests/test_presets.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_presets.py`:

```python
def test_recover_config_emits_provider_and_server_fields(tmp_path):
    import subprocess
    import sys
    from pathlib import Path
    import yaml

    from conversation2sql.presets import MODEL_PROFILES

    prof = MODEL_PROFILES["qwen35"]
    snapshot = {
        "pipeline": {"baseline": "no_tool"},
        "predictor": {
            "model_name": prof["predictor_model_name"],
            "model_provider": "hosted_vllm",
            "enable_thinking": None,
        },
        "user_simulator": {"model_name": "x", "model_provider": "openai"},
    }
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(snapshot))

    presets_py = Path("src/conversation2sql/presets.py").resolve()
    out = subprocess.run(
        [sys.executable, str(presets_py), "recover-config", "--run-dir", str(tmp_path)],
        capture_output=True, text=True, check=True,
    ).stdout
    fields = out.split("\0")
    assert fields[0] == "hosted_vllm"                       # provider first
    assert fields[1] == prof["predictor_model_name"]        # model name
    assert fields[2] == str(prof["max_model_len"])          # max model len
    assert fields[3] in ("true", "false")                   # resolved thinking
    assert "--reasoning-parser" in fields[4:]               # serve args follow
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_presets.py -k recover_config -v`
Expected: FAIL — argparse exits non-zero (`invalid choice: 'recover-config'`), so `check=True` raises `CalledProcessError`.

- [ ] **Step 3: Implement the subcommand**

In `src/conversation2sql/presets.py`, add the command function (place it just above `def main(`):

```python
def _cmd_recover_config(args: argparse.Namespace) -> None:
    """Emit NUL-delimited fields for a resume launch, read from a run's snapshot:

        provider \\0 model_name \\0 max_model_len \\0 enable_thinking \\0 <vllm serve args...>

    provider comes first so bash decides whether to start a local vLLM server.
    yaml is imported lazily: this command runs under the venv (recover_payload.sh
    sources the venv first), while the module itself stays third-party-free so the
    system python3 can run `server-config`.
    """
    import yaml  # lazy: keep module import third-party-free for system python3

    config_path = os.path.join(args.run_dir, "config.yaml")
    with open(config_path, encoding="utf-8") as f:
        snapshot = yaml.safe_load(f)

    predictor = snapshot["predictor"]
    provider = predictor["model_provider"]
    model_name = predictor["model_name"]
    enable_thinking = predictor.get("enable_thinking")
    baseline = snapshot.get("pipeline", {}).get("baseline", "no_tool")

    profile = profile_for_model_name(model_name)
    think = resolve_effective_thinking(profile, enable_thinking)
    server_args = resolve_server_args(
        profile, think, args.tp, args.dp, args.base_work, baseline
    )
    prof = _require_profile(profile)
    fields = [
        provider,
        model_name,
        str(prof["max_model_len"]),
        "true" if think else "false",
        *server_args,
    ]
    sys.stdout.write("\0".join(fields))
```

Then register it in `main`, right after the `server-config` subparser block (before `args = parser.parse_args(argv)`):

```python
    rc = sub.add_parser(
        "recover-config",
        help="Emit provider + vllm serve config read from a run's config.yaml snapshot.",
    )
    rc.add_argument("--run-dir", required=True, help="Run directory containing config.yaml.")
    rc.add_argument("--tp", type=int, default=1, help="tensor-parallel-size")
    rc.add_argument("--dp", type=int, default=1, help="data-parallel-size")
    rc.add_argument("--base-work", default=os.environ.get("BASE_WORK", ""))
    rc.set_defaults(func=_cmd_recover_config)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_presets.py -k recover_config -v`
Expected: PASS

- [ ] **Step 5: Verify the module still imports under bare system python (no venv)**

Run: `python3 -c "import ast; ast.parse(open('src/conversation2sql/presets.py').read()); print('parse-ok')"`
Expected: prints `parse-ok` (sanity that no top-level third-party import was added).

- [ ] **Step 6: Commit**

```bash
git add src/conversation2sql/presets.py tests/test_presets.py
git commit -m "feat(presets): add recover-config subcommand for resume launches

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 7: Add `bash_scripts/recover_payload.sh`

**Files:**
- Create: `bash_scripts/recover_payload.sh`
- Test: `tests/test_recover_payload_dryrun.py` (new)

- [ ] **Step 1: Write the failing test**

Create `tests/test_recover_payload_dryrun.py`:

```python
import subprocess
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "bash_scripts" / "recover_payload.sh"


def _write_snapshot(run_dir: Path, provider: str):
    run_dir.mkdir(parents=True, exist_ok=True)
    snapshot = {
        "pipeline": {"baseline": "no_tool"},
        "predictor": {
            "model_name": "Qwen/Qwen3.5-9B",
            "model_provider": provider,
            "enable_thinking": None,
        },
        "user_simulator": {"model_name": "x", "model_provider": "openai"},
    }
    (run_dir / "config.yaml").write_text(yaml.safe_dump(snapshot))


def _dry_run(run_dir: Path):
    env = {
        "PATH": "/usr/bin:/bin:/usr/local/bin",
        "DRY_RUN": "1",
        "RESUME_DIR": str(run_dir),
        "BASE_WORK": str(REPO),
    }
    return subprocess.run(
        ["bash", str(SCRIPT)], env=env, capture_output=True, text=True, cwd=str(REPO)
    )


def test_dryrun_emits_resume_run_command(tmp_path):
    run_dir = tmp_path / "run"
    _write_snapshot(run_dir, provider="hosted_vllm")
    out = _dry_run(run_dir).stdout
    assert "--config" in out
    assert str(run_dir / "config.yaml") in out
    assert "--output_folder" in out
    assert str(run_dir) in out
    assert "--resume true" in out
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_recover_payload_dryrun.py -v`
Expected: FAIL — script does not exist yet (non-zero exit, empty stdout).

- [ ] **Step 3: Create the payload script**

Create `bash_scripts/recover_payload.sh` with exactly this content:

```bash
#!/bin/bash
#SBATCH -A wjx@h100
#SBATCH -C h100
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=2
#SBATCH --output=./logs/rl/%j.out
#SBATCH --nodes=1
#SBATCH --qos=qos_gpu_h100-t3
#SBATCH --time=05:00:00
#SBATCH --cpus-per-task=100
#SBATCH --open-mode=append

# ---------------------------------------------------------------------------
# Resume/recover payload: re-run only the missing (instance_id, iteration)
# pairs of a previous eval run, appending into that same run directory.
#
# Inputs (env vars):
#   RESUME_DIR  – the run directory to resume (must contain config.yaml)  (required)
#   TP / DP     – vLLM tensor/data-parallel size                          (default: 1)
#
# It replays the run's own config.yaml snapshot via --config and overrides only
# output_folder, --resume, and (for hosted_vllm) a fresh vLLM api-base. The model
# server profile is reverse-mapped from the snapshot by presets.py recover-config.
#
# Launch via `just recover <run_dir>`. Set DRY_RUN=1 to print and exit.
# ---------------------------------------------------------------------------

set -Eeuo pipefail

export BASE_WORK="${BASE_WORK:-/workspaces/conversation2SQL}"
export MY_SLURM_JOB_ID="${MY_SLURM_JOB_ID:-local}"
RESUME_DIR="${RESUME_DIR:?set RESUME_DIR to the run directory to resume (must contain config.yaml)}"
TP="${TP:-1}"
DP="${DP:-1}"

if [ ! -f "${RESUME_DIR}/config.yaml" ]; then
  echo "[recover_payload] No config.yaml in RESUME_DIR='${RESUME_DIR}'" >&2
  exit 1
fi

# Bring up the venv + shared exports (uv sync, activate .venv, source .env).
# This also means `recover-config` below runs under the venv, where PyYAML exists.
source "${BASE_WORK}/bash_scripts/utils/utils_evaluate.sh"

# Snapshot -> provider + model + max-model-len + thinking + vllm serve args.
mapfile -d '' _RC < <(
  python "${BASE_WORK}/src/conversation2sql/presets.py" recover-config \
    --run-dir "${RESUME_DIR}" \
    --tp "${TP}" --dp "${DP}" \
    --base-work "${BASE_WORK}"
)
if (( ${#_RC[@]} < 5 )) || [ -z "${_RC[0]}" ]; then
  echo "[recover_payload] Failed to resolve recover config from ${RESUME_DIR}/config.yaml" >&2
  exit 1
fi
PROVIDER="${_RC[0]}"
MODEL_NAME="${_RC[1]}"
MAX_MODEL_LEN="${_RC[2]}"
ENABLE_THINKING="${_RC[3]}"
SERVER_ARGS=("${_RC[@]:4}")
export PREDICTOR_MODEL_NAME="${MODEL_NAME}"

RUN_ARGS=(
  --config "${RESUME_DIR}/config.yaml"
  --output_folder "${RESUME_DIR}"
  --resume true
)

if [ "${DRY_RUN:-0}" = "1" ]; then
  echo "[DRY-RUN] recover provider=${PROVIDER} model=${MODEL_NAME} resume_dir=${RESUME_DIR}"
  printf '[DRY-RUN] conv2sql run'; printf ' %s' "${RUN_ARGS[@]}"; printf '\n'
  exit 0
fi

if [ "${PROVIDER}" = "hosted_vllm" ]; then
  source "${BASE_WORK}/bash_scripts/utils/vllm_server.sh"
  start_vllm_server "$MODEL_NAME" "$MAX_MODEL_LEN" "${SERVER_ARGS[@]}"
  RUN_ARGS+=(--predictor_vllm_api_base "${PREDICTOR_VLLM_API_BASE}")
  if [ -n "${USER_SIMULATOR_VLLM_API_BASE:-}" ]; then
    RUN_ARGS+=(--user_simulator_vllm_api_base "${USER_SIMULATOR_VLLM_API_BASE}")
  fi
else
  unset PREDICTOR_VLLM_API_BASE      || true
  unset USER_SIMULATOR_VLLM_API_BASE || true
fi

run_suite() { :; }  # placeholder guard: recovery does not use run_suite's dir creation
uv run conv2sql run "${RUN_ARGS[@]}"
```

Note: the `run_suite() { :; }` line is a deliberate no-op guard so a future reader doesn't accidentally call the dir-creating `run_suite` here; results must go to `RESUME_DIR`. If you prefer, omit it — it has no runtime effect since we never call `run_suite`.

- [ ] **Step 4: Make it executable**

Run: `chmod +x bash_scripts/recover_payload.sh`

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest tests/test_recover_payload_dryrun.py -v`
Expected: PASS

Note on the dry-run path: in DRY_RUN mode the script still sources `utils_evaluate.sh` (which runs `uv sync` and activates the venv) and calls `recover-config`. This is acceptable for the test (the repo venv exists). If `uv sync` is too heavy in CI, the test can be marked slow later; for now it mirrors how `test_eval_payload_dryrun.py` exercises the real script.

- [ ] **Step 6: Commit**

```bash
git add bash_scripts/recover_payload.sh tests/test_recover_payload_dryrun.py
git commit -m "feat(eval): add recover_payload.sh to resume a run from its snapshot

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 8: Add the `just recover` recipe

**Files:**
- Modify: `justfile`

- [ ] **Step 1: Add the dispatch variable**

In `justfile`, just after the existing `dispatch := "..."` line (around line 34), add:

```just
# Same dispatcher, but runs the resume/recover payload instead of eval.
dispatch_recover := "bash bash_scripts/submit_and_log.sh bash_scripts/recover_payload.sh"
```

- [ ] **Step 2: Add the recipe**

In `justfile`, add this recipe (place it after the `eval` recipe block, before `dry`):

```just
# ── recover ────────────────────────────────────────────────────────────────────
# Re-run only the missing/errored (instance_id, iteration) pairs of a previous run,
# appending results into that same run directory. Replays the run's own config.yaml
# snapshot, so the original model/variant/baseline settings are reused automatically.
#
#   just recover results/2026_06_01/13_55_45__no_tool__Qwen3.5-9B__ddl
#
# run_dir must be the leaf directory that contains config.yaml (and results_iter*.jsonl).
[arg("run_dir", help="run directory to resume (the dir holding config.yaml)")]
recover run_dir:
    #!/usr/bin/env bash
    set -Eeuo pipefail
    if [ ! -f "{{run_dir}}/config.yaml" ]; then
        echo "[recover] No config.yaml in '{{run_dir}}' — pass the leaf run dir." >&2
        exit 1
    fi
    export RESUME_DIR="{{run_dir}}"
    if [ "{{runner}}" = "slurm" ]; then
        {{dispatch_recover}} "recover_$(basename '{{run_dir}}')"
    else
        {{dispatch_recover}}
    fi
```

- [ ] **Step 3: Verify the recipe parses and shows up**

Run: `just --list | grep recover`
Expected: a `recover run_dir` line appears.

- [ ] **Step 4: Verify the recipe rejects a bad dir (no submission)**

Run: `just recover /tmp/does-not-exist-xyz; echo "exit=$?"`
Expected: prints the `No config.yaml` error and `exit=1`.

- [ ] **Step 5: Commit**

```bash
git add justfile
git commit -m "feat(just): add 'just recover <run_dir>' resume recipe

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 9: Final verification + docs

**Files:**
- Modify: `.claude/CLAUDE.md` (or the nearest relevant CLAUDE.md) — one-line mention of recover.

- [ ] **Step 1: Run the full test suite**

Run: `uv run pytest tests/`
Expected: PASS (no failures). If any pre-existing unrelated tests fail, note them but do not silence new failures introduced by this work.

- [ ] **Step 2: Type-check**

Run: `uv run pyrefly check`
Expected: no new errors in the files this plan touched.

- [ ] **Step 3: Document the command**

In `.claude/CLAUDE.md`, under the `just` usage section (near the `just eval` bullets), add a line:

```markdown
  just recover <run_dir>    # re-run only the missing/errored instances of a prior run, appending into <run_dir>
```

- [ ] **Step 4: Commit**

```bash
git add .claude/CLAUDE.md
git commit -m "docs: mention 'just recover' for resuming failed eval runs

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Self-review notes (already applied)

- **Spec coverage:** Layer 1 (`resume` flag + `_load_completed_pairs` + skip logic + snapshot preservation) → Tasks 2-4. Snapshot key bug → Task 1. Layer 2 (`recover-config`, `profile_for_model_name`, `recover_payload.sh`, `just recover`) → Tasks 5-8. Testing requirements → embedded per task + Task 9.
- **Type consistency:** `_load_completed_pairs(output_folder: Path, num_iterations: int) -> set[tuple[str,int]]` and `profile_for_model_name(model_name: str) -> str` are referenced with matching signatures everywhere. `recover-config` emits `provider \0 model_name \0 max_model_len \0 thinking \0 <serve args>` and `recover_payload.sh` unpacks indices `[0..3]` + `[4:]` accordingly.
- **Known wart (documented, not fixed):** stale lines in `results_error.jsonl` for now-fixed instances; downstream treats `results_iter*` as source of truth.
