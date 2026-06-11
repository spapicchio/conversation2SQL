# Eval Config Flow (CLI-flag bridge + Python presets) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Route every per-run eval parameter to Python as CLI flags (highest-priority layer), moving model/variant config logic out of bash `case` statements into a typed, tested Python presets module — fixing the YAML-shadows-env override bug and making new ablation knobs cost one Pydantic field.

**Architecture:** A new `presets.py` maps `model-profile`/`variant` names to lists of CLI-flag strings keyed by the exact dest names `PydanticParser` expects. `cli.py run` gains `--model-profile`/`--variant` options that expand presets and prepend them ahead of passthrough args (argparse last-wins lets explicit overrides beat the profile). `eval_payload.sh` shrinks its model `case` to server-launch-only outputs and forwards everything else as CLI flags via a `RUN_ARGS` array threaded through `run_suite "$@"`. The YAML demotes to static defaults.

**Tech Stack:** Python 3.12, Pydantic, Typer, argparse (`PydanticParser`), pytest (`asyncio_mode=auto`), bash, `just` 1.51, run everything via `uv run`.

---

## File Structure

- **Create** `src/conversation2sql/presets.py` — `MODEL_PROFILES`, `VARIANTS` data tables + `resolve_profile`, `resolve_variant`, `expand_presets` pure functions returning CLI-arg lists.
- **Create** `tests/test_presets.py` — unit tests for the resolvers.
- **Modify** `src/conversation2sql/cli.py` — add `--model-profile`/`--variant` to `run`, expand + prepend presets.
- **Modify** `tests/test_cli_run.py` — add preset-expansion tests.
- **Modify** `bash_scripts/eval_payload.sh` — server-only `case`, drop sampling/reader env exports, build `RUN_ARGS`, dry-run prints them.
- **Create** `tests/test_eval_payload_dryrun.py` — GPU-free dry-run integration test of the bash wiring.
- **Modify** `bash_scripts/utils/utils_evaluate.sh` — `run_suite` forwards `"$@"` + `--output_folder` + api-base flags.
- **Modify** `justfile` — `extra=""` arg on `eval`/`dry`, forwarded; `sequential` threads it.
- **Modify** `configs/eval_pipeline_config.yaml` — trim to static fields.
- **Modify** `.claude/CLAUDE.md`, `src/conversation2sql/CLAUDE.md`, `bash_scripts/README.md` — document the new flow.

Reference values (from current `bash_scripts/eval_payload.sh`):

| profile | model_name | max_len | think temp/top_p/top_k/pres/rep | non-think temp/top_p/top_k/pres/rep | default_thinking |
|---|---|---|---|---|---|
| qwen35 | `Qwen/Qwen3.5-9B` | 50000 | 0.6/0.95/20/0.0/1.0 | 1.0/0.95/20/1.5/1.0 | true |
| gemma4-12B | `google/gemma-4-12B-it` | 32000 | 1.0/0.95/64/0.0/1.0 | 1.0/0.95/64/0.0/1.0 | true |

| variant | schema_type | gt_tables | gt_kb | linearized |
|---|---|---|---|---|
| all_db_all_kb | ddl | false | false | false |
| all_db_all_kb_linearized | ddl | false | false | true |
| all_db_toon_all_kb | toon | false | false | false |
| all_db_toon_all_kb_linearized | toon | false | false | true |
| gt_db_all_kb_linearized | ddl | true | false | true |
| gt_db_gt_kb_linearized | ddl | true | true | true |
| gt_db_gt_kb | ddl | true | true | false |

CLI-dest names (predictor sampling fields are ambiguous → `predictor_*`; reader flags are unique → bare): `predictor_model_name`, `predictor_temperature`, `predictor_top_p`, `predictor_top_k`, `predictor_presence_penalty`, `predictor_repetition_penalty`; `database_schema_type`, `read_only_gt_tables`, `read_only_gt_kb`, `is_kb_linearized`.

---

## Task 1: Presets module

**Files:**
- Create: `src/conversation2sql/presets.py`
- Test: `tests/test_presets.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_presets.py
import pytest

from conversation2sql.presets import (
    expand_presets,
    resolve_profile,
    resolve_variant,
)


def test_resolve_variant_flags():
    assert resolve_variant("gt_db_gt_kb_linearized") == [
        "--database_schema_type", "ddl",
        "--read_only_gt_tables", "true",
        "--read_only_gt_kb", "true",
        "--is_kb_linearized", "true",
    ]


def test_resolve_variant_toon():
    flags = resolve_variant("all_db_toon_all_kb")
    assert "--database_schema_type" in flags
    assert flags[flags.index("--database_schema_type") + 1] == "toon"
    assert flags[flags.index("--read_only_gt_tables") + 1] == "false"


def test_resolve_variant_unknown_lists_valid_keys():
    with pytest.raises(ValueError) as exc:
        resolve_variant("nope")
    assert "all_db_all_kb" in str(exc.value)


def test_resolve_profile_qwen_thinking():
    flags = resolve_profile("qwen35", enable_thinking=True)
    assert flags[flags.index("--predictor_model_name") + 1] == "Qwen/Qwen3.5-9B"
    assert flags[flags.index("--predictor_temperature") + 1] == "0.6"
    assert flags[flags.index("--predictor_presence_penalty") + 1] == "0.0"


def test_resolve_profile_qwen_non_thinking():
    flags = resolve_profile("qwen35", enable_thinking=False)
    assert flags[flags.index("--predictor_temperature") + 1] == "1.0"
    assert flags[flags.index("--predictor_presence_penalty") + 1] == "1.5"


def test_resolve_profile_default_thinking_per_model():
    # qwen defaults to thinking, gemma4-12B to thinking
    qwen = resolve_profile("qwen35", enable_thinking=None)
    assert qwen[qwen.index("--predictor_temperature") + 1] == "0.6"
    gemma = resolve_profile("gemma4-12B", enable_thinking=None)
    assert gemma[gemma.index("--predictor_top_k") + 1] == "64"


def test_resolve_profile_unknown_lists_valid_keys():
    with pytest.raises(ValueError) as exc:
        resolve_profile("nope", enable_thinking=True)
    assert "qwen35" in str(exc.value)


def test_expand_presets_composes_profile_then_variant():
    flags = expand_presets("qwen35", "all_db_all_kb", enable_thinking=True)
    assert flags.index("--predictor_model_name") < flags.index("--database_schema_type")


def test_expand_presets_allows_none_selectors():
    # Only a variant, no profile
    assert expand_presets(None, "all_db_all_kb", enable_thinking=None) == resolve_variant(
        "all_db_all_kb"
    )
    # Neither selector → empty
    assert expand_presets(None, None, enable_thinking=None) == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_presets.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'conversation2sql.presets'`

- [ ] **Step 3: Write the implementation**

```python
# src/conversation2sql/presets.py
"""Named presets for eval model profiles and dataset variants.

Each preset resolves to a list of CLI-flag strings keyed by the exact dest
names PydanticParser expects (predictor sampling fields are ambiguous across
ConfigPredictor/ConfigUserSimulator so they carry the `predictor_` prefix; the
reader schema flags are unique so they use bare names). Returning flags — not a
dict — lets cli.py splice presets straight into the argv it hands PydanticParser,
where the CLI layer wins over YAML.
"""
from __future__ import annotations

# --- Model profiles --------------------------------------------------------
# Sampling params differ by thinking mode for qwen; gemma is identical either way.
# `default_thinking` is used when the caller does not pass enable_thinking.
MODEL_PROFILES: dict[str, dict] = {
    "qwen35": {
        "predictor_model_name": "Qwen/Qwen3.5-9B",
        "default_thinking": True,
        "thinking": {
            "predictor_temperature": "0.6",
            "predictor_top_p": "0.95",
            "predictor_top_k": "20",
            "predictor_presence_penalty": "0.0",
            "predictor_repetition_penalty": "1.0",
        },
        "non_thinking": {
            "predictor_temperature": "1.0",
            "predictor_top_p": "0.95",
            "predictor_top_k": "20",
            "predictor_presence_penalty": "1.5",
            "predictor_repetition_penalty": "1.0",
        },
    },
    "gemma4-12B": {
        "predictor_model_name": "google/gemma-4-26B-A4B-it",
        "default_thinking": False,
        "thinking": {
            "predictor_temperature": "1.0",
            "predictor_top_p": "0.95",
            "predictor_top_k": "64",
            "predictor_presence_penalty": "0.0",
            "predictor_repetition_penalty": "1.0",
        },
        "non_thinking": {
            "predictor_temperature": "1.0",
            "predictor_top_p": "0.95",
            "predictor_top_k": "64",
            "predictor_presence_penalty": "0.0",
            "predictor_repetition_penalty": "1.0",
        },
    },
}

# --- Dataset variants ------------------------------------------------------
# Maps each variant to the 4 reader schema flags. Values are "true"/"false"
# strings so PydanticParser's bool coercion handles them.
VARIANTS: dict[str, dict[str, str]] = {
    "all_db_all_kb": {"database_schema_type": "ddl", "read_only_gt_tables": "false", "read_only_gt_kb": "false", "is_kb_linearized": "false"},
    "all_db_all_kb_linearized": {"database_schema_type": "ddl", "read_only_gt_tables": "false", "read_only_gt_kb": "false", "is_kb_linearized": "true"},
    "all_db_toon_all_kb": {"database_schema_type": "toon", "read_only_gt_tables": "false", "read_only_gt_kb": "false", "is_kb_linearized": "false"},
    "all_db_toon_all_kb_linearized": {"database_schema_type": "toon", "read_only_gt_tables": "false", "read_only_gt_kb": "false", "is_kb_linearized": "true"},
    "gt_db_all_kb_linearized": {"database_schema_type": "ddl", "read_only_gt_tables": "true", "read_only_gt_kb": "false", "is_kb_linearized": "true"},
    "gt_db_gt_kb_linearized": {"database_schema_type": "ddl", "read_only_gt_tables": "true", "read_only_gt_kb": "true", "is_kb_linearized": "true"},
    "gt_db_gt_kb": {"database_schema_type": "ddl", "read_only_gt_tables": "true", "read_only_gt_kb": "true", "is_kb_linearized": "false"},
}

# Order in which reader flags are emitted (stable output for tests/readability).
_VARIANT_FLAG_ORDER = (
    "database_schema_type",
    "read_only_gt_tables",
    "read_only_gt_kb",
    "is_kb_linearized",
)


def _dict_to_flags(values: dict[str, str], order: tuple[str, ...] | None = None) -> list[str]:
    keys = order if order is not None else tuple(values)
    flags: list[str] = []
    for key in keys:
        flags.extend([f"--{key}", values[key]])
    return flags


def resolve_variant(name: str) -> list[str]:
    """Return CLI flags for a dataset variant, e.g. ['--database_schema_type', 'ddl', ...]."""
    if name not in VARIANTS:
        raise ValueError(
            f"Unknown variant {name!r}; valid: {', '.join(sorted(VARIANTS))}"
        )
    return _dict_to_flags(VARIANTS[name], _VARIANT_FLAG_ORDER)


def resolve_profile(name: str, enable_thinking: bool | None) -> list[str]:
    """Return CLI flags for a model profile's predictor sampling params.

    When ``enable_thinking`` is None, the profile's ``default_thinking`` is used.
    """
    if name not in MODEL_PROFILES:
        raise ValueError(
            f"Unknown model profile {name!r}; valid: {', '.join(sorted(MODEL_PROFILES))}"
        )
    prof = MODEL_PROFILES[name]
    think = prof["default_thinking"] if enable_thinking is None else enable_thinking
    sampling = prof["thinking" if think else "non_thinking"]
    flags = ["--predictor_model_name", prof["predictor_model_name"]]
    flags.extend(_dict_to_flags(sampling))
    return flags


def expand_presets(
    model_profile: str | None,
    variant: str | None,
    enable_thinking: bool | None,
) -> list[str]:
    """Compose profile + variant flags. Either selector may be None."""
    flags: list[str] = []
    if model_profile is not None:
        flags.extend(resolve_profile(model_profile, enable_thinking))
    if variant is not None:
        flags.extend(resolve_variant(variant))
    return flags
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_presets.py -q`
Expected: PASS (9 passed)

- [ ] **Step 5: Type-check**

Run: `uv run pyrefly check src/conversation2sql/presets.py`
Expected: no errors

- [ ] **Step 6: Commit**

```bash
git add src/conversation2sql/presets.py tests/test_presets.py
git commit -m "feat(presets): model-profile and variant preset resolvers

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 2: Wire presets into `cli.py run`

**Files:**
- Modify: `src/conversation2sql/cli.py:39-61` (the `run` command)
- Test: `tests/test_cli_run.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_cli_run.py`:

```python
def test_run_expands_model_profile_and_variant():
    with (
        patch("conversation2sql.cli.PydanticParser") as mock_cls,
        patch("conversation2sql.cli.workflow_evaluation_pipeline"),
    ):
        mock_parser = MagicMock()
        mock_parser.parse_args_and_config.return_value = (
            MagicMock(), MagicMock(), MagicMock(), MagicMock()
        )
        mock_cls.return_value = mock_parser
        result = runner.invoke(
            app,
            [
                "run", "--config", "foo.yaml",
                "--model-profile", "qwen35",
                "--variant", "all_db_all_kb",
                "--predictor_enable_thinking", "true",
            ],
        )
    assert result.exit_code == 0, result.output
    passed = mock_parser.parse_args_and_config.call_args.args[0]
    # config first, then expanded preset flags, then the passthrough extras
    assert passed[:2] == ["--config", "foo.yaml"]
    assert "--predictor_model_name" in passed
    assert passed[passed.index("--predictor_model_name") + 1] == "Qwen/Qwen3.5-9B"
    assert passed[passed.index("--predictor_temperature") + 1] == "0.6"  # thinking
    assert passed[passed.index("--database_schema_type") + 1] == "ddl"
    # passthrough flag preserved at the end
    assert passed[-2:] == ["--predictor_enable_thinking", "true"]


def test_run_non_thinking_changes_sampling():
    with (
        patch("conversation2sql.cli.PydanticParser") as mock_cls,
        patch("conversation2sql.cli.workflow_evaluation_pipeline"),
    ):
        mock_parser = MagicMock()
        mock_parser.parse_args_and_config.return_value = (
            MagicMock(), MagicMock(), MagicMock(), MagicMock()
        )
        mock_cls.return_value = mock_parser
        result = runner.invoke(
            app,
            ["run", "--model-profile", "qwen35",
             "--predictor_enable_thinking", "false"],
        )
    assert result.exit_code == 0, result.output
    passed = mock_parser.parse_args_and_config.call_args.args[0]
    assert passed[passed.index("--predictor_temperature") + 1] == "1.0"
    assert passed[passed.index("--predictor_presence_penalty") + 1] == "1.5"


def test_run_without_presets_unchanged():
    with (
        patch("conversation2sql.cli.PydanticParser") as mock_cls,
        patch("conversation2sql.cli.workflow_evaluation_pipeline"),
    ):
        mock_parser = MagicMock()
        mock_parser.parse_args_and_config.return_value = (
            MagicMock(), MagicMock(), MagicMock(), MagicMock()
        )
        mock_cls.return_value = mock_parser
        result = runner.invoke(app, ["run", "--config", "foo.yaml", "--baseline", "no_tool"])
    assert result.exit_code == 0, result.output
    mock_parser.parse_args_and_config.assert_called_once_with(
        ["--config", "foo.yaml", "--baseline", "no_tool"]
    )
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_cli_run.py -q`
Expected: FAIL — new tests error on unknown option `--model-profile` (exit_code != 0) / missing expansion.

- [ ] **Step 3: Implement**

In `src/conversation2sql/cli.py`, add the import near the top (after the existing `conversation2sql` imports):

```python
from conversation2sql.presets import expand_presets
```

Add a small helper above `run`:

```python
def _extract_enable_thinking(extra: list[str]) -> bool | None:
    """Pull the --predictor_enable_thinking value out of passthrough args, if present."""
    for flag in ("--predictor_enable_thinking", "--predictor-enable-thinking"):
        if flag in extra:
            i = extra.index(flag)
            if i + 1 < len(extra):
                return extra[i + 1].strip().lower() in ("1", "true", "t", "yes", "y")
    return None
```

Replace the `run` signature and body. Current (`cli.py:39-61`):

```python
def run(
    ctx: typer.Context,
    config: Optional[Path] = typer.Option(None, "--config", help="Path to YAML config file."),
    help: bool = typer.Option(False, "--help", "-h", help="Show this message and exit."),
) -> None:
    """Run an evaluation experiment."""
    load_dotenv(".env")
    litellm.suppress_debug_info = True
    warnings.filterwarnings("ignore", message="Pydantic serializer warnings", category=UserWarning)

    if help and config is None:
        typer.echo(ctx.get_help())
        raise typer.Exit()

    extra = ctx.args
    args = (["--config", str(config)] if config else []) + extra
    if help and config is not None:
        args.append("--help")
    parser = PydanticParser(
        [ConfigPipeline, ConfigReader, ConfigPredictor, ConfigUserSimulator]
    )
    cfg = parser.parse_args_and_config(args)
    workflow_evaluation_pipeline(*cfg)
```

New:

```python
def run(
    ctx: typer.Context,
    config: Optional[Path] = typer.Option(None, "--config", help="Path to YAML config file."),
    model_profile: Optional[str] = typer.Option(
        None, "--model-profile", help="Named model preset (e.g. qwen35, gemma4-12B) — expands to predictor sampling flags."
    ),
    variant: Optional[str] = typer.Option(
        None, "--variant", help="Named dataset variant (e.g. all_db_all_kb) — expands to reader schema flags."
    ),
    help: bool = typer.Option(False, "--help", "-h", help="Show this message and exit."),
) -> None:
    """Run an evaluation experiment."""
    load_dotenv(".env")
    litellm.suppress_debug_info = True
    warnings.filterwarnings("ignore", message="Pydantic serializer warnings", category=UserWarning)

    if help and config is None:
        typer.echo(ctx.get_help())
        raise typer.Exit()

    extra = ctx.args
    # Presets are prepended so explicit passthrough flags (which come later in
    # argv) win — argparse keeps the last occurrence of a repeated flag.
    preset_args = expand_presets(
        model_profile, variant, _extract_enable_thinking(extra)
    )
    args = (["--config", str(config)] if config else []) + preset_args + extra
    if help and config is not None:
        args.append("--help")
    parser = PydanticParser(
        [ConfigPipeline, ConfigReader, ConfigPredictor, ConfigUserSimulator]
    )
    cfg = parser.parse_args_and_config(args)
    workflow_evaluation_pipeline(*cfg)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_cli_run.py -q`
Expected: PASS (all, including the 3 pre-existing tests)

- [ ] **Step 5: Commit**

```bash
git add src/conversation2sql/cli.py tests/test_cli_run.py
git commit -m "feat(cli): expand --model-profile/--variant presets into CLI args

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 3: Shrink `eval_payload.sh` to server-only + RUN_ARGS

**Files:**
- Modify: `bash_scripts/eval_payload.sh`
- Test: `tests/test_eval_payload_dryrun.py`

- [ ] **Step 1: Write the failing dry-run test**

```python
# tests/test_eval_payload_dryrun.py
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "bash_scripts" / "eval_payload.sh"


def _dry_run(**env_overrides):
    env = {
        "PATH": "/usr/bin:/bin",
        "DRY_RUN": "1",
        "MODEL": "qwen35",
        "VARIANT": "all_db_all_kb",
        "BASELINE": "no_tool",
        "PREDICTOR_MODEL_PROVIDER": "hosted_vllm",
        "CONCURRENCY": "16",
        "NUM_ITERATIONS": "1",
    }
    env.update(env_overrides)
    return subprocess.run(
        ["bash", str(SCRIPT)], env=env, capture_output=True, text=True, cwd=str(REPO)
    )


def test_dryrun_emits_profile_and_variant_flags():
    out = _dry_run().stdout
    assert "--model-profile qwen35" in out
    assert "--variant all_db_all_kb" in out
    assert "--baseline no_tool" in out


def test_dryrun_forwards_extra_passthrough():
    out = _dry_run(EXTRA="--predictor_top_p 0.8 --reader_user_patience_budget 4").stdout
    assert "--predictor_top_p 0.8" in out
    assert "--reader_user_patience_budget 4" in out


def test_dryrun_no_longer_exports_reader_schema_env():
    # The reader-flag env exports were removed; the run command carries them as
    # CLI flags via the --variant preset instead.
    out = _dry_run().stdout
    assert "DATABASE_SCHEMA_TYPE=" not in out
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_eval_payload_dryrun.py -q`
Expected: FAIL — current dry-run prints `conv2sql run --config ... (all params read from env)`, not the `--model-profile`/`--variant`/`EXTRA` flags.

- [ ] **Step 3: Edit `eval_payload.sh`**

(a) In the model `case`, **delete the predictor sampling export block** (`eval_payload.sh:101-110`):

```bash
# Export predictor params with the names PydanticParser expects from the env.
# Ambiguous fields (shared with ConfigUserSimulator) get the section prefix.
export PREDICTOR_MODEL_NAME="${MODEL_NAME}"
export PREDICTOR_MODEL_PROVIDER="${PREDICTOR_MODEL_PROVIDER}"
export PREDICTOR_TEMPERATURE="${TEMPERATURE}"
export PREDICTOR_TOP_P="${TOP_P}"
export PREDICTOR_TOP_K="${TOP_K}"
export PREDICTOR_PRESENCE_PENALTY="${PRESENCE_PENALTY}"
export PREDICTOR_REPETITION_PENALTY="${REPETITION_PENALTY}"
export PREDICTOR_ENABLE_THINKING="${ENABLE_THINKING}"
```

Replace it with just the bits the bash side still needs (server build_run_slug reads `PREDICTOR_MODEL_NAME`; the provider is forwarded as a flag later):

```bash
# Only the model NAME is still needed bash-side (build_run_slug + vllm serve).
# All predictor sampling params now come from the Python --model-profile preset.
export PREDICTOR_MODEL_NAME="${MODEL_NAME}"
```

(b) **Delete the entire variant `case` block and reader exports** (`eval_payload.sh:113-139`, from the `# Variant -> the 4 run_suite condition flags.` comment through `export IS_KB_LINEARIZED="${IS_LIN}"`). The variant → schema-flag mapping now lives in `presets.py`. Keep nothing from this block.

(c) Replace the pipeline-export block (`eval_payload.sh:141-145`) and build the run-args array. Replace:

```bash
# Export pipeline params (all unique fields, no section prefix needed).
export BASELINE="${BASELINE}"
export CONCURRENCY="${CONCURRENCY}"
export NUM_ITERATIONS="${NUM_ITERATIONS}"
export DEBUG="${DEBUG}"
```

with:

```bash
# ---------------------------------------------------------------------------
# Build the CLI args forwarded to `conv2sql run`. Everything per-run goes
# through the CLI (highest-priority layer) so the static YAML cannot shadow it.
# The model profile + variant expand to predictor/reader flags inside Python.
# ---------------------------------------------------------------------------
RUN_ARGS=(
  --model-profile "${MODEL}"
  --variant "${VARIANT}"
  --baseline "${BASELINE}"
  --predictor_model_provider "${PREDICTOR_MODEL_PROVIDER}"
  --predictor_enable_thinking "${ENABLE_THINKING}"
  --concurrency "${CONCURRENCY}"
  --num-iterations "${NUM_ITERATIONS}"
  --debug "${DEBUG}"
)
# Ad-hoc ablation overrides from `just ... --extra "..."`. Word-split on spaces;
# values containing spaces are out of scope.
if [ -n "${EXTRA:-}" ]; then
  read -ra _EXTRA_ARR <<< "${EXTRA}"
  RUN_ARGS+=("${_EXTRA_ARR[@]}")
fi
```

(d) Replace the dry-run block (`eval_payload.sh:151-157`):

```bash
if [ "${DRY_RUN:-0}" = "1" ]; then
  echo "[DRY-RUN] model=${MODEL} variant=${VARIANT} baseline=${BASELINE} provider=${PREDICTOR_MODEL_PROVIDER} concurrency=${CONCURRENCY} num_iterations=${NUM_ITERATIONS} gpus=${CUDA_VISIBLE_DEVICES}"
  printf '[DRY-RUN] vllm serve %q --max-model-len %q' "${MODEL_NAME}" "${MAX_MODEL_LEN}"
  printf ' %q' "${SERVER_ARGS[@]}"; printf '\n'
  echo "[DRY-RUN] conv2sql run --config configs/eval_pipeline_config.yaml  (all params read from env)"
  exit 0
fi
```

with (prints the resolved run args):

```bash
if [ "${DRY_RUN:-0}" = "1" ]; then
  echo "[DRY-RUN] model=${MODEL} variant=${VARIANT} baseline=${BASELINE} provider=${PREDICTOR_MODEL_PROVIDER} concurrency=${CONCURRENCY} num_iterations=${NUM_ITERATIONS} gpus=${CUDA_VISIBLE_DEVICES}"
  printf '[DRY-RUN] vllm serve %q --max-model-len %q' "${MODEL_NAME}" "${MAX_MODEL_LEN}"
  printf ' %q' "${SERVER_ARGS[@]}"; printf '\n'
  printf '[DRY-RUN] conv2sql run --config configs/eval_pipeline_config.yaml'
  printf ' %s' "${RUN_ARGS[@]}"; printf '\n'
  exit 0
fi
```

(e) Replace the final `run_suite` call (`eval_payload.sh:178`):

```bash
run_suite
```

with:

```bash
run_suite "${RUN_ARGS[@]}"
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_eval_payload_dryrun.py -q`
Expected: PASS (3 passed)

Also smoke-check by eye:
Run: `just dry --variant gt_db_gt_kb --extra "--predictor_top_k 30"`
Expected: a `[DRY-RUN] conv2sql run ...` line containing `--model-profile qwen35 --variant gt_db_gt_kb ... --predictor_top_k 30`.

- [ ] **Step 5: Commit**

```bash
git add bash_scripts/eval_payload.sh tests/test_eval_payload_dryrun.py
git commit -m "refactor(eval): forward run params as CLI flags, drop env exports

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 4: `run_suite` forwards args + fixes output/api-base shadowing

**Files:**
- Modify: `bash_scripts/utils/utils_evaluate.sh` (the `run_suite` function)

- [ ] **Step 1: Edit `run_suite`**

Replace the invocation block at the end of `run_suite` (the `CUDA_VISIBLE_DEVICES=... uv run conv2sql run --config ...` lines):

```bash
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
  TOKENIZERS_PARALLELISM=true \
  VLLM_WORKER_MULTIPROC_METHOD=spawn \
  OUTPUT_FOLDER="${run_dir}" \
  uv run conv2sql run \
    --config "${BASE_WORK}/configs/eval_pipeline_config.yaml"
```

with:

```bash
  # Per-run params arrive as CLI flags (passed through from eval_payload's
  # RUN_ARGS via "$@"). output_folder and the vLLM api-base are computed here,
  # so pass them as flags too — this is also what stops the static YAML from
  # shadowing them.
  local extra_flags=("$@")
  extra_flags+=(--output_folder "${run_dir}")
  if [ -n "${PREDICTOR_VLLM_API_BASE:-}" ]; then
    extra_flags+=(--predictor_vllm_api_base "${PREDICTOR_VLLM_API_BASE}")
  fi
  if [ -n "${USER_SIMULATOR_VLLM_API_BASE:-}" ]; then
    extra_flags+=(--user_simulator_vllm_api_base "${USER_SIMULATOR_VLLM_API_BASE}")
  fi

  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
  TOKENIZERS_PARALLELISM=true \
  VLLM_WORKER_MULTIPROC_METHOD=spawn \
  uv run conv2sql run \
    --config "${BASE_WORK}/configs/eval_pipeline_config.yaml" \
    "${extra_flags[@]}"
```

Note: `BASELINE` is still read by `build_run_slug` from the environment (eval_payload still has `BASELINE` in scope), so the slug is unaffected.

- [ ] **Step 2: Verify the dry-run path is unaffected and tests still green**

Run: `uv run pytest tests/test_eval_payload_dryrun.py -q`
Expected: PASS (run_suite is not reached in dry-run, but confirm no regression)

- [ ] **Step 3: Commit**

```bash
git add bash_scripts/utils/utils_evaluate.sh
git commit -m "fix(eval): pass output_folder + vllm api-base as flags, forward run args

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 5: `justfile` `--extra` passthrough

**Files:**
- Modify: `justfile` (the `eval`, `dry`, and `sequential` recipes)

- [ ] **Step 1: Add `extra` to `eval`**

In the `eval` recipe, add the arg attribute after the `num_iterations` `[arg(...)]` line:

```
[arg("extra", long="extra", help="extra flags forwarded verbatim to `conv2sql run`, quoted (e.g. --extra \"--predictor_top_p 0.8\")")]
```

Add `extra=""` to the recipe signature (after `num_iterations="1"`):

```
eval variant="all_db_all_kb" model="qwen35" gpus="1" debug="false" provider="hosted_vllm" baseline="no_tool" concurrency="16" num_iterations="1" extra="":
```

In the `eval` recipe body, add this export alongside the other `export` lines (after `export DEBUG="{{debug}}"`):

```bash
    export EXTRA="{{extra}}"
```

- [ ] **Step 2: Add `extra` to `dry`**

Add the arg attribute after the `dry` recipe's `num_iterations` `[arg(...)]`:

```
[arg("extra", long="extra", help="extra flags forwarded verbatim to `conv2sql run`")]
```

Change the `dry` signature and body to thread `EXTRA`:

```
dry variant="all_db_all_kb" model="qwen35" provider="hosted_vllm" baseline="no_tool" concurrency="16" num_iterations="1" extra="":
    DRY_RUN=1 MODEL="{{model}}" VARIANT="{{variant}}" BASELINE="{{baseline}}" PREDICTOR_MODEL_PROVIDER="{{provider}}" CONCURRENCY="{{concurrency}}" NUM_ITERATIONS="{{num_iterations}}" EXTRA="{{extra}}" bash bash_scripts/eval_payload.sh
```

- [ ] **Step 3: Thread `extra` through `sequential`**

Add the arg attribute after `sequential`'s `num_iterations` `[arg(...)]`:

```
[arg("extra", long="extra", help="extra flags forwarded verbatim to each variant's `conv2sql run`")]
```

Add `extra=""` to the `sequential` signature (before `*variants`):

```
sequential model="qwen35" gpus="1" provider="hosted_vllm" baseline="no_tool" concurrency="16" num_iterations="1" extra="" *variants:
```

In the `sequential` body, update the delegated `just eval` call to forward `--extra` (the line beginning `if ! out=$(just eval variant=...`):

```bash
        if ! out=$(just eval variant="${variant}" model="{{model}}" gpus="{{gpus}}" provider="{{provider}}" baseline="{{baseline}}" concurrency="{{concurrency}}" --num-iterations "{{num_iterations}}" --extra "{{extra}}" 2>&1); then
```

- [ ] **Step 4: Verify recipes parse and the extra flows through**

Run: `just dry --variant all_db_all_kb --extra "--predictor_top_p 0.8"`
Expected: dry-run output line contains `--model-profile qwen35 --variant all_db_all_kb` and `--predictor_top_p 0.8`.

Run: `just --list`
Expected: recipes list prints without a parse error.

- [ ] **Step 5: Commit**

```bash
git add justfile
git commit -m "feat(just): --extra passthrough for ad-hoc ablation flags

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 6: Trim YAML + update docs

**Files:**
- Modify: `configs/eval_pipeline_config.yaml`
- Modify: `.claude/CLAUDE.md`, `src/conversation2sql/CLAUDE.md`, `bash_scripts/README.md`

- [ ] **Step 1: Trim the YAML to static fields**

Replace `configs/eval_pipeline_config.yaml` with (removes predictor sampling, reader schema flags, baseline/concurrency/num_iterations/output_folder, predictor_vllm_api_base — all now CLI-supplied; keeps static infra + user-sim defaults):

```yaml
# eval_pipeline_config.yaml
# STATIC defaults for the conversation2SQL evaluation pipeline.
#
# Per-run knobs (baseline, model sampling, dataset variant/schema flags,
# concurrency, num_iterations, output_folder, vllm api-base) are NOT here —
# they arrive as CLI flags from `just`/`eval_payload.sh` (the highest-priority
# layer in PydanticParser). Putting them here would silently shadow those flags.

reader:
  dataset_name_jsonl: 'data/bird_interact/bird-interact-lite/bird_interact_data_GT.jsonl'
  dataset_path: 'data/bird_interact/bird-interact-lite'
  filter_query_category: true
  db_dsn_template: 'postgresql://root:123123@localhost:5432/{database}' # Lite DB
  user_patience_budget: 1

predictor:
  max_new_tokens: 20000

user_simulator:
  model_name: 'gpt-5.4-mini-2026-03-17'
  model_provider: 'openai'
  temperature: 1.0
  max_new_tokens: 500
```

- [ ] **Step 2: Smoke-test that a real (non-dry) parse still builds configs**

Run: `uv run conv2sql run --config configs/eval_pipeline_config.yaml --model-profile qwen35 --variant all_db_all_kb --baseline no_tool --debug false --help`
Expected: exits cleanly (help path) with no parser error — confirms the trimmed YAML + preset flags parse together.

- [ ] **Step 3: Update docs**

In `.claude/CLAUDE.md`, under the eval/launch section, add a sentence:

```
Per-run parameters are passed to `conv2sql run` as **CLI flags** (the highest-priority
layer): `eval_payload.sh` builds a `RUN_ARGS` array (`--model-profile`, `--variant`,
`--baseline`, …) and `just eval ... --extra "<flags>"` forwards ad-hoc overrides.
Model sampling and variant→schema-flag mappings live in `src/conversation2sql/presets.py`,
not in bash. `configs/eval_pipeline_config.yaml` holds only static defaults.
```

In `src/conversation2sql/CLAUDE.md`, under "Gotchas", append:

```
- `presets.py` maps `--model-profile`/`--variant` names to CLI flags; `cli.py run`
  expands and prepends them so explicit `--extra` overrides win (argparse last-wins).
- Config priority is unchanged (defaults → env → YAML → CLI); per-run values arrive as
  CLI, so the YAML (now static-only) never shadows them.
```

In `bash_scripts/README.md`, add a short note (near the model×variant matrix) that model sampling params now come from `presets.py` and that `eval_payload.sh`'s `case` only builds the `vllm serve` args; mention the `--extra` passthrough.

- [ ] **Step 4: Full test + type-check**

Run: `uv run pytest`
Expected: all pass (per CLAUDE.md "always run pytest after changes").

Run: `uv run pyrefly check`
Expected: clean.

- [ ] **Step 5: Commit**

```bash
git add configs/eval_pipeline_config.yaml .claude/CLAUDE.md src/conversation2sql/CLAUDE.md bash_scripts/README.md
git commit -m "docs+config: trim YAML to static defaults, document CLI-flag flow

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Self-Review notes

- **Spec coverage:** presets.py (Task 1) ✓; cli.py expansion (Task 2) ✓; eval_payload server-only + RUN_ARGS + EXTRA (Task 3) ✓; run_suite forwards args + output_folder/api-base fix (Task 4) ✓; justfile --extra (Task 5) ✓; YAML trim + docs (Task 6) ✓. Non-goal "no priority flip" honored (no PydanticParser change). `main.py` left unchanged per spec.
- **Override semantics:** presets prepended, passthrough appended → argparse last-wins gives explicit overrides priority (Task 2 test `test_run_expands_...` asserts the `--config` head and passthrough tail).
- **Naming consistency:** dest names `predictor_*` (ambiguous) and bare reader flags are used identically in presets.py, the cli tests, and eval_payload's `--predictor_model_provider`/`--predictor_enable_thinking`.
- **Known limitation (documented):** `EXTRA` is word-split on spaces; flag values containing spaces are unsupported.
