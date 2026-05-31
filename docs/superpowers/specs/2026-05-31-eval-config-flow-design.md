# Eval config flow: CLI-flag bridge + Python presets

**Date:** 2026-05-31
**Status:** Approved design, pending implementation plan

## Problem

Parameters flow `justfile` → `submit_and_log.sh` → `eval_payload.sh` → `conv2sql run`
→ `PydanticParser` → config models. The bash layer exports per-run parameters as
**environment variables** (`BASELINE`, `DATABASE_SCHEMA_TYPE`, `PREDICTOR_MODEL_NAME`,
…), but `PydanticParser`'s priority is `defaults → env → YAML → CLI`. Because the YAML
(`configs/eval_pipeline_config.yaml`) sits **above** env and re-declares those same
fields, every value the bash scripts set is silently overwritten by the static YAML.

Symptoms (all the same root cause — YAML shadows env):
- `just eval --baseline tools_user` runs as `no_tool` (YAML `pipeline.baseline`).
- `run_suite`'s computed `OUTPUT_FOLDER` is overridden by YAML `output_folder: results`.
- `vllm_server.sh`'s discovered `PREDICTOR_VLLM_API_BASE` is overridden by the YAML.

Secondary problems for the stated goal (adding many ablation knobs):
- Adding one tunable touches `config_input.py` + 3 `justfile` recipes + `eval_payload.sh`,
  and depends on an implicit env-name-to-field-name match (with a section-prefix rule
  for ambiguous fields) that fails silently when wrong.
- Real config logic (model profile → sampling params, variant → schema flags) lives in
  bash `case` statements: stringly-typed, untested, duplicated knowledge.

## Goals

- Fix the override bug.
- Make adding a future ablation knob cost **one Pydantic field** and nothing else.
- Move config *logic* (presets) into Python where it is typed and `pytest`-able.
- Keep bash responsible only for what it must own: launching vLLM, dispatch (tmux/SLURM),
  GPUs, API keys.

## Non-goals

- No priority flip in `PydanticParser` (CLI already wins; flipping risks stray-env
  surprises). We route per-run params through the CLI layer instead.
- No bash→Python coupling for server launch (keeps `dry` venv-free and server launch
  independent of Python importing cleanly).
- No change to `main.py`'s behavior; the bash flow uses `conv2sql run`.

## Core idea

Route every per-run parameter to Python as **CLI flags** — the highest-priority layer in
`PydanticParser` — instead of env vars. This sidesteps the YAML-override bug with no
priority change. The YAML demotes to static infra defaults.

Resulting priority, unchanged: `defaults → env → YAML → CLI`. Per-run values arrive as
CLI, so they win.

## Components

### 1. `src/conversation2sql/presets.py` (new)

Two data tables keyed by the exact CLI-dest names `PydanticParser` expects (so expansion
needs no field-name translation):

- `MODEL_PROFILES["qwen35"]` → predictor sampling params: `predictor_model_name`,
  `predictor_temperature`, `predictor_top_p`, `predictor_top_k`,
  `predictor_presence_penalty`, `predictor_repetition_penalty`. qwen's sampling differs by
  thinking mode, so resolution takes the flag:
  `resolve_profile(name: str, enable_thinking: bool) -> list[str]`.
- `VARIANTS["all_db_all_kb"]` → the 4 reader flags: `database_schema_type`,
  `read_only_gt_tables`, `read_only_gt_kb`, `is_kb_linearized`.
  `resolve_variant(name: str) -> list[str]`.
- `expand_presets(model_profile, variant, enable_thinking) -> list[str]` composes both.
- Unknown key → `ValueError` listing valid keys (mirrors current bash behavior).
- Functions return arg lists like `["--predictor_temperature", "0.6", ...]`. Pure → unit
  testable. Reusable from both `cli.py` and (optionally, later) `main.py`.

Format is a Python module, not YAML: the repo is Pydantic/typed throughout, and this gives
type-checking and `pytest` coverage of the config logic.

### 2. `src/conversation2sql/cli.py`

`run` gains two typer options: `--model-profile` and `--variant`. It expands them via
`presets.expand_presets(...)` into a CLI-arg list, **prepends** that list to the user's
passthrough extra args, and hands the whole list to `PydanticParser`. Because argparse
takes the **last** occurrence of a repeated flag, an explicit later
`--predictor_temperature 0.9` (from `--extra`) still overrides the profile value.

Omitting both options reproduces today's behavior — existing `tests/test_cli_run.py`
stays green (backward compatible).

The thinking flag needed by `expand_presets` is read from the forwarded
`--predictor_enable_thinking` value (parsed out of the incoming args, defaulting per the
config default when absent).

### 3. `bash_scripts/eval_payload.sh`

- Model `case` shrinks to **server-only** outputs: `MODEL_NAME`, `MAX_MODEL_LEN`,
  `ENABLE_THINKING`, `DEFAULT_PARAMS`, `SERVER_ARGS` (used to launch / dry-print vLLM).
- **Delete** all `PREDICTOR_*` sampling exports and the entire variant `case` + the
  `DATABASE_SCHEMA_TYPE` / `READ_ONLY_*` / `IS_KB_LINEARIZED` exports.
- Build a `RUN_ARGS` bash array:
  `--model-profile $MODEL --variant $VARIANT --baseline $BASELINE
   --predictor_model_provider $PREDICTOR_MODEL_PROVIDER
   --predictor_enable_thinking $ENABLE_THINKING
   --concurrency $CONCURRENCY --num-iterations $NUM_ITERATIONS --debug $DEBUG`,
  then word-split `$EXTRA` onto the end (`read -ra` — values with embedded spaces are out
  of scope and documented as such).
- Pass the array to `run_suite "${RUN_ARGS[@]}"`.
- `DRY_RUN` path prints the resolved `conv2sql run` command including `RUN_ARGS`.

### 4. `bash_scripts/utils/utils_evaluate.sh` (`run_suite`)

`run_suite` appends `"$@"` to the `conv2sql run` invocation, plus `--output_folder
"$run_dir"`, and `--predictor_vllm_api_base "$PREDICTOR_VLLM_API_BASE"` /
`--user_simulator_vllm_api_base "$USER_SIMULATOR_VLLM_API_BASE"` when those are set
(hosted_vllm). This also fixes the `output_folder` and api-base instances of the override
bug. Drop the `OUTPUT_FOLDER` env export (now a flag).

### 5. `justfile`

`eval` and `dry` gain an `extra=""` arg, exported as `EXTRA` so it rides the
`export -p` env snapshot in `submit_and_log.sh` through to `eval_payload.sh`. `sequential`
forwards `--extra` to each delegated `eval`. Usage:

```
just eval --variant all_db_all_kb --extra "--predictor_top_p 0.8 --reader_user_patience_budget 4"
```

A quoted `--extra` string (not bare `--flag` passthrough): `just 1.51.0` parses loose
`--flags` as recipe options and has no clean `--` passthrough into a variadic.

### 6. YAML cleanup (`configs/eval_pipeline_config.yaml`)

Trim to genuinely static fields: dataset paths, `db_dsn_template`, `user_patience_budget`,
`filter_query_category`, the `user_simulator` block, `max_new_tokens`. **Remove** the
fields now owned by presets/bash (predictor sampling, reader schema flags, `baseline`,
`concurrency`, `num_iterations`, `output_folder`, `predictor_vllm_api_base`) so nothing can
silently shadow the CLI values.

## Data flow (after)

```
just eval --variant V --extra "..."
  → env (transport only): EXTRA, MODEL, VARIANT, BASELINE, PROVIDER, …
  → submit_and_log.sh: export -p snapshot → tmux/sbatch → eval_payload.sh
  → eval_payload.sh:
        bash case → MODEL_NAME/SERVER_ARGS  (launch vLLM)
        bash builds RUN_ARGS=[--model-profile, --variant, --baseline, …] + $EXTRA
  → run_suite "${RUN_ARGS[@]}"
  → conv2sql run --config <static.yaml> --output_folder <run_dir> [--*_vllm_api_base ...] <RUN_ARGS>
  → cli.run: expand presets → prepend → PydanticParser (defaults→env→YAML→CLI; CLI wins)
```

## Testing

- `presets.py`: valid key → expected flag list; qwen thinking vs non-thinking sampling
  differ; unknown key → `ValueError` naming valid keys.
- `cli.run`: `--model-profile qwen35 --variant all_db_all_kb` produces the expected arg
  list passed into `PydanticParser` (extend the existing mock-based tests); a later
  explicit `--predictor_temperature` overrides the profile value.
- `dry` recipe prints a command containing the expanded flags (light smoke check).
- Full `uv run pytest` green; `uv run pyrefly check` clean.

## Cost to add a future ablation knob

Add the Pydantic field to `config_input.py` → done. Use it ad hoc via
`--extra "--my_flag val"`. Promote it to a named `MODEL_PROFILES` / `VARIANTS` entry only
when it becomes a standing axis. Zero `justfile` / bash edits per knob.

## Docs to update

- `.claude/CLAUDE.md` and `src/conversation2sql/CLAUDE.md`: the flow + that per-run params
  now arrive as CLI flags (YAML is static defaults).
- `bash_scripts/README.md`: the `--extra` passthrough and the model/variant preset move.
