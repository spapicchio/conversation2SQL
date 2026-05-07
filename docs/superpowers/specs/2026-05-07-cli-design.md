# CLI to Launch Experiments — Design Spec

**Date:** 2026-05-07  
**Status:** Approved

---

## Overview

Add a `conv2sql` installed CLI command with two subcommands — `run` and `results` — built on Typer. The existing `main.py` entry point and `PydanticParser` are left untouched; the CLI is a thin layer on top.

---

## Architecture

### New file

`src/conversation2sql/cli.py` — Typer app with two subcommands.

```
src/conversation2sql/
  cli.py          ← new: Typer app (run + results)
  cli_parser.py   ← unchanged
  config_input.py ← unchanged
  main.py         ← unchanged (direct python entrypoint still works)
```

### New dependency

`typer[all]>=0.12` added to `pyproject.toml` dependencies. This pulls in `rich` (table rendering) and `shellingham` (shell-completion support).

### Entry point

```toml
[project.scripts]
conv2sql = "conversation2sql.cli:app"
```

After `uv sync`, `conv2sql` is available in the activated venv.

---

## Subcommand: `run`

### Signature

```
conv2sql run [--config PATH] [EXTRA_ARGS...]
```

### Behaviour

- `--config PATH` is an explicit `typer.Option` that accepts a path to a YAML config file.
- The command is decorated with `typer.Context(allow_extra_args=True, ignore_unknown_options=True)`.
- All remaining args from `ctx.args` are forwarded verbatim to `PydanticParser.parse_args_and_config()`, preserving the full existing override surface (`--baseline`, `--debug`, `--predictor_model_name`, `--reader_user_patience_budget`, etc.).
- The resolved config tuple is passed directly to `workflow_evaluation_pipeline`.

### Help behaviour

- `conv2sql run --help` shows the Typer-generated help (clean, minimal).
- `conv2sql run --config foo.yaml --help` passes `--help` to PydanticParser's argparse, listing every config field with its type and default.

### Implementation sketch

```python
@app.command()
def run(
    ctx: typer.Context,
    config: Optional[Path] = typer.Option(None, "--config", help="Path to YAML config file."),
) -> None:
    extra = ctx.args
    args = (["--config", str(config)] if config else []) + extra
    parser = PydanticParser(
        [ConfigPipeline, ConfigReader, ConfigPredictor, ConfigUserSimulator]
    )
    cfg = parser.parse_args_and_config(args)
    workflow_evaluation_pipeline(*cfg)
```

### Examples

```bash
conv2sql run --config configs/eval_pipeline_config.yaml
conv2sql run --config configs/eval_pipeline_config.yaml --baseline no_tool --debug
conv2sql run --config configs/eval_pipeline_config.yaml --predictor_model_name gpt-4o --predictor_model_provider openai
```

---

## Subcommand: `results`

### Signature

```
conv2sql results [--dir PATH] [--run BASELINE/DATE/TIME]
```

### Options

| Option | Default | Description |
|---|---|---|
| `--dir` | `results` | Root folder to scan for past runs |
| `--run` | `None` | Drill into a specific run (format: `BASELINE/YYYY_MM_DD/HH_MM_SS`) |

### Default behaviour (no `--run`)

Walks `<dir>/<baseline>/<YYYY_MM_DD>/<HH_MM_SS>/` and for each run directory that contains both `results_smaller.jsonl` and `config.yaml`, extracts:

| Column | Source |
|---|---|
| baseline | `config.yaml → pipeline.baseline` |
| date | directory name (`YYYY_MM_DD`) |
| time | directory name (`HH_MM_SS`) |
| model | `config.yaml → predictor.model_name` |
| tasks | count of records in `results_smaller.jsonl` |
| exec_accuracy | mean of `execution_accuracy` field across all records |
| avg_cost | mean of `total_cost` field across all records |

Results are rendered as a `rich.table.Table` and sorted by date/time descending.

### Drill-down behaviour (`--run BASELINE/DATE/TIME`)

Prints the same aggregate columns for the single run, followed by a per-task breakdown table:

| Column | Source |
|---|---|
| instance_id | record field |
| exec_accuracy | record field |
| total_cost | record field |
| submit_msg | last `submit_sql` tool message content (truncated to 80 chars) |

### Examples

```bash
conv2sql results
conv2sql results --dir /tmp/my_results
conv2sql results --run bird_full/2026_05_07/12_00_00
```

---

## `pyproject.toml` changes

```toml
[project.scripts]
conv2sql = "conversation2sql.cli:app"

# Added to [project.dependencies]:
"typer[all]>=0.12",
```

---

## Testing

Two new unit test files, no live DB or API keys required:

### `tests/test_cli_run.py`

Uses `typer.testing.CliRunner`. Mocks `workflow_evaluation_pipeline`. Verifies:

- `conv2sql run --config foo.yaml` calls `workflow_evaluation_pipeline` with a config tuple where `config_pipeline` reflects YAML values.
- Extra flags (e.g. `--baseline no_tool`) are forwarded and override YAML values.
- Missing `--config` falls back to `PydanticParser` defaults without error.

### `tests/test_cli_results.py`

Creates a temporary `results/` tree with fixture `results_smaller.jsonl` + `config.yaml` files. Verifies:

- Default invocation renders a table containing expected baseline name, model, and accuracy.
- `--run` drill-down renders per-task rows.
- `--dir` pointing to a non-existent folder exits with a clear error message.

---

## Constraints & non-goals

- `main.py` is **not removed or modified**. `uv run python main.py` continues to work.
- No interactive prompts or shell wizards — the CLI remains scriptable.
- No `compare` subcommand in this iteration (YAGNI; can be added later).
- `config` validation subcommand is out of scope for this iteration.
