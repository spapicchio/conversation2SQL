# Design: main_run_analysis.py + classify-turns CLI command

**Date:** 2026-05-07  
**Branch:** feat/baselines-ablation  
**Status:** Approved

---

## Overview

Add a `main_run_analysis.py` orchestration module to `src/conversation2sql/eval_framework/` and a `classify-turns` command to the existing `typer` CLI in `src/conversation2sql/cli.py`.

The command classifies every AI turn in one or more `results_smaller.jsonl` files using the two-level `TurnClassifier` (Level 2 rules + Level 1 LLM judge), writes enriched JSONL output, and prints aggregate statistics as Rich tables.

---

## Architecture

Follows the existing pattern:

- `eval_framework/main_pipe_workflow.py` ← orchestrates the evaluation pipeline  
- `eval_framework/main_run_analysis.py` ← orchestrates the classification analysis  
- `cli.py run` ← thin wrapper calling `workflow_evaluation_pipeline`  
- `cli.py classify-turns` ← thin wrapper calling `workflow_classification_pipeline`

---

## `main_run_analysis.py`

### Public API

```python
def workflow_classification_pipeline(
    inputs: list[Path],
    output: Path,
    model_str: str,                # "provider/model" LiteLLM format
    tool_categories_path: Path,
) -> AnalysisSummary:
```

### Steps

1. **Load** tool categories from `tool_categories_path` via `load_tool_categories`.
2. **Build** model via `utils_create_model` (temperature=0.0, max_tokens=1024); wrap with `TurnClassifier`.
3. **Iterate** over each `input` file:
   - Parse records (handle both compact JSONL and pretty-printed JSON, same logic as `scripts/classify_turns.py`).
   - For each record, walk messages in order:
     - On `role=="ai"`: call `classifier.classify_turn(message_index, ai_msg, prior_failed_submit)`.
     - On `role=="tool"` with `tool_name=="submit_sql"`: update `prior_failed_submit` if `passed==False`.
   - Append `{"turn_classifications": [...]}` to the record and write to `output`.
4. **Collect aggregates** incrementally:
   - `l2_counts: Counter[str]` — counts per Level 2 category
   - `l1_counts: Counter[str]` — counts per Level 1 category
   - `confidence_counts: Counter[str]` — counts per confidence label
   - `per_instance: dict[str, InstanceStats]` — per instance_id: n_turns, l1_counts
5. **Return** `AnalysisSummary(l2_counts, l1_counts, confidence_counts, per_instance, total_records, total_errors)`.

### `AnalysisSummary` dataclass

```python
@dataclass
class InstanceStats:
    n_turns: int
    l1_counts: Counter[str]

@dataclass
class AnalysisSummary:
    l2_counts: Counter[str]
    l1_counts: Counter[str]
    confidence_counts: Counter[str]
    per_instance: dict[str, InstanceStats]
    total_records: int
    total_errors: int
```

### Error handling

- Unreadable input files: print to stderr, increment `total_errors`, continue.
- Per-record classification errors: print `ERROR [instance_id]: ...` to stderr, increment `total_errors`, skip the record (do not write partial output).

### Output path default

If `output` is `None` at the call site and only one input is given, default to `input.parent / "results_classified.jsonl"`; otherwise default to `Path("results_classified.jsonl")`. This default is computed in `cli.py` before calling the pipeline function.

---

## `cli.py` — `classify-turns` command

```python
@app.command("classify-turns")
def classify_turns(
    inputs: list[Path] = typer.Argument(..., help="Input results_smaller.jsonl file(s)."),
    output: Optional[Path] = typer.Option(None, "--output", help="Output JSONL path."),
    model: str = typer.Option("openai/gpt-4o-mini", "--model", help="LiteLLM model string."),
    tool_categories: Path = typer.Option(
        Path("configs/turn_classifier/tool_categories.yaml"),
        "--tool-categories",
        help="Path to tool_categories.yaml.",
    ),
) -> None:
    """Classify agent turns in BIRD-Interact result traces and show aggregate statistics."""
```

The handler:
1. Resolves the output path default.
2. Calls `workflow_classification_pipeline(inputs, output, model, tool_categories)`.
3. Prints four Rich tables using `console.print`:
   - **L2 Category Distribution** — category, count, %
   - **L1 Category Distribution** — category, count, %
   - **Confidence Distribution** — level, count, %
   - **Per-Instance Breakdown** — instance_id, n_turns, top L1 category (by count)
4. Prints a summary line: `Done — {total_records} records, {total_errors} errors → {output}`.

---

## Rich table format

Each distribution table has three columns: `category` (left), `count` (right), `%` (right).  
The per-instance table has four columns: `instance_id` (left), `n_turns` (right), `top_l1` (left), `top_l1_count` (right).

---

## What does NOT change

- `scripts/classify_turns.py` is left as-is (standalone script still works).
- No changes to `TurnClassifier`, `Level1Classification`, `TurnClassification`, or `rules.py`.
- No changes to existing CLI commands (`run`, `results`).

---

## Testing

No new test files are required by this design. The orchestration logic in `main_run_analysis.py` relies on `TurnClassifier` which is already tested. Integration can be verified manually with:

```bash
uv run c2sql classify-turns results/bird_full/*/results_smaller.jsonl --model openai/gpt-4o-mini
```
