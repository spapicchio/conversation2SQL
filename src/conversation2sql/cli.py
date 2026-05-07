from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Optional

import litellm
import typer
import yaml
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

from conversation2sql.cli_parser import PydanticParser
from conversation2sql.config_input import (
    ConfigPipeline,
    ConfigPredictor,
    ConfigReader,
    ConfigUserSimulator,
)
from conversation2sql.eval_framework.main_pipe_workflow import workflow_evaluation_pipeline
from conversation2sql.eval_framework.main_run_analysis import (
    print_summary_tables,
    workflow_classification_pipeline,
)

app = typer.Typer(help="conversation2SQL — evaluate LLM agents on BIRD-Interact.")
console = Console()


@app.command(context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def run(
    ctx: typer.Context,
    config: Optional[Path] = typer.Option(None, "--config", help="Path to YAML config file."),
) -> None:
    """Run an evaluation experiment."""
    load_dotenv(".env")
    litellm.suppress_debug_info = True
    warnings.filterwarnings("ignore", message="Pydantic serializer warnings", category=UserWarning)

    extra = ctx.args
    args = (["--config", str(config)] if config else []) + extra
    parser = PydanticParser(
        [ConfigPipeline, ConfigReader, ConfigPredictor, ConfigUserSimulator]
    )
    cfg = parser.parse_args_and_config(args)
    workflow_evaluation_pipeline(*cfg)


def _load_run(run_dir: Path) -> tuple[dict, list[dict]] | None:
    config_path = run_dir / "config.yaml"
    jsonl_path = run_dir / "results_smaller.jsonl"
    if not config_path.exists() or not jsonl_path.exists():
        return None
    config = yaml.safe_load(config_path.read_text())
    records = [
        json.loads(line)
        for line in jsonl_path.read_text().splitlines()
        if line.strip()
    ]
    return config, records


def _show_summary_table(results_dir: Path) -> None:
    if not results_dir.exists():
        typer.echo(f"Results directory not found: {results_dir}", err=True)
        raise typer.Exit(1)

    rows = []
    for baseline_dir in sorted(results_dir.iterdir()):
        if not baseline_dir.is_dir():
            continue
        for date_dir in sorted(baseline_dir.iterdir()):
            if not date_dir.is_dir():
                continue
            for time_dir in sorted(date_dir.iterdir()):
                if not time_dir.is_dir():
                    continue
                loaded = _load_run(time_dir)
                if loaded is None or not loaded[1]:
                    continue
                config, records = loaded
                n = len(records)
                exec_acc = sum(r.get("execution_accuracy", 0) for r in records) / n
                avg_cost = sum(r.get("total_cost", 0) for r in records) / n
                rows.append((
                    config.get("pipeline", {}).get("baseline", "?"),
                    date_dir.name,
                    time_dir.name,
                    config.get("predictor", {}).get("model_name", "?"),
                    n,
                    exec_acc,
                    avg_cost,
                ))

    rows.sort(key=lambda r: (r[1], r[2]), reverse=True)

    table = Table(title="Evaluation Results")
    for col, justify in [
        ("baseline", "left"), ("date", "left"), ("time", "left"),
        ("model", "left"), ("tasks", "right"), ("exec_acc", "right"), ("avg_cost", "right"),
    ]:
        table.add_column(col, justify=justify)

    for baseline, date, time_s, model, n, exec_acc, avg_cost in rows:
        table.add_row(baseline, date, time_s, model, str(n), f"{exec_acc:.2f}", f"{avg_cost:.4f}")

    console.print(table)


def _submit_msg(record: dict) -> str:
    for msg in reversed(record.get("messages", [])):
        if isinstance(msg, dict) and msg.get("name") == "submit_sql":
            return str(msg.get("content", ""))[:80]
    return ""


def _show_run_drilldown(results_dir: Path, run_spec: str) -> None:
    run_dir = results_dir / Path(run_spec)
    loaded = _load_run(run_dir)
    if loaded is None:
        typer.echo(f"Run not found or incomplete: {run_dir}", err=True)
        raise typer.Exit(1)

    config, records = loaded
    n = len(records)
    exec_acc = sum(r.get("execution_accuracy", 0) for r in records) / n if n else 0
    avg_cost = sum(r.get("total_cost", 0) for r in records) / n if n else 0
    baseline = config.get("pipeline", {}).get("baseline", "?")
    model = config.get("predictor", {}).get("model_name", "?")

    console.print(
        f"[bold]{baseline}[/bold]  model={model}  tasks={n}"
        f"  exec_acc={exec_acc:.2f}  avg_cost={avg_cost:.4f}"
    )

    table = Table(title=f"Per-task: {run_spec}")
    table.add_column("instance_id")
    table.add_column("exec_accuracy", justify="right")
    table.add_column("total_cost", justify="right")
    table.add_column("submit_msg")

    for r in records:
        table.add_row(
            str(r.get("instance_id", "?")),
            str(r.get("execution_accuracy", "?")),
            f"{r.get('total_cost', 0):.4f}",
            _submit_msg(r),
        )

    console.print(table)


@app.command()
def results(
    dir: Path = typer.Option(Path("results"), "--dir", help="Results folder root."),
    run: Optional[str] = typer.Option(
        None, "--run", help="Drill into a specific run: BASELINE/YYYY_MM_DD/HH_MM_SS."
    ),
) -> None:
    """Inspect past evaluation results."""
    if run:
        _show_run_drilldown(dir, run)
    else:
        _show_summary_table(dir)


@app.command("classify-turns")
def classify_turns(
    inputs: list[Path] = typer.Argument(..., help="Input results_smaller.jsonl file(s)."),
    output: Optional[Path] = typer.Option(None, "--output", help="Output JSONL path."),
    model: str = typer.Option(
        "openai/gpt-4o-mini",
        "--model",
        help="LiteLLM model string (provider/model).",
    ),
    tool_categories: Path = typer.Option(
        Path("configs/turn_classifier/tool_categories.yaml"),
        "--tool-categories",
        help="Path to tool_categories.yaml.",
    ),
) -> None:
    """Classify agent turns in BIRD-Interact result traces and show aggregate statistics."""
    load_dotenv(".env")
    if output is None:
        output = (
            inputs[0].parent / "results_classified.jsonl"
            if len(inputs) == 1
            else Path("results_classified.jsonl")
        )
    summary = workflow_classification_pipeline(inputs, output, model, tool_categories)
    print_summary_tables(summary, output)
