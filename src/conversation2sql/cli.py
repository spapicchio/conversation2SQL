from __future__ import annotations

import warnings
from pathlib import Path
from typing import Optional

import litellm
import typer
from dotenv import load_dotenv

from conversation2sql.cli_parser import PydanticParser
from conversation2sql.presets import expand_presets
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


def _extract_enable_thinking(extra: list[str]) -> bool | None:
    """Pull the --predictor_enable_thinking value out of passthrough args, if present.

    Presets must know the thinking mode *before* PydanticParser parses the
    passthrough flags (it drives which sampling block expand_presets emits), so
    we scan the raw argv for the flag here rather than reading the parsed config.
    """
    for flag in ("--predictor_enable_thinking", "--predictor-enable-thinking"):
        if flag in extra:
            i = extra.index(flag)
            if i + 1 < len(extra):
                return extra[i + 1].strip().lower() in ("1", "true", "t", "yes", "y")
    return None


def _extract_baseline(extra: list[str]) -> str:
    """Pull the --baseline value out of passthrough args (default: no_tool).

    Presets need the baseline to decide the default thinking mode (tool
    baselines default to non-thinking) before PydanticParser runs.
    """
    for flag in ("--baseline", "--pipeline_baseline"):
        if flag in extra:
            i = extra.index(flag)
            if i + 1 < len(extra):
                return extra[i + 1].strip()
    return "no_tool"


@app.command(
    context_settings={
        "allow_extra_args": True,
        "ignore_unknown_options": True,
        "help_option_names": [],
    }
)
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
        model_profile, variant, _extract_enable_thinking(extra), _extract_baseline(extra)
    )
    args = (["--config", str(config)] if config else []) + preset_args + extra
    if help and config is not None:
        args.append("--help")
    parser = PydanticParser(
        [ConfigPipeline, ConfigReader, ConfigPredictor, ConfigUserSimulator]
    )
    cfg = parser.parse_args_and_config(args)
    workflow_evaluation_pipeline(*cfg)


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
