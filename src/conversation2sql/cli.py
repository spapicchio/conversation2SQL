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


@app.command()
def results(
    results_dir: Optional[Path] = typer.Option(None, "--results-dir", help="Path to results directory."),
) -> None:
    """Display evaluation results. (Not yet implemented.)"""
    raise NotImplementedError("The 'results' command is not yet implemented.")
