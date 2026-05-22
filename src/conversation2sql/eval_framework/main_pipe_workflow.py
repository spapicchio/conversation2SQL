import asyncio
import json
import re
import threading
from datetime import datetime
from pathlib import Path
from typing import Callable

import tqdm
import tqdm.asyncio
import yaml

from conversation2sql.config_input import (
    ConfigReader,
    ConfigPredictor,
    ConfigUserSimulator,
    ConfigPipeline,
)
from conversation2sql.eval_framework.agents import (
    run_agent_bird_baseline,
    run_baseline_no_tool,
)
from conversation2sql.eval_framework.agents.utils import utils_create_model
from conversation2sql.eval_framework.dataset_readers import load_bird_interact_as_tasks
from conversation2sql.eval_framework.state import TaskData
from conversation2sql.logger import get_logger

logger = get_logger(__name__)


def _build_run_slug(
    config_pipeline: ConfigPipeline,
    config_reader: ConfigReader,
    config_predictor: ConfigPredictor,
) -> str:
    """Build a short descriptive suffix from key config params for run folder naming.

    Encodes model name, schema type, and enabled flags so each run folder is
    self-documenting. Baseline is a separate directory level, not in the slug.
    """
    parts: list[str] = [str(config_pipeline.baseline)]
    model_slug = re.sub(
        r"[^A-Za-z0-9._-]", "-", config_predictor.model_name.split("/")[-1]
    )
    parts.append(model_slug)
    parts.append(config_reader.database_schema_type)
    if config_reader.is_kb_linearized:
        parts.append("lin")
    if config_reader.read_only_gt_tables:
        parts.append("gt-db")
    if config_reader.read_only_gt_kb:
        parts.append("gt-kb")
    return "__".join(parts)


def _resolve_baseline_settings(baseline: str) -> tuple[bool, Callable, bool]:
    """Map ConfigPipeline.baseline → (make_data_ambiguous, runner, needs_user_sim).

    no_tool    -> clean query, no agent loop, no user-sim
    tools_only -> clean query, agent without ask_user, no user-sim
    tools_user -> clean query, agent with ask_user, user-sim required
    bird_full  -> ambiguous query, agent with ask_user, user-sim required
    """
    table = {
        "no_tool": (False, run_baseline_no_tool, False),
        "tools_only": (False, run_agent_bird_baseline, False),
        "tools_user": (False, run_agent_bird_baseline, True),
        "bird_full": (True, run_agent_bird_baseline, True),
    }
    if baseline not in table:
        raise ValueError(
            f"Unknown baseline: {baseline!r}; expected one of {list(table)}"
        )
    return table[baseline]


def workflow_evaluation_pipeline(
    config_pipeline: ConfigPipeline,
    config_reader: ConfigReader,
    config_predictor: ConfigPredictor,
    config_user: ConfigUserSimulator,
) -> list[dict]:
    logger.info(f"config_pipeline: {config_pipeline}")
    logger.info(f"config_reader: {config_reader}")
    logger.info(f"config_predictor: {config_predictor}")
    logger.info(f"config_user: {config_user}")

    # Resolve baseline settings and override make_data_ambiguous
    forced_amb, runner, needs_user_sim = _resolve_baseline_settings(
        config_pipeline.baseline
    )
    if config_reader.make_data_ambiguous != forced_amb:
        logger.warning(
            "baseline=%s forces make_data_ambiguous=%s; overriding ConfigReader.make_data_ambiguous=%s",
            config_pipeline.baseline,
            forced_amb,
            config_reader.make_data_ambiguous,
        )
    config_reader = config_reader.model_copy(update={"make_data_ambiguous": forced_amb})

    # Build output path: <base>/<baseline>/<YYYY_MM_DD>/<HH_MM_SS>__<slug>/
    # This makes each run folder self-documenting without relying on the caller
    # to embed timestamps or params in the path.
    now = datetime.now()
    slug = _build_run_slug(config_pipeline, config_reader, config_predictor)
    output_folder = (
        Path(config_pipeline.output_folder)
        / now.strftime("%Y_%m_%d")
        / f"{now.strftime('%H_%M_%S')}__{slug}"
    )
    config_pipeline = config_pipeline.model_copy(
        update={"output_folder": str(output_folder)}
    )

    # save config in the output folder
    _save_configs_as_yaml(
        output_folder=output_folder,
        config_pipeline=config_pipeline,
        config_reader=config_reader,
        config_predictor=config_predictor,
        config_user=config_user,
    )

    file_result = output_folder / "results.jsonl"

    # initialize models (API based)
    model_agent, (model_user_parsing, model_user_generator) = _init_models(
        config_predictor,
        config_user,
        needs_user_sim=needs_user_sim,
    )

    # read dataset
    dataset: list[TaskData] = load_bird_interact_as_tasks(**config_reader.model_dump())

    if config_pipeline.debug:
        dataset = dataset[:10]
        logger.info("Debug mode is ON - using only the first 10 tasks from the dataset")

    try:
        result = asyncio.run(
            _run_tasks_concurrently(
                dataset=dataset,
                runner=runner,
                model_agent=model_agent,
                model_user_parsing=model_user_parsing,
                model_user_generator=model_user_generator,
                baseline=config_pipeline.baseline,
                config_predictor=config_predictor,
                config_user=config_user,
                config_pipeline=config_pipeline,
                config_reader=config_reader,
                file_result=file_result,
                concurrency=config_pipeline.concurrency,
            )
        )
    except Exception as e:
        logger.error(f"Error occurred: {e}")
        response_error = {"error": str(e)}
        output = file_result.parent / f"{file_result.stem}_error.jsonl"
        _save_record(response=response_error, output_path_jsonl=output)
        logger.info(f"Saved ERROR to {output}")
        raise e

    return result


async def _run_tasks_concurrently(
    dataset: list[TaskData],
    runner: Callable,
    model_agent,
    model_user_parsing,
    model_user_generator,
    baseline: str,
    config_predictor: ConfigPredictor,
    config_user: ConfigUserSimulator,
    config_pipeline: ConfigPipeline,
    config_reader: ConfigReader,
    file_result: Path,
    concurrency: int,
) -> list[dict]:
    sem = asyncio.Semaphore(concurrency)
    file_lock = threading.Lock()

    async def _process_one(task: TaskData) -> dict:
        async with sem:
            if baseline == "no_tool":
                response = await asyncio.to_thread(runner, task, model_agent)
            else:
                response = await asyncio.to_thread(
                    runner,
                    task,
                    model_agent,
                    model_user_parsing,
                    model_user_generator,
                    enable_ask_user=(baseline in ("tools_user", "bird_full")),
                )
        task_output = {
            "config_predictor": config_predictor.model_dump(),
            "config_user": config_user.model_dump(),
            "config_pipeline": config_pipeline.model_dump(),
            "config_reader": config_reader.model_dump(),
            **task.model_dump(),
            **response,
        }
        with file_lock:
            _save_record(response=task_output, output_path_jsonl=file_result)
        logger.info(f"Saved response for task_id={task.instance_id} to {file_result}")
        return task_output

    coros = [_process_one(task) for task in dataset]
    return list(
        await tqdm.asyncio.tqdm.gather(*coros, desc=f"Inference with {baseline}")
    )


def _init_models(
    config_predictor: ConfigPredictor,
    config_user: ConfigUserSimulator,
    needs_user_sim: bool,
) -> tuple:
    model_agent = utils_create_model(
        model_name=config_predictor.model_name,
        model_provider=config_predictor.model_provider,
        temperature=config_predictor.temperature,
        max_tokens=config_predictor.max_new_tokens,
        top_p=config_predictor.top_p,
        top_k=config_predictor.top_k,
        api_base=config_predictor.predictor_vllm_api_base,
        min_p=config_predictor.min_p,
        presence_penalty=config_predictor.presence_penalty,
        repetition_penalty=config_predictor.repetition_penalty,
        enable_thinking=config_predictor.enable_thinking,
        reasoning_effort=config_predictor.reasoning_effort,
    )
    if not needs_user_sim:
        return model_agent, (None, None)

    model_user_parsing = utils_create_model(
        model_name=config_user.model_name,
        model_provider=config_user.model_provider,
        temperature=config_user.temperature,
        max_tokens=config_user.max_new_tokens,
        api_base=config_user.user_simulator_vllm_api_base,
    )
    model_user_generator = utils_create_model(
        model_name=config_user.model_name,
        model_provider=config_user.model_provider,
        temperature=config_user.temperature,
        max_tokens=config_user.max_new_tokens,
    )
    return model_agent, (model_user_parsing, model_user_generator)


def _save_configs_as_yaml(
    output_folder: Path,
    config_pipeline: ConfigPipeline,
    config_reader: ConfigReader,
    config_predictor: ConfigPredictor,
    config_user: ConfigUserSimulator,
) -> None:
    output_folder.mkdir(parents=True, exist_ok=True)
    configs = {
        "pipeline": config_pipeline.model_dump(mode="json"),
        "reader": config_reader.model_dump(mode="json"),
        "predictor": config_predictor.model_dump(mode="json"),
        "user": config_user.model_dump(mode="json"),
    }
    config_path = output_folder / "config.yaml"
    with config_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(configs, f, sort_keys=False, allow_unicode=True)
    logger.info(f"Saved configs to {config_path}")


def _save_record(response: dict, output_path_jsonl: Path):
    with output_path_jsonl.open(
        "a", encoding="utf-8"
    ) as f:  # "a" = append line by line
        f.write(json.dumps(response, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    _config_pipeline = ConfigPipeline()
    _config_reader = ConfigReader()
    _config_predictor = ConfigPredictor()
    _config_user = ConfigUserSimulator()
    workflow_evaluation_pipeline(
        config_pipeline=_config_pipeline,
        config_reader=_config_reader,
        config_predictor=_config_predictor,
        config_user=_config_user,
    )
