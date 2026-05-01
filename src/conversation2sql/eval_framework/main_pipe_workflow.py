import copy
import json
from datetime import datetime, timezone
from pathlib import Path

import tqdm
import yaml
from langchain_litellm import ChatLiteLLM

from conversation2sql.config_input import ConfigReader, ConfigPredictor, ConfigUserSimulator, ConfigPipeline
from conversation2sql.eval_framework.agent import run_agent
from conversation2sql.eval_framework.agent.agent_code_state import CustomAgentState
from conversation2sql.eval_framework.dataset_readers import load_bird_interact_as_tasks
from conversation2sql.eval_framework.state import TaskData
from conversation2sql.logger import get_logger

logger = get_logger(__name__)

LAUNCH_TS = datetime.now(timezone.utc)  # fixed timestamp for this process run
LAUNCH_DAY = LAUNCH_TS.strftime("%Y_%m_%d")
LAUNCH_HOUR = LAUNCH_TS.strftime("%H_%M_%S")


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

    output_folder = config_pipeline.output_folder

    # save config in output folder
    _save_configs_as_yaml(
        output_folder=Path(output_folder),
        config_pipeline=config_pipeline,
        config_reader=config_reader,
        config_predictor=config_predictor,
        config_user=config_user,
    )

    file_result = Path(output_folder) / 'results.jsonl'
    # initialize models (API based)
    model_agent, (model_user_parsing, model_user_generator) = _init_models(config_predictor, config_user)

    # read dataset
    dataset: list[TaskData] = load_bird_interact_as_tasks(**config_reader.model_dump())
    # Process dataset
    i = -1
    result = []
    try:
        for i, task in tqdm.tqdm(enumerate(dataset), desc="Processing dataset"):
            response: CustomAgentState = run_agent(task, model_agent, model_user_parsing, model_user_generator)
            task_output = {
                "config_predictor": config_predictor.model_dump(),
                "config_user": config_user.model_dump(),
                "config_pipeline": config_pipeline.model_dump(),
                "config_reader": config_reader.model_dump(),
                **task.model_dump(),
                **response
            }
            _save_record(response=task_output, output_path_jsonl=file_result)
            logger.info(f"Saved response for task_id={task.instance_id} to {file_result}")

            result.append(task_output)

            if config_pipeline.debug:
                break

    except Exception as e:
        logger.error(f"Error occurred: {e}")
        response_error = {'error': str(e), 'last_processed_task': i}
        output = file_result.parent / f"{file_result.stem}_error.jsonl"
        _save_record(response=response_error, output_path_jsonl=output)
        logger.info(f"Saved ERROR to {output}")
        raise e

    return result


def _create_model(model_name: str, model_provider: str, temperature: float,
                  max_tokens: int, top_p: float | None = None):
    # LiteLLM uses "{provider}/{model}" format
    # https://docs.litellm.ai/docs/providers
    litellm_model = f"{model_provider}/{model_name}"
    kwargs = dict(model=litellm_model, temperature=temperature, max_tokens=max_tokens)
    if top_p is not None:
        kwargs["top_p"] = top_p

    return ChatLiteLLM(**kwargs)


def _init_models(config_predictor: ConfigPredictor,
                 config_user: ConfigUserSimulator) -> tuple:
    model_agent = _create_model(
        model_name=config_predictor.model_name,
        model_provider=config_predictor.model_provider,
        temperature=config_predictor.temperature,
        top_p=config_predictor.top_p,
        max_tokens=config_predictor.max_new_tokens,
    )
    model_user_parsing = _create_model(
        model_name=config_user.model_name,
        model_provider=config_user.model_provider,
        temperature=config_user.temperature,
        max_tokens=config_user.max_new_tokens,
    )
    model_user_generator = _create_model(
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
    # include in the parent dir also the date
    with output_path_jsonl.open("a", encoding="utf-8") as f:  # "a" = append line by line
        f.write(json.dumps(response, ensure_ascii=False) + "\n")

    if 'error' not in response:
        output_path_smaller = output_path_jsonl.parent / f"{output_path_jsonl.stem}_smaller.jsonl"
        keep_vars = [
            "config_predictor",
            "config_user",
            "config_pipeline",
            "config_reader",
            'instance_id', 'selected_database', 'amb_user_query', 'sol_sql', 'sql_query_conditions',
            'not_ambiguos_query', 'gt_knowledge_base', 'category', 'initial_user_patience',
            'updated_user_patience',
            'total_cost',
            'total_tokens',
            'mean_prompt_tokens',
            'mean_completion_tokens',
            'tool_calls_in_order', 'messages',
            'execution_accuracy'
        ]
        smaller_response = {var: copy.deepcopy(response[var]) for var in keep_vars}
        with output_path_smaller.open("a", encoding="utf-8") as f:
            f.write(json.dumps(smaller_response, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    _config_pipeline = ConfigPipeline()
    _config_reader = ConfigReader()
    _config_predictor = ConfigPredictor()
    _config_user = ConfigUserSimulator()
    workflow_evaluation_pipeline(
        config_pipeline=_config_pipeline,
        config_reader=_config_reader,
        config_predictor=_config_predictor,
        config_user=_config_user
    )
