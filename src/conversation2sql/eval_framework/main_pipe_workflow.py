import json
from pathlib import Path

import tqdm
from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel

from conversation2sql.config_input import ConfigReader, ConfigPredictor, ConfigUserSimulator, ConfigPipeline
from conversation2sql.eval_framework.agent import run_agent
from conversation2sql.eval_framework.agent.agent_code_state import CustomAgentState
from conversation2sql.eval_framework.dataset_readers import load_bird_interact_as_tasks
from conversation2sql.eval_framework.state import TaskData
from conversation2sql.logger import get_logger

logger = get_logger(__name__)

def workflow_evaluation_pipeline(
        config_pipeline: ConfigPipeline,
        config_reader: ConfigReader,
        config_predictor: ConfigPredictor,
        config_user: ConfigUserSimulator,
) -> TaskData:
    logger.info(f"config_pipeline: {config_pipeline}")
    logger.info(f"config_reader: {config_reader}")
    logger.info(f"config_predictor: {config_predictor}")
    logger.info(f"config_user: {config_user}")

    # initialize models (API based)
    model_agent, (model_user_parsing, model_user_generator) = _init_models(config_predictor, config_user)
    # read dataset
    dataset: list[TaskData] = load_bird_interact_as_tasks(**config_reader.model_dump())

    # Process dataset
    for task in tqdm.tqdm(dataset, desc="Processing dataset"):
        response: CustomAgentState = run_agent(model_agent, model_user_parsing, model_user_generator, task,
                                               config_predictor=config_predictor)
        _save_record(response=response, output_path_jsonl=config_pipeline.output)
        if config_pipeline.debug:
            break

    return ...


def _init_models(config_predictor: ConfigPredictor,
                 config_user: ConfigUserSimulator) -> tuple[BaseChatModel, ...]:
    # https://reference.langchain.com/python/langchain/chat_models/base/init_chat_model

    model_agent = init_chat_model(
        model=config_predictor.model_name,
        model_provider=config_predictor.model_provider,
        temperature=config_predictor.temperature,
        top_p=config_predictor.top_p,
        max_tokens=config_predictor.max_new_tokens,
    )

    model_user_parsing = init_chat_model(
        model=config_user.model_name,
        model_provider=config_user.model_provider,
        temperature=config_user.temperature,
        max_tokens=config_user.max_new_tokens,
    )

    model_user_generator = init_chat_model(
        model=config_user.model_name,
        model_provider=config_user.model_provider,
        temperature=config_user.temperature,
        max_tokens=config_user.max_new_tokens,
    )
    return model_agent, (model_user_parsing, model_user_generator)


def _save_record(response, output_path_jsonl):
    output_path_jsonl = Path(output_path_jsonl)
    output_path_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with output_path_jsonl.open("a", encoding="utf-8") as f:  # "a" = append line by line
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
        config_user=_config_user
    )
