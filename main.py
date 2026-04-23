from dotenv import load_dotenv

from conversation2sql.cli_parser import PydanticParser
from conversation2sql.config_input import ConfigReader, ConfigPredictor, ConfigPipeline, ConfigUserSimulator
from conversation2sql.eval_framework.main_pipe_workflow import workflow_evaluation_pipeline

load_dotenv('.env')


def main_launch_eval():
    parser = PydanticParser([ConfigPipeline, ConfigReader, ConfigPredictor, ConfigUserSimulator])
    config_pipeline, config_reader, config_predictor, config_user = parser.parse_args_and_config()

    workflow_evaluation_pipeline(
        config_pipeline,
        config_reader,
        config_predictor,
        config_user,
    )


if __name__ == "__main__":
    main_launch_eval()
