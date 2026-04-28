import warnings

import litellm
from dotenv import load_dotenv

from conversation2sql.cli_parser import PydanticParser
from conversation2sql.config_input import ConfigReader, ConfigPredictor, ConfigPipeline, ConfigUserSimulator
from conversation2sql.eval_framework.main_pipe_workflow import workflow_evaluation_pipeline

load_dotenv('.env')

# Suppress litellm's "Provider List" stderr noise. It fires when the cost
# calculator calls get_llm_provider() with a bare model name (e.g. the
# OpenRouter response echoes "qwen/qwen3.6-flash" without the provider prefix).
# Side-effect: cost_usd in output will be 0 for OpenRouter models anyway.
litellm.suppress_debug_info = True

# LangChain + Pydantic v2 mismatch: AIMessage with list content doesn't match
# the discriminated-union types used internally; serialization still works.
warnings.filterwarnings("ignore", message="Pydantic serializer warnings", category=UserWarning)


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
