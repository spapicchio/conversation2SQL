from conversation2sql.cli_parser import PydanticParser
from conversation2sql.config_input import ConfigReader, ConfigPredictor, ConfigScorer
from conversation2sql.eval.workflow import EvalPipelineInput
from conversation2sql.eval import workflow_evaluation_pipeline


def main_launch_eval():
    parser = PydanticParser([ConfigReader, ConfigPredictor, ConfigScorer])
    config_reader, config_pred, config_scorer = parser.parse_args_and_config()
    data_input = EvalPipelineInput(
        config_reader=config_reader,
        config_predictor=config_pred,
        config_scorer=config_scorer
    )
    data_output: EvalPipelineInput = workflow_evaluation_pipeline.invoke(  # pyrefly: ignore
        input=data_input,
        config={"configurable": {"thread_id": "eval-run"}},
    )
    print(data_output.dataset[0])


if __name__ == "__main__":
    main_launch_eval()
