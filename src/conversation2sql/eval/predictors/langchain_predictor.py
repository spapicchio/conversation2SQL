
from conversation2sql.config_input import ConfigPredictor
from conversation2sql.eval import Sample, SampleWithPred, BasePredictor
from conversation2sql.eval.predictors.langchain_agent_factory import LangChainAgentFactory
from conversation2sql.eval.registry import predictor_registry
from conversation2sql.logger import get_logger


@predictor_registry.register
class LangChainPredictor(BasePredictor):
    """Returns a canned AIMessage for every input — no LLM call required."""

    def __init__(self, config_predictor: ConfigPredictor, *args, **kwargs):
        super().__init__(config_predictor, *args, **kwargs)
        self.agent_factory = LangChainAgentFactory(config_predictor)
        self.logger = get_logger(__name__)

    def predict(self, samples: list[Sample]) -> list[SampleWithPred]:
        prompts = [
            {'messages': sample.messages}
            for sample in samples if isinstance(sample.messages, list)
        ]
        if len(prompts) != len(samples):
            raise ValueError("All samples must have a string data_input for LangChainPredictor.")

        agent = self.agent_factory.create_agent()
        responses  = agent.batch(
            prompts,
            config={'max_concurrency': 5}
        )
        return [
            SampleWithPred(messages=response, **sample.model_dump(exclude='messages'))
            for sample, response in zip(samples, responses)
        ]
