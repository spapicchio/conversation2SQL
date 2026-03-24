from langchain_community.adapters.openai import convert_message_to_dict

from conversation2sql.config_input import ConfigPredictor
from conversation2sql.eval import Sample, SampleWithPred, BasePredictor
from conversation2sql.eval.predictors.langchain_agent_factory import LangChainAgentFactory
from conversation2sql.eval.interfaces import CustomAgentState
from conversation2sql.eval.registry import predictor_registry
from conversation2sql.logger import get_logger


@predictor_registry.register
class LangChainPredictor(BasePredictor):
    """Returns a canned AIMessage for every input — no LLM call required."""

    def __init__(self, config_predictor: ConfigPredictor, *args, **kwargs):
        super().__init__(config_predictor, *args, **kwargs)
        self.config_predictor = config_predictor
        self.agent_factory = LangChainAgentFactory(config_predictor)
        self.logger = get_logger(__name__)

    def predict(self, samples: list[Sample]) -> list[SampleWithPred]:
        output = []
        agent = self.agent_factory.create_agent()

        for sample in samples:
            if not isinstance(sample.messages, list):
                raise ValueError("All samples must have a list of messages for LangChainPredictor.")
            agent_state: CustomAgentState = {
                'messages': sample.messages,  # pyrefly: ignore
                'user_patience': self.config_predictor.user_patience_budget,
            }
            response = agent.invoke(agent_state, context=sample.user_context)  # pyrefly: ignore
            output.append(
                SampleWithPred(
                    messages=[convert_message_to_dict(msg) for msg in response['messages']],
                    **sample.model_dump(exclude={'messages'})
                )
            )

        return output
