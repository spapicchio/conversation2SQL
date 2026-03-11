from langchain.agents import create_agent
from langchain.agents.middleware import ModelCallLimitMiddleware, ToolCallLimitMiddleware
from langchain.chat_models import init_chat_model  # pyrefly: ignore
from langchain_core.language_models import BaseChatModel  # pyrefly: ignore
from langgraph.graph.state import CompiledStateGraph

from conversation2sql.config_input import ConfigPredictor
from conversation2sql.eval.registry import tool_registry


class LangChainAgentFactory:
    def __init__(self, config_predictor: ConfigPredictor, *args, **kwargs):
        self.config_predictor = config_predictor

    def create_agent(self) -> CompiledStateGraph:
        """Build an agent with the given tools (all registered tools if None)."""
        model = self._create_model()
        tools = [tool_registry.get(tool) for tool in self.config_predictor.tool_names]
        agent = create_agent(
            model=model,
            tools=tools,
            middleware=[
                ModelCallLimitMiddleware(run_limit=3),
                ToolCallLimitMiddleware(
                    run_limit=1,
                    # Max 1 tool call per conversation round. If the assistant calls more than 1 tool, only the first one will be executed.
                    thread_limit=1  # Max 10 tool calls across all conversation
                ),
            ]
        )
        return agent

    def _create_model(self, *args, **kwargs) -> BaseChatModel:
        return init_chat_model(model=self.config_predictor.model_name,
                               model_provider=self.config_predictor.model_provider,
                               temperature=self.config_predictor.temperature,
                               max_tokens=self.config_predictor.max_new_tokens,
                               top_p=self.config_predictor.top_p,
                               # top_k=self.config_predictor.top_k
                               )
