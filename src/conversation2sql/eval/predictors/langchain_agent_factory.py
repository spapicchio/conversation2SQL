from __future__ import annotations

from functools import cache

from langchain.agents import create_agent
from langchain.agents.middleware import ModelCallLimitMiddleware, ToolCallLimitMiddleware
from langchain.chat_models import init_chat_model  # pyrefly: ignore
from langchain_core.language_models import BaseChatModel  # pyrefly: ignore
from langgraph.graph.state import CompiledStateGraph

from conversation2sql.config_input import ConfigPredictor
from conversation2sql.eval.interfaces import CustomAgentState, ToolUserContext
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
            state_schema=CustomAgentState,
            context_schema=ToolUserContext,  # immutable, cannot be changed in the
            middleware=[  # pyrefly: ignore
                ModelCallLimitMiddleware(run_limit=self.config_predictor.user_patience_budget + 5),
                ToolCallLimitMiddleware(
                    # Maximum tool calls per single invocation (one user message → response cycle).
                    # Resets with each new user message.
                    run_limit=self.config_predictor.user_patience_budget,
                    # Maximum tool calls across all runs in a thread (conversation).
                    # Persists across multiple invocations with the same thread ID.
                    # Requires a checkpointer to maintain state. None means no thread limit.
                    thread_limit=self.config_predictor.user_patience_budget * 3,
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


@cache
def get_cached_model(model_name, model_provider, temperature, max_tokens, top_p, top_k):
    return init_chat_model(model=model_name,
                           model_provider=model_provider,
                           temperature=temperature,
                           max_tokens=max_tokens,
                           top_p=top_p,
                           top_k=top_k)
