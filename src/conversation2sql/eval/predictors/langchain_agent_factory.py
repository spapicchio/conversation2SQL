from __future__ import annotations

from functools import cache

from langchain.agents import AgentState as LangChainAgentState
from langchain.agents import create_agent
from langchain.agents.middleware import ModelCallLimitMiddleware, ToolCallLimitMiddleware
from langchain.chat_models import init_chat_model  # pyrefly: ignore
from langchain_core.language_models import BaseChatModel  # pyrefly: ignore
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel

from conversation2sql.config_input import ConfigPredictor
from conversation2sql.eval import Sample
from conversation2sql.eval.registry import tool_registry


# This TypedDict is used for the short memory into a conversation with tools
# The budget is the number tool interaction the model can do based on the user patience
# This is used to keep a state that can be modified in the conversation
class CustomAgentState(LangChainAgentState):
    user_patience: float  # must be set with invoke


# This is used to give context to the tool and cannot be modified
class ToolUserContext(BaseModel):
    """Used to define the LLM as a user. Must be specified if tool 'ask_user' is used."""
    # User simulator config
    template_params: dict
    user_simulator_prompt_folder: str
    user_simulator_system_prompt: str | None = None
    user_simulator_user_prompt: str

    sample: Sample | None = None  # the current sample being evaluated, for use in the user simulator prompts


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
