"""Regression: the seeded patience budget must survive the channel reducer.

`updated_user_patience` carries a `min` reducer (to merge concurrent terminal
writes). LangGraph maps a reduced channel to a BinaryOperatorAggregate whose
initial value is `typ()`. If the annotated type were a bare `float`, that init
would be `0.0`, and seeding the channel with the real budget would merge as
`min(0.0, task_budget) == 0.0` — the conversation would start already exhausted
and the first tool call would be blocked. The union type keeps the channel
uninitialised so the seed is taken verbatim.
"""
from typing import Any, Optional

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END

from langchain.agents import create_agent
from langchain.agents.middleware import (
    ModelCallLimitMiddleware,
    ToolCallLimitMiddleware,
)
from conversation2sql.eval_framework.agents.bird_baseline.agent_callback import (
    tool_wrapper_patience_and_submit,
    wrap_model_append_tool_message,
    check_budget_limit,
)
from conversation2sql.eval_framework.agents.bird_baseline.agent_code_state import (
    CustomAgentState,
)


def test_seeded_budget_is_not_zeroed_by_reducer_init():
    """Directly seed the channel; it must read back the budget, not 0.0."""

    def node(state):
        return {}

    builder = StateGraph(CustomAgentState)
    builder.add_node("n", node)
    builder.add_edge(START, "n")
    builder.add_edge("n", END)
    graph = builder.compile()

    out = graph.invoke(
        {
            "messages": [],
            "initial_user_patience": 12,
            "updated_user_patience": 12,
            "tool_called_patience": [],
        }
    )
    assert out["updated_user_patience"] == 12


class _ScriptedModel(BaseChatModel):
    """Returns a pre-scripted AIMessage per invocation."""

    script: list
    idx: dict

    def __init__(self, script):
        super().__init__(script=script, idx={"i": 0})

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: Optional[list[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        i = self.idx["i"]
        self.idx["i"] = i + 1
        msg = self.script[min(i, len(self.script) - 1)]
        return ChatResult(generations=[ChatGeneration(message=msg)])

    def bind_tools(self, tools, **kwargs):
        return self


def _stub_tool(name):
    @tool(name)
    def f(query: str = "x") -> str:
        """stub"""
        return f"ok from {name}"

    return f


def test_first_tool_call_is_not_budget_blocked():
    """End-to-end through the real middleware: a full-budget agent must run its
    first tool instead of immediately hitting the 'Budget exhausted' block."""
    import json

    @tool("submit_sql")
    def submit_sql(sql: str = "SELECT 1") -> str:
        """submit"""
        return json.dumps({"passed": True, "message": "done"})

    script = [
        AIMessage(content="", tool_calls=[{"name": "get_schema", "args": {"query": "x"}, "id": "t1"}]),
        AIMessage(content="", tool_calls=[{"name": "execute_sql", "args": {"query": "x"}, "id": "t2"}]),
        AIMessage(content="", tool_calls=[{"name": "submit_sql", "args": {"sql": "SELECT 1"}, "id": "t3"}]),
    ]
    budget = 12
    agent = create_agent(
        _ScriptedModel(script),
        [_stub_tool("get_schema"), _stub_tool("execute_sql"), submit_sql],
        state_schema=CustomAgentState,
        middleware=[
            ModelCallLimitMiddleware(run_limit=budget + 5),
            ToolCallLimitMiddleware(run_limit=budget + 5, thread_limit=budget * 2),
            check_budget_limit,
            wrap_model_append_tool_message,
            tool_wrapper_patience_and_submit,
        ],
    )

    result = agent.invoke(
        {
            "messages": [HumanMessage(content="do it")],
            "initial_user_patience": budget,
            "updated_user_patience": budget,
            "tool_called_patience": [],
        }
    )

    contents = [str(m.content) for m in result["messages"]]
    assert any("ok from get_schema" in c for c in contents), contents
    assert not any("Budget exhausted" in c for c in contents), contents
