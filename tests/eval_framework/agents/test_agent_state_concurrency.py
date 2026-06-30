"""Regression tests for concurrent writes to the patience channel.

When the agent emits more than one tool call in a single assistant turn,
LangGraph runs every tool call in the *same* super-step. Each call goes
through ``tool_wrapper_patience_and_submit``, and the block / terminal-submit
branches both write ``updated_user_patience``. If that channel is a plain
``LastValue`` it rejects the second write with ``InvalidUpdateError``
(INVALID_CONCURRENT_GRAPH_UPDATE). The field carries an ``Annotated[..., min]``
reducer so concurrent writes merge to the most-terminal (smallest) value.
"""

from langgraph.graph import StateGraph, START, END

from conversation2sql.eval_framework.agents.bird_baseline.agent_code_state import (
    CustomAgentState,
)


def _build_fan_out_graph():
    """Two nodes that both write ``updated_user_patience`` in one super-step."""

    def blocked_a(state):
        return {"updated_user_patience": -1}

    def blocked_b(state):
        return {"updated_user_patience": -2}

    builder = StateGraph(CustomAgentState)
    builder.add_node("blocked_a", blocked_a)
    builder.add_node("blocked_b", blocked_b)
    builder.add_edge(START, "blocked_a")
    builder.add_edge(START, "blocked_b")
    builder.add_edge("blocked_a", END)
    builder.add_edge("blocked_b", END)
    return builder.compile()


def test_concurrent_patience_writes_do_not_raise():
    graph = _build_fan_out_graph()

    result = graph.invoke(
        {
            "messages": [],
            "initial_user_patience": 26,
            "updated_user_patience": 26,
            "tool_called_patience": [],
        }
    )

    # The terminal value (-2) must win over the block value (-1): min reducer.
    assert result["updated_user_patience"] == -2


def test_concurrent_identical_block_writes_merge():
    """Two blocked tool calls in one step both write -1 -> merges to -1."""

    def blocked_a(state):
        return {"updated_user_patience": -1}

    def blocked_b(state):
        return {"updated_user_patience": -1}

    builder = StateGraph(CustomAgentState)
    builder.add_node("blocked_a", blocked_a)
    builder.add_node("blocked_b", blocked_b)
    builder.add_edge(START, "blocked_a")
    builder.add_edge(START, "blocked_b")
    builder.add_edge("blocked_a", END)
    builder.add_edge("blocked_b", END)
    graph = builder.compile()

    result = graph.invoke(
        {
            "messages": [],
            "initial_user_patience": 26,
            "updated_user_patience": 26,
            "tool_called_patience": [],
        }
    )

    assert result["updated_user_patience"] == -1
