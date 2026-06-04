from operator import add
from typing import Annotated

from langchain.agents import AgentState


def _keep_lowest_patience(current: float, incoming: float) -> float:
    """Reducer for `updated_user_patience`: keep the most-terminal (smallest) value.

    (A 2-arg wrapper rather than the `min` builtin, whose C signature LangGraph
    can't introspect as an `(a, b) -> c` reducer.)
    """
    return min(current, incoming)


class CustomAgentState(AgentState):
    initial_user_patience: float
    # Reducer keeps the smallest write. Patience is monotonically non-increasing
    # within a run, so this is a no-op for the normal single-write decrement path.
    # It also lets the channel tolerate >1 write in a single super-step, which
    # happens when the model emits multiple tool calls in one turn and several hit
    # the block (-1) / terminal-submit (-2) branches in
    # `tool_wrapper_patience_and_submit` (the terminal value wins). Without a
    # reducer this raises InvalidUpdateError (INVALID_CONCURRENT_GRAPH_UPDATE).
    updated_user_patience: Annotated[float, _keep_lowest_patience]
    tool_called_patience: Annotated[list[float | int], add]
