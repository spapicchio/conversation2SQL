from typing import Annotated

from langchain.agents import AgentState


def _patience_reducer(left: float | None, right: float) -> float:
    """Combine concurrent patience writes.

    Why: tool middleware emits a delta (-cost). With multiple parallel tool
    calls in one step, LangGraph's default LastValue channel rejects the
    second write — we need an Annotated reducer instead.
    How to apply: the first write (initial state) sets the absolute value;
    subsequent writes are deltas that are summed into it, clamped at -1.
    """
    if left is None:
        return right
    return max(left + right, -1)


class CustomAgentState(AgentState):
    initial_user_patience: float
    updated_user_patience: Annotated[float, _patience_reducer]
