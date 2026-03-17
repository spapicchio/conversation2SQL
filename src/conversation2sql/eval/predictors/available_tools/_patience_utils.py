"""Shared patience-tracking helpers for Bird-Interact tools.

Usage inside any tool function:

    note = deduct_and_note(runtime, cost=1)
    return my_output + note
"""

from langgraph.prebuilt import ToolRuntime

from conversation2sql.eval.interfaces import UserContext
from conversation2sql.eval.predictors.langchain_agent_factory import AgentState

PATIENCE_TOTAL: int = 10


def deduct_and_note(runtime: ToolRuntime[UserContext, AgentState], cost: float) -> str:
    """Subtract *cost* from ``runtime.state['user_patience']`` and return the
    ``[SYSTEM NOTE: …]`` string that must be appended to every tool response."""
    runtime.state["user_patience"] = max(0.0, runtime.state["user_patience"] - cost)
    remaining = runtime.state["user_patience"]
    return f"\n[SYSTEM NOTE: Remaining user patience: {remaining}/{PATIENCE_TOTAL}]"
