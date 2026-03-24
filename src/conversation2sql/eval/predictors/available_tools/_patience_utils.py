"""Shared patience-tracking helpers for Bird-Interact tools.

User patience represents a finite budget of interaction turns available before
the simulated user loses patience and the episode terminates. Each tool call
deducts a cost from the remaining patience stored in ``runtime.state``.

Patience starts at ``PATIENCE_TOTAL`` (default: 10) and is decremented by each
tool invocation. When it reaches 0 the agent should wrap up — the user simulator
will stop responding cooperatively.

Usage inside any tool function::

    note = deduct_and_note(runtime, cost=1)
    return my_output + note

The returned ``note`` string must be appended to every tool response so the
language model is continuously aware of the remaining budget.

Constants:
    PATIENCE_TOTAL: The maximum (and initial) patience value for an episode.
"""

from langgraph.prebuilt import ToolRuntime

from conversation2sql.eval.interfaces import CustomAgentState, ToolUserContext

PATIENCE_TOTAL: int = 10


def deduct_and_note(runtime: ToolRuntime[ToolUserContext, CustomAgentState], cost: float) -> str:
    """Subtract *cost* from ``runtime.state['user_patience']`` and return the
    ``[SYSTEM NOTE: …]`` string that must be appended to every tool response."""
    runtime.state["user_patience"] = max(0.0, runtime.state["user_patience"] - cost)
    remaining = runtime.state["user_patience"]
    return f"\n[SYSTEM NOTE: Remaining user patience: {remaining}/{PATIENCE_TOTAL}]"
