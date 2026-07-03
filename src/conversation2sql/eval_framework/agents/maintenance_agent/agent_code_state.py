"""maintenance_agent state: bird patience fields, mirrors DeepAgentCustomState."""
from __future__ import annotations

from conversation2sql.eval_framework.agents.bird_baseline.agent_code_state import (
    CustomAgentState,
)


class MaintenanceAgentCustomState(CustomAgentState):
    """The three patience fields (from CustomAgentState), nothing more.

    Mirrors DeepAgentCustomState: the workspace lives on disk (ISSUE.md,
    docs/, queries/, tests/), not in agent state, and the system prompt
    travels as the leading message in the initial state.
    """
