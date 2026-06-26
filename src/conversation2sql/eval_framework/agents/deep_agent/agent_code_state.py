"""Merged agent state: deepagents filesystem (`files`) + bird patience fields."""
from __future__ import annotations

from deepagents.middleware.filesystem import FilesystemState

from conversation2sql.eval_framework.agents.bird_baseline.agent_code_state import (
    CustomAgentState,
)


class DeepAgentCustomState(FilesystemState, CustomAgentState):
    """Both bases are TypedDicts extending AgentState; MRO merges their keys:
    `files` (FilesystemState) + the three patience fields (CustomAgentState)."""
