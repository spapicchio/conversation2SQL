"""deep_agent state: bird patience fields (no FS, no captured system prompt)."""
from __future__ import annotations

from conversation2sql.eval_framework.agents.bird_baseline.agent_code_state import (
    CustomAgentState,
)


class DeepAgentCustomState(CustomAgentState):
    """The three patience fields (from CustomAgentState), nothing more.

    The system prompt now travels as the leading message in the initial state
    (so it lands in the message history naturally), so there is no
    `captured_system_prompt` to reconstruct it for logging. There is also no
    deepagents `files` state — the catalog is on disk, explored via `bash`."""
