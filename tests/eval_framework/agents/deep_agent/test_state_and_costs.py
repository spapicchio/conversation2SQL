from conversation2sql.eval_framework.agents.deep_agent.agent_code_state import (
    DeepAgentCustomState,
)


def test_state_drops_files_and_captured_prompt():
    # The deep_agent state is just the patience fields (from CustomAgentState):
    # no deepagents `files` state (the catalog is on disk), and no
    # `captured_system_prompt` — the system prompt now travels as the leading
    # message in the initial state, so it lands in the history without capture.
    keys = DeepAgentCustomState.__annotations__.keys()
    assert "captured_system_prompt" not in keys
    assert "files" not in keys
