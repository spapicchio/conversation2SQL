from conversation2sql.eval_framework.main_pipe_workflow import (
    _resolve_baseline_settings,
)
from conversation2sql.eval_framework.agents import run_agent_deep_agent


def test_deep_agent_baseline_resolves_to_runner():
    forced_amb, runner, needs_user_sim = _resolve_baseline_settings("deep_agent")
    assert forced_amb is True
    assert needs_user_sim is True
    assert runner is run_agent_deep_agent
