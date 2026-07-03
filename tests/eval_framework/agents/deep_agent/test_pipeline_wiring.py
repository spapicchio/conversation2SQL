from conversation2sql.eval_framework.main_pipe_workflow import (
    _resolve_baseline_settings,
)
from conversation2sql.eval_framework.agents import run_agent_deep_agent


def test_deep_agent_baseline_resolves_to_runner():
    # deep_agent defaults to the clean (non-ambiguous) query with no ask_user /
    # user-sim, to validate bash + KB reading in isolation.
    forced_amb, runner, needs_user_sim = _resolve_baseline_settings("deep_agent")
    assert forced_amb is False
    assert needs_user_sim is False
    assert runner is run_agent_deep_agent
