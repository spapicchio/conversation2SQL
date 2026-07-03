from conversation2sql.eval_framework.main_pipe_workflow import (
    _resolve_baseline_settings,
)
from conversation2sql.eval_framework.agents import run_agent_maintenance


def test_maintenance_agent_baseline_resolves_to_runner():
    # maintenance_agent's entire point is ambiguity resolution through the
    # issue thread, so — unlike deep_agent — it defaults ambiguity on.
    forced_amb, runner, needs_user_sim = _resolve_baseline_settings("maintenance_agent")
    assert forced_amb is True
    assert needs_user_sim is True
    assert runner is run_agent_maintenance
