"""Tests for the terminal-state messaging of the patience-budget middleware.

A *passing* ``submit_sql`` and a *forced* (out-of-budget) ``submit_sql`` both end
the episode, but they are different outcomes. The before-model guard used to emit
a single "User patience exhausted" note for every terminal state, which mislabels
a clean, successful submit as an exhaustion. These tests pin the distinct
behaviour for the two terminal reasons.
"""

import json
from types import SimpleNamespace

from langchain_core.messages import AIMessage, ToolMessage

from conversation2sql.eval_framework.agents.bird_baseline.agent_callback import (
    PATIENCE_BLOCKED,
    PATIENCE_SUBMIT_EXHAUSTED,
    PATIENCE_SUBMIT_PASSED,
    check_budget_limit,
    tool_wrapper_patience_and_submit,
)


def _guard(patience):
    return check_budget_limit.before_model({"updated_user_patience": patience}, None)


def _exhaustion_text(result) -> str:
    msgs = (result or {}).get("messages", [])
    return " ".join(str(m.content) for m in msgs).lower()


class TestCheckBudgetLimit:
    def test_passing_submit_ends_without_exhaustion_note(self):
        result = _guard(PATIENCE_SUBMIT_PASSED)
        assert result["jump_to"] == "end"
        assert "patience exhausted" not in _exhaustion_text(result)

    def test_forced_submit_keeps_exhaustion_note(self):
        result = _guard(PATIENCE_SUBMIT_EXHAUSTED)
        assert result["jump_to"] == "end"
        assert "patience exhausted" in _exhaustion_text(result)

    def test_blocked_state_does_not_end_the_run(self):
        assert _guard(PATIENCE_BLOCKED) is None

    def test_live_budget_does_not_end_the_run(self):
        assert _guard(5.0) is None


def _run_submit(*, passed: bool, patience: float):
    """Drive ``tool_wrapper_patience_and_submit`` for one ``submit_sql`` call."""
    response = ToolMessage(
        content=json.dumps({"passed": passed, "message": "stub"}),
        tool_call_id="call-1",
        name="submit_sql",
    )
    request = SimpleNamespace(
        tool_call={"name": "submit_sql", "id": "call-1"},
        runtime=SimpleNamespace(state={"updated_user_patience": patience}),
    )
    command = tool_wrapper_patience_and_submit.wrap_tool_call(
        request, lambda _req: response
    )
    return command.update


class TestSubmitTerminalSentinel:
    def test_passing_submit_marks_success_sentinel(self):
        update = _run_submit(passed=True, patience=10.0)
        assert update["updated_user_patience"] == PATIENCE_SUBMIT_PASSED

    def test_out_of_budget_failed_submit_marks_exhausted_sentinel(self):
        update = _run_submit(passed=False, patience=-1.0)
        assert update["updated_user_patience"] == PATIENCE_SUBMIT_EXHAUSTED

    def test_failed_submit_with_budget_left_is_not_terminal(self):
        # Wrong SQL but budget remains: the agent should be allowed to retry,
        # so patience is not driven to a terminal sentinel.
        update = _run_submit(passed=False, patience=10.0)
        assert "updated_user_patience" not in update
        assert update["tool_called_patience"]  # cost recorded for the retry
