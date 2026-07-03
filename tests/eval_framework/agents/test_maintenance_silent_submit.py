"""Tests for make_tool_wrapper_patience_and_submit_silent — the maintenance_agent
variant of the patience/submit wrapper whose submit tool carries no pass/fail
signal (see docs/superpowers/specs/2026-07-03-maintenance-agent-baseline-design.md).
Mirrors tests/eval_framework/agents/test_budget_terminal_state.py's patterns for
the bird_baseline factory.
"""
from types import SimpleNamespace

from langchain_core.messages import ToolMessage
from langgraph.types import Command

from conversation2sql.eval_framework.agents.bird_baseline.agent_callback import (
    PATIENCE_BLOCKED,
    PATIENCE_SUBMIT_EXHAUSTED,
    PATIENCE_SUBMIT_PASSED,
    make_tool_wrapper_patience_and_submit_silent,
)


def _wrapper():
    return make_tool_wrapper_patience_and_submit_silent(
        {"bash": 1.0, "submit": 3.0}, submit_tool_name="submit"
    )


def _run(wrapper, tool_name, patience, response):
    request = SimpleNamespace(
        tool_call={"name": tool_name, "id": "call-1"},
        runtime=SimpleNamespace(state={"updated_user_patience": patience}),
    )
    return wrapper.wrap_tool_call(request, lambda _req: response)


class TestSubmitAlwaysTerminal:
    def test_submit_with_budget_left_marks_passed_sentinel(self):
        response = ToolMessage(content="Submitted.", tool_call_id="call-1", name="submit")
        command = _run(_wrapper(), "submit", 5.0, response)
        assert command.update["updated_user_patience"] == PATIENCE_SUBMIT_PASSED

    def test_submit_after_block_marks_exhausted_sentinel(self):
        response = ToolMessage(content="Submitted.", tool_call_id="call-1", name="submit")
        command = _run(_wrapper(), "submit", -1.0, response)
        assert command.update["updated_user_patience"] == PATIENCE_SUBMIT_EXHAUSTED

    def test_submit_is_never_blocked_even_when_over_cost(self):
        # submit's own cost (3.0) exceeds remaining budget (1.0), but submit must
        # still run — the wrapper only blocks non-submit tools.
        response = ToolMessage(content="Submitted.", tool_call_id="call-1", name="submit")
        command = _run(_wrapper(), "submit", 1.0, response)
        assert command.update["updated_user_patience"] == PATIENCE_SUBMIT_PASSED


class TestNonSubmitToolBlocking:
    def test_blocked_tool_message_names_the_configured_submit_tool(self):
        response = ToolMessage(content="ignored", tool_call_id="call-1", name="bash")
        command = _run(_wrapper(), "bash", 0.5, response)
        assert command.update["updated_user_patience"] == PATIENCE_BLOCKED
        blocked_message = command.update["messages"][0]
        assert "call submit now" in blocked_message.content.lower()

    def test_live_budget_tool_records_cost(self):
        response = ToolMessage(content="output", tool_call_id="call-1", name="bash")
        command = _run(_wrapper(), "bash", 5.0, response)
        assert command.update["tool_called_patience"] == [1.0]


class TestCommandReturningTool:
    def test_command_update_is_preserved_and_cost_recorded(self):
        tool_msg = ToolMessage(content="wrote file", tool_call_id="c1", name="write_query")
        response = Command(update={"messages": [tool_msg], "files": {"answer.sql": "x"}})
        request = SimpleNamespace(
            tool_call={"name": "write_query", "id": "c1"},
            runtime=SimpleNamespace(state={"updated_user_patience": 5.0}),
        )
        out = _wrapper().wrap_tool_call(request, lambda _req: response)
        assert isinstance(out, Command)
        assert out.update["files"] == {"answer.sql": "x"}
        # "write_query" is absent from the {"bash": 1.0, "submit": 3.0} cost
        # table passed to _wrapper(), so it must default to 0.0.
        assert out.update["tool_called_patience"] == [0.0]
