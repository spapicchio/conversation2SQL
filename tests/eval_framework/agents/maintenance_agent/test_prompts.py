from conversation2sql.eval_framework.agents.maintenance_agent.prompts import (
    build_maintenance_agent_messages,
)


def _render(**overrides):
    params = {
        "total_budget": 20,
        "amb_user_query": "How many active users?",
    }
    params.update(overrides)
    return build_maintenance_agent_messages(params)


def test_prompt_describes_workspace_layout():
    system = _render()[0]["content"]
    assert "ISSUE.md" in system
    assert "docs/database_overview.md" in system
    assert "docs/tables/" in system
    assert "docs/knowledge_base/" in system
    assert "queries/answer.sql" in system
    assert "tests/test_contract.py" in system


def test_prompt_describes_tools_and_costs():
    system = _render()[0]["content"]
    assert "write_query" in system
    assert "run_tests" in system
    assert "submit" in system
    assert "bash" in system
    assert "psql" in system


def test_prompt_includes_ticket_and_budget():
    msgs = _render()
    joined = " ".join(m["content"] for m in msgs)
    assert "How many active users?" in joined
    assert "20" in joined


def test_comment_on_issue_absent_by_default():
    system = _render()[0]["content"]
    assert "comment_on_issue" not in system


def test_comment_on_issue_surfaced_when_enabled():
    system = _render(enable_ask_user=True)[0]["content"]
    assert "comment_on_issue" in system
    assert "ambiguous" in system


def test_submit_is_described_as_silent_and_terminal():
    system = _render()[0]["content"]
    assert "no pass/fail feedback" in system
