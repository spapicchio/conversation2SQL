from conversation2sql.eval_framework.agents.deep_agent.prompts import (
    build_deep_agent_messages,
)


def _render(**overrides):
    params = {
        "total_budget": 20,
        "amb_user_query": "How many active users?",
    }
    params.update(overrides)
    return build_deep_agent_messages(params)


def test_prompt_describes_bash_catalog_and_psql():
    msgs = _render()
    system = msgs[0]["content"]
    # The catalog lives in the cwd and is explored via the bash tool.
    assert "bash" in system
    assert "tables/" in system
    assert "knowledge_base/" in system
    assert "database_overview.md" in system
    # SQL is run via psql through the same tool.
    assert "psql" in system
    # No get_schema-style tools, and no legacy virtual-filesystem /db paths.
    assert "get_schema" not in system
    assert "/db/" not in system


def test_prompt_includes_user_query_and_budget():
    msgs = _render()
    joined = " ".join(m["content"] for m in msgs)
    assert "How many active users?" in joined
    assert "20" in joined


def test_subagent_tool_not_in_prompt():
    assert "delegate an isolated sub-task" not in _render()[0]["content"]
    assert "task:" not in _render()[0]["content"]
