from conversation2sql.eval_framework.agents.deep_agent.prompts import (
    build_deep_agent_messages,
)


def _render(**overrides):
    params = {
        "total_budget": 20,
        "amb_user_query": "How many active users?",
        "enable_fs_write": False,
        "enable_todos": False,
        "enable_subagents": False,
    }
    params.update(overrides)
    return build_deep_agent_messages(params)


def test_prompt_mentions_db_filesystem_paths():
    msgs = _render()
    system = msgs[0]["content"]
    assert "/db/schema.sql" in system
    assert "read_file" in system
    # No get_schema-style tools are advertised.
    assert "get_schema" not in system


def test_prompt_includes_user_query_and_budget():
    msgs = _render()
    joined = " ".join(m["content"] for m in msgs)
    assert "How many active users?" in joined
    assert "20" in joined


def test_todos_block_only_when_enabled():
    assert "write_todos" not in _render()[0]["content"]
    assert "write_todos" in _render(enable_todos=True)[0]["content"]
