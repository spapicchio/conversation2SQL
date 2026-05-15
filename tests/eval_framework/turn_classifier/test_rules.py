"""Unit tests for turn_classifier.rules — deterministic Level 2 classification."""
from __future__ import annotations

import pytest

from conversation2sql.eval_framework.turn_classifier.rules import (
    classify_level2,
    load_tool_categories,
)

# Inline fixture — no YAML dependency in unit tests
CATEGORIES = {
    "submit_sql": "SQL_SUBMISSION",
    "ask_user": "USER_INTERACTION",
    "execute_sql": "DB_EXPLORATION",
    "get_schema": "DB_EXPLORATION",
    "get_knowledge_definition": "KNOWLEDGE_LOOKUP",
}


def _tc(name: str) -> dict:
    """Build a minimal tool_call dict."""
    return {"tool_name": name, "arguments": {}, "tool_cost": 0}


# --- single-tool cases ---


def test_sql_submission():
    cat, tools = classify_level2([_tc("submit_sql")], CATEGORIES, has_content=False)
    assert cat == "SQL_SUBMISSION"
    assert tools == ["submit_sql"]


def test_user_interaction():
    cat, tools = classify_level2([_tc("ask_user")], CATEGORIES, has_content=False)
    assert cat == "USER_INTERACTION"
    assert tools == ["ask_user"]


def test_db_exploration_single():
    cat, tools = classify_level2([_tc("execute_sql")], CATEGORIES, has_content=True)
    assert cat == "DB_EXPLORATION"


def test_knowledge_lookup():
    cat, tools = classify_level2([_tc("get_knowledge_definition")], CATEGORIES, has_content=False)
    assert cat == "KNOWLEDGE_LOOKUP"


# --- no-tool cases ---


def test_no_action_empty_content():
    cat, tools = classify_level2([], CATEGORIES, has_content=False)
    assert cat == "NO_ACTION"
    assert tools == []


def test_text_only():
    cat, tools = classify_level2([], CATEGORIES, has_content=True)
    assert cat == "TEXT_ONLY"
    assert tools == []


# --- multi-tool cases ---


def test_db_exploration_multi():
    calls = [_tc("execute_sql"), _tc("get_schema")]
    cat, tools = classify_level2(calls, CATEGORIES, has_content=False)
    assert cat == "DB_EXPLORATION"
    assert set(tools) == {"execute_sql", "get_schema"}


def test_mixed():
    calls = [_tc("execute_sql"), _tc("get_knowledge_definition")]
    cat, tools = classify_level2(calls, CATEGORIES, has_content=False)
    assert cat == "MIXED"


# --- priority ---


def test_sql_submission_priority_over_user():
    calls = [_tc("submit_sql"), _tc("ask_user")]
    cat, _ = classify_level2(calls, CATEGORIES, has_content=False)
    assert cat == "SQL_SUBMISSION"


def test_sql_submission_priority_over_db():
    calls = [_tc("submit_sql"), _tc("execute_sql")]
    cat, _ = classify_level2(calls, CATEGORIES, has_content=False)
    assert cat == "SQL_SUBMISSION"


def test_user_interaction_priority_over_db():
    calls = [_tc("ask_user"), _tc("execute_sql")]
    cat, _ = classify_level2(calls, CATEGORIES, has_content=False)
    assert cat == "USER_INTERACTION"


# --- unknown tool ---


def test_unknown_tool():
    cat, tools = classify_level2([_tc("mystery_tool")], CATEGORIES, has_content=False)
    assert cat == "UNKNOWN"
    assert "mystery_tool" in tools


def test_load_tool_categories(tmp_path):
    yaml_content = "tool_categories:\n  submit_sql: SQL_SUBMISSION\n"
    p = tmp_path / "cats.yaml"
    p.write_text(yaml_content)
    cats = load_tool_categories(p)
    assert cats == {"submit_sql": "SQL_SUBMISSION"}
