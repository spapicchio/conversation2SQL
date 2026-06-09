import pytest

from conversation2sql.eval_framework.agents.bird_baseline.agent_code import (
    _select_db_tools,
)
from conversation2sql.eval_framework.state import TaskData


def _task(**flags) -> TaskData:
    # model_construct skips validation/required fields: _select_db_tools only
    # reads the two ablation booleans off the task.
    return TaskData.model_construct(
        enable_psql_console=flags.get("psql", False),
        enable_table_schema_tools=flags.get("table_tools", False),
    )


def test_default_db_tools():
    names = {t.name for t in _select_db_tools(_task())}
    assert names == {"execute_sql", "get_schema"}


def test_table_schema_tools_added():
    names = {t.name for t in _select_db_tools(_task(table_tools=True))}
    assert names == {"execute_sql", "get_schema", "get_table_names", "get_table_schema"}


def test_psql_console_replaces_db_tools():
    names = {t.name for t in _select_db_tools(_task(psql=True))}
    assert names == {"psql_console"}


def test_both_flags_raise():
    with pytest.raises(ValueError, match="mutually exclusive"):
        _select_db_tools(_task(psql=True, table_tools=True))


from conversation2sql.eval_framework.agents.bird_baseline.prompts import (
    build_bird_interact_agent_messages,
)


def _system_prompt(**params) -> str:
    base = {
        "total_budget": 20,
        "amb_user_query": "q",
        "enable_ask_user": False,
        "enable_table_schema_tools": False,
        "enable_psql_console": False,
    }
    base.update(params)
    return build_bird_interact_agent_messages(base)[0]["content"]


def test_prompt_default_lists_execute_sql_not_psql():
    text = _system_prompt()
    assert "execute_sql" in text
    assert "psql_console" not in text


def test_prompt_psql_mode_lists_psql_not_execute_sql():
    text = _system_prompt(enable_psql_console=True)
    assert "psql_console" in text
    assert "execute_sql" not in text
    assert "get_schema" not in text


def test_prompt_psql_mode_explains_how_to_explore():
    text = _system_prompt(enable_psql_console=True)
    assert "First explore the database with psql_console" in text
    assert "\\dt" in text and "\\d <table>" in text


def test_prompt_default_mode_keeps_generic_exploration_tip():
    text = _system_prompt()
    assert "First explore the database schema, column meanings" in text
    assert "psql_console" not in text
