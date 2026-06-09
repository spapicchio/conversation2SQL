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
