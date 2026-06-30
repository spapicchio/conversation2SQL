"""Shared fixtures for the deep_agent tests.

Re-expose the existing tools-test fixtures (`task_data`, `make_chat_model`,
`column_meanings`, `masked_agent_kb`) so the deep_agent tests can reuse the same
minimal `TaskData` construction without duplicating it.
"""
from __future__ import annotations

import pytest

from tests.eval_framework.tools.conftest import (  # noqa: F401
    column_meanings,
    make_chat_model,
    masked_agent_kb,
    task_data,
)


@pytest.fixture
def make_minimal_task_kwargs(column_meanings, masked_agent_kb):  # noqa: F811
    """Return a callable producing the minimum required `TaskData` kwargs."""

    def _factory(**overrides) -> dict:
        kwargs = dict(
            instance_id="task-1",
            selected_database="mydb",
            amb_user_query="give me users",
            sol_sql=["SELECT id FROM users;"],
            not_ambiguos_query="give me all user ids",
            task_question="give me all user ids",
            task_budget=10,
            db_dsn="postgresql://test:test@localhost:5432/mydb",
            ddl_database_schema="CREATE TABLE users (id INT, name TEXT);",
            masked_agent_kb=masked_agent_kb,
            column_meanings=column_meanings,
            user_query_ambiguity={"Amb": "user vs all_users"},
            clean_up_sqls=[],
            preprocess_sql=[],
            test_cases=[],
            category="Query",
            sql_query_conditions={"order": False},
        )
        kwargs.update(overrides)
        return kwargs

    return _factory
