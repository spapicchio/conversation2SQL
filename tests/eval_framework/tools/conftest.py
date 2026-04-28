"""Shared fixtures for the eval_framework tools tests."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from conversation2sql.eval_framework.state import (
    ColumnMeaningEntry,
    ExternalKnowledgeEntry,
    TaskData,
)


@pytest.fixture
def column_meanings() -> dict[str, ColumnMeaningEntry]:
    return {
        "mydb|users|id": ColumnMeaningEntry(column_meaning="user primary key"),
        "mydb|users|name": ColumnMeaningEntry(
            column_meaning="display name",
            fields_meaning={"format": "free text"},
        ),
    }


@pytest.fixture
def masked_agent_kb() -> dict[str, ExternalKnowledgeEntry]:
    return {
        "active_user": ExternalKnowledgeEntry(
            id=1,
            knowledge="active_user",
            description="A user who logged in within 30 days.",
            definition="login_date >= NOW() - INTERVAL '30 days'",
            type="domain_knowledge",
            children_knowledge=[-1],
        ),
        "revenue": ExternalKnowledgeEntry(
            id=2,
            knowledge="revenue",
            description="Sum of completed orders.",
            definition="SUM(order.total) WHERE order.status = 'completed'",
            type="calculation_knowledge",
            children_knowledge=[-1],
        ),
    }


@pytest.fixture
def task_data(column_meanings, masked_agent_kb) -> TaskData:
    return TaskData(
        instance_id="task-1",
        selected_database="mydb",
        amb_user_query="give me users",
        sol_sql=["SELECT id FROM users;"],
        not_ambiguos_query="give me all user ids",
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


@pytest.fixture
def make_chat_model():
    """Build a ``BaseChatModel`` mock whose ``invoke`` returns a message with ``content``."""

    def _factory(content: str) -> MagicMock:
        model = MagicMock()
        model.invoke.return_value = MagicMock(content=content)
        return model

    return _factory
