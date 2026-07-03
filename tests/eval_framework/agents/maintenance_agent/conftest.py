"""Shared fixtures for the maintenance_agent tests.

Re-expose the existing tools-test fixtures (`task_data`, `make_chat_model`,
`column_meanings`, `masked_agent_kb`) so the maintenance_agent tests can reuse
the same minimal `TaskData` construction without duplicating it — mirrors
tests/eval_framework/agents/deep_agent/conftest.py.
"""
from __future__ import annotations

from tests.eval_framework.tools.conftest import (  # noqa: F401
    column_meanings,
    make_chat_model,
    masked_agent_kb,
    task_data,
)
